import time
import random
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    def remaining():
        return deadline - time.time()

    if k != 3:
        return None

    n_min = 384
    n_max = 1000
    
    rng = random.Random(42)

    def get_girth(n, adj):
        """
        Returns the girth of the graph.
        """
        min_girth = float('inf')
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    elif dist[v] >= dist[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                        # Early exit if we found a very small cycle?
                        # We only care if it's < g.
                        if min_girth < g:
                            return min_girth
        return min_girth if min_girth != float('inf') else g

    def generate_random_3_reg(n):
        """
        Generates a random 3-regular graph using configuration model.
        Returns adj list or None if failed (e.g. multi-edges).
        """
        stubs = []
        for i in range(n):
            stubs.extend([i, i, i])
        
        for _ in range(100):
            rng.shuffle(stubs)
            adj = [[] for _ in range(n)]
            edges = set()
            valid = True
            for i in range(0, n * 3, 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in edges:
                    valid = False
                    break
                edges.add(key)
                adj[u].append(v)
                adj[v].append(u)
            if valid:
                return adj
        return None

    def sa_optimize(n, max_time):
        """
        Runs Simulated Annealing to minimize girth (maximize min cycle length).
        Returns (n, edges) if girth >= g, else None.
        """
        adj = generate_random_3_reg(n)
        if adj is None:
            return None
            
        # Initial cost
        current_girth = get_girth(n, adj)
        if current_girth >= g:
            return (n, [(min(u,v), max(u,v)) for u in range(n) for v in adj[u] if u < v])
            
        T = 1.0
        cooling = 0.995
        start = time.time()
        
        while time.time() - start < max_time:
            # Pick two random edges
            # To pick random edges, we can pick random u, then random v in adj[u]
            # But we need two *distinct* edges.
            
            # Collect all edges
            # This is O(n). We can do it once? No, adj changes.
            # We can pick u1, v1 = random edge.
            # Then pick u2, v2 = random edge, ensuring (u2,v2) != (u1,v1) and (u2,v2) != (v1,u1).
            
            u1 = rng.randrange(n)
            if not adj[u1]: continue
            v1 = rng.choice(adj[u1])
            
            u2 = rng.randrange(n)
            if not adj[u2]: continue
            v2 = rng.choice(adj[u2])
            
            # Check if edges are same or share vertices in a way that invalidates 2-opt
            # Edges (u1,v1) and (u2,v2)
            # If they share a vertex, e.g., v1 == u2, the swap is different.
            # Standard 2-opt on regular graphs:
            # Remove (u1,v1) and (u2,v2).
            # Add (u1,u2) and (v1,v2) OR (u1,v2) and (v1,u2).
            # Must ensure new edges don't exist and no loops.
            
            e1 = (min(u1,v1), max(u1,v1))
            e2 = (min(u2,v2), max(u2,v2))
            
            if e1 == e2:
                continue
            
            # Try swap 1: (u1,u2) and (v1,v2)
            valid_swap = False
            new_edges = []
            
            # Check validity of (u1, u2)
            if u1 != u2 and u2 not in adj[u1]:
                # Check validity of (v1, v2)
                if v1 != v2 and v2 not in adj[v1]:
                    # Also ensure (u1,u2) and (v1,v2) are distinct
                    if (min(u1,u2), max(u1,u2)) != (min(v1,v2), max(v1,v2)):
                        valid_swap = True
                        new_edges = [(u1, u2), (v1, v2)]
            
            if not valid_swap:
                # Try swap 2: (u1,v2) and (v1,u2)
                if u1 != v2 and v2 not in adj[u1]:
                    if v1 != u2 and u2 not in adj[v1]:
                        if (min(u1,v2), max(u1,v2)) != (min(v1,u2), max(v1,u2)):
                            valid_swap = True
                            new_edges = [(u1, v2), (v1, u2)]
            
            if not valid_swap:
                T *= cooling
                continue
                
            # Apply swap
            # Remove old edges
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            
            # Add new edges
            for a, b in new_edges:
                adj[a].append(b)
                adj[b].append(a)
                
            new_girth = get_girth(n, adj)
            
            if new_girth >= g:
                # Success
                return (n, [(min(u,v), max(u,v)) for u in range(n) for v in adj[u] if u < v])
                
            # Cost difference
            # We want to maximize girth.
            # Let cost = 100 - girth (lower is better)
            old_cost = 100 - current_girth
            new_cost = 100 - new_girth
            
            delta = new_cost - old_cost
            if delta < 0 or rng.random() < math.exp(-delta / T):
                current_girth = new_girth
            else:
                # Revert
                for a, b in new_edges:
                    adj[a].remove(b)
                    adj[b].remove(a)
                adj[u1].append(v1)
                adj[v1].append(u1)
                adj[u2].append(v2)
                adj[v2].append(u2)
                
            T *= cooling
            
        return None

    for n in range(n_min, n_max + 1, 2):
        if remaining() < 15.0:
            break
        
        # Budget for this n
        budget = min(10.0, remaining() - 10.0)
        if budget <= 0:
            break
            
        result = sa_optimize(n, budget)
        if result is not None:
            return result

    return None
