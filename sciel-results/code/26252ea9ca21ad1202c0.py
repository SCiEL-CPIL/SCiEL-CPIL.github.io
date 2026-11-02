import random
import time
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for safety

    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def generate_random_regular_graph(n, k):
        # Stub matching method
        while True:
            stubs = []
            for i in range(n):
                stubs.extend([i] * k)
            random.shuffle(stubs)
            adj = [set() for _ in range(n)]
            edges = []
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
            if valid:
                return adj, edges

    def find_short_cycle(adj, n, g):
        # BFS from each node to find a cycle of length < g
        # We only need to search up to depth g//2
        max_depth = g // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            parent[start] = -2 # Mark start
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and dist[v] < dist[u]:
                        # Found a back edge to a closer node
                        # Cycle length is dist[u] + dist[v] + 1
                        # If this is < g, we have a short cycle
                        # We need to reconstruct the cycle to perform a switch
                        if dist[u] + dist[v] + 1 < g:
                            # Reconstruct path from u to LCA
                            path_u = []
                            curr = u
                            while curr != v and curr != -2:
                                path_u.append(curr)
                                curr = parent[curr]
                            # path_u is u -> ... -> v (if v is ancestor)
                            # Actually, v is an ancestor of u if dist[v] < dist[u]
                            # The cycle is u -> ... -> v -> (edge) -> u
                            # Wait, the edge is (u, v).
                            # The path is u -> parent[u] -> ... -> v
                            # Length of path is dist[u] - dist[v]
                            # Total cycle length = (dist[u] - dist[v]) + 1
                            # This is < g by check above.
                            
                            # We have a cycle. Let's extract edges.
                            cycle_edges = []
                            curr = u
                            while curr != v:
                                cycle_edges.append((curr, parent[curr]))
                                curr = parent[curr]
                            cycle_edges.append((u, v))
                            
                            return cycle_edges
        return None

    def try_find_graph(n, k, g, time_limit):
        start_t = time.time()
        # Try a few random instances
        for attempt in range(5):
            if time.time() - start_t > time_limit:
                break
            
            adj, edges = generate_random_regular_graph(n, k)
            
            # Local search
            while time.time() - start_t < time_limit:
                cycle = find_short_cycle(adj, n, g)
                if not cycle:
                    return n, edges
                
                # Perform a 2-switch to break the cycle
                # Pick an edge (u, v) in the cycle
                # Pick another edge (x, y) not in the cycle and not sharing vertices with u, v
                # Replace (u, v) and (x, y) with (u, x) and (v, y)
                
                edge_to_break = random.choice(cycle)
                u, v = edge_to_break
                
                # Find a candidate edge (x, y)
                # To be efficient, we can pick a random edge and check
                # We need x, y not in {u, v} and (u, x), (v, y) not in adj
                # And (u, y), (v, x) not in adj (if we choose that pairing)
                
                swapped = False
                for _ in range(100): # Limit random tries for swap
                    x, y = random.choice(edges)
                    if x in (u, v) or y in (u, v):
                        continue
                    
                    # Check if new edges exist
                    if x in adj[u] or y in adj[v]:
                        continue
                    if y in adj[u] or x in adj[v]:
                        continue
                    
                    # Perform swap
                    # Remove (u, v) and (x, y)
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[x].remove(y)
                    adj[y].remove(x)
                    
                    # Add (u, x) and (v, y)
                    adj[u].add(x)
                    adj[x].add(u)
                    adj[v].add(y)
                    adj[y].add(v)
                    
                    # Update edges list
                    # This is O(E), which is slow. 
                    # For simplicity and given time constraints, we might skip updating the list 
                    # and just rely on adj for checks, but we need the list for return.
                    # We can rebuild edges at the end if successful.
                    # For now, we just modify adj.
                    swapped = True
                    break
                
                if not swapped:
                    # If we can't find a valid swap easily, break and try new random graph
                    break
            
            # If we exited the while loop without returning, check if we have a valid graph
            # The loop breaks if time runs out or if we fail to swap.
            # If we failed to swap, we should probably restart.
            # If time runs out, we should check if current adj is valid.
            if time.time() - start_t > time_limit:
                break
            
            # Re-check girth
            if not find_short_cycle(adj, n, g):
                # Rebuild edges list
                final_edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            final_edges.append((u, v))
                return n, final_edges

    # Strategy: Try n values starting from a lower bound
    # Moore bound for (5,9) is 428.
    # Known best is 1116.
    # Let's try a range of n.
    # We will allocate time budget to different n.
    
    # If k=5, g=9
    # Try n from 1000 to 6000 in steps.
    # But generating and searching is expensive.
    
    # Let's try a few specific n values.
    # Start with n = 1100, 1200, 1500, 2000, 3000, 5000.
    
    n_candidates = [1000, 1100, 1200, 1500, 2000, 3000, 5000]
    
    # Filter n_candidates to be at least Moore bound and even (if k is odd, n*k must be even, so n must be even)
    if k % 2 == 1:
        n_candidates = [n for n in n_candidates if n % 2 == 0]
    
    # Sort by n
    n_candidates.sort()
    
    # Allocate time: 10s per candidate?
    # Total 115s. 7 candidates * 10s = 70s. Leaves margin.
    
    for n in n_candidates:
        if time.time() > deadline - 10: # Stop if less than 10s left
            break
        
        # Try to find a graph of size n
        # Time limit for this n: 10s
        result = try_find_graph(n, k, g, 10)
        if result:
            return result
    
    # Fallback to LUW
    return luw_construct(k, g)

if __name__ == "__main__":
    # For local testing
    n, edges = construct(5, 9)
    print(f"Found graph with n={n}, edges={len(edges)}")
