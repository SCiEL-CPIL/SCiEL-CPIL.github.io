"""Combined approach: set-based adjacency with repair local search for K4-free, alpha<=5 graphs."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115  # safety margin before 150s kill
    random.seed(42)
    
    def find_k4(adj, n):
        """Find one K4. adj is list of sets. Returns tuple or None."""
        for i in range(n):
            for j in adj[i]:
                if j <= i:
                    continue
                common = adj[i] & adj[j]
                common = {k for k in common if k > j}
                for k in common:
                    for l in common:
                        if l > k and l in adj[k]:
                            return (i, j, k, l)
        return None

    def find_indep6(adj, n):
        """Find one independent 6-set. Returns tuple or None."""
        # Use branching with pruning
        def backtrack(remaining, current):
            if len(current) >= 6:
                return current
            if not remaining:
                return None
            if len(current) + len(remaining) < 6:
                return None
            # Pick vertex with highest degree in remaining for better pruning
            v = max(remaining, key=lambda x: len(adj[x] & remaining))
            # Try including v
            result = backtrack(remaining - {v} - adj[v], current + [v])
            if result:
                return result
            # Try excluding v
            return backtrack(remaining - {v}, current)
        
        return backtrack(set(range(n)), [])

    def is_valid(adj, n):
        """Check K4-free and alpha < 6."""
        if find_k4(adj, n) is not None:
            return False
        if find_indep6(adj, n) is not None:
            return False
        return True

    def make_adj_from_edges(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append([i, j])
        return edges

    # Start with Paley graph on 17 vertices
    def paley_graph(n):
        squares = {x * x % 17 for x in range(1, 17)}
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % 17 in squares:
                    adj[i].add(j)
                    adj[j].add(i)
        return adj

    # Initial valid graph
    n = 17
    adj = paley_graph(n)
    best_n = n
    best_adj = [set(s) for s in adj]
    
    if not is_valid(adj, n):
        # Fallback to a small valid graph
        n = 8
        adj = [set() for _ in range(n)]
        for i in range(4):
            for j in range(4, 8):
                adj[i].add(j)
                adj[j].add(i)
        best_n = n
        best_adj = [set(s) for s in adj]

    # Main search loop: try to grow and repair
    while time.time() < deadline:
        # Try to add a vertex
        if n < 40:
            new_n = n + 1
            success = False
            for attempt in range(100):
                if time.time() > deadline - 5:
                    break
                
                trial_adj = [set(s) for s in adj]
                trial_adj.append(set())
                
                # Randomly connect new vertex
                for i in range(n):
                    if random.random() < 0.5:
                        trial_adj[i].add(new_n - 1)
                        trial_adj[new_n - 1].add(i)
                
                if is_valid(trial_adj, new_n):
                    adj = trial_adj
                    n = new_n
                    best_n = n
                    best_adj = [set(s) for s in adj]
                    success = True
                    break
            
            if success:
                continue
        
        # Local search: repair violations
        improved = False
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            
            # Check for K4
            k4 = find_k4(adj, n)
            if k4:
                i, j, k, l = k4
                pair_options = [(i, j), (i, k), (i, l), (j, k), (j, l), (k, l)]
                u, v = random.choice(pair_options)
                if v in adj[u]:
                    adj[u].discard(v)
                    adj[v].discard(u)
                improved = True
                continue
            
            # Check for independent 6-set
            indep6 = find_indep6(adj, n)
            if indep6:
                u, v = random.sample(list(indep6), 2)
                adj[u].add(v)
                adj[v].add(u)
                improved = True
                continue
            
            # No violations - try to grow
            if n < 40:
                new_n = n + 1
                trial_adj = [set(s) for s in adj]
                trial_adj.append(set())
                for i in range(n):
                    if random.random() < 0.5:
                        trial_adj[i].add(new_n - 1)
                        trial_adj[new_n - 1].add(i)
                
                if is_valid(trial_adj, new_n):
                    adj = trial_adj
                    n = new_n
                    best_n = n
                    best_adj = [set(s) for s in adj]
                    improved = True
                    break
            else:
                break
        
        if not improved and n >= best_n:
            # Try random edge flips to escape local optimum
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                
                if v in adj[u]:
                    adj[u].discard(v)
                    adj[v].discard(u)
                else:
                    adj[u].add(v)
                    adj[v].add(u)
                
                if is_valid(adj, n):
                    # Try to grow
                    if n < 40:
                        new_n = n + 1
                        trial_adj = [set(s) for s in adj]
                        trial_adj.append(set())
                        for i in range(n):
                            if random.random() < 0.5:
                                trial_adj[i].add(new_n - 1)
                                trial_adj[new_n - 1].add(i)
                        
                        if is_valid(trial_adj, new_n):
                            adj = trial_adj
                            n = new_n
                            best_n = n
                            best_adj = [set(s) for s in adj]
                            break
                else:
                    # Revert
                    if v in adj[u]:
                        adj[u].add(v)
                        adj[v].add(u)
                    else:
                        adj[u].discard(v)
                        adj[v].discard(u)

    # Return best result
    edges = edges_from_adj(best_adj, best_n)
    return {'n': best_n, 'edges': edges}
# EVOLVE-BLOCK-END
