import random
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    original_edges = set()
    for u, v in P['edges']:
        original_edges.add((min(u, v), max(u, v)))
    
    best_cycles = []
    deadline = time.time() + 50.0 # Safety margin
    
    # We will try to find 3 cycles.
    # We use a randomized restart strategy.
    # Each restart, we shuffle the neighbor lists slightly or use a random seed.
    
    for trial in range(200): # Enough trials
        if time.time() > deadline:
            break
            
        # Reset graph
        adj = {i: set() for i in range(n)}
        for u, v in P['edges']:
            adj[u].add(v)
            adj[v].add(u)
        
        cycles_found = []
        success = True
        
        for cycle_idx in range(3):
            if time.time() > deadline:
                success = False
                break
                
            # Find one Hamiltonian cycle in current adj
            # Use randomized DFS
            found_cycle = None
            
            # Randomize start vertex? Or fix 0 for symmetry breaking?
            # Fixing start to 0 reduces symmetry.
            start = 0
            
            # Shuffle neighbors for randomization
            # We can create a list of lists for neighbors to shuffle
            adj_list = {i: list(adj[i]) for i in range(n)}
            for i in range(n):
                random.shuffle(adj_list[i])
                
            # DFS with backtracking
            path = [start]
            visited = {start}
            
            # We need a recursive function or iterative with stack
            # Recursive is easier to write, depth 21 is fine.
            
            def dfs(u, depth):
                if depth == n:
                    if start in adj_list[u]:
                        return path[:]
                    return None
                for v in adj_list[u]:
                    if v == start and depth < n - 1:
                        continue
                    if v not in visited:
                        visited.add(v)
                        path.append(v)
                        res = dfs(v, depth + 1)
                        if res:
                            return res
                        path.pop()
                        visited.remove(v)
                return None
            
            # To avoid deep recursion limits or speed issues, 
            # we can add a simple heuristic: 
            # Sort neighbors by degree in the *remaining* graph?
            # That's expensive to maintain. 
            # Simple randomization usually works for n=21.
            
            # Let's try the DFS
            found_cycle = dfs(start, 1)
            
            if found_cycle is None:
                # Maybe try a different start? Or just fail this trial.
                # With randomization, different trials will have different adj_list orders.
                success = False
                break
            else:
                cycles_found.append(found_cycle)
                # Remove edges of this cycle from adj
                for i in range(len(found_cycle)):
                    u = found_cycle[i]
                    v = found_cycle[(i + 1) % len(found_cycle)]
                    adj[u].remove(v)
                    adj[v].remove(u)
                    # Update adj_list for next iteration? 
                    # We rebuild adj_list at the start of the loop for the next cycle? 
                    # No, we are inside the cycle_idx loop.
                    # We need to update the adj_list used for the NEXT cycle search.
                    # The dfs uses adj_list. We should rebuild adj_list after removing edges.
        
        if success and len(cycles_found) > len(best_cycles):
            best_cycles = cycles_found
            
        if len(best_cycles) == 3:
            break
            
    # Convert to required format: list of lists of ints
    # Check types
    result = []
    for c in best_cycles:
        result.append([int(v) for v in c])
        
    return result
