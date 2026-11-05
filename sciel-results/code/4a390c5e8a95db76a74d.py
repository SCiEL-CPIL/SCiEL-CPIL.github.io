import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find connected components
    visited_comp = [False] * n
    components = []
    for i in range(n):
        if not visited_comp[i]:
            comp = []
            stack = [i]
            visited_comp[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for neighbor in adj[node]:
                    if not visited_comp[neighbor]:
                        visited_comp[neighbor] = True
                        stack.append(neighbor)
            components.append(comp)
    
    # Sort components by size, largest first
    components.sort(key=len, reverse=True)
    
    best_path = []
    deadline = time.time() + 50.0 # Safety margin
    
    # We focus on the largest component (Giant Component)
    if not components:
        return [0]
        
    target_comp = components[0]
    target_set = set(target_comp)
    
    # Helper to count unvisited neighbors in the component
    def get_unvisited_neighbors(node, visited):
        return [nb for nb in adj[node] if nb in target_set and nb not in visited]

    # Try to find a Hamiltonian path in the target component using DFS + Warnsdorff's Rule
    # If that fails or is too slow, we might need a fallback, but let's try this first.
    
    # Optimization: If the component is small, we can be more exhaustive.
    # If it's large, Warnsdorff is key.
    
    import random
    random.seed(42) # Deterministic for reproducibility in testing, but we can shuffle starts
    
    best_len = 0
    best_path_found = []
    
    # Try starting from each node in the component? 
    # Or just a few? With Warnsdorff, starting from any node might work.
    # Let's try all nodes if the component is small (< 50), otherwise just a few or all with time check.
    
    starts = target_comp
    if len(target_comp) > 30:
        # Shuffle to diversify, but keep deterministic? 
        # Actually, Warnsdorff is deterministic given a start. 
        # Different starts might lead to different path lengths if the graph isn't Hamiltonian.
        # Let's try all starts if time permits.
        pass
    
    for start in starts:
        if time.time() > deadline:
            break
            
        visited = {start}
        path = [start]
        
        # Recursive DFS with backtracking
        # We use a stack to avoid recursion limit issues, but recursion is cleaner.
        # N=100, recursion depth 100 is fine.
        
        def dfs(node, visited, path):
            nonlocal best_len, best_path_found
            
            if len(path) > best_len:
                best_len = len(path)
                best_path_found = path[:]
                
            # If we have visited all nodes in the component, we are done
            if len(visited) == len(target_comp):
                return True # Found Hamiltonian path
            
            if time.time() > deadline:
                return False # Time out
            
            neighbors = get_unvisited_neighbors(node, visited)
            
            if not neighbors:
                return False # Dead end
            
            # Warnsdorff's Rule: Sort neighbors by number of unvisited neighbors
            # The one with the fewest unvisited neighbors is the best candidate
            # (to avoid creating isolated nodes)
            neighbors.sort(key=lambda x: len(get_unvisited_neighbors(x, visited)))
            
            for nb in neighbors:
                visited.add(nb)
                path.append(nb)
                
                if dfs(nb, visited, path):
                    return True
                
                path.pop()
                visited.remove(nb)
                
                # If time is running out, return early
                if time.time() > deadline:
                    return False
                    
            return False
        
        dfs(start, visited, path)
        
        if best_len == len(target_comp):
            break # Found Hamiltonian path, can't do better
            
    if best_path_found:
        return best_path_found
    
    # Fallback: If DFS didn't find anything good (unlikely), return a simple greedy path
    # Greedy: Start from node with min degree, always go to neighbor with min degree
    if not best_path:
         # Just return the first component's nodes in BFS order? No, must be a path.
         # Let's do a simple greedy walk
         visited = set()
         current = target_comp[0]
         path = [current]
         visited.add(current)
         
         while True:
             if time.time() > deadline:
                 break
             neighbors = [nb for nb in adj[current] if nb in target_set and nb not in visited]
             if not neighbors:
                 break
             # Pick neighbor with smallest degree (greedy)
             neighbors.sort(key=lambda x: len(adj[x]))
             current = neighbors[0]
             path.append(current)
             visited.add(current)
             
         if path:
             return path
             
    return [0] # Should not reach here if graph is non-empty

# EVOLVE-BLOCK-END
