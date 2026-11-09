import time
import random
import sys

sys.setrecursionlimit(2000)

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Find largest connected component
    visited = [False] * n
    comp_sizes = []
    for i in range(n):
        if not visited[i]:
            # BFS/DFS to find component
            stack = [i]
            visited[i] = True
            comp = []
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        stack.append(nei)
            comp_sizes.append((len(comp), comp))
            
    if not comp_sizes:
        return [0] # Fallback, though graph is non-empty
        
    # Sort by size, get largest
    comp_sizes.sort(reverse=True)
    largest_comp = comp_sizes[0][1]
    comp_set = set(largest_comp)
    
    # Map vertices to indices in the component for faster lookup? 
    # Or just use the original indices.
    
    best_path = []
    start_time = time.time()
    time_limit = 85.0 # 5s margin
    
    # We will try to find a path in the largest component.
    # Heuristic: DFS with Warnsdorff's rule.
    
    # To improve search, we can try starting from different nodes.
    # Shuffle the component nodes for random restarts.
    import random
    rng = random.Random(12345) # Fixed seed for reproducibility within the run? 
    # Actually, the task says "random.Random(748)" is used for graph generation.
    # We can use any random seed for our search strategy.
    
    nodes_to_try = largest_comp[:]
    rng.shuffle(nodes_to_try)
    
    # We can also try to extend a path found by a simple greedy algorithm.
    # But let's stick to DFS.
    
    def dfs(node, path, visited_set):
        nonlocal best_path
        nonlocal start_time
        nonlocal time_limit
        
        if time.time() - start_time > time_limit:
            return False # Timeout signal
        
        if len(path) > len(best_path):
            best_path = path[:]
            
        # Check for completion or dead end
        # If we visited all nodes in the component, we are done
        if len(path) == len(comp_set):
            return True # Found Hamiltonian path!
            
        # Get unvisited neighbors
        candidates = [nei for nei in adj[node] if nei not in visited_set]
        
        if not candidates:
            return False
            
        # Warnsdorff's rule: sort by number of unvisited neighbors
        # We want to visit the node with the fewest options first to avoid trapping
        def score(nei):
            # Count unvisited neighbors of nei
            unvisited_nei = len([x for x in adj[nei] if x not in visited_set])
            return (unvisited_nei, nei)
            
        candidates.sort(key=score)
        
        for nei in candidates:
            visited_set.add(nei)
            path.append(nei)
            
            # Recurse
            # We can prune if we know we can't beat the best path
            # But for now, let's just go deep.
            dfs(nei, path, visited_set)
            
            path.pop()
            visited_set.remove(nei)
            
            if time.time() - start_time > time_limit:
                return False
                
    # Run DFS from each start node in the shuffled list until time runs out
    for start_node in nodes_to_try:
        if time.time() - start_time > time_limit:
            break
            
        visited_set = {start_node}
        path = [start_node]
        dfs(start_node, path, visited_set)
        
        # If we found a path covering the whole component, we can stop
        if len(best_path) == len(comp_set):
            break
            
    return best_path
