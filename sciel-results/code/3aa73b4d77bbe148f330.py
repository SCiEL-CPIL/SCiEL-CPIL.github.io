import time
import sys
import random

# Set recursion limit to be safe, though we manage depth manually via stack or recursion
sys.setrecursionlimit(20000)

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
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
                for nei in adj[node]:
                    if not visited_comp[nei]:
                        visited_comp[nei] = True
                        stack.append(nei)
            components.append(comp)
            
    # Select largest component
    if not components:
        return [0] # Fallback, shouldn't happen
        
    largest_comp = max(components, key=len)
    comp_nodes = set(largest_comp)
    comp_size = len(largest_comp)
    
    # If component is small, we can be more exhaustive
    # If large, we rely on heuristics and pruning
    
    best_path = []
    best_len = 0
    start_time = time.time()
    time_limit = 50.0 # Keep well under 60s
    
    # Initial greedy pass to get a decent lower bound
    for start in largest_comp:
        path = [start]
        seen = {start}
        while True:
            last = path[-1]
            neighbors = [x for x in adj[last] if x not in seen]
            if not neighbors:
                break
            # Pick neighbor with fewest unvisited neighbors (Warnsdorff-like) to avoid traps? 
            # Or just pick one. Let's pick one that keeps options open (most unvisited neighbors)
            # Actually, for longest path, Warnsdorff (min degree) is standard for Hamiltonian.
            # Let's try min degree unvisited.
            def unvisited_deg(node):
                return len([x for x in adj[node] if x not in seen])
            
            next_node = min(neighbors, key=lambda x: (unvisited_deg(x), x))
            path.append(next_node)
            seen.add(next_node)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
            
    # Now, try to improve with DFS
    # We only search within the largest component
    
    # Sort neighbors for each node in the component to have a consistent order
    # We can sort by degree or just keep as is
    
    # DFS function
    # To avoid recursion depth issues and overhead, we could use iterative, 
    # but for N=100, recursion is fine if limit is set.
    
    def dfs(path, seen):
        nonlocal best_path, best_len
        
        # Pruning: if current path + remaining nodes in component <= best_len, stop
        # Note: 'remaining' is comp_size - len(path)
        # This is an optimistic bound.
        if len(path) + (comp_size - len(path)) <= best_len:
            return
            
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
            
        # Check time
        if time.time() - start_time > time_limit:
            return
            
        last = path[-1]
        # Get unvisited neighbors in the component
        candidates = [x for x in adj[last] if x not in seen and x in comp_nodes]
        
        # Heuristic: Sort candidates.
        # Try neighbors that have fewer unvisited neighbors first (Warnsdorff)
        # This tends to lead to Hamiltonian paths if they exist.
        # For longest path, it's also a good bet to force the path to "close" or extend into dead ends late.
        
        # Calculate unvisited degree for sorting
        def get_unvisited_deg(node):
            return len([x for x in adj[node] if x not in seen and x in comp_nodes])
        
        candidates.sort(key=lambda x: (get_unvisited_deg(x), x))
        
        for next_node in candidates:
            # Additional pruning: if next_node is a leaf in the unvisited subgraph 
            # and we haven't finished, it might be a good idea to go there, 
            # but if we have other options, maybe delay it?
            # Warnsdorff says go to the one with fewest options.
            
            path.append(next_node)
            seen.add(next_node)
            
            dfs(path, seen)
            
            path.pop()
            seen.remove(next_node)
            
            # Check time after each branch
            if time.time() - start_time > time_limit:
                return

    # We might want to try multiple starting points or randomize the order
    # The greedy pass already tried all starts.
    # The DFS starting from a single start might not find the global optimum if the graph is complex.
    # However, with N=100 and sparsity, the search space is manageable if pruned well.
    # Let's run DFS from a few promising starts (e.g., nodes with high degree in the component).
    
    # Identify high degree nodes in the component
    degs = [(len([x for x in adj[node] if x in comp_nodes]), node) for node in largest_comp]
    degs.sort(reverse=True)
    
    # Try DFS from top 5 nodes
    for _, start_node in degs[:5]:
        if time.time() - start_time > time_limit:
            break
        dfs([start_node], {start_node})
        
    # If we still have time, try randomized restarts
    # Shuffle the order of nodes and try DFS from a few more
    random.seed(748) # Pinned seed for reproducibility if needed, but task says random.Random(748) is used for graph gen.
    # The task says "locally generated... is pinned". The graph is fixed.
    # We can use any randomization for search.
    
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5:
        # Try a few random starts
        random_starts = random.sample(largest_comp, min(10, comp_size))
        for start_node in random_starts:
            if time.time() - start_time > time_limit:
                break
            dfs([start_node], {start_node})

    return best_path
