import time
import random
import sys

# EVOLVE-BLOCK-START
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
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        stack.append(nei)
            components.append(comp)
            
    # Sort components by size, largest first
    components.sort(key=len, reverse=True)
    
    best_path = []
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # We will search in the largest component
    # If there are multiple large components, we might check the top 2-3
    # But usually the largest contains the longest path
    
    def dfs(u, visited_set, path):
        nonlocal best_path
        if len(path) > len(best_path):
            best_path = path[:]
            
        # Pruning: if max possible length is not better than best, stop
        # Max possible length = len(path) + (size_of_component - len(visited_set))
        # But we need to know which component we are in.
        # Since we start from a component, visited_set is within that component.
        # Let comp_size be the size of the component we are searching in.
        
        for v in adj[u]:
            if v not in visited_set:
                # Heuristic: try neighbors with fewer unvisited neighbors first (Warnsdorff)
                # This is a static ordering if we sort once, but dynamic is better.
                # For speed, we can just iterate and rely on the DFS structure.
                # To improve, we can sort neighbors by degree at the start.
                pass
                
    # Let's use a more controlled search:
    # 1. Precompute degrees.
    # 2. For the largest component, run a DFS with pruning.
    # 3. To handle time, we can limit the recursion depth or use iterative deepening if needed.
    # 4. Randomized restarts.
    
    largest_comp = components[0]
    comp_size = len(largest_comp)
    comp_set = set(largest_comp)
    
    # Precompute degrees within the component
    deg = {u: 0 for u in largest_comp}
    for u in largest_comp:
        for v in adj[u]:
            if v in comp_set:
                deg[u] += 1
                
    # Sort neighbors for each node in the component by degree (ascending)
    # This helps Warnsdorff's rule
    sorted_adj = {u: sorted([v for v in adj[u] if v in comp_set], key=lambda x: deg[x]) for u in largest_comp}
    
    # DFS with pruning
    # We use a stack to avoid recursion depth issues, but recursion is fine for N=100.
    # However, Python recursion limit might be an issue if we go deep, but 100 is fine.
    # We need to track visited set. A bitmask or a set. Set is O(1) average.
    
    def search(start):
        nonlocal best_path
        visited_set = {start}
        path = [start]
        
        def dfs(u):
            nonlocal best_path
            if len(path) > len(best_path):
                best_path = path[:]
                
            # Pruning
            # If len(path) + (comp_size - len(visited_set)) <= len(best_path):
            #     return
            
            for v in sorted_adj[u]:
                if v not in visited_set:
                    # Check time
                    if time.time() > deadline:
                        return True # Signal to stop
                    visited_set.add(v)
                    path.append(v)
                    if dfs(v):
                        return True
                    path.pop()
                    visited_set.remove(v)
            return False
            
        dfs(start)
        
    # Run search from each vertex in the largest component?
    # That might be too slow if the component is large.
    # Let's try from a few strategic starts:
    # 1. The vertex with the highest degree in the component.
    # 2. The vertex with the lowest degree (leaf-like).
    # 3. Random vertices.
    
    starts = list(largest_comp)
    # Sort starts by degree, maybe try high degree first? Or low?
    # Usually, starting from leaves is good for finding long paths in trees.
    # In graphs, it's mixed.
    # Let's just iterate through all, but check time.
    
    # To make it faster, we can randomize the order of starts.
    random.shuffle(starts)
    
    for start in starts:
        if time.time() > deadline:
            break
        search(start)
        
    # If we have time, try the second largest component too
    if len(components) > 1 and time.time() < deadline - 5:
        second_comp = components[1]
        comp_size2 = len(second_comp)
        comp_set2 = set(second_comp)
        deg2 = {u: 0 for u in second_comp}
        for u in second_comp:
            for v in adj[u]:
                if v in comp_set2:
                    deg2[u] += 1
        sorted_adj2 = {u: sorted([v for v in adj[u] if v in comp_set2], key=lambda x: deg2[x]) for u in second_comp}
        
        def search2(start):
            nonlocal best_path
            visited_set = {start}
            path = [start]
            
            def dfs(u):
                nonlocal best_path
                if len(path) > len(best_path):
                    best_path = path[:]
                for v in sorted_adj2[u]:
                    if v not in visited_set:
                        if time.time() > deadline:
                            return True
                        visited_set.add(v)
                        path.append(v)
                        if dfs(v):
                            return True
                        path.pop()
                        visited_set.remove(v)
                return False
            dfs(start)
            
        starts2 = list(second_comp)
        random.shuffle(starts2)
        for start in starts2:
            if time.time() > deadline:
                break
            search2(start)
            
    return best_path
# EVOLVE-BLOCK-END
