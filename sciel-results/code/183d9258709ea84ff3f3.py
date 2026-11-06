"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from collections import deque

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
            queue = deque([i])
            visited[i] = True
            while queue:
                node = queue.popleft()
                comp.append(node)
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            components.append(comp)
    
    # Find the largest component
    largest_comp = max(components, key=len)
    comp_set = set(largest_comp)
    comp_size = len(largest_comp)
    
    if comp_size <= 2:
        return largest_comp

    # Beam Search for Longest Path
    # Heuristic: "Matching Slack" proxy -> number of unvisited neighbors in the component
    # We prioritize extending paths where the new vertex has fewer options (Warnsdorff's rule)
    # This helps avoid creating isolated subgraphs or trapping the path end.
    
    beam_width = 50
    # Initialize beam with single vertices
    # Prefer starting at low-degree vertices (likely endpoints)
    vertices_in_comp = list(largest_comp)
    vertices_in_comp.sort(key=lambda v: len(adj[v] & comp_set))
    
    # Start with the lowest degree vertices
    initial_paths = [[v] for v in vertices_in_comp[:beam_width]]
    
    best_path = max(initial_paths, key=len)
    
    # Beam search loop
    current_beam = initial_paths
    
    for step in range(comp_size - 1):
        next_beam = []
        for path in current_beam:
            last = path[-1]
            used = set(path)
            # Candidates: neighbors of last in comp, not in used
            candidates = [v for v in adj[last] if v in comp_set and v not in used]
            
            if not candidates:
                # Dead end, path terminates here
                continue
            
            for v in candidates:
                new_path = path + [v]
                next_beam.append(new_path)
        
        if not next_beam:
            break
        
        # Score and select top beam_width paths
        # Primary score: length
        next_beam.sort(key=lambda p: len(p), reverse=True)
        current_beam = next_beam[:beam_width]
        
        # Update best path
        if current_beam:
            current_best = max(current_beam, key=len)
            if len(current_best) > len(best_path):
                best_path = current_best
                if len(best_path) == comp_size:
                    return best_path
        
    return best_path
# EVOLVE-BLOCK-END
