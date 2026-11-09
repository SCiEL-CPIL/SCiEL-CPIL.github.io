import time
import random
import sys

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    visited_comp = [False] * n
    largest_comp = []
    for i in range(n):
        if not visited_comp[i]:
            stack = [i]
            comp = []
            visited_comp[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if not visited_comp[nei]:
                        visited_comp[nei] = True
                        stack.append(nei)
            if len(comp) > len(largest_comp):
                largest_comp = comp
                
    if not largest_comp:
        return [0]
        
    best_path = []
    best_len = 0
    
    def greedy_warnsdorff(start_node):
        visited = {start_node}
        path = [start_node]
        while True:
            last = path[-1]
            neighbors = [x for x in adj[last] if x not in visited]
            if not neighbors:
                break
            next_node = min(neighbors, key=lambda x: (len(adj[x] - visited), x))
            path.append(next_node)
            visited.add(next_node)
        return path

    for i in largest_comp:
        path = greedy_warnsdorff(i)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path

    def extend_path(path, visited):
        path = list(path)
        visited = set(visited)
        while True:
            last = path[-1]
            neighbors = [x for x in adj[last] if x not in visited]
            if not neighbors:
                break
            next_node = min(neighbors, key=lambda x: (len(adj[x] - visited), x))
            path.append(next_node)
            visited.add(next_node)
        while True:
            first = path[0]
            neighbors = [x for x in adj[first] if x not in visited]
            if not neighbors:
                break
            next_node = min(neighbors, key=lambda x: (len(adj[x] - visited), x))
            path.insert(0, next_node)
            visited.add(next_node)
        return path

    if best_path:
        visited = set(best_path)
        improved = extend_path(best_path, visited)
        if len(improved) > best_len:
            best_path = improved
            best_len = len(best_path)
            
    if time.time() > deadline:
        return best_path

    # Beam Search
    def beam_search(start_path, beam_width, time_limit):
        # start_path is a list
        # Returns best path found
        current_beam = [(list(start_path), set(start_path))]
        best_found = list(start_path)
        
        while time.time() < time_limit:
            next_beam = []
            for path, visited in current_beam:
                last = path[-1]
                neighbors = [x for x in adj[last] if x not in visited]
                if not neighbors:
                    continue
                for x in neighbors:
                    new_path = path + [x]
                    new_visited = visited | {x}
                    # Heuristic: prefer longer paths, then those with more options
                    # Score: len(path) * 100 + len(adj[x] - new_visited)
                    score = len(new_path) * 100 + len(adj[x] - new_visited)
                    next_beam.append((new_path, new_visited, score))
                    
            if not next_beam:
                break
                
            # Sort by score descending
            next_beam.sort(key=lambda item: item[2], reverse=True)
            
            # Keep top beam_width
            current_beam = [(p, v) for p, v, s in next_beam[:beam_width]]
            
            # Update best
            for p, v in current_beam:
                if len(p) > len(best_found):
                    best_found = list(p)
                    
        return best_found

    # Run Beam Search from the best greedy path
    if best_path:
        # Try different beam widths
        for bw in [10, 20, 50]:
            if time.time() > deadline - 5.0:
                break
            t_start = time.time()
            # Limit time per beam search run
            t_limit = min(deadline - 5.0, t_start + 10.0)
            found = beam_search(best_path, bw, t_limit)
            if len(found) > best_len:
                best_path = found
                best_len = len(best_path)
                
    # Final extension
    if best_path and time.time() < deadline - 2.0:
        visited = set(best_path)
        improved = extend_path(best_path, visited)
        if len(improved) > best_len:
            best_path = improved
            best_len = len(best_path)
            
    return best_path
# EVOLVE-BLOCK-END
