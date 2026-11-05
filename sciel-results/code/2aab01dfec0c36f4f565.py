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
    
    if not components:
        return [0]
        
    target_comp = components[0]
    target_set = set(target_comp)
    
    deadline = time.time() + 55.0 # Safety margin (90s limit)
    
    # Helper to count unvisited neighbors in the component
    def get_unvisited_neighbors(node, visited):
        return [nb for nb in adj[node] if nb in target_set and nb not in visited]

    best_len = 0
    best_path_found = []
    
    # Try starting from each node in the component
    # Order starts by degree (ascending) - Warnsdorff works better from low-degree nodes
    starts = sorted(target_comp, key=lambda x: len(adj[x]))
    
    for start in starts:
        if time.time() > deadline:
            break
            
        visited = {start}
        path = [start]
        
        # Iterative DFS with backtracking and Warnsdorff's Rule
        # Stack contains (node, visited_set, path_list, next_neighbor_index)
        # We use a list to simulate recursion for better control
        
        stack = [(start, visited, path, 0)]
        
        while stack:
            if time.time() > deadline:
                break
                
            node, visited, path, next_idx = stack[-1]
            
            if len(path) > best_len:
                best_len = len(path)
                best_path_found = path[:]
                
            if len(visited) == len(target_comp):
                return best_path_found # Found Hamiltonian path
            
            neighbors = get_unvisited_neighbors(node, visited)
            
            if not neighbors:
                # Dead end, backtrack
                stack.pop()
                if stack:
                    # Remove the node from the parent's path and visited
                    parent_node, parent_visited, parent_path, _ = stack[-1]
                    # The last node in parent_path is the one we came from
                    # Actually, we need to be careful with shared references
                    # Let's use a different approach: copy-on-write or explicit state
                    # For now, let's just break if we can't continue easily
                    pass
                continue
            
            # Warnsdorff's Rule: Sort neighbors by number of unvisited neighbors
            neighbors.sort(key=lambda x: len(get_unvisited_neighbors(x, visited)))
            
            if next_idx < len(neighbors):
                nb = neighbors[next_idx]
                # Update stack top with incremented index
                stack[-1] = (node, visited, path, next_idx + 1)
                
                # Move to neighbor
                new_visited = visited | {nb}
                new_path = path + [nb]
                stack.append((nb, new_visited, new_path, 0))
            else:
                # No more neighbors, backtrack
                stack.pop()
                
        # If we exited the while loop normally (stack empty), continue to next start
        # If we found a Hamiltonian path, we already returned
        
        if best_len == len(target_comp):
            break # Found Hamiltonian path, can't do better
            
    if best_path_found:
        return best_path_found
    
    # Fallback: If DFS didn't find anything good, return a simple greedy path
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
        neighbors.sort(key=lambda x: len(adj[x]))
        current = neighbors[0]
        path.append(current)
        visited.add(current)
        
    if path:
        return path
        
    return [0] # Should not reach here if graph is non-empty

# EVOLVE-BLOCK-END
