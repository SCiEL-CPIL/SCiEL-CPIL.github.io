import sys
import time
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
            
    best_path = []
    
    # Helper to reconstruct path from parent pointers
    def get_path(parents, end_node):
        path = []
        curr = end_node
        while curr is not None:
            path.append(curr)
            curr = parents[curr]
        return path[::-1]

    # Helper to find diameter of a tree
    def find_tree_diameter(nodes, adj, comp_set):
        # Pick an arbitrary node, find farthest node
        start = nodes[0]
        dist = {start: 0}
        parents = {start: None}
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v in comp_set and v not in dist:
                    dist[v] = dist[u] + 1
                    parents[v] = u
                    queue.append(v)
        
        farthest_node = max(dist, key=dist.get)
        
        # Find farthest node from farthest_node
        dist2 = {farthest_node: 0}
        parents2 = {farthest_node: None}
        queue = deque([farthest_node])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v in comp_set and v not in dist2:
                    dist2[v] = dist2[u] + 1
                    parents2[v] = u
                    queue.append(v)
                    
        end_node = max(dist2, key=dist2.get)
        path = get_path(parents2, end_node)
        return path

    # Helper to find cycle in unicyclic component
    def find_cycle(nodes, adj, comp_set):
        # Use DFS to find back edge
        color = {u: 0 for u in nodes} # 0: white, 1: gray, 2: black
        parent = {u: None for u in nodes}
        cycle = []
        
        def dfs(u):
            color[u] = 1
            for v in adj[u]:
                if v not in comp_set:
                    continue
                if color[v] == 0:
                    parent[v] = u
                    if dfs(v):
                        return True
                elif color[v] == 1:
                    # Found back edge u -> v
                    # Reconstruct cycle
                    cycle_nodes = [u]
                    curr = u
                    while curr != v:
                        curr = parent[curr]
                        cycle_nodes.append(curr)
                    cycle.extend(cycle_nodes)
                    return True
            color[u] = 2
            return False
            
        for u in nodes:
            if color[u] == 0:
                if dfs(u):
                    break
        return cycle

    for comp in components:
        comp_set = set(comp)
        num_nodes = len(comp)
        num_edges = sum(len(adj[u] & comp_set) for u in comp) // 2
        
        if num_edges < num_nodes:
            # Tree
            path = find_tree_diameter(comp, adj, comp_set)
            if len(path) > len(best_path):
                best_path = path
        elif num_edges == num_nodes:
            # Unicyclic
            cycle = find_cycle(comp, adj, comp_set)
            if not cycle:
                # Should not happen, fallback to tree diameter (might be wrong if graph is disconnected? No, it's a component)
                path = find_tree_diameter(comp, adj, comp_set)
                if len(path) > len(best_path):
                    best_path = path
                continue
            
            cycle_set = set(cycle)
            
            # Calculate depth of tree attached to each cycle node
            depths = {}
            for c in cycle:
                # Find max depth in tree attached to c, excluding cycle edges
                max_depth = 0
                # BFS/DFS from c, avoiding other cycle nodes
                visited_local = {c}
                queue = deque([(c, 0)])
                while queue:
                    node, d = queue.popleft()
                    if d > max_depth:
                        max_depth = d
                    for neighbor in adj[node]:
                        if neighbor in comp_set and neighbor not in visited_local and neighbor not in cycle_set:
                            visited_local.add(neighbor)
                            queue.append((neighbor, d + 1))
                depths[c] = max_depth
            
            # Also need to consider diameters of the attached trees
            # The longest path in an attached tree is its diameter.
            # However, the path might go from a leaf in tree A to a leaf in tree B via the cycle.
            # Or it might be entirely within one attached tree.
            
            # 1. Max diameter of any attached tree
            # We can approximate this by checking pairs of nodes in the same attached tree?
            # Actually, the diameter of the attached tree rooted at c is not just 2*depth[c].
            # It's the longest path within that tree.
            # Let's compute diameter for each attached tree.
            
            max_tree_diam_path = []
            for c in cycle:
                # Get nodes in tree attached to c
                tree_nodes = set()
                queue = deque([c])
                visited_local = {c}
                while queue:
                    node = queue.popleft()
                    tree_nodes.add(node)
                    for neighbor in adj[node]:
                        if neighbor in comp_set and neighbor not in visited_local and neighbor not in cycle_set:
                            visited_local.add(neighbor)
                            queue.append(neighbor)
                
                if len(tree_nodes) > 1:
                    # Find diameter of this tree
                    # Pick arbitrary node, find farthest
                    start = list(tree_nodes)[0]
                    dist = {start: 0}
                    parents = {start: None}
                    q = deque([start])
                    while q:
                        u = q.popleft()
                        for v in adj[u]:
                            if v in tree_nodes and v not in dist:
                                dist[v] = dist[u] + 1
                                parents[v] = u
                                q.append(v)
                    farthest = max(dist, key=dist.get)
                    
                    dist2 = {farthest: 0}
                    parents2 = {farthest: None}
                    q = deque([farthest])
                    while q:
                        u = q.popleft()
                        for v in adj[u]:
                            if v in tree_nodes and v not in dist2:
                                dist2[v] = dist2[u] + 1
                                parents2[v] = u
                                q.append(v)
                    end = max(dist2, key=dist2.get)
                    path = get_path(parents2, end)
                    if len(path) > len(max_tree_diam_path):
                        max_tree_diam_path = path
            
            # 2. Paths going through the cycle
            # For each pair of cycle nodes (u, v), calculate path length
            # Path: Leaf_u -> ... -> u -> cycle -> v -> ... -> Leaf_v
            # Length = depths[u] + dist_cycle(u, v) + depths[v] + 1
            # We need to store the actual path to return it.
            # We can reconstruct it by combining the deepest path in u's tree, the cycle path, and the deepest path in v's tree.
            
            # First, get the deepest leaf and path for each cycle node
            deepest_leaf = {}
            path_to_leaf = {}
            for c in cycle:
                if depths[c] == 0:
                    deepest_leaf[c] = c
                    path_to_leaf[c] = [c]
                else:
                    # Re-find the leaf that gives max depth
                    # We did BFS above, let's redo it to store parents
                    parents_local = {c: None}
                    dist_local = {c: 0}
                    q = deque([c])
                    while q:
                        u = q.popleft()
                        for v in adj[u]:
                            if v in comp_set and v not in dist_local and v not in cycle_set:
                                dist_local[v] = dist_local[u] + 1
                                parents_local[v] = u
                                q.append(v)
                    leaf = max(dist_local, key=dist_local.get)
                    deepest_leaf[c] = leaf
                    path_to_leaf[c] = get_path(parents_local, leaf)
            
            best_cycle_path = []
            best_cycle_len = 0
            
            # Map cycle node to index for distance calculation
            cycle_list = cycle # ordered list
            C = len(cycle_list)
            cycle_index = {node: i for i, node in enumerate(cycle_list)}
            
            for i, u in enumerate(cycle_list):
                for j, v in enumerate(cycle_list):
                    if i == j:
                        continue
                    # Distance on cycle between u and v
                    # Two arcs: clockwise and counter-clockwise
                    # We take the longer one to maximize path length
                    # dist_cw = (j - i) % C
                    # dist_ccw = (i - j) % C
                    # Max edges = max(dist_cw, dist_ccw)
                    dist_cw = (j - i) % C
                    dist_ccw = (i - j) % C
                    max_dist = max(dist_cw, dist_ccw)
                    
                    # Total edges = depths[u] + max_dist + depths[v]
                    # Total vertices = Total edges + 1
                    total_verts = depths[u] + max_dist + depths[v] + 1
                    
                    if total_verts > best_cycle_len:
                        best_cycle_len = total_verts
                        # Reconstruct path
                        # Part 1: path from leaf_u to u (reversed to go leaf->u)
                        p1 = path_to_leaf[u][::-1]
                        
                        # Part 2: path on cycle from u to v
                        # If we took dist_cw (which is j-i), path is cycle_list[i], cycle_list[i+1]...
                        # If we took dist_ccw (which is i-j), path is cycle_list[i], cycle_list[i-1]...
                        # We need to know which arc was longer.
                        if dist_cw == max_dist:
                            # Go forward from i to j
                            p2 = [cycle_list[(i + k) % C] for k in range(dist_cw + 1)]
                        else:
                            # Go backward from i to j
                            p2 = [cycle_list[(i - k) % C] for k in range(dist_ccw + 1)]
                        
                        # Part 3: path from v to leaf_v
                        p3 = path_to_leaf[v]
                        
                        # Combine: p1 ends at u. p2 starts at u. p3 starts at v.
                        # Remove duplicate u and v
                        combined = p1 + p2[1:] + p3[1:]
                        
                        if len(combined) > len(best_cycle_path):
                            best_cycle_path = combined
            
            # Compare best_tree_path and best_cycle_path
            if len(best_cycle_path) > len(best_path):
                best_path = best_cycle_path
            if len(max_tree_diam_path) > len(best_path):
                best_path = max_tree_diam_path
                
        else:
            # Complex graph: Use heuristic DFS
            # Simple DFS with pruning
            best_local = []
            for start in comp:
                # Greedy DFS
                path = [start]
                seen = {start}
                while True:
                    last = path[-1]
                    neighbors = [n for n in adj[last] if n in comp_set and n not in seen]
                    if not neighbors:
                        break
                    # Choose neighbor with fewest unvisited neighbors (greedy)
                    # Or just pick one
                    # Let's try to be a bit smarter: pick neighbor that leads to deepest unvisited part?
                    # For now, simple greedy
                    next_node = min(neighbors, key=lambda x: len([y for y in adj[x] if y in comp_set and y not in seen]))
                    path.append(next_node)
                    seen.add(next_node)
                
                if len(path) > len(best_local):
                    best_local = path
            
            if len(best_local) > len(best_path):
                best_path = best_local

    return best_path
