import time
import sys
from collections import deque

def construct():
    start_time = time.time()
    time_limit = 85.0

    P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]}

    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
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
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            components.append(comp)
    
    components.sort(key=len, reverse=True)
    
    best_path = []
    
    def get_longest_path_in_tree(comp, adj, root):
        """
        Returns the longest path (diameter) in the tree formed by comp, 
        assuming the tree is rooted at root and we want the diameter 
        that might pass through root or not.
        Actually, we just want the diameter of the tree.
        """
        if len(comp) == 1:
            return [comp[0]]
        
        # BFS from root to find farthest node u
        dist = {root: 0}
        parent = {root: None}
        q = deque([root])
        farthest = root
        while q:
            u = q.popleft()
            if dist[u] > dist[farthest]:
                farthest = u
            for v in adj[u]:
                if v in comp and v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
        
        # BFS from u to find farthest node v
        dist2 = {farthest: 0}
        parent2 = {farthest: None}
        q = deque([farthest])
        farthest2 = farthest
        while q:
            u = q.popleft()
            if dist2[u] > dist2[farthest2]:
                farthest2 = u
            for v in adj[u]:
                if v in comp and v not in dist2:
                    dist2[v] = dist2[u] + 1
                    parent2[v] = u
                    q.append(v)
        
        # Reconstruct path
        path = []
        cur = farthest2
        while cur is not None:
            path.append(cur)
            cur = parent2[cur]
        path.reverse()
        return path

    def get_height_and_path(comp, adj, root, forbidden_edges):
        """
        Computes the longest path starting from root in the tree 'comp'
        without using 'forbidden_edges'.
        Returns (length, path).
        """
        if root not in comp:
            return 0, []
        
        # BFS/DFS to find farthest node from root
        visited = {root}
        stack = [(root, [root])]
        farthest = root
        max_dist = 0
        parent = {root: None}
        
        # We need to traverse the tree structure.
        # Since it's a tree, we can just DFS.
        def dfs(u, dist, path):
            nonlocal farthest, max_dist
            if dist > max_dist:
                max_dist = dist
                farthest = u
            for v in adj[u]:
                if v in comp and v not in visited:
                    # Check if edge (u,v) is forbidden
                    if (u, v) in forbidden_edges or (v, u) in forbidden_edges:
                        continue
                    visited.add(v)
                    parent[v] = u
                    dfs(v, dist + 1, path + [v])
        
        # Use iterative DFS to avoid recursion limits
        stack = [(root, 0, [root])]
        parent = {root: None}
        while stack:
            u, d, path = stack.pop()
            if d > max_dist:
                max_dist = d
                farthest = u
            for v in adj[u]:
                if v in comp and v not in visited:
                    if (u, v) in forbidden_edges or (v, u) in forbidden_edges:
                        continue
                    visited.add(v)
                    parent[v] = u
                    stack.append((v, d + 1, path + [v]))
        
        # Reconstruct path to farthest
        if farthest == root:
            return 0, [root]
        
        path = []
        cur = farthest
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return max_dist, path

    def solve_unicyclic(comp, adj):
        """
        Finds the longest path in a unicyclic component.
        """
        comp_set = set(comp)
        # Find the cycle
        # Degrees within the component
        deg = {u: 0 for u in comp}
        for u in comp:
            for v in adj[u]:
                if v in comp_set:
                    deg[u] += 1
        
        # Nodes with degree > 2 are on the cycle
        cycle_nodes = [u for u in comp if deg[u] > 2]
        
        # If no node has degree > 2, it's a pure cycle? 
        # In a unicyclic graph, if all nodes have degree 2, it's a simple cycle.
        # If some have degree 2 and some > 2, the > 2 ones are on the cycle.
        # Actually, nodes on the cycle have degree at least 2. Nodes not on the cycle have degree 1 in the "cycle graph" but higher in the tree.
        # Wait, in the original graph, cycle nodes have degree >= 2. Tree nodes have degree >= 1.
        # The cycle is the set of nodes that remain if we repeatedly remove leaves (degree 1).
        
        # Let's find the cycle by pruning leaves.
        remaining = set(comp)
        deg_rem = {u: deg[u] for u in comp}
        
        queue = deque([u for u in comp if deg_rem[u] == 1])
        
        while queue:
            u = queue.popleft()
            remaining.remove(u)
            for v in adj[u]:
                if v in remaining:
                    deg_rem[v] -= 1
                    if deg_rem[v] == 1:
                        queue.append(v)
        
        cycle_nodes = list(remaining)
        
        if not cycle_nodes:
            # Should not happen for unicyclic
            return get_longest_path_in_tree(comp, adj, comp[0])
        
        # Order the cycle nodes
        # Pick a start node
        start = cycle_nodes[0]
        cycle_path = [start]
        prev = None
        curr = start
        visited_cycle = {start}
        
        # Find neighbors of start in cycle
        next_node = None
        for v in adj[start]:
            if v in remaining and v != prev:
                next_node = v
                break
        
        curr = next_node
        while curr != start:
            cycle_path.append(curr)
            visited_cycle.add(curr)
            prev = curr
            next_node = None
            for v in adj[curr]:
                if v in remaining and v != prev:
                    next_node = v
                    break
            curr = next_node
            if curr is None:
                break # Error
        
        # Now we have the cycle in order: cycle_path
        # For each node in cycle_path, find the height of the tree attached to it (excluding cycle edges)
        cycle_len = len(cycle_path)
        cycle_edges = set()
        for i in range(cycle_len):
            u = cycle_path[i]
            v = cycle_path[(i + 1) % cycle_len]
            cycle_edges.add((u, v))
            cycle_edges.add((v, u))
            
        heights = [0] * cycle_len
        height_paths = [None] * cycle_len
        
        for i, u in enumerate(cycle_path):
            # The tree attached to u is the component containing u in the graph 
            # with cycle edges removed.
            # We can just do a DFS from u, not crossing cycle edges.
            # Since it's a tree, the height is the max distance from u.
            # We already have get_height_and_path.
            # But we need to be careful: the "comp" for this tree is just the nodes reachable from u without cycle edges.
            # However, get_height_and_path takes a comp set. 
            # Let's just use a simple DFS that avoids cycle edges.
            
            # Actually, we can just use the original adj and a visited set, 
            # and avoid cycle edges.
            
            visited = {u}
            stack = [(u, 0, [u])]
            max_dist = 0
            farthest = u
            parent = {u: None}
            
            while stack:
                node, dist, path = stack.pop()
                if dist > max_dist:
                    max_dist = dist
                    farthest = node
                for v in adj[node]:
                    if v in comp_set and v not in visited:
                        if (node, v) in cycle_edges:
                            continue
                        visited.add(v)
                        parent[v] = node
                        stack.append((v, dist + 1, path + [v]))
                        
            heights[i] = max_dist
            # Reconstruct path from u to farthest
            if farthest == u:
                height_paths[i] = [u]
            else:
                path = []
                cur = farthest
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                height_paths[i] = path
        
        # Now find the best pair (i, j) on the cycle to maximize heights[i] + heights[j] + dist(i, j)
        # dist(i, j) is the shortest path distance along the cycle.
        # Since it's a cycle, dist(i, j) = min(|i-j|, cycle_len - |i-j|)
        
        best_len = 0
        best_path_global = []
        
        # Also consider paths entirely within one attached tree (diameter)
        # The diameter of the tree attached to u is not necessarily 2*heights[i].
        # It's the longest path in that tree.
        # We can compute the diameter of each attached tree.
        for i, u in enumerate(cycle_path):
            # Find diameter of tree attached to u
            # We can use get_longest_path_in_tree on the subset of nodes reachable from u without cycle edges.
            # But we need the subset.
            # Let's just use the height and assume the diameter is bounded by 2*height? No.
            # Let's compute diameter properly for each attached tree.
            
            # Find nodes in this tree
            tree_nodes = set()
            stack = [u]
            tree_nodes.add(u)
            while stack:
                node = stack.pop()
                for v in adj[node]:
                    if v in comp_set and v not in tree_nodes:
                        if (node, v) in cycle_edges:
                            continue
                        tree_nodes.add(v)
                        stack.append(v)
                        
            # Find diameter of this tree
            if len(tree_nodes) > 1:
                # BFS from arbitrary node in tree_nodes to find farthest
                start_node = next(iter(tree_nodes))
                dist = {start_node: 0}
                parent = {start_node: None}
                q = deque([start_node])
                farthest1 = start_node
                while q:
                    node = q.popleft()
                    if dist[node] > dist[farthest1]:
                        farthest1 = node
                    for v in adj[node]:
                        if v in tree_nodes and v not in dist:
                            dist[v] = dist[node] + 1
                            parent[v] = node
                            q.append(v)
                            
                dist2 = {farthest1: 0}
                parent2 = {farthest1: None}
                q = deque([farthest1])
                farthest2 = farthest1
                while q:
                    node = q.popleft()
                    if dist2[node] > dist2[farthest2]:
                        farthest2 = node
                    for v in adj[node]:
                        if v in tree_nodes and v not in dist2:
                            dist2[v] = dist2[node] + 1
                            parent2[v] = node
                            q.append(v)
                            
                # Reconstruct path
                path = []
                cur = farthest2
                while cur is not None:
                    path.append(cur)
                    cur = parent2[cur]
                path.reverse()
                
                if len(path) > best_len:
                    best_len = len(path)
                    best_path_global = path
        
        # Now consider paths using the cycle
        # Maximize heights[i] + heights[j] + dist(i, j)
        # We can duplicate the cycle to handle the wrap-around
        # Let H[i] = heights[i]
        # We want max_{i, j} H[i] + H[j] + dist(i, j)
        # dist(i, j) is the distance along the cycle.
        
        # We can iterate over all pairs? cycle_len is small (<= 100).
        # O(N^2) is fine.
        
        for i in range(cycle_len):
            for j in range(cycle_len):
                if i == j:
                    continue
                # Distance along cycle
                d1 = abs(i - j)
                d2 = cycle_len - d1
                d = min(d1, d2)
                
                total_len = heights[i] + heights[j] + d
                
                if total_len > best_len:
                    best_len = total_len
                    # Reconstruct path
                    # Path from height_paths[i] (ends at cycle_path[i])
                    # Then cycle path from cycle_path[i] to cycle_path[j]
                    # Then height_paths[j] reversed (starts at cycle_path[j])
                    
                    p1 = height_paths[i]
                    p2 = height_paths[j]
                    
                    # p1 ends at cycle_path[i]
                    # p2 starts at cycle_path[j]
                    
                    # Determine cycle segment
                    # We need the shortest path from i to j on the cycle.
                    # There are two directions.
                    # Let's construct both and pick the one with length d.
                    
                    # Direction 1: i -> i+1 -> ... -> j (mod cycle_len)
                    # Direction 2: i -> i-1 -> ... -> j (mod cycle_len)
                    
                    seg1 = []
                    if d1 == d:
                        if i < j:
                            for k in range(i, j+1):
                                seg1.append(cycle_path[k])
                        else:
                            for k in range(i, cycle_len):
                                seg1.append(cycle_path[k])
                            for k in range(0, j+1):
                                seg1.append(cycle_path[k])
                    else:
                        if i > j:
                            for k in range(i, j-1, -1):
                                seg1.append(cycle_path[k])
                        else:
                            for k in range(i, -1, -1):
                                seg1.append(cycle_path[k])
                            for k in range(cycle_len-1, j-1, -1):
                                seg1.append(cycle_path[k])
                    
                    # Check length of seg1
                    # len(seg1) - 1 should be d
                    # But we need to ensure it's the shortest one.
                    # Since we picked d = min(d1, d2), we just need to pick the correct direction.
                    
                    # Let's just construct the path based on the shorter direction.
                    # If d1 == d, go forward. Else go backward.
                    
                    if d1 == d:
                        # Forward
                        if i < j:
                            cycle_seg = cycle_path[i:j+1]
                        else:
                            cycle_seg = cycle_path[i:] + cycle_path[:j+1]
                    else:
                        # Backward
                        if i > j:
                            cycle_seg = cycle_path[j:i+1][::-1]
                        else:
                            cycle_seg = cycle_path[j:] + cycle_path[:i+1][::-1]
                            # Wait, cycle_path[:i+1][::-1] is i, i-1, ..., 0.
                            # cycle_path[j:] is j, ..., N-1.
                            # So j, ..., N-1, i, ..., 0.
                            # This is correct for backward from i to j?
                            # i -> i-1 -> ... -> 0 -> N-1 -> ... -> j.
                            # Yes.
                            
                    # Combine
                    # p1 is leaf -> ... -> cycle_path[i]
                    # cycle_seg is cycle_path[i] -> ... -> cycle_path[j]
                    # p2 is cycle_path[j] -> ... -> leaf
                    # So we need to reverse p2? No, p2 is [cycle_path[j], ..., leaf].
                    # We want to go from cycle_path[j] to leaf.
                    # So we append p2[1:]? No, p2[0] is cycle_path[j], which is already the end of cycle_seg.
                    # So we append p2[1:].
                    
                    full_path = p1 + cycle_seg[1:] + p2[1:]
                    
                    if len(full_path) > len(best_path_global):
                        best_path_global = full_path
        
        return best_path_global

    for comp in components:
        if time.time() - start_time > time_limit:
            break
            
        if len(comp) <= len(best_path):
            continue
            
        comp_set = set(comp)
        edge_count = sum(len(adj[u]) for u in comp) // 2
        
        if edge_count == len(comp) - 1:
            # Tree
            path = get_longest_path_in_tree(comp, adj, comp[0])
            if len(path) > len(best_path):
                best_path = path
        elif edge_count == len(comp):
            # Unicyclic
            path = solve_unicyclic(comp, adj)
            if len(path) > len(best_path):
                best_path = path
        else:
            # Complex component (multi-cyclic)
            # Use heuristic DFS as fallback
            # But these should be rare and small.
            # We can try a simple DFS with pruning.
            # For now, let's just use the tree diameter heuristic on a spanning tree?
            # Or run the old DFS?
            # Let's run a quick DFS.
            
            # Given time constraints, if this happens, we might just skip or use a simple heuristic.
            # Let's try to find a long path using a greedy approach.
            # Start from a leaf of the spanning tree?
            
            # Let's just use the old DFS logic for this case, but with a tight time limit.
            # Actually, let's just try to find the diameter of a spanning tree.
            # It's a lower bound.
            
            # Find a spanning tree (BFS tree)
            visited = {comp[0]}
            parent = {comp[0]: None}
            q = deque([comp[0]])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v in comp_set and v not in visited:
                        visited.add(v)
                        parent[v] = u
                        q.append(v)
                        
            # Now find diameter of this tree
            # We can use get_longest_path_in_tree on the tree edges.
            # But get_longest_path_in_tree uses adj.
            # We need to restrict to tree edges.
            
            # Let's just use the tree nodes and the parent pointers to reconstruct?
            # No, let's just run BFS on the tree edges.
            
            # Build tree adjacency
            tree_adj = {u: [] for u in comp}
            for u in comp:
                if parent[u] is not None:
                    tree_adj[u].append(parent[u])
                    tree_adj[parent[u]].append(u)
                    
            # Find diameter
            # BFS from comp[0]
            dist = {comp[0]: 0}
            p1 = {comp[0]: None}
            q = deque([comp[0]])
            farthest1 = comp[0]
            while q:
                u = q.popleft()
                if dist[u] > dist[farthest1]:
                    farthest1 = u
                for v in tree_adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        p1[v] = u
                        q.append(v)
                        
            dist2 = {farthest1: 0}
            p2 = {farthest1: None}
            q = deque([farthest1])
            farthest2 = farthest1
            while q:
                u = q.popleft()
                if dist2[u] > dist2[farthest2]:
                    farthest2 = u
                for v in tree_adj[u]:
                    if v not in dist2:
                        dist2[v] = dist2[u] + 1
                        p2[v] = u
                        q.append(v)
                        
            path = []
            cur = farthest2
            while cur is not None:
                path.append(cur)
                cur = p2[cur]
            path.reverse()
            
            if len(path) > len(best_path):
                best_path = path

    return best_path
