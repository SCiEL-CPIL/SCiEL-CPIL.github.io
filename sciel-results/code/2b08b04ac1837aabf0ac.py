import sys
import time
import random
from collections import deque

# Set recursion limit just in case
sys.setrecursionlimit(2000)

# The graph data is provided in the seed program's P dict
# We will extract it here or assume it's available?
# The prompt says "The finite path score certifies... A locally generated... is pinned."
# The seed program has P defined. I should copy P from the seed program to ensure I have the exact graph.
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
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            components.append(comp)
            
    best_path = []
    
    for comp in components:
        if len(comp) < 2:
            continue
            
        # 1. Peel degree-1 nodes to find 2-core and tails
        # We work on a subgraph induced by 'comp'
        # deg[node] is degree within the current subgraph
        deg = {node: len([nb for nb in adj[node] if nb in comp]) for node in comp}
        # Actually, adj[node] contains all neighbors. We need to count only those in comp.
        # Since comp is a connected component, all neighbors of a node in comp are in comp.
        # So deg[node] = len(adj[node]) is correct for the initial state.
        deg = {node: len(adj[node]) for node in comp}
        
        queue = deque([node for node in comp if deg[node] == 1])
        removed = set()
        max_tail = {node: 0 for node in comp} # Max length of path starting at node, going into removed trees, excluding node
        
        # tail_len[u] is the length of the path starting at u, going into removed trees, INCLUDING u.
        # When u is a leaf, tail_len[u] = 1.
        # When u is removed, we update its neighbor v.
        
        # We need to store the "current tail length" for nodes that are in the queue or about to be.
        # Let's use a dict or just update max_tail.
        
        # Initialize tail_len for leaves
        tail_len = {node: 1 if deg[node] == 1 else 0 for node in comp}
        
        while queue:
            u = queue.popleft()
            if u in removed: continue
            removed.add(u)
            
            # Find the neighbor v that is not removed
            v = None
            for nb in adj[u]:
                if nb in comp and nb not in removed:
                    v = nb
                    break
            
            if v is None:
                # This should not happen if deg > 1, but if deg was 1 and the only neighbor was removed?
                # No, we only remove u when deg[u] becomes 1 (or initially 1).
                # If v is None, u was isolated? No, it was in a component.
                # If the component is a single edge u-v, both deg 1.
                # u removed, v becomes deg 0?
                # If v becomes deg 0, it is not pushed to queue (deg != 1).
                # So v stays in core? No, if deg 0, it's isolated.
                # In a tree, this happens at the end.
                continue
                
            # Update max_tail for v
            # The path v -> u -> ... has length 1 (for u) + (tail_len[u] - 1) for the rest?
            # tail_len[u] includes u.
            # So the number of nodes added to v's tail is tail_len[u].
            if tail_len[u] > max_tail[v]:
                max_tail[v] = tail_len[u]
            
            # Update degree of v
            deg[v] -= 1
            
            # If v becomes degree 1, it will be removed next
            if deg[v] == 1:
                # v's new tail length is 1 (itself) + max_tail[v] (the best tail it has accumulated)
                tail_len[v] = 1 + max_tail[v]
                queue.append(v)
                
        core_nodes = [node for node in comp if node not in removed]
        
        if not core_nodes:
            # The component is a tree (or forest, but it's a component so tree)
            # The longest path is the diameter.
            # We can find it by BFS from any leaf.
            # Or simply, the max tail_len found during peeling?
            # Actually, if the core is empty, the last node removed had the max tail?
            # Let's just compute diameter of the tree using standard BFS.
            # Pick any node, BFS to find farthest node A.
            # BFS from A to find farthest node B.
            # Distance A-B is diameter.
            
            # Find a leaf
            start_node = None
            for node in comp:
                if len(adj[node]) == 1:
                    start_node = node
                    break
            if start_node is None:
                start_node = comp[0]
                
            # BFS 1
            dist = {start_node: 0}
            q = deque([start_node])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        q.append(v)
            a = max(dist, key=dist.get)
            
            # BFS 2
            dist2 = {a: 0}
            q = deque([a])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in dist2:
                        dist2[v] = dist2[u] + 1
                        q.append(v)
            b = max(dist2, key=dist2.get)
            
            # Reconstruct path
            # We need the actual path to return it.
            # Re-run BFS from a to b, keeping parents.
            parent = {a: None}
            q = deque([a])
            while q:
                u = q.popleft()
                if u == b: break
                for v in adj[u]:
                    if v not in parent:
                        parent[v] = u
                        q.append(v)
                        
            path = []
            curr = b
            while curr is not None:
                path.append(curr)
                curr = parent[curr]
            path.reverse()
            
            if len(path) > len(best_path):
                best_path = path
        
        else:
            # Core is not empty
            # Find longest path in core, considering tails
            # Core subgraph
            core_adj = {u: [v for v in adj[u] if v in core_nodes] for u in core_nodes}
            
            # DFS to find best path in core
            # State: current_node, visited_in_core, current_path
            # Score: len(current_path) + max_tail[start] + max_tail[current_node]
            # We want to maximize this.
            
            # We can use a simple DFS with pruning.
            # Best score so far
            best_score = 0
            best_core_path = []
            
            def dfs(u, visited_core, path):
                nonlocal best_score, best_core_path
                # Calculate current potential score
                # The path so far is 'path'. Start is path[0]. End is u.
                # Score = len(path) + max_tail[path[0]] + max_tail[u]
                # Note: max_tail excludes the node itself.
                score = len(path) + max_tail[path[0]] + max_tail[u]
                if score > best_score:
                    best_score = score
                    best_core_path = path[:]
                    
                # Pruning:
                # Max possible additional length is (number of unvisited core nodes)
                # If score + (len(core_nodes) - len(path)) <= best_score, prune
                if score + (len(core_nodes) - len(path)) <= best_score:
                    return
                
                for v in core_adj[u]:
                    if v not in visited_core:
                        visited_core.add(v)
                        path.append(v)
                        dfs(v, visited_core, path)
                        path.pop()
                        visited_core.remove(v)
                        
            # Try all start nodes in core
            for start in core_nodes:
                visited_core = {start}
                dfs(start, visited_core, [start])
                
            # Reconstruct full path
            if best_core_path:
                start_core = best_core_path[0]
                end_core = best_core_path[-1]
                
                # Find the tail paths
                # We need to find the actual path corresponding to max_tail[start_core]
                # This requires backtracking through the peeling process.
                # This is complex. 
                # Alternative: Just return the core path if tails are 0?
                # No, we must include tails to maximize length.
                
                # Let's find the tail path for a given core node.
                # We need to know which removed nodes formed the max_tail.
                # We didn't store the parent pointers during peeling.
                # We can re-run the peeling logic to find the specific path?
                # Or, we can just do a DFS from the core node into the removed nodes?
                # The removed nodes form trees attached to the core.
                # So, from start_core, we can do a DFS in the original graph, avoiding core nodes, to find the longest path.
                
                def find_tail_path(core_node, avoid_core):
                    # DFS to find longest path starting at core_node, going into non-core nodes
                    # Returns the list of nodes (excluding core_node)
                    best_tail = []
                    
                    def dfs_tail(u, visited):
                        nonlocal best_tail
                        current_path = [u]
                        # Update best
                        if len(current_path) > len(best_tail):
                            best_tail = current_path[:]
                        
                        for v in adj[u]:
                            if v not in visited and v not in avoid_core:
                                visited.add(v)
                                dfs_tail(v, visited)
                                visited.remove(v)
                                
                    dfs_tail(core_node, {core_node})
                    return best_tail # This includes core_node? No, we start at core_node, but we want the path *after* core_node.
                    # Wait, dfs_tail starts at u=core_node. current_path=[core_node].
                    # We want the path of *removed* nodes.
                    # So we should return best_tail[1:] if best_tail[0] == core_node.
                    # But dfs_tail explores neighbors.
                    # Let's adjust: Start from neighbors of core_node that are not in core.
                    
                # Simpler: Just do a DFS from start_core in the full graph, but force the first step to be in the core path direction?
                # No, the path is Tail1 + CorePath + Tail2.
                # We can construct it by:
                # 1. Get Tail1: Longest path starting at start_core, not using core nodes (except start_core).
                # 2. Get Tail2: Longest path starting at end_core, not using core nodes (except end_core).
                
                def get_tail(start_node, core_set):
                    # Find longest path starting at start_node, going into non-core nodes
                    # Returns list of nodes [start_node, n1, n2, ...]
                    best = [start_node]
                    def dfs_t(u, visited):
                        nonlocal best
                        # visited contains start_node
                        for v in adj[u]:
                            if v not in visited and v not in core_set:
                                visited.add(v)
                                # Try path u -> v
                                # We need to find the longest path from v
                                # This is just a tree, so we can do a simple DFS
                                path_so_far = [v]
                                def dfs_sub(w, vis):
                                    for x in adj[w]:
                                        if x not in vis and x not in core_set:
                                            vis.add(x)
                                            # Recurse
                                            # This is getting messy.
                                            pass
                                # Let's just do a standard longest path in the tree component attached to start_node
                                # The attached component is a tree.
                                # We can find the diameter of the tree formed by removing core nodes?
                                # No, we want the longest path *starting* at start_node.
                                # Since it's a tree, the longest path starting at a node is the max distance to any leaf in that tree.
                                # So we just need the max depth from start_node in the non-core tree.
                                
                                # BFS/DFS from start_node into non-core nodes
                                max_depth = 0
                                def find_max_depth(node, depth, vis):
                                    nonlocal max_depth
                                    if depth > max_depth:
                                        max_depth = depth
                                        # We also need the path
                                        # This is inefficient.
                                        
                                # Let's just do a DFS that returns the longest path list.
                                def longest_path_from(node, vis, core_set):
                                    # Returns list of nodes
                                    best_path_local = [node]
                                    for nb in adj[node]:
                                        if nb not in vis and nb not in core_set:
                                            vis.add(nb)
                                            sub_path = longest_path_from(nb, vis, core_set)
                                            vis.remove(nb)
                                            if len(sub_path) + 1 > len(best_path_local):
                                                best_path_local = [node] + sub_path
                                    return best_path_local
                                    
                                tail = longest_path_from(start_node, {start_node}, core_set)
                                return tail
                                
                tail1 = get_tail(start_core, set(core_nodes))
                tail2 = get_tail(end_core, set(core_nodes))
                
                # Combine
                # tail1 is [start_core, ...]
                # core_path is [start_core, ..., end_core]
                # tail2 is [end_core, ...]
                
                # Remove start_core from tail1 and tail2 to avoid duplication
                # tail1_rest = tail1[1:]
                # tail2_rest = tail2[1:]
                
                # But wait, tail1 and tail2 might share nodes?
                # No, they are in disjoint trees attached to different core nodes (or same, but then we can't use both).
                # If start_core == end_core, we can only use one tail.
                if start_core == end_core:
                    # Choose the longer tail
                    if len(tail1) > len(tail2):
                        full_path = tail1 + core_path[1:] # core_path is [start]
                        # Wait, core_path is just [start_core]
                        # So full_path = tail1
                    else:
                        full_path = tail2
                else:
                    # tail1: [start_core, a, b]
                    # core_path: [start_core, c, d, end_core]
                    # tail2: [end_core, e, f]
                    # We want: [a, b, start_core, c, d, end_core, e, f]
                    # So reverse tail1? No, tail1 is start_core -> ...
                    # Path must be simple.
                    # tail1 is a path starting at start_core.
                    # So the order is: (tail1 reversed?) No.
                    # If tail1 is [start_core, a], then a is connected to start_core.
                    # Path: a -> start_core -> ...
                    # So we reverse tail1 to get [a, start_core].
                    # Then remove start_core from the end of reversed tail1?
                    # reversed_tail1 = [a, start_core]
                    # core_path = [start_core, ..., end_core]
                    # reversed_tail2 = [f, e, end_core]
                    # Combine: [a, start_core] + core_path[1:] + reversed_tail2[1:]?
                    # No.
                    # Let's just concatenate:
                    # tail1_rev = tail1[::-1] # [a, start_core]
                    # tail2_rev = tail2[::-1] # [f, e, end_core]
                    # full_path = tail1_rev[:-1] + core_path + tail2_rev[:-1]
                    # Wait, tail1_rev[:-1] is [a]. core_path is [start_core, ..., end_core]. tail2_rev[:-1] is [f, e].
                    # Result: [a, start_core, ..., end_core, f, e].
                    # This is a valid path.
                    
                    tail1_rev = tail1[::-1]
                    tail2_rev = tail2[::-1]
                    
                    # tail1_rev ends with start_core
                    # tail2_rev ends with end_core
                    # We want to remove the duplicate core nodes.
                    # tail1_part = tail1_rev[:-1]
                    # tail2_part = tail2_rev[:-1]
                    
                    full_path = tail1_rev[:-1] + core_path + tail2_rev[:-1]
                    
                    if len(full_path) > len(best_path):
                        best_path = full_path
                        
    return best_path

# Test locally
# print(construct())
