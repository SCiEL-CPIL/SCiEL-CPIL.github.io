import time
import sys

N = 100
EDGES = [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]

# Build adjacency list
ADJ = [[] for _ in range(N)]
for u, v in EDGES:
    ADJ[u].append(v)
    ADJ[v].append(u)

# Find connected components
def find_components():
    visited = [False] * N
    comps = []
    for i in range(N):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in ADJ[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        stack.append(nei)
            comps.append(comp)
    return comps

def get_edges_in_component(comp):
    comp_set = set(comp)
    edges = 0
    for u in comp:
        for v in ADJ[u]:
            if v in comp_set and u < v:
                edges += 1
    return edges

def bfs_farthest(start, comp_set):
    """Returns (farthest_node, distance, path_to_farthest)"""
    dist = {start: 0}
    parent = {start: None}
    queue = [start]
    farthest = start
    max_dist = 0
    while queue:
        node = queue.pop(0)
        if dist[node] > max_dist:
            max_dist = dist[node]
            farthest = node
        for nei in ADJ[node]:
            if nei in comp_set and nei not in dist:
                dist[nei] = dist[node] + 1
                parent[nei] = node
                queue.append(nei)
    
    # Reconstruct path
    path = []
    curr = farthest
    while curr is not None:
        path.append(curr)
        curr = parent[curr]
    path.reverse()
    return farthest, max_dist, path

def tree_diameter(comp):
    """Find diameter of a tree component. Returns (length, path)"""
    if not comp:
        return 0, []
    comp_set = set(comp)
    # Pick an arbitrary node
    start = comp[0]
    # BFS to find farthest
    far1, dist1, path1 = bfs_farthest(start, comp_set)
    # BFS from far1 to find farthest
    far2, dist2, path2 = bfs_farthest(far1, comp_set)
    return dist2, path2

def longest_path_in_component(comp, deadline):
    """
    Find longest path in a component.
    If it's a tree, use diameter.
    If it has cycles, use DFS with pruning.
    """
    if not comp:
        return 0, []
    
    comp_set = set(comp)
    n_comp = len(comp)
    e_comp = get_edges_in_component(comp)
    
    # If it's a tree (or forest part, but comp is connected so it's a tree)
    if e_comp == n_comp - 1:
        length, path = tree_diameter(comp)
        return length, path
        
    # If it has cycles, we need to search.
    # Since n_comp is likely small or the graph is sparse, DFS is viable.
    # We use a global best path tracker.
    best_path = []
    
    # Optimization: If the component is large and has cycles, 
    # we can try to find a long path by modifying the tree diameter approach?
    # No, better to just DFS.
    
    # To speed up DFS, we can use Warnsdorff's rule.
    # We also prune if the remaining unvisited nodes are not connected to the current end.
    
    # Precompute adjacency for this component only for faster lookup?
    # ADJ is global, but we filter by comp_set.
    
    def dfs(node, visited, path):
        nonlocal best_path
        if time.time() > deadline:
            return
        
        if len(path) > len(best_path):
            best_path = path[:]
        
        # If we've visited all nodes, we can't go further
        if len(path) == n_comp:
            return
            
        # Check if we can prune: 
        # If the unvisited nodes are not connected to 'node', we stop.
        # This is expensive to check every step (BFS/DFS on unvisited).
        # However, for small components or sparse graphs, it might be worth it.
        # Let's skip this for now and rely on Warnsdorff + time limit.
        # If n_comp is large (e.g., > 50) and has cycles, this might time out.
        # But given the edge count, large components are likely trees.
        
        # Get unvisited neighbors
        neighbors = [n for n in ADJ[node] if n in comp_set and n not in visited]
        
        # Sort by Warnsdorff's rule: fewest unvisited neighbors first
        # This helps in finding long paths (avoiding dead ends early)
        def unvisited_count(n):
            return len([x for x in ADJ[n] if x in comp_set and x not in visited])
        
        neighbors.sort(key=unvisited_count)
        
        for nei in neighbors:
            visited.add(nei)
            path.append(nei)
            dfs(nei, visited, path)
            path.pop()
            visited.remove(nei)
            
            if time.time() > deadline:
                return

    # Start DFS from each node? 
    # Or just from one? 
    # For finding the *longest* path, starting from every node is safest but slow.
    # However, in a sparse graph, the longest path is likely to have endpoints with degree 1 or 2.
    # Let's try starting from all nodes in the component.
    # To save time, we can limit the DFS depth or use a heuristic.
    # Given the time limit of 90s, and N=100, if one component is a tree (99 nodes), 
    # the DFS would be exponential and fail. 
    # BUT, we already handled the tree case!
    # So we only run DFS on components with cycles.
    # How big can a cyclic component be?
    # If there is 1 cycle, it's likely a small loop attached to a tree.
    # The component size might be large (e.g., 50 nodes) but the "cyclic part" is small.
    # However, DFS on the whole component is still exponential in the size of the tree part if not careful.
    
    # Better approach for "Tree + Cycles":
    # 1. Find the tree diameter (ignoring cycle edges? No, that's hard).
    # 2. Just run DFS but with a node budget or time check.
    
    # Let's refine the DFS to be smarter.
    # If the component is large, say > 30 nodes, and has cycles, 
    # we can try to remove the "extra" edges to make it a tree, find diameter, 
    # then try to extend it using the removed edges.
    
    # For now, let's just run DFS with a strict time limit per component.
    # Since the tree case is handled exactly, we only risk time on cyclic components.
    # If a cyclic component is huge, we might not find the exact longest path, 
    # but we will find a good one (longer than diameter of the underlying tree).
    
    # To avoid hanging on a large cyclic component:
    # We'll run DFS from each node, but break early if we've exceeded time.
    
    for start in comp:
        if time.time() > deadline:
            break
        visited = {start}
        path = [start]
        dfs(start, visited, path)
        
    return len(best_path), best_path

def construct():
    t0 = time.time()
    deadline = t0 + 75.0 # Safety margin
    
    comps = find_components()
    
    best_overall_path = []
    
    for comp in comps:
        if time.time() > deadline:
            break
        length, path = longest_path_in_component(comp, deadline)
        if len(path) > len(best_overall_path):
            best_overall_path = path
            
    # Fallback if something went wrong
    if not best_overall_path:
        best_overall_path = [0]
        
    return best_overall_path
