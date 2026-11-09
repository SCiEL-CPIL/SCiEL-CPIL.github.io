"""
Hybrid approach: 
1. Structural Initialization via Maximum Bipartite Matching (Hopcroft-Karp) 
   to find a maximum set of vertex-disjoint edges (a 2-regular-ish subgraph).
2. Path Stitching to merge the resulting disjoint paths/cycles into a single path.
3. Local Search and DFS Extension to refine and maximize length.
"""
import random
import time
from collections import deque

N = 100
EDGES = [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]

ADJ = [set() for _ in range(N)]
DEG = [0] * N
for u, v in EDGES:
    ADJ[u].add(v)
    ADJ[v].add(u)
    DEG[u] += 1
    DEG[v] += 1

def find_components():
    visited = set()
    comps = []
    for s in range(N):
        if s not in visited:
            comp = set()
            stack = [s]
            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u)
                comp.add(u)
                for v in ADJ[u]:
                    if v not in visited:
                        stack.append(v)
            comps.append(comp)
    return comps

def hopcroft_karp(bipartite_adj):
    """
    Hopcroft-Karp algorithm for maximum bipartite matching.
    bipartite_adj: list of lists, where bipartite_adj[u] contains v for u in left, v in right.
    Returns: list of length len(left), where match_left[u] = v (or -1 if unmatched)
    """
    n_left = len(bipartite_adj)
    n_right = max([max(adj) for adj in bipartite_adj] + [0]) + 1
    
    match_left = [-1] * n_left
    match_right = [-1] * n_right
    dist = [0] * n_left
    
    def bfs():
        queue = deque()
        for u in range(n_left):
            if match_left[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = float('inf')
        
        found = False
        while queue:
            u = queue.popleft()
            for v in bipartite_adj[u]:
                if match_right[v] == -1:
                    found = True
                elif dist[match_right[v]] == float('inf'):
                    dist[match_right[v]] = dist[u] + 1
                    queue.append(match_right[v])
        return found

    def dfs(u):
        for v in bipartite_adj[u]:
            if match_right[v] == -1 or (dist[match_right[v]] == dist[u] + 1 and dfs(match_right[v])):
                match_left[u] = v
                match_right[v] = u
                return True
        dist[u] = float('inf')
        return False

    matching_size = 0
    while bfs():
        for u in range(n_left):
            if match_left[u] == -1:
                if dfs(u):
                    matching_size += 1
    return match_left

def get_initial_path_from_matching(comp_vertices):
    """
    Constructs a bipartite graph for the subgraph induced by comp_vertices,
    finds max matching, and stitches the resulting paths/cycles into a single path.
    """
    # Map local indices for bipartite graph construction
    # Left side: 0..n-1 (v_in), Right side: 0..n-1 (v_out)
    # Edge (u, v) in G implies edges (u_out, v_in) and (v_out, u_in) in Bipartite
    # Actually, standard reduction:
    # Vertex v splits into v_left and v_right.
    # Edge (u,v) in G adds edge (u_right, v_left) and (v_right, u_left) in Bipartite.
    # A matching M in Bipartite corresponds to a set of edges in G such that no two share a vertex.
    
    verts = sorted(comp_vertices)
    n = len(verts)
    idx_map = {v: i for i, v in enumerate(verts)}
    
    # Bipartite adjacency: Left nodes 0..n-1, Right nodes 0..n-1
    # Let Left represent "out" of u, Right represent "in" of v?
    # If we pick edge (u_right, v_left), it means we use edge (u,v) in G.
    # Constraint: u can only have one outgoing and one incoming in the matching.
    # So we need to ensure that if (u_right, v_left) is matched, then no other edge incident to u is matched.
    # This is exactly what bipartite matching on the split graph does.
    
    bip_adj = [[] for _ in range(n)]
    
    for u in verts:
        u_idx = idx_map[u]
        for v in ADJ[u]:
            if v in idx_map:
                v_idx = idx_map[v]
                # Add edge from u_right (left side u_idx) to v_left (right side v_idx)
                # And edge from v_right (left side v_idx) to u_left (right side u_idx)
                bip_adj[u_idx].append(v_idx)
                bip_adj[v_idx].append(u_idx)
    
    # Run Hopcroft-Karp
    match_left = hopcroft_karp(bip_adj)
    
    # Reconstruct the set of edges in G
    # If match_left[u_idx] == v_idx, it means edge (u, v) is selected?
    # Careful: bip_adj[u_idx] contains v_idx if edge (u,v) exists.
    # If match_left[u_idx] == v_idx, then (u_right, v_left) is matched. This corresponds to directed edge u->v.
    # Since the graph is undirected, we just need to collect the undirected edges.
    
    selected_edges = set()
    for u_idx in range(n):
        v_idx = match_left[u_idx]
        if v_idx != -1:
            u = verts[u_idx]
            v = verts[v_idx]
            selected_edges.add((min(u, v), max(u, v)))
    
    # Now we have a set of edges forming a subgraph where max degree is 2.
    # This subgraph is a collection of disjoint paths and cycles.
    # We want to merge them into a single path.
    
    # Build adjacency for the selected edges
    sel_adj = {v: set() for v in verts}
    for u, v in selected_edges:
        sel_adj[u].add(v)
        sel_adj[v].add(u)
    
    # Find all paths and cycles in this subgraph
    visited_nodes = set()
    components = []
    
    for start in verts:
        if start in visited_nodes:
            continue
        
        # Traverse the component
        if not sel_adj[start]:
            # Isolated vertex
            visited_nodes.add(start)
            components.append([start])
            continue
            
        # Determine if it's a path or cycle
        # A path has 0 or 2 nodes of degree 1. A cycle has 0 nodes of degree 1 (all degree 2).
        # We can just traverse from a node of degree 1 if it exists, otherwise pick any node and break the cycle.
        
        path = []
        # Find a node with degree 1 to start if possible
        start_node = None
        for v in verts:
            if v in sel_adj and len(sel_adj[v]) == 1 and v not in visited_nodes:
                start_node = v
                break
        
        if start_node is None:
            # It's a cycle or all nodes are visited?
            # If it's a cycle, pick any unvisited node in this component
            # We need to identify the component first
            comp_stack = [start]
            comp_nodes = set()
            while comp_stack:
                u = comp_stack.pop()
                if u in comp_nodes:
                    continue
                comp_nodes.add(u)
                for v in sel_adj[u]:
                    if v not in comp_nodes:
                        comp_stack.append(v)
            
            # Check if any node in comp_nodes has degree 1
            has_degree_1 = any(len(sel_adj[v]) == 1 for v in comp_nodes)
            if has_degree_1:
                for v in comp_nodes:
                    if len(sel_adj[v]) == 1:
                        start_node = v
                        break
            else:
                # It's a cycle. Break it by picking a node and removing an edge?
                # For now, just treat it as a path by picking a start and traversing until we return to start.
                # We will handle cycles by breaking one edge arbitrarily in the stitching phase.
                # For initial construction, let's just extract the cycle as a list.
                start_node = comp_nodes.pop()
                path.append(start_node)
                visited_nodes.add(start_node)
                prev = start_node
                curr = start_node
                while True:
                    # Find next node
                    nexts = [v for v in sel_adj[curr] if v != prev]
                    if not nexts:
                        break
                    curr = nexts[0]
                    if curr == start_node:
                        break
                    path.append(curr)
                    visited_nodes.add(curr)
                    prev = curr
                components.append(path)
                continue
        
        # It's a path
        curr = start_node
        visited_nodes.add(curr)
        path.append(curr)
        prev = None
        while True:
            nexts = [v for v in sel_adj[curr] if v != prev]
            if not nexts:
                break
            curr = nexts[0]
            if curr in visited_nodes:
                break # Should not happen for a path
            path.append(curr)
            visited_nodes.add(curr)
            prev = curr
        components.append(path)
        
    # Now we have a list of paths (and potentially cycles represented as paths).
    # We want to merge these into a single path.
    # We can do this by looking for connections between the ends of these paths in the original graph.
    
    # Sort components by length (descending) to prioritize long chains
    components.sort(key=len, reverse=True)
    
    # Stitching
    # We have a set of disjoint paths. We want to connect them.
    # Two paths P1 = [a, ..., b] and P2 = [c, ..., d] can be connected if:
    # (b, c) is an edge in G, or (b, d) is an edge in G, etc.
    # If we connect them, we form a longer path.
    
    # We will try to greedily merge components.
    # Represent each component as a list. 
    # We can reverse a path if needed.
    
    active_components = [p[:] for p in components if p]
    
    # To make merging easier, we can try to find edges between endpoints of different components.
    # Build a graph of components where nodes are components and edges exist if endpoints are adjacent in G.
    
    # This is a bit complex. A simpler heuristic:
    # Pick the longest component as the base.
    # Try to extend it by finding other components that can be attached to its ends.
    
    if not active_components:
        return [verts[0]] if verts else []
        
    base_path = active_components.pop(0)
    base_set = set(base_path)
    
    # Try to extend base_path
    changed = True
    while changed and active_components:
        changed = False
        # Check if we can attach any other component to the start or end of base_path
        # We need to check if there is an edge between an endpoint of base_path and an endpoint of another component.
        
        # To allow reversal, we consider both orientations of the other component.
        
        for i, other in enumerate(active_components):
            if not other:
                continue
            # Try connecting base_path[-1] to other[0] or other[-1]
            # And base_path[0] to other[0] or other[-1]
            
            # 1. base_end -> other_start
            if base_path[-1] in ADJ[other[0]]:
                # Merge: base + other
                base_path.extend(other)
                active_components.pop(i)
                changed = True
                break
            # 2. base_end -> other_end (reverse other)
            if base_path[-1] in ADJ[other[-1]]:
                # Merge: base + reversed(other)
                base_path.extend(reversed(other))
                active_components.pop(i)
                changed = True
                break
            # 3. base_start -> other_start (reverse base? No, just prepend)
            if base_path[0] in ADJ[other[0]]:
                # Merge: reversed(other) + base
                base_path = other + base_path
                active_components.pop(i)
                changed = True
                break
            # 4. base_start -> other_end (reverse other, prepend)
            if base_path[0] in ADJ[other[-1]]:
                # Merge: reversed(other) + base
                base_path = list(reversed(other)) + base_path
                active_components.pop(i)
                changed = True
                break
        
        # If we couldn't attach directly, maybe we can insert a component in the middle?
        # That's harder. Let's stick to end-attachments for now.
        # If no changes, we might be stuck. 
        # We could try to break a cycle in 'other' to make it a path if it was a cycle?
        # The initial extraction already treated cycles as paths (by breaking one edge implicitly in traversal? 
        # Actually, my cycle traversal logic above might have produced a closed list or a path.
        # If it produced a path, it has endpoints. If it produced a cycle, it's a list where first and last are adjacent?
        # The code above: `if curr == start_node: break` ensures it's a path in the list representation, 
        # but the edge (last, first) exists in the graph. 
        # So we can use that edge to connect to other things?
        # Actually, if it's a cycle, we can break any edge. 
        # For simplicity, if we are stuck, we might just return the longest path found.
        
        # To improve, if we are stuck, we can try to "break" a cycle component into a path 
        # and try again. But for now, let's see if this works.
        
        # If no component was merged in this pass, break to avoid infinite loop
        if not changed:
             # Try to merge by breaking a cycle?
             # Identify cycles: a component where first and last are adjacent in G.
             # If we break it, we get a path.
             # Let's try to break any cycle in active_components and re-evaluate.
             broken = False
             for i, comp in enumerate(active_components):
                 if len(comp) > 1 and comp[0] in ADJ[comp[-1]]:
                     # It's a cycle. Break it.
                     # Remove the last node from the end and prepend it? No, that changes the path.
                     # Just treat it as a path. The edge (last, first) is not used in the path list.
                     # So it's already a valid path list. 
                     # The issue is connectivity. 
                     # Maybe we can rotate the path? 
                     # For now, just break the loop.
                     pass
             break

    # If there are still active components, we can't attach them easily.
    # We return the base_path.
    # Note: This might not include all vertices.
    
    return base_path

def greedy_warnsdorff(start, rng):
    """Warnsdorff's rule: pick neighbor with fewest unvisited neighbors."""
    seen = {start}
    path = [start]
    while True:
        cand = ADJ[path[-1]] - seen
        if not cand:
            break
        v = min(cand, key=lambda w: (len(ADJ[w] - seen), rng.random()))
        path.append(v)
        seen.add(v)
    return path

def extend_both_ends(path, seen):
    """Extend path from both ends using Warnsdorff."""
    path = list(path)
    seen = set(seen)
    improved = True
    while improved:
        improved = False
        # Extend front
        while True:
            cand = ADJ[path[0]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: (len(ADJ[w] - seen), w))
            path.insert(0, v)
            seen.add(v)
            improved = True
        # Extend back
        while True:
            cand = ADJ[path[-1]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: (len(ADJ[w] - seen), w))
            path.append(v)
            seen.add(v)
            improved = True
    return path

def local_improve(path, seen, rng, deadline):
    """Try to insert unvisited vertices into the path, extend ends."""
    path = list(path)
    seen = set(seen)
    unvisited = set(range(N)) - seen
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try inserting each unvisited vertex
        for u in list(unvisited):
            found = False
            for i in range(len(path) - 1):
                if path[i] in ADJ[u] and path[i+1] in ADJ[u]:
                    path.insert(i + 1, u)
                    seen.add(u)
                    unvisited.remove(u)
                    improved = True
                    found = True
                    break
            # Also try at ends
            if not found:
                if path[0] in ADJ[u]:
                    path.insert(0, u)
                    seen.add(u)
                    unvisited.remove(u)
                    improved = True
                elif path[-1] in ADJ[u]:
                    path.append(u)
                    seen.add(u)
                    unvisited.remove(u)
                    improved = True
        # Extend ends with Warnsdorff
        while True:
            cand = ADJ[path[0]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: (len(ADJ[w] - seen), w))
            path.insert(0, v)
            seen.add(v)
            improved = True
        while True:
            cand = ADJ[path[-1]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: (len(ADJ[w] - seen), w))
            path.append(v)
            seen.add(v)
            improved = True
    return path

def dfs_search(start, seen, path, best, deadline, max_depth, node_count):
    """DFS to find longer path from current state with node budget."""
    if time.time() > deadline or node_count[0] > 500000:
        return
    if len(path) > len(best):
        best.clear()
        best.extend(path)
    if len(path) >= max_depth:
        return
    cand = ADJ[path[-1]] - seen
    # Warnsdorff ordering
    cand = sorted(cand, key=lambda w: (len(ADJ[w] - seen), w))
    for v in cand:
        node_count[0] += 1
        path.append(v)
        seen.add(v)
        dfs_search(start, seen, path, best, deadline, max_depth, node_count)
        path.pop()
        seen.remove(v)
        if time.time() > deadline or node_count[0] > 500000:
            return

def construct():
    t0 = time.time()
    deadline = t0 + 75  # Safety margin
    
    comps = find_components()
    comps.sort(key=len, reverse=True)
    
    best_path = []
    rng = random.Random(42)
    
    for comp in comps:
        if time.time() > deadline:
            break
        comp_size = len(comp)
        comp_list = sorted(comp)
        target = comp_size
        
        # Phase 1: Matching-based initialization
        # This provides a strong structural lower bound
        initial_path = get_initial_path_from_matching(comp)
        if len(initial_path) > len(best_path):
            best_path = initial_path[:]
            
        # Extend the matching-based path
        if best_path:
            extended = extend_both_ends(best_path, set(best_path))
            if len(extended) > len(best_path):
                best_path = extended
        
        # Phase 2: Local improvement
        if len(best_path) > 0 and len(best_path) < target:
            if time.time() > deadline:
                break
            p = local_improve(best_path, set(best_path), rng, deadline)
            if len(p) > len(best_path):
                best_path = p
        
        # Phase 3: Greedy restarts if matching didn't find a long path
        # We keep this as a fallback or to find even longer paths if matching was suboptimal
        if len(best_path) < target:
            for start in comp_list:
                if time.time() > deadline:
                    break
                p = greedy_warnsdorff(start, rng)
                p = extend_both_ends(p, set(p))
                if len(p) > len(best_path):
                    best_path = p[:]
                    # Immediate local improvement on new best
                    if len(best_path) < target:
                        p2 = local_improve(best_path, set(best_path), rng, deadline)
                        if len(p2) > len(best_path):
                            best_path = p2
        
        # Phase 4: DFS refinement
        if len(best_path) < target:
            remaining_time = deadline - time.time()
            if remaining_time > 0.5:
                # Try DFS from endpoints of current best
                starts = [best_path[0], best_path[-1]] if best_path else [comp_list[0]]
                for start in starts:
                    if time.time() > deadline:
                        break
                    remaining = deadline - time.time()
                    if remaining < 0.1:
                        break
                    budget = min(2.0, remaining * 0.25)
                    dl = time.time() + budget
                    node_count = [0]
                    local_best = list(best_path)
                    dfs_search(start, {start}, [start], local_best, dl, target, node_count)
                    if len(local_best) > len(best_path):
                        best_path = local_best
                        if len(best_path) < target and time.time() < deadline:
                            p2 = local_improve(best_path, set(best_path), rng, deadline)
                            if len(p2) > len(best_path):
                                best_path = p2
                        if len(best_path) >= target:
                            break
    
    # Final safety check
    if not best_path:
        best_path = [0]
    
    # Verify path validity
    for i in range(len(best_path) - 1):
        if best_path[i+1] not in ADJ[best_path[i]]:
            return [0]
    if len(set(best_path)) != len(best_path):
        seen = set()
        fixed = []
        for v in best_path:
            if v not in seen:
                fixed.append(v)
                seen.add(v)
        return fixed
    
    return best_path

if __name__ == "__main__":
    # Quick self-test
    p = construct()
    print(f"Found path of length {len(p)}")
    print(p)
