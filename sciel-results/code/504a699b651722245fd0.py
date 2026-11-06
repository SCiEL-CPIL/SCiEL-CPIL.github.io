import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adj(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def verify_and_fix(paths, edges):
        """Ensure paths are simple, use valid edges, edge-disjoint, cover all edges."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        used = set()
        valid_paths = []
        for p in paths:
            # Ensure simplicity
            if len(p) != len(set(p)):
                seen = set()
                fixed = []
                for v in p:
                    if v in seen:
                        break
                    seen.add(v)
                    fixed.append(v)
                p = fixed
            if len(p) < 2:
                continue
            ok = True
            for i in range(len(p) - 1):
                e = tuple(sorted((p[i], p[i+1])))
                if e not in edge_set or e in used:
                    ok = False
                    break
            if ok:
                for i in range(len(p) - 1):
                    used.add(tuple(sorted((p[i], p[i+1]))))
                valid_paths.append(p)
        # Add any missing edges as single-edge paths
        for e in edge_set:
            if e not in used:
                valid_paths.append([e[0], e[1]])
                used.add(e)
        return valid_paths

    def max_bipartite_matching(n_left, n_right, adj_list):
        """
        Hopcroft-Karp algorithm for maximum bipartite matching.
        adj_list[i] is a list of nodes in the right set connected to node i in the left set.
        Returns:
        match_l: array where match_l[u] = v if u is matched to v, else -1
        match_r: array where match_r[v] = u if v is matched to u, else -1
        """
        INF = float('inf')
        match_l = [-1] * n_left
        match_r = [-1] * n_right
        dist = [0] * n_left

        def bfs():
            queue = deque()
            for u in range(n_left):
                if match_l[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = INF
            found = False
            while queue:
                u = queue.popleft()
                for v in adj_list[u]:
                    if match_r[v] != -1:
                        if dist[match_r[v]] == INF:
                            dist[match_r[v]] = dist[u] + 1
                            queue.append(match_r[v])
                    else:
                        found = True
            return found

        def dfs(u):
            for v in adj_list[u]:
                if match_r[v] == -1 or (dist[match_r[v]] == dist[u] + 1 and dfs(match_r[v])):
                    match_l[u] = v
                    match_r[v] = u
                    return True
            dist[u] = INF
            return False

        matching = 0
        while bfs():
            for u in range(n_left):
                if match_l[u] == -1:
                    if dfs(u):
                        matching += 1
        return match_l, match_r

    def decompose_via_matching(edges, deadline):
        """
        1. Build bipartite graph of edge stubs.
        2. Find max matching.
        3. Reconstruct trails.
        4. Split trails into simple paths if necessary.
        """
        if not edges:
            return []
        
        # Map each undirected edge to two directed stubs
        # Left side: tail u, head v (edge index i, direction 0)
        # Right side: tail v, head u (edge index i, direction 1)
        # Actually, let's define:
        # Left node (i, 0) represents traversing edge i from u_i to v_i
        # Right node (j, 1) represents traversing edge j from v_j to u_j
        # We want to match Left(i, 0) to Right(j, 1) if head of i == tail of j.
        # head of i is v_i. tail of j is v_j.
        # So match if v_i == v_j.
        
        m = len(edges)
        # n_left = m, n_right = m
        # Left index i corresponds to edge i (u_i, v_i)
        # Right index j corresponds to edge j (u_j, v_j) but reversed? 
        # Let's be precise.
        # A path is a sequence of edges e1, e2, ... ek such that end(e1)=start(e2).
        # Let's represent an edge e_i = (u_i, v_i).
        # Stub L_i: enters u_i, leaves v_i. (Wait, standard matching for path cover usually matches "end of i" to "start of j")
        # Let's use:
        # Left node i: represents the "end" of edge i (vertex v_i).
        # Right node j: represents the "start" of edge j (vertex u_j).
        # Match L_i to R_j if v_i == u_j.
        # This chains i -> j.
        
        # However, we must ensure we don't chain an edge to itself in a way that creates a 2-cycle if we aren't careful, 
        # but since we are matching distinct stubs, L_i and R_i are distinct nodes.
        # We must prevent matching L_i to R_i? 
        # If we match L_i to R_i, it means we go from v_i to u_i. This is traversing edge i backwards.
        # But edge i is undirected. If we use edge i, we can traverse it either way.
        # In the bipartite graph formulation for undirected graphs:
        # We create 2 nodes for each edge: one for direction u->v (L) and one for v->u (R).
        # Let's stick to:
        # Node L_i: "I am finishing edge i at vertex v_i"
        # Node R_j: "I am starting edge j at vertex u_j"
        # Edge in bipartite graph: (L_i, R_j) exists if v_i == u_j AND i != j? 
        # Actually, if i == j, v_i == u_j implies u_i == v_i, which is a self-loop. 
        # Assuming no self-loops (random graphs usually don't have them, but instance generator might? 
        # zip(order, order[1:]) creates a path, no self loops. Random edges u<v, no self loops. So no self loops.)
        
        # So, construct adj_list for Hopcroft-Karp
        # n_left = m, n_right = m
        adj_list = [[] for _ in range(m)]
        
        # Precompute vertex to edge indices for fast lookup
        # end_vertex_to_edges[v] = list of edge indices i where v_i == v
        # start_vertex_to_edges[u] = list of edge indices j where u_j == u
        
        end_map = defaultdict(list)
        start_map = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            end_map[v].append(i)
            start_map[u].append(i)
            
        for i, (u, v) in enumerate(edges):
            # L_i corresponds to ending at v
            # We can connect to any R_j that starts at v
            for j in start_map[v]:
                if i != j: # Cannot chain an edge to itself in a simple path decomposition context easily without care, but matching allows it. 
                           # If we match L_i to R_i, it means we traverse i, then immediately traverse i again? No, that would use edge i twice.
                           # The matching selects a set of edges in the bipartite graph.
                           # If (L_i, R_i) is in matching, it implies we use "end of i" and "start of i".
                           # This doesn't necessarily mean we use edge i twice. It means the path segment ...->i->i->... is formed?
                           # No, the path is formed by the matching edges.
                           # If L_i is matched to R_j, it means edge i is followed by edge j.
                           # If L_i is matched to R_i, edge i is followed by edge i. This is impossible in a simple path (no repeated edges).
                           # So we MUST exclude i == j.
                           pass
                if i != j:
                    adj_list[i].append(j)

        match_l, match_r = max_bipartite_matching(m, m, adj_list)
        
        # Reconstruct paths
        # match_l[i] = j means i is followed by j.
        # match_r[j] = i means i is followed by j.
        
        # Find starts: Nodes i where match_l[i] == -1 (no predecessor)
        # Wait, match_l is for Left nodes (Ends). If match_l[i] == -1, it means the "End" of i is not used to start another edge.
        # This doesn't tell us if i is a start.
        # A start node is one whose "Start" stub (R_i) is not matched. i.e., match_r[i] == -1.
        
        starts = [i for i in range(m) if match_r[i] == -1]
        
        paths = []
        visited_edges = [False] * m
        
        for start_edge in starts:
            if visited_edges[start_edge]:
                continue
            
            path_edges = []
            curr = start_edge
            while curr != -1:
                if visited_edges[curr]:
                    break # Safety
                visited_edges[curr] = True
                path_edges.append(curr)
                # Next edge is determined by match_l[curr]
                curr = match_l[curr]
            
            # Convert edge indices to vertex list
            if path_edges:
                verts = []
                # Determine orientation
                # We need to know the order of vertices.
                # For the first edge, we know it starts at u_start.
                # We can trace the vertices.
                # Let's just build the vertex list by following the edges.
                # This requires knowing the orientation of each edge in the path.
                # match_l[i] = j means edge i ends at v_i, and edge j starts at u_j, and v_i == u_j.
                # So the sequence of edges e1, e2, ... ek is consistent.
                
                # Reconstruct vertices:
                # e1 = (u1, v1). Since it's a start, it must be traversed u1->v1.
                # e2 = (u2, v2). It follows e1, so v1 == u2. Traversed u2->v2.
                # So we can just list u1, v1(=u2), v2(=u3)...
                
                verts.append(edges[path_edges[0]][0])
                for k in range(len(path_edges) - 1):
                    # Current edge is path_edges[k], next is path_edges[k+1]
                    # End of current is edges[path_edges[k]][1]
                    # Start of next is edges[path_edges[k+1]][0]
                    # They should be equal.
                    verts.append(edges[path_edges[k]][1])
                # Add the end of the last edge
                verts.append(edges[path_edges[-1]][1])
                
                paths.append(verts)
                
        # Check for any unvisited edges (should not happen if matching covers all "starts" correctly, 
        # but if there are cycles in the matching that don't connect to a start, they form closed trails)
        # A closed trail in the matching means a cycle of edges e1->e2->...->ek->e1.
        # In this case, no edge has match_r[i] == -1. So it won't be in 'starts'.
        # We need to handle these cycles.
        for i in range(m):
            if not visited_edges[i]:
                # Found a cycle
                path_edges = []
                curr = i
                while not visited_edges[curr]:
                    visited_edges[curr] = True
                    path_edges.append(curr)
                    curr = match_l[curr]
                    
                if path_edges:
                    # It's a cycle. We can break it at any point to make it a path.
                    # Let's just take the sequence as is, but it's a cycle.
                    # To make it a simple path, we can just list the vertices.
                    # Since it's a cycle, the last vertex connects to the first.
                    # We can just output the vertices in the order of the edges.
                    verts = []
                    verts.append(edges[path_edges[0]][0])
                    for k in range(len(path_edges) - 1):
                        verts.append(edges[path_edges[k]][1])
                    # The last edge's end connects back to the first edge's start.
                    # We just append the end of the last edge.
                    verts.append(edges[path_edges[-1]][1])
                    paths.append(verts)
                    
        return paths

    def local_search_improve(paths, edges, deadline):
        """
        Try to reduce the number of paths by splitting and recombining.
        Move: Take a path P, split it into P1 and P2. Try to merge P1 with another path Q to form Q1, and P2 with R to form R1, such that total count decreases.
        Simpler Move: Try to merge two existing paths if they share an endpoint and the merged path is simple.
        """
        current_paths = [list(p) for p in paths]
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        improved = True
        max_iter = 200
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < deadline - 2:
            improved = False
            iter_count += 1
            new_paths = []
            used = [False] * len(current_paths)
            
            for i in range(len(current_paths)):
                if used[i]:
                    continue
                p1 = current_paths[i]
                merged = False
                for j in range(len(current_paths)):
                    if i == j or used[j]:
                        continue
                    p2 = current_paths[j]
                    
                    # Try p1 + p2[1:]
                    if p1[-1] == p2[0]:
                        merged_path = p1 + p2[1:]
                        if len(merged_path) == len(set(merged_path)) and all(
                            tuple(sorted((merged_path[k], merged_path[k+1]))) in edge_set
                            for k in range(len(merged_path) - 1)
                        ):
                            new_paths.append(merged_path)
                            used[j] = True
                            merged = True
                            improved = True
                            break
                            
                    # Try p2 + p1[1:]
                    if p2[-1] == p1[0]:
                        merged_path = p2 + p1[1:]
                        if len(merged_path) == len(set(merged_path)) and all(
                            tuple(sorted((merged_path[k], merged_path[k+1]))) in edge_set
                            for k in range(len(merged_path) - 1)
                        ):
                            new_paths.append(merged_path)
                            used[j] = True
                            merged = True
                            improved = True
                            break
                            
                if not merged:
                    new_paths.append(p1)
            
            current_paths = new_paths
            
            # If no simple merge worked, try a "split and recombine" heuristic?
            # For now, simple merge is the primary local search.
            # If we want to be more aggressive, we could try to break a long path and re-attach ends.
            # But simple merge is a good baseline.
            
        return current_paths

    def decompose_one(edges, deadline):
        if not edges:
            return []
        
        # Strategy 1: Bipartite Matching (Deterministic, strong lower bound)
        if time.time() < deadline - 10:
            paths = decompose_via_matching(edges, deadline)
            paths = verify_and_fix(paths, edges)
            paths = local_search_improve(paths, edges, deadline)
            best_count = len(paths)
            best_paths = paths
        else:
            best_paths = [[u, v] for u, v in edges]
            best_count = len(best_paths)
            
        # Strategy 2: Randomized Greedy (Fallback/Competitor)
        # Run a few quick greedy attempts to see if we can beat the matching solution
        # (Matching gives min trails. Greedy might find a better *simple* path packing if matching required splits)
        
        def greedy_decompose(edges, rng, deadline):
            remaining = set(tuple(sorted(e)) for e in edges)
            if not remaining:
                return []
            adj = defaultdict(set)
            for u, v in remaining:
                adj[u].add(v)
                adj[v].add(u)
            degree = defaultdict(int)
            for u, v in remaining:
                degree[u] += 1
                degree[v] += 1
            paths = []
            while remaining and time.time() < deadline:
                # Pick start
                odd_verts = [v for v in adj if degree.get(v, 0) % 2 == 1 and adj[v]]
                if odd_verts:
                    start = rng.choice(odd_verts)
                else:
                    cands = [v for v in adj if adj[v]]
                    if not cands:
                        break
                    cands.sort(key=lambda v: degree.get(v, 0), reverse=True)
                    start = rng.choice(cands[:min(5, len(cands))])
                    
                # Extend
                path = [start]
                visited = {start}
                while True:
                    cur = path[-1]
                    nbs = [nb for nb in adj[cur] if nb not in visited and (tuple(sorted((cur, nb)))) in remaining]
                    if not nbs:
                        break
                    # Warnsdorff
                    def score(nb):
                        return sum(1 for x in adj[nb] if x not in visited and (tuple(sorted((nb, x)))) in remaining)
                    nbs.sort(key=score)
                    choice = rng.choice(nbs[:min(3, len(nbs))])
                    path.append(choice)
                    visited.add(choice)
                    
                if len(path) < 2:
                    nbs = [nb for nb in adj[start] if (tuple(sorted((start, nb)))) in remaining]
                    if nbs:
                        path = [start, rng.choice(nbs)]
                    else:
                        e = next(iter(remaining))
                        path = [e[0], e[1]]
                        
                # Remove edges
                path_edges = set()
                for i in range(len(path) - 1):
                    e = tuple(sorted((path[i], path[i+1])))
                    if e in remaining:
                        path_edges.add(e)
                if not path_edges:
                    e = next(iter(remaining))
                    path = [e[0], e[1]]
                    path_edges = {e}
                    
                for e in path_edges:
                    remaining.discard(e)
                    u, v = e
                    adj[u].discard(v)
                    adj[v].discard(u)
                    degree[u] = max(0, degree.get(u, 0) - 1)
                    degree[v] = max(0, degree.get(v, 0) - 1)
                for v in list(adj.keys()):
                    if not adj[v]:
                        del adj[v]
                        
                paths.append(path)
                
            return paths

        # Run a few greedy restarts if time permits
        if time.time() < deadline - 5:
            for seed in range(5):
                if time.time() > deadline - 2:
                    break
                rng = random.Random(seed * 7919 + 13)
                g_paths = greedy_decompose(edges, rng, deadline - 2)
                g_paths = verify_and_fix(g_paths, edges)
                g_paths = local_search_improve(g_paths, edges, deadline - 1)
                if len(g_paths) < best_count:
                    best_count = len(g_paths)
                    best_paths = g_paths
                    
        return best_paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    for i, spec in enumerate(specs):
        if time.time() > deadline:
            break
        edges = instance(*spec)
        remaining_time = deadline - time.time()
        # Allocate time proportional to graph size? Or equal?
        # Graphs are small, so equal allocation is fine.
        per = remaining_time / (3 - len(results))
        paths = decompose_one(edges, time.time() + per)
        results.append(paths)

    while len(results) < 3:
        spec = specs[len(results)]
        edges = instance(*spec)
        results.append([[u, v] for u, v in edges])

    return results
