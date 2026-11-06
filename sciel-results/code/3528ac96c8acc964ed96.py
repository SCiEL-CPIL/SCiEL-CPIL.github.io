import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = set()
        for i in range(n - 1):
            u, v = order[i], order[i + 1]
            edges.add((min(u, v), max(u, v)))
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return list(edges)

    def beam_search_longest_path(adj, curr_edges, start, beam_width=15, max_depth=100):
        """Find longest simple path starting from 'start' using beam search."""
        if not curr_edges:
            return None
        
        # Check if start has any available edges
        has_edge = False
        for nb in adj[start]:
            e = (min(start, nb), max(start, nb))
            if e in curr_edges:
                has_edge = True
                break
        if not has_edge:
            return None
        
        beam = [([start], {start}, start)]
        local_best = [start]
        
        for d in range(max_depth):
            new_beam = []
            for path, visited, node in beam:
                valid_neighbors = []
                for nb in adj[node]:
                    if nb in visited:
                        continue
                    e = (min(node, nb), max(node, nb))
                    if e in curr_edges:
                        valid_neighbors.append(nb)
                
                if not valid_neighbors:
                    if len(path) > len(local_best):
                        local_best = path[:]
                    continue
                
                # Score neighbors using Warnsdorff's rule
                scored = []
                for nb in valid_neighbors:
                    count = 0
                    for m in adj[nb]:
                        if m in visited:
                            continue
                        e2 = (min(nb, m), max(nb, m))
                        if e2 in curr_edges:
                            count += 1
                    scored.append((count, nb))
                scored.sort()
                
                for _, nb in scored[:beam_width]:
                    new_path = path + [nb]
                    new_visited = visited | {nb}
                    new_beam.append((new_path, new_visited, nb))
            
            if not new_beam:
                break
            
            # Keep top beam_width by path length
            new_beam.sort(key=lambda x: -len(x[0]))
            beam = new_beam[:beam_width]
            
            longest = max(len(p) for p, v, n_ in beam)
            if longest > len(local_best):
                local_best = beam[0][0][:]
        
        return local_best if len(local_best) > 1 else None

    def greedy_extract_paths(n, adj, edges, max_restarts=80, beam_width=15):
        """Greedy path extraction with multiple restarts."""
        remaining_edges = set(edges)
        best_paths = None
        best_count = float('inf')
        
        for r in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
            if best_count <= 1:
                break
                
            curr_edges = set(remaining_edges)
            curr_paths = []
            
            # Randomize start selection order
            starts = list(range(n))
            random.shuffle(starts)
            
            while curr_edges:
                if time.time() > deadline - 2.0:
                    curr_paths = None
                    break
                
                # Find active starts (vertices with remaining edges)
                active_starts = []
                for s in starts:
                    for nb in adj[s]:
                        e = (min(s, nb), max(s, nb))
                        if e in curr_edges:
                            active_starts.append(s)
                            break
                
                if not active_starts:
                    break
                
                # Try to find longest path from several candidate starts
                best_path = None
                candidates = active_starts[:min(10, len(active_starts))]
                
                for s in candidates:
                    path = beam_search_longest_path(adj, curr_edges, s, beam_width=beam_width)
                    if path and (best_path is None or len(path) > len(best_path)):
                        best_path = path
                
                if best_path and len(best_path) > 1:
                    curr_paths.append(best_path)
                    for i in range(len(best_path) - 1):
                        u, v = best_path[i], best_path[i + 1]
                        e = (min(u, v), max(u, v))
                        curr_edges.discard(e)
                else:
                    # Single edge fallback
                    if curr_edges:
                        e = next(iter(curr_edges))
                        curr_paths.append(list(e))
                        curr_edges.discard(e)
            
            if curr_paths and len(curr_paths) < best_count:
                best_count = len(curr_paths)
                best_paths = curr_paths[:]
        
        if best_paths is None:
            best_paths = [list(e) for e in edges]
        
        return best_paths

    def validate_and_fix(edge_set, paths):
        """Ensure paths are simple, use valid edges, and are edge-disjoint."""
        used = set()
        ok_paths = []
        for vp in paths:
            newp = []
            ok = True
            for k in range(len(vp)):
                vtx = vp[k]
                if newp and vtx == newp[-1]:
                    continue
                if not newp:
                    newp.append(vtx)
                else:
                    a, b = newp[-1], vtx
                    e = (min(a, b), max(a, b))
                    if e not in edge_set or e in used:
                        ok = False
                        break
                    used.add(e)
                    newp.append(vtx)
            if ok and newp:
                ok_paths.append(newp)
        
        # Add any uncovered edges as singletons
        for e in edge_set:
            if e not in used:
                ok_paths.append(list(e))
                used.add(e)
        return ok_paths

    def try_merge_paths(paths, edge_set):
        """Try to merge two paths if their endpoints are adjacent."""
        improved = True
        while improved and (time.time() < deadline - 5.0):
            improved = False
            new_paths = list(paths)
            
            for ia in range(len(new_paths)):
                if improved:
                    break
                for ib in range(len(new_paths)):
                    if ia == ib or improved:
                        continue
                    A = new_paths[ia]
                    B = new_paths[ib]
                    
                    # Try B + A: check if B's end connects to A's start
                    if len(B) >= 1 and len(A) >= 1:
                        b_end = B[-1]
                        a_start = A[0]
                        e = (min(b_end, a_start), max(b_end, a_start))
                        if e in edge_set:
                            # Check if this edge is already used in A or B
                            edge_used = False
                            for vp in (A, B):
                                for k in range(len(vp) - 1):
                                    x, y = vp[k], vp[k+1]
                                    if (min(x,y), max(x,y)) == e:
                                        edge_used = True
                                        break
                                if edge_used:
                                    break
                            if not edge_used:
                                newp = B + A[1:]
                                # Validate simplicity
                                seen = set()
                                valid = True
                                for v in newp:
                                    if v in seen:
                                        valid = False
                                        break
                                    seen.add(v)
                                if valid:
                                    new_paths = [p for k, p in enumerate(new_paths) if k != ia and k != ib]
                                    new_paths.append(newp)
                                    paths = new_paths
                                    improved = True
                                    break
                    
                    # Try A + B: check if A's end connects to B's start
                    if len(A) >= 1 and len(B) >= 1:
                        a_end = A[-1]
                        b_start = B[0]
                        e = (min(a_end, b_start), max(a_end, b_start))
                        if e in edge_set:
                            edge_used = False
                            for vp in (A, B):
                                for k in range(len(vp) - 1):
                                    x, y = vp[k], vp[k+1]
                                    if (min(x,y), max(x,y)) == e:
                                        edge_used = True
                                        break
                                if edge_used:
                                    break
                            if not edge_used:
                                newp = A + B[1:]
                                seen = set()
                                valid = True
                                for v in newp:
                                    if v in seen:
                                        valid = False
                                        break
                                    seen.add(v)
                                if valid:
                                    new_paths = [p for k, p in enumerate(new_paths) if k != ia and k != ib]
                                    new_paths.append(newp)
                                    paths = new_paths
                                    improved = True
                                    break
        
        return paths

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    all_results = []

    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        edge_set = set(edges)
        
        if not edges:
            all_results.append([])
            continue
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Greedy path extraction with beam search
        paths = greedy_extract_paths(n, adj, edges, max_restarts=80, beam_width=15)
        
        # Validate and fix
        paths = validate_and_fix(edge_set, paths)
        
        # Local improvement: merge paths
        paths = try_merge_paths(paths, edge_set)
        
        all_results.append(paths)

    return all_results
