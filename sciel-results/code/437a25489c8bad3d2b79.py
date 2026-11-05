import time
import random
from collections import defaultdict, deque

def construct():
    t0 = time.time()
    DEADLINE = 145.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_one(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []

        edge_set = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # 1. Greedy Initial Path Construction
        # We want to create simple paths that cover all edges.
        # Strategy: Start from a vertex, extend as far as possible without revisiting vertices.
        # To maximize path length, we prioritize extending from the end with more available neighbors.
        
        remaining_edges = set(edges)
        paths = []
        
        # Pre-compute degrees for initial heuristic
        deg = defaultdict(int)
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            
        # Sort vertices by degree descending to start from high-degree nodes
        vertices_by_deg = sorted(range(n), key=lambda x: -deg[x])
        
        # We will repeatedly pick a starting point and build a path.
        # To ensure we cover all edges, we pick the next start point from the remaining edges.
        
        # Helper to get available neighbors
        def get_avail_neighbors(u):
            nbrs = []
            for v in adj[u]:
                e = (u, v) if u < v else (v, u)
                if e in remaining_edges:
                    nbrs.append(v)
            return nbrs

        def remove_edge(u, v):
            e = (u, v) if u < v else (v, u)
            remaining_edges.discard(e)

        # Build initial paths
        # We use a simple greedy: pick an edge, extend it.
        # To make it better, we pick the edge with the highest sum of degrees of its endpoints.
        
        # However, a simpler and often effective method is:
        # 1. Find a vertex with max degree.
        # 2. Start a path from it, extend greedily.
        # 3. Repeat.
        
        # Let's use a randomized greedy construction with multiple restarts and keep the best.
        # Since the graphs are small (n<=30), we can afford a few restarts.
        
        best_paths = None
        best_len = float('inf')
        
        rng = random.Random(seed + 100)
        
        for restart in range(50): # 50 restarts is cheap for n=30
            if time.time() > DEADLINE - 20:
                break
                
            rem = set(edges)
            curr_paths = []
            
            # Shuffle vertices to randomize start points
            v_order = list(range(n))
            rng.shuffle(v_order)
            
            # We need to ensure we pick edges that are still remaining.
            # A robust way: iterate through all edges, if edge is in rem, start a path.
            # But we want long paths.
            
            # Heuristic: Start from high degree vertices in rem.
            # Calculate current degrees in rem
            rem_deg = defaultdict(int)
            for u, v in rem:
                rem_deg[u] += 1
                rem_deg[v] += 1
            
            # Sort candidates by rem_deg
            # We will pick the vertex with highest rem_deg, then pick a neighbor.
            
            while rem:
                # Find vertex with max degree in rem
                # If no vertices have degree > 0, something is wrong (isolated edges?)
                # But graph is connected, so if rem is not empty, some vertex has deg > 0.
                
                # Optimization: maintain a list of active vertices?
                # For n=30, scanning is fine.
                max_deg = 0
                start_v = -1
                for v in range(n):
                    if rem_deg[v] > max_deg:
                        max_deg = rem_deg[v]
                        start_v = v
                        
                if start_v == -1:
                    # Should not happen if rem is not empty
                    break
                    
                # Pick a neighbor to start the path
                # Choose neighbor with highest degree? Or random?
                # Random helps avoid local minima.
                nbrs = []
                for v in adj[start_v]:
                    e = (start_v, v) if start_v < v else (v, start_v)
                    if e in rem:
                        nbrs.append(v)
                        
                if not nbrs:
                    # Isolated vertex in rem? Impossible if rem_deg > 0.
                    # Unless rem_deg calculation is stale.
                    # Re-calc?
                    # Let's just break to be safe, but this shouldn't happen.
                    break
                
                next_v = rng.choice(nbrs)
                remove_edge(start_v, next_v)
                rem_deg[start_v] -= 1
                rem_deg[next_v] -= 1
                
                path = [start_v, next_v]
                
                # Extend path
                # We can extend from front or back.
                # Prefer the end with more options (higher degree in rem).
                
                while True:
                    # Check back end
                    back_v = path[-1]
                    back_nbrs = []
                    for v in adj[back_v]:
                        if v not in path: # Simple path constraint
                            e = (back_v, v) if back_v < v else (v, back_v)
                            if e in rem:
                                back_nbrs.append(v)
                    
                    # Check front end
                    front_v = path[0]
                    front_nbrs = []
                    for v in adj[front_v]:
                        if v not in path:
                            e = (front_v, v) if front_v < v else (v, front_v)
                            if e in rem:
                                front_nbrs.append(v)
                    
                    if not back_nbrs and not front_nbrs:
                        break
                    
                    # Choose which end to extend
                    # Heuristic: extend the side with more options
                    if len(back_nbrs) >= len(front_nbrs) and back_nbrs:
                        next_v = rng.choice(back_nbrs)
                        path.append(next_v)
                        remove_edge(back_v, next_v)
                        rem_deg[back_v] -= 1
                        rem_deg[next_v] -= 1
                    elif front_nbrs:
                        next_v = rng.choice(front_nbrs)
                        path.insert(0, next_v)
                        remove_edge(front_v, next_v)
                        rem_deg[front_v] -= 1
                        rem_deg[next_v] -= 1
                    else:
                        # Only back_nbrs
                        next_v = rng.choice(back_nbrs)
                        path.append(next_v)
                        remove_edge(back_v, next_v)
                        rem_deg[back_v] -= 1
                        rem_deg[next_v] -= 1
                        
                curr_paths.append(path)
                
            # Update best
            if len(curr_paths) < best_len:
                best_len = len(curr_paths)
                best_paths = [list(p) for p in curr_paths]
                
            # Early exit if we hit the theoretical lower bound
            # Lower bound: max(1, odd_vertices / 2)
            odd_count = sum(1 for v in range(n) if deg[v] % 2 == 1)
            lb = max(1, odd_count // 2)
            if best_len == lb:
                break

        if best_paths is None:
            # Fallback: just return edges as paths
            return [[u, v] for u, v in edges]
            
        paths = best_paths

        # 2. Merging Phase
        # Iteratively merge paths that share endpoints, if the result is simple.
        
        # To speed up, we maintain a map: vertex -> list of (path_index, is_end)
        # is_end: True if it's the last vertex, False if it's the first.
        
        def try_merge(paths, max_iters=500):
            # We will use a randomized greedy approach with multiple passes.
            # Randomizing the order of path checks helps escape local minima.
            
            current_paths = [list(p) for p in paths]
            
            # Map vertex to list of (idx, end_flag)
            # end_flag: 0 for start, 1 for end
            endpoint_map = defaultdict(list)
            
            def rebuild_map():
                endpoint_map.clear()
                for i, p in enumerate(current_paths):
                    if not p: continue
                    endpoint_map[p[0]].append((i, 0))
                    endpoint_map[p[-1]].append((i, 1))
                    
            rebuild_map()
            
            improved = True
            iter_count = 0
            rng_merge = random.Random(seed + 200)
            
            while improved and iter_count < max_iters:
                if time.time() > DEADLINE - 5:
                    break
                improved = False
                iter_count += 1
                
                # Collect merge candidates: vertices that are endpoints of >= 2 distinct paths
                candidates = []
                for v, entries in endpoint_map.items():
                    # Unique path indices
                    idxs = set(idx for idx, _ in entries)
                    if len(idxs) >= 2:
                        candidates.append(v)
                        
                if not candidates:
                    break
                    
                rng_merge.shuffle(candidates)
                
                for v in candidates:
                    # Re-fetch entries as map might have changed
                    entries = endpoint_map[v]
                    if not entries:
                        continue
                    idxs = list(set(idx for idx, _ in entries))
                    if len(idxs) < 2:
                        continue
                    
                    # Try to merge pairs from idxs
                    # We only need to try a few pairs per vertex per iteration
                    # Let's pick the first two distinct paths involving v
                    
                    # Find two paths that share v as an endpoint
                    p1_idx = -1
                    p2_idx = -1
                    p1_is_end = False
                    p2_is_end = False
                    
                    # We need to find two *different* paths.
                    # entries contains (idx, flag).
                    # We need idx1 != idx2.
                    
                    found = False
                    for i in range(len(entries)):
                        idx1, flag1 = entries[i]
                        if p1_idx == -1:
                            p1_idx = idx1
                            p1_is_end = (flag1 == 1)
                        else:
                            if idx1 != p1_idx:
                                p2_idx = idx1
                                p2_is_end = (flag1 == 1)
                                found = True
                                break
                                
                    if not found:
                        continue
                        
                    if p1_idx == -1 or p2_idx == -1:
                        continue
                        
                    p1 = current_paths[p1_idx]
                    p2 = current_paths[p2_idx]
                    
                    if not p1 or not p2:
                        continue
                        
                    # Construct merged path
                    # Cases:
                    # 1. p1 ends at v, p2 starts at v: p1 + p2
                    # 2. p1 ends at v, p2 ends at v: p1 + reversed(p2)
                    # 3. p1 starts at v, p2 starts at v: reversed(p1) + p2
                    # 4. p1 starts at v, p2 ends at v: p2 + p1
                    
                    new_path = None
                    if p1_is_end and not p2_is_end:
                        new_path = p1 + p2
                    elif p1_is_end and p2_is_end:
                        new_path = p1 + list(reversed(p2))
                    elif not p1_is_end and not p2_is_end:
                        new_path = list(reversed(p1)) + p2
                    else: # p1 starts, p2 ends
                        new_path = p2 + p1
                        
                    # Check simplicity
                    if len(set(new_path)) == len(new_path):
                        # Valid merge
                        current_paths[p1_idx] = new_path
                        current_paths[p2_idx] = [] # Mark as empty
                        
                        # Rebuild map is expensive, but necessary for correctness.
                        # We can do incremental updates, but for n=30, rebuild is fine.
                        rebuild_map()
                        improved = True
                        break # Restart candidate loop with new map state
                        
            # Filter out empty paths
            return [p for p in current_paths if p]

        paths = try_merge(paths)
        
        # 3. Local Search / Edge Swapping (Optional but powerful)
        # If we have time, try to improve further by swapping edges between paths.
        # This is complex. Given the deadline, we might skip if merging worked well.
        # For now, let's do a second pass of merging with different randomization.
        
        if time.time() < DEADLINE - 10:
            paths = try_merge(paths, max_iters=200)
            
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > DEADLINE - 10:
            # Fallback: return edges as paths
            n, p, seed = spec
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        
        n, p, seed = spec
        paths = solve_one(n, p, seed)
        results.append(paths)
        
    return results

if __name__ == "__main__":
    r = construct()
    for i, lst in enumerate(r):
        print(i, len(lst))
    print("Total:", sum(len(lst) for lst in r))
