import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_graph(edges):
        if not edges:
            return []
        
        # Build adjacency and degree
        n = max(max(e) for e in edges) + 1 if edges else 0
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Initialize paths: each edge is a path
        # Path is a list of vertices
        paths = []
        for u, v in edges:
            paths.append([u, v])
            
        # Map from vertex to list of (path_index, is_start)
        # is_start True if vertex is the first in the path
        # We need to know if we can merge.
        # A path P = [v0, v1, ..., vk] can be extended at v0 by a path ending at v0?
        # No, if P1 ends at u and P2 starts at u, we can merge P1 + P2.
        # Or P1 starts at u and P2 ends at u, merge P2 + P1.
        
        # To simplify, let's just maintain a list of paths.
        # And a dict: vertex -> list of path indices that have this vertex as an endpoint.
        # For each path, we know its start and end.
        
        path_ends = {} # path_idx -> (start, end)
        
        def update_maps(idx, p, start, end):
            path_ends[idx] = (start, end)
            # Remove from old maps if needed? 
            # It's easier to rebuild or maintain carefully.
            # Given small size, rebuilding or careful removal is fine.
            
        # Let's use a list of paths and rebuild endpoint maps each iteration?
        # n is small (30), edges ~ 30-50. Paths ~ 50.
        # Iterations ~ 50.
        # Rebuilding maps is O(E). Total O(E^2). Very fast.
        
        while True:
            if time.time() > deadline:
                break
                
            # Rebuild endpoint map
            # vertex -> list of (path_idx, position) where position 0 is start, 1 is end
            end_map = defaultdict(list)
            for i, p in enumerate(paths):
                if not p: continue
                s, e = p[0], p[-1]
                end_map[s].append((i, 0))
                end_map[e].append((i, 1))
                
            best_merge = None
            best_score = -1
            
            # Find all valid merges
            for u, entries in end_map.items():
                if len(entries) < 2:
                    continue
                
                # We need to pick two entries (i1, pos1) and (i2, pos2)
                # such that one is start and one is end?
                # Actually, we can merge if one path ends at u and the other starts at u.
                # So we need one entry with pos=1 (end) and one with pos=0 (start).
                
                starts = [e for e in entries if e[1] == 0]
                ends = [e for e in entries if e[1] == 1]
                
                if not starts or not ends:
                    continue
                
                for i_s, _ in starts:
                    for i_e, _ in ends:
                        if i_s == i_e:
                            continue
                        p_s = paths[i_s]
                        p_e = paths[i_e]
                        
                        # Merge p_e + p_s
                        # p_e ends at u, p_s starts at u
                        new_p = p_e + p_s[1:]
                        
                        # Check if simple
                        if len(new_p) != len(set(new_p)):
                            continue
                        
                        # Score: length of new path
                        score = len(new_p)
                        
                        if score > best_score:
                            best_score = score
                            best_merge = (i_e, i_s, new_p)
            
            if best_merge is None:
                break
            
            i_e, i_s, new_p = best_merge
            # Remove old paths, add new
            # Order matters for index stability, but we rebuild maps anyway.
            # Let's just replace one and remove the other.
            # Remove i_s, replace i_e with new_p.
            # Note: i_e and i_s are indices in the current list.
            # We need to be careful if we remove i_s first, i_e index might change.
            # Let's construct a new list of paths.
            
            new_paths = []
            for i, p in enumerate(paths):
                if i == i_e or i == i_s:
                    continue
                new_paths.append(p)
            
            new_paths.append(new_p)
            paths = new_paths
            
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            # Fallback: return whatever we have or empty? 
            # Better to return a valid cover if possible, but we don't have one.
            # The task requires valid paths. 
            # If we time out, we should return the best valid result.
            # But we are in the middle of solving one graph.
            # Let's just break and return what we have for previous graphs, 
            # and for this one, maybe return edge-by-edge if we have time?
            # Actually, the loop above is fast. Timeout is unlikely.
            pass
        
        edges = instance(*spec)
        paths = solve_graph(edges)
        results.append(paths)
        
    return results
