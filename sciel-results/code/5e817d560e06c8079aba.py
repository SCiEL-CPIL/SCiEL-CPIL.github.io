"""Snake-in-the-box in the 9-dimensional hypercube - CSP-inspired structural approach."""
N = 9
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    best_path = None
    best_len = 0
    
    bit_masks = [1 << b for b in range(N)]
    
    def _is_valid_extension(v, path, used, check_range=None):
        """Check if vertex v can be appended to path maintaining induced path property.
        check_range: if specified, only check against path[check_range[0]:check_range[1]]
        """
        if v in used:
            return False
        if check_range is None:
            # Check against all except the last (consecutive)
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    return False
        else:
            lo, hi = check_range
            for i in range(lo, hi):
                if (v ^ path[i]).bit_count() == 1:
                    return False
        return True
    
    def _greedy_build(start_vertex, rng, deadline, use_warnsdorff=True):
        """Build a snake greedily using Warnsdorff heuristic."""
        path = [start_vertex]
        used = {start_vertex}
        
        while time.time() < deadline:
            cur = path[-1]
            candidates = []
            for b in range(N):
                v = cur ^ bit_masks[b]
                if v in used:
                    continue
                if not _is_valid_extension(v, path, used):
                    continue
                
                if use_warnsdorff:
                    # Count onward moves for heuristic
                    onward = 0
                    for b2 in range(N):
                        nv = v ^ bit_masks[b2]
                        if nv in used:
                            continue
                        # nv must be valid if added after v
                        # Check against path[:-1] (since v will be last, path[:-1] are non-consecutive to nv)
                        # Actually need to check nv against path[:-1] because when nv is added, 
                        # path will be [..., cur, v, nv], so nv's non-consecutive neighbors are path[:-1]
                        # Wait, no: path is [..., cur], we add v, then nv.
                        # When nv is added, path becomes [..., cur, v, nv].
                        # nv is consecutive to v, so nv must differ from all in path[:-1] by >= 2 bits.
                        # path[:-1] here means path without the last element, which after adding v would be [..., cur].
                        # But we're checking before adding, so check nv against path (current) since 
                        # after adding v, the "previous" path for nv's check is path + [v], 
                        # and nv must differ from path + [v] minus v = path by >= 2 bits... 
                        # No wait: nv must differ from all non-consecutive vertices.
                        # Non-consecutive to nv are all vertices except v (its predecessor).
                        # Those are path[0:-1] if we consider the path before adding v... 
                        # Let me just check nv against current path (which doesn't include v yet).
                        ok = True
                        for p in path:
                            if (nv ^ p).bit_count() == 1:
                                ok = False
                                break
                        if ok:
                            onward += 1
                    candidates.append((onward, b, v))
                else:
                    candidates.append((0, b, v))
            
            if not candidates:
                break
            
            if use_warnsdorff:
                candidates.sort(key=lambda x: x[0])
                _, b, v = candidates[0]
            else:
                _, b, v = candidates[rng.randint(0, len(candidates)-1)]
            
            path.append(v)
            used.add(v)
        
        return path
    
    def _bidirectional_greedy(start1, start2, rng, deadline):
        """Build two snakes from start1 and start2, try to connect them."""
        path1 = _greedy_build(start1, rng, deadline)
        path2 = _greedy_build(start2, rng, deadline)
        
        # Try to connect end of path1 to start of path2 or vice versa
        # Also try reversing path2
        
        def try_connect(p1, p2):
            """Try to concatenate p1 + p2 where last of p1 is adjacent to first of p2."""
            if not p1 or not p2:
                return []
            last1 = p1[-1]
            first2 = p2[0]
            if (last1 ^ first2).bit_count() != 1:
                return []
            
            # Check that all vertices in p2 are valid against p1[:-1]
            for v in p2:
                for p in p1[:-1]:
                    if (v ^ p).bit_count() == 1:
                        return []
            # Also check p2[-1] against... no, p2 is already a valid snake.
            # We just need cross-checks: each v in p2 must not be dist 1 from any p in p1[:-1]
            # and each p in p1[:-1] must not be dist 1 from any v in p2.
            # The first check is done above. Also need: no vertex in p2 is dist 1 from p1[-1] 
            # except p2[0] (which is consecutive). And p2[0] must not be dist 1 from p1[:-2].
            # Actually p2[0] is consecutive to p1[-1], so p2[0] must differ from p1[:-1] by >= 2.
            # And p2[1] must differ from p1[:-1] by >= 2, etc.
            # We checked v in p2 against p1[:-1]. That covers it.
            # But also need to check that vertices in p1 (except last) aren't dist 1 from p2[0].
            # p1[-1] is consecutive to p2[0], so p1[-2] etc must be dist >= 2 from p2[0].
            # That's covered by checking p2[0] against p1[:-1].
            return p1 + p2
        
        # Try path1 + path2
        connected = try_connect(path1, path2)
        if connected and len(connected) > best_len:
            return connected
        
        # Try path1 + reversed path2
        rev_path2 = path2[::-1]
        connected = try_connect(path1, rev_path2)
        if connected and len(connected) > best_len:
            return connected
        
        # Try reversed path1 + path2
        rev_path1 = path1[::-1]
        connected = try_connect(rev_path1, path2)
        if connected and len(connected) > best_len:
            return connected
        
        # Try reversed path1 + reversed path2
        connected = try_connect(rev_path1, rev_path2)
        if connected and len(connected) > best_len:
            return connected
        
        return []
    
    def _local_search_extend(path, rng, deadline):
        """Try to extend a path from both ends using local search."""
        path = path[:]
        used = set(path)
        improved = True
        
        while improved and time.time() < deadline - 1:
            improved = False
            for end in [0, 1]:
                if end == 0:
                    cur = path[0]
                    # Check against path[1:]
                else:
                    cur = path[-1]
                    # Check against path[:-1]
                
                candidates = []
                for b in range(N):
                    v = cur ^ bit_masks[b]
                    if v in used:
                        continue
                    
                    if end == 0:
                        # v will be prepended. v is consecutive to cur=path[0].
                        # v must differ from path[1:] by >= 2 bits.
                        valid = True
                        for p in path[1:]:
                            if (v ^ p).bit_count() == 1:
                                valid = False
                                break
                        if not valid:
                            continue
                    else:
                        # v will be appended. v is consecutive to cur=path[-1].
                        # v must differ from path[:-1] by >= 2 bits.
                        valid = True
                        for p in path[:-1]:
                            if (v ^ p).bit_count() == 1:
                                valid = False
                                break
                        if not valid:
                            continue
                    
                    # Count onward moves
                    onward = 0
                    for b2 in range(N):
                        nv = v ^ bit_masks[b2]
                        if nv in used:
                            continue
                        if end == 0:
                            # nv would be prepended after v. nv consecutive to v.
                            # nv must differ from path[2:] by >= 2 bits (since v and path[0] are between nv and rest)
                            # Actually nv's non-consecutive are path[1:] (since nv-v-path[0]-path[1]-...)
                            # Wait: new path would be [nv, v, path[0], path[1], ...]
                            # nv consecutive to v. Non-consecutive to nv: path[0], path[1], ...
                            # So check nv against path[0:] = path
                            # But v is used, so nv != v (guaranteed by bit flip)
                            ok = True
                            for p in path:
                                if (nv ^ p).bit_count() == 1:
                                    ok = False
                                    break
                            if ok:
                                onward += 1
                        else:
                            # nv would be appended after v. New path: [..., path[-1], v, nv]
                            # nv consecutive to v. Non-consecutive to nv: path[:-1]
                            ok = True
                            for p in path[:-1]:
                                if (nv ^ p).bit_count() == 1:
                                    ok = False
                                    break
                            if ok:
                                onward += 1
                    
                    candidates.append((onward, b, v))
                
                if candidates:
                    candidates.sort(key=lambda x: x[0])
                    _, b, v = candidates[0]
                    if end == 0:
                        path.insert(0, v)
                        used.add(v)
                    else:
                        path.append(v)
                        used.add(v)
                    improved = True
        
        return path
    
    def _csp_style_build(start_vertex, rng, deadline):
        """Build snake using CSP-inspired variable ordering and forward checking.
        Uses a priority queue approach: at each step, pick the neighbor that 
        constrains the future least (most flexible choice first - opposite of Warnsdorff).
        This explores a different region of solution space."""
        path = [start_vertex]
        used = {start_vertex}
        
        while time.time() < deadline:
            cur = path[-1]
            candidates = []
            for b in range(N):
                v = cur ^ bit_masks[b]
                if v in used:
                    continue
                if not _is_valid_extension(v, path, used):
                    continue
                
                # Count how many vertices would become "blocked" (dist 1 from v) 
                # that are currently unvisited - this is a constraint measure
                constraints = 0
                for b2 in range(N):
                    nv = v ^ bit_masks[b2]
                    if nv not in used:
                        # nv would be blocked from being adjacent to future vertices
                        constraints += 1
                
                candidates.append((constraints, b, v))
            
            if not candidates:
                break
            
            # Pick the one with MOST constraints (most constrained first - like Warnsdorff)
            # Or pick randomly among top few for diversity
            candidates.sort(key=lambda x: x[0], reverse=True)
            if rng.random() < 0.7:
                _, b, v = candidates[0]
            else:
                k = min(3, len(candidates))
                _, b, v = candidates[rng.randint(0, k-1)]
            
            path.append(v)
            used.add(v)
        
        return path
    
    # Phase 1: Multiple greedy builds with Warnsdorff
    rng = random.Random(12345)
    
    starts = [0, 1, (1 << N) - 1, 0x155, 0xAAA]
    for _ in range(30):
        starts.append(rng.randint(0, (1 << N) - 1))
    
    for sv in starts:
        if time.time() > deadline - 10:
            break
        path = _greedy_build(sv, rng, deadline - 5)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
    
    # Phase 2: CSP-style builds
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        sv = rng.randint(0, (1 << N) - 1)
        path = _csp_style_build(sv, rng, deadline - 5)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
    
    # Phase 3: Bidirectional construction and splicing
    if best_path is None:
        best_path = _greedy_build(0, rng, deadline - 5)
        best_len = len(best_path)
    
    for _ in range(30):
        if time.time() > deadline - 5:
            break
        sv1 = rng.randint(0, (1 << N) - 1)
        sv2 = rng.randint(0, (1 << N) - 1)
        if sv1 == sv2:
            continue
        connected = _bidirectional_greedy(sv1, sv2, rng, deadline - 2)
        if connected and len(connected) > best_len:
            best_len = len(connected)
            best_path = connected
    
    # Phase 4: Local search extension on best path
    if best_path:
        extended = _local_search_extend(best_path, rng, deadline - 2)
        if len(extended) > best_len:
            best_len = len(extended)
            best_path = extended
    
    # Phase 5: More diverse random restarts with local search
    for _ in range(50):
        if time.time() > deadline - 2:
            break
        sv = rng.randint(0, (1 << N) - 1)
        path = _greedy_build(sv, rng, deadline - 1)
        path = _local_search_extend(path, rng, deadline - 1)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
    
    # Final: ensure we have at least a valid path
    if best_path is None or len(best_path) < 2:
        best_path = [0, 1]
        best_len = 2
    
    return best_path
