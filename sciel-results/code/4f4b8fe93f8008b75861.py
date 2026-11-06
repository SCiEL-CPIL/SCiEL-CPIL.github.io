"""Snake-in-the-box in the 9-dimensional hypercube - Bitmask backtracking + local refinement."""
N = 9
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Precompute neighbor masks: neighbor_mask[v] has bits set for all vertices at Hamming distance 1 from v
    neighbor_mask = [0] * (1 << N)
    for v in range(1 << N):
        mask = 0
        for b in range(N):
            nv = v ^ (1 << b)
            mask |= (1 << nv)
        neighbor_mask[v] = mask

    bit_masks = [1 << b for b in range(N)]

    def _backtrack_search(start_vertex, search_deadline, rng=None, max_depth=None):
        """Backtracking search with Warnsdorff heuristic using bitmask state.
        
        State:
        - path: list of vertices
        - forbidden_mask: int bitmask, bit i set means vertex i cannot be next
          forbidden_mask = used_mask | union_{i=0}^{k-1} neighbor_mask[path[i]]
          (vertices at dist 1 from all non-last vertices in path, plus all used vertices)
        
        When extending by v:
        - v must not be in forbidden_mask
        - new_forbidden = forbidden_mask | (1 << v) | neighbor_mask[path[-1]]
          (path[-1] becomes non-consecutive, so its neighbors become forbidden)
        """
        if rng is None:
            rng = random.Random(42)
        
        best_path = [start_vertex]
        best_len = 1
        
        # Iterative backtracking to avoid recursion depth issues
        # Stack entries: (path, forbidden_mask, candidate_index, candidates_list)
        
        def _search_iterative():
            nonlocal best_path, best_len
            
            # Use explicit stack for iterative DFS
            # Each stack frame: (path_copy, forbidden_mask, next_candidate_idx, sorted_candidates)
            # We'll use a more memory-efficient approach with a single path and undo
            
            path = [start_vertex]
            forbidden = (1 << start_vertex)
            
            # We'll use a stack of (forbidden_mask, candidate_list, next_idx)
            # When we go deeper, we save state; when we backtrack, we restore
            
            stack = []
            cur_forbidden = forbidden
            cur_path = path
            
            while True:
                if time.time() > search_deadline:
                    break
                
                cur = cur_path[-1]
                
                # Generate candidates
                candidates = []
                for b in range(N):
                    v = cur ^ bit_masks[b]
                    if cur_forbidden & (1 << v):
                        continue
                    # Compute onward moves
                    new_forb = cur_forbidden | (1 << v) | neighbor_mask[cur]
                    onward = 0
                    for b2 in range(N):
                        nv = v ^ bit_masks[b2]
                        if not (new_forb & (1 << nv)):
                            onward += 1
                    candidates.append((onward, v))
                
                if not candidates:
                    # Dead end, backtrack
                    if not stack:
                        break
                    # Restore previous state
                    cur_forbidden, saved_candidates, next_idx = stack.pop()
                    cur_path.pop()  # Remove the vertex we added
                    # Now try next candidate from saved_candidates starting at next_idx
                    if next_idx < len(saved_candidates):
                        _, v = saved_candidates[next_idx]
                        cur_path.append(v)
                        cur_forbidden |= (1 << v) | neighbor_mask[cur_path[-2]]
                        # Save state for potential backtracking
                        # We need to save the remaining candidates
                        remaining = saved_candidates[next_idx+1:]
                        stack.append((cur_forbidden, saved_candidates, next_idx+1))
                        # Wait, this is getting complicated. Let me use a cleaner approach.
                        # Actually, let me restart with a cleaner iterative approach.
                        break  # This approach is getting messy, let me use recursion with increased limit
                    else:
                        # No more candidates at this level, keep backtracking
                        continue
                
                # Sort by Warnsdorff (fewest onward first)
                candidates.sort(key=lambda x: x[0])
                
                # Try the best candidate
                _, v = candidates[0]
                
                # Save state for backtracking
                remaining = candidates[1:]
                stack.append((cur_forbidden, remaining, 0))
                
                # Move forward
                cur_path.append(v)
                cur_forbidden |= (1 << v) | neighbor_mask[cur]
                
                if len(cur_path) > best_len:
                    best_len = len(cur_path)
                    best_path = cur_path[:]
                
                if max_depth is not None and len(cur_path) >= max_depth:
                    # Restore and try next
                    if stack:
                        cur_forbidden, remaining, _ = stack.pop()
                        cur_path.pop()
                    else:
                        break
                    if remaining:
                        _, v2 = remaining[0]
                        cur_path.append(v2)
                        cur_forbidden |= (1 << v2) | neighbor_mask[cur_path[-2]]
                        # Save for further backtracking
                        stack.append((cur_forbidden, remaining[1:], 0))
                    else:
                        # Keep backtracking
                        continue
                
                # Continue loop to go deeper
        
        # Use recursive approach with sys.setrecursionlimit
        import sys
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(10000)
        
        def _search_recursive(path, forbidden_mask, depth):
            nonlocal best_path, best_len
            
            if time.time() > search_deadline:
                return
            
            if depth > best_len:
                best_len = depth
                best_path = path[:]
            
            if max_depth is not None and depth >= max_depth:
                return
            
            cur = path[-1]
            
            # Find candidate next vertices
            candidates = []
            for b in range(N):
                v = cur ^ bit_masks[b]
                if forbidden_mask & (1 << v):
                    continue
                new_forbidden = forbidden_mask | (1 << v) | neighbor_mask[cur]
                onward = 0
                for b2 in range(N):
                    nv = v ^ bit_masks[b2]
                    if not (new_forbidden & (1 << nv)):
                        onward += 1
                candidates.append((onward, v))
            
            if not candidates:
                return
            
            # Sort by onward moves (Warnsdorff: fewest onward first)
            candidates.sort(key=lambda x: x[0])
            
            # Try each candidate
            for onward, v in candidates:
                if time.time() > search_deadline:
                    return
                new_forbidden = forbidden_mask | (1 << v) | neighbor_mask[cur]
                path.append(v)
                _search_recursive(path, new_forbidden, depth + 1)
                path.pop()
        
        _search_recursive([start_vertex], (1 << start_vertex), 1)
        sys.setrecursionlimit(old_limit)
        
        return best_path

    def _greedy_extend(path, deadline, rng=None):
        """Greedily extend a path using Warnsdorff with bitmask state."""
        if rng is None:
            rng = random.Random(42)
        
        if not path:
            return path
        
        # Reconstruct forbidden mask from path
        forbidden_mask = 0
        for v in path:
            forbidden_mask |= (1 << v)
        for v in path[:-1]:
            forbidden_mask |= neighbor_mask[v]
        
        cur = path[-1]
        while time.time() < deadline:
            candidates = []
            for b in range(N):
                v = cur ^ bit_masks[b]
                if forbidden_mask & (1 << v):
                    continue
                new_forbidden = forbidden_mask | (1 << v) | neighbor_mask[cur]
                onward = 0
                for b2 in range(N):
                    nv = v ^ bit_masks[b2]
                    if not (new_forbidden & (1 << nv)):
                        onward += 1
                candidates.append((onward, v))
            
            if not candidates:
                break
            
            candidates.sort(key=lambda x: x[0])
            _, v = candidates[0]
            
            path.append(v)
            forbidden_mask |= (1 << v) | neighbor_mask[cur]
            cur = v
        
        return path

    def _local_search_extend(path, deadline, rng=None):
        """Try to extend a path from both ends using local search with bitmask state."""
        if rng is None:
            rng = random.Random(42)
        
        path = path[:]
        improved = True
        
        while improved and time.time() < deadline - 0.5:
            improved = False
            for end in [0, 1]:
                if time.time() > deadline - 0.5:
                    break
                
                if end == 0:
                    cur = path[0]
                    # For prepending, forbidden for new vertex v:
                    # v must not be used, and v must not be at dist 1 from path[1:]
                    # Forbidden mask for prepending: used_mask | union_{i=1}^{k} neighbor_mask[path[i]]
                    forbidden_mask = 0
                    for v in path:
                        forbidden_mask |= (1 << v)
                    for v in path[1:]:
                        forbidden_mask |= neighbor_mask[v]
                else:
                    cur = path[-1]
                    # For appending, forbidden for new vertex v:
                    # v must not be used, and v must not be at dist 1 from path[:-1]
                    forbidden_mask = 0
                    for v in path:
                        forbidden_mask |= (1 << v)
                    for v in path[:-1]:
                        forbidden_mask |= neighbor_mask[v]
                
                candidates = []
                for b in range(N):
                    v = cur ^ bit_masks[b]
                    if forbidden_mask & (1 << v):
                        continue
                    
                    # Count onward moves
                    if end == 0:
                        # If we prepend v, new path is [v, path[0], path[1], ...]
                        # Next prepend candidate nv: must not be used (including v), 
                        # and must not be at dist 1 from path[1:] (since v is between nv and path[0])
                        # Wait: nv consecutive to v. Non-consecutive to nv: path[0], path[1], ...
                        # So forbidden for nv after prepending v:
                        #   used_mask | (1<<v) | union_{i=0}^{k} neighbor_mask[path[i]]
                        # But path[0] is now consecutive to v, not to nv.
                        # Actually: new path = [nv, v, path[0], path[1], ...]
                        # nv consecutive to v. Non-consecutive to nv: path[0], path[1], ...
                        # So forbidden = (old_used | {v}) | union_{i=0}^{k} neighbor_mask[path[i]]
                        # = forbidden_mask | (1 << v) | neighbor_mask[path[0]]
                        # Wait, forbidden_mask already has union_{i=1}^{k} neighbor_mask[path[i]]
                        # So we need to add neighbor_mask[path[0]] and (1<<v)
                        new_forbidden = forbidden_mask | (1 << v) | neighbor_mask[path[0]]
                    else:
                        # If we append v, new path is [..., path[-1], v]
                        # Next append candidate nv: must not be used (including v),
                        # and must not be at dist 1 from path[:-1] (since v is between nv and path[-1])
                        # New path = [..., path[-1], v, nv]
                        # nv consecutive to v. Non-consecutive to nv: path[:-1]
                        # So forbidden = (old_used | {v}) | union_{i=0}^{k-1} neighbor_mask[path[i]]
                        # = forbidden_mask | (1 << v) | neighbor_mask[path[-1]]
                        new_forbidden = forbidden_mask | (1 << v) | neighbor_mask[path[-1]]
                    
                    onward = 0
                    for b2 in range(N):
                        nv = v ^ bit_masks[b2]
                        if not (new_forbidden & (1 << nv)):
                            onward += 1
                    
                    candidates.append((onward, v))
                
                if candidates:
                    candidates.sort(key=lambda x: x[0])
                    _, v = candidates[0]
                    if end == 0:
                        path.insert(0, v)
                    else:
                        path.append(v)
                    improved = True
        
        return path

    # Phase 1: Backtracking search from multiple starts
    rng = random.Random(12345)
    
    starts = [0, 1, (1 << N) - 1, 0x155, 0x1AA, 0x0FF, 0x1F0, 0x123, 0x1AB]
    for _ in range(15):
        starts.append(rng.randint(0, (1 << N) - 1))
    
    best_path = None
    best_len = 0
    
    for sv in starts:
        if time.time() > deadline - 40:
            break
        # Give each backtracking search a time budget
        search_deadline = min(deadline - 5, time.time() + 3.0)
        path = _backtrack_search(sv, search_deadline, rng)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
    
    # Phase 2: Greedy extension with bitmask state from random starts
    for _ in range(30):
        if time.time() > deadline - 20:
            break
        sv = rng.randint(0, (1 << N) - 1)
        path = _greedy_extend([sv], deadline - 5, rng)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
    
    # Phase 3: Local search extension on best path
    if best_path:
        extended = _local_search_extend(best_path, deadline - 5, rng)
        if len(extended) > best_len:
            best_len = len(extended)
            best_path = extended
    
    # Phase 4: More diverse random restarts with local search
    for _ in range(40):
        if time.time() > deadline - 3:
            break
        sv = rng.randint(0, (1 << N) - 1)
        path = _greedy_extend([sv], deadline - 2, rng)
        path = _local_search_extend(path, deadline - 2, rng)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
    
    # Phase 5: Final local search on best
    if best_path:
        extended = _local_search_extend(best_path, deadline - 1, rng)
        if len(extended) > best_len:
            best_len = len(extended)
            best_path = extended

    # Final: ensure we have at least a valid path
    if best_path is None or len(best_path) < 2:
        best_path = [0, 1]
        best_len = 2
    
    return best_path
