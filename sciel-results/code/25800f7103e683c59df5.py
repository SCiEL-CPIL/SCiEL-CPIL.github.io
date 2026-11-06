"""Snake-in-the-box Q_9: Warnsdorff backtracking + bidirectional splicing."""
import time
import random
import sys

N = 9
SIZE = 1 << N  # 512
TIME_LIMIT = 140  # safety margin below 150s hard kill

# Precompute neighbors list and neighbor bitmask for each vertex
NEIGHBORS = []
NEIGHBOR_MASKS = [0] * SIZE
for v in range(SIZE):
    nbs = [v ^ (1 << b) for b in range(N)]
    NEIGHBORS.append(nbs)
    m = 0
    for nb in nbs:
        m |= (1 << nb)
    NEIGHBOR_MASKS[v] = m

VERTEX_MASKS = [1 << v for v in range(SIZE)]


def construct():
    start_time = time.time()
    random.seed()  # fresh randomness

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    def warnsdorff_backtrack(start_vertex, max_depth, max_backtracks=200000):
        """
        Warnsdorff's rule with backtracking.
        At each step, among available neighbors, prefer the one with the
        fewest available neighbors (to avoid dead-ends). If a dead-end is
        reached, backtrack and try the next candidate.
        Returns the longest path found.
        """
        # State: path (list), used_mask (int)
        # We use iterative backtracking with a stack of (vertex, candidates_list, index)
        # For efficiency, we track the mask of used + forbidden.
        
        best_path = [start_vertex]
        best_len = 1
        
        # Iterative DFS with Warnsdorff ordering
        # Stack entry: (vertex, available_neighbors_list, next_index_to_try, mask_at_this_level)
        # Actually, let's use a simpler recursive-style iterative approach:
        # path = list of vertices
        # mask = used + forbidden (as before)
        # We'll implement backtracking by keeping a stack of decision points.
        
        path = [start_vertex]
        mask = VERTEX_MASKS[start_vertex]
        # Stack of (vertex, sorted_candidates, next_index, mask_before_this_choice)
        # When we hit a dead end, we pop and try the next candidate.
        # For memory efficiency with max_depth up to ~200, this is fine.
        
        # Actually, a cleaner iterative backtracking:
        # We maintain path and mask. At each step, we compute available neighbors
        # sorted by Warnsdorff score (fewest future options first).
        # We try them in order. If none work (dead end), we backtrack one step.
        
        # For backtracking, we need to save the state before each choice.
        # State: (path_length, mask, current_vertex)
        # We save: for each step i, the list of candidates tried and the mask before choosing.
        
        # Let's use a stack-based approach:
        # stack[i] = (vertex_i, candidates_sorted, next_idx, mask_before_i)
        # path[0] = start, mask starts as VERTEX_MASKS[start]
        
        stack = []  # Each entry: (vertex, candidates_list, next_idx, mask_before)
        
        # Initial: we're at start_vertex, need to choose next
        # Compute candidates for start_vertex
        cur = start_vertex
        cur_mask = NEIGHBOR_MASKS[cur]
        avail = cur_mask & ~mask
        if avail:
            candidates = _get_sorted_candidates(avail, mask, cur)
            stack.append((cur, candidates, 0, mask))
        
        backtracks = 0
        max_backtracks = max_backtracks
        
        while stack and backtracks < max_backtracks:
            if time_left() < 2.0:
                break
            
            cur_v, candidates, idx, mask_before = stack[-1]
            
            if idx < len(candidates):
                # Try next candidate
                next_v = candidates[idx]
                stack[-1] = (cur_v, candidates, idx + 1, mask_before)
                
                # Extend: add next_v
                path.append(next_v)
                new_mask = mask_before | NEIGHBOR_MASKS[cur_v] | VERTEX_MASKS[next_v]
                
                # Update best
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = list(path)
                
                if len(path) >= max_depth:
                    # Can't extend further due to depth limit
                    path.pop()
                    continue
                
                # Check if we can extend from next_v
                next_mask = NEIGHBOR_MASKS[next_v]
                next_avail = next_mask & ~new_mask
                if next_avail:
                    next_candidates = _get_sorted_candidates(next_avail, new_mask, next_v)
                    stack.append((next_v, next_candidates, 0, new_mask))
                else:
                    # Dead end at next_v
                    path.pop()
                    backtracks += 1
            else:
                # No more candidates at this level, backtrack
                path.pop()
                stack.pop()
                backtracks += 1
        
        return best_path

    def _get_sorted_candidates(avail_mask, current_mask, current_vertex):
        """
        Get available neighbors sorted by Warnsdorff score:
        fewer available neighbors = higher priority (to avoid dead-ends).
        """
        candidates = []
        a = avail_mask
        while a:
            lb = a & (-a)
            v = lb.bit_length() - 1
            a ^= lb
            # Count how many neighbors of v are available (not in current_mask, not current_vertex)
            v_avail = NEIGHBOR_MASKS[v] & ~current_mask & ~VERTEX_MASKS[current_vertex]
            score = v_avail.bit_count()
            candidates.append((score, v.bit_count(), v))
        # Sort: fewer available neighbors first (Warnsdorff), then lower popcount
        candidates.sort(key=lambda x: (x[0], x[1]))
        return [c[2] for c in candidates]

    def verify_snake(path):
        """Verify validity of the snake using index map for O(L*N) check."""
        if not path:
            return False
        L = len(path)
        if len(set(path)) != L:
            return False
        # Build vertex -> index map
        idx = {}
        for i, v in enumerate(path):
            idx[v] = i
        # For each vertex, check all its 9 neighbors
        for i, v in enumerate(path):
            for nb in NEIGHBORS[v]:
                if nb in idx:
                    j = idx[nb]
                    if abs(i - j) != 1:
                        return False
        return True

    def try_splice(path1, path2):
        """
        Try to splice two paths together.
        If an endpoint of path1 is adjacent to an endpoint of path2,
        and the spliced path is valid, return the spliced path.
        """
        # Try all 4 combinations of endpoints
        for p1_end in [0, -1]:
            for p2_end in [0, -1]:
                v1 = path1[p1_end]
                v2 = path2[p2_end]
                if (v1 ^ v2) & ((v1 ^ v2) - 1) == 0:  # differ by exactly one bit
                    # Build spliced path
                    if p1_end == 0:
                        seg1 = list(reversed(path1))
                    else:
                        seg1 = list(path1)
                    if p2_end == 0:
                        seg2 = list(reversed(path2))
                    else:
                        seg2 = list(path2)
                    
                    # seg1 ends with v1, seg2 starts with v2
                    # Check they differ by one bit (already checked)
                    spliced = seg1 + seg2[1:]
                    
                    # Quick check: no duplicates
                    if len(set(spliced)) != len(spliced):
                        continue
                    
                    # Full verification (expensive but necessary)
                    if verify_snake(spliced):
                        return spliced
        return None

    def greedy_extend(path, direction='forward', max_steps=50):
        """Simple greedy extension from one end."""
        path = list(path)
        for _ in range(max_steps):
            if time_left() < 2.0:
                break
            if direction == 'forward':
                cur = path[-1]
            else:
                cur = path[0]
            
            # Build mask
            mask = 0
            for v in path:
                mask |= VERTEX_MASKS[v]
            for i in range(1, len(path) - 1):
                mask |= NEIGHBOR_MASKS[path[i]]
            
            avail = NEIGHBOR_MASKS[cur] & ~mask
            if avail == 0:
                break
            
            candidates = _get_sorted_candidates(avail, mask, cur)
            if not candidates:
                break
            
            next_v = candidates[0]
            if direction == 'forward':
                path.append(next_v)
            else:
                path.insert(0, next_v)
        
        return path

    best_path = None
    best_len = 0

    # Phase 1: Warnsdorff backtracking from multiple starts (70% of time)
    deadline1 = start_time + time_left() * 0.7
    
    # First: deterministic from 0
    p = warnsdorff_backtrack(0, max_depth=250, max_backtracks=100000)
    if len(p) > best_len:
        best_len = len(p)
        best_path = p

    # Collect several paths for splicing
    all_paths = [p]
    
    attempts = 0
    while time.time() < deadline1:
        sv = random.randint(0, SIZE - 1)
        # Vary depth and backtrack budget
        depth = random.choice([150, 200, 250, 300])
        bt = random.choice([50000, 100000, 200000])
        p = warnsdorff_backtrack(sv, max_depth=depth, max_backtracks=bt)
        attempts += 1
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
        if len(p) > 50:
            all_paths.append(p)
        # Keep only top 20 paths for splicing
        if len(all_paths) > 20:
            all_paths.sort(key=lambda x: -len(x))
            all_paths = all_paths[:20]

    # Phase 2: Splicing (20% of time)
    deadline2 = start_time + time_left() * 0.9
    if len(all_paths) >= 2:
        # Try splicing pairs
        tried = set()
        while time.time() < deadline2:
            i = random.randint(0, len(all_paths) - 1)
            j = random.randint(0, len(all_paths) - 1)
            if i == j:
                continue
            key = (min(i, j), max(i, j))
            if key in tried:
                continue
            tried.add(key)
            
            spliced = try_splice(all_paths[i], all_paths[j])
            if spliced and len(spliced) > best_len:
                best_len = len(spliced)
                best_path = spliced
                all_paths.append(spliced)
                if len(all_paths) > 20:
                    all_paths.sort(key=lambda x: -len(x))
                    all_paths = all_paths[:20]

    # Phase 3: Greedy extension on best path (10% of time)
    if time_left() > 5.0:
        # Extend forward
        p_ext = greedy_extend(best_path, 'forward', max_steps=100)
        if len(p_ext) > best_len:
            best_len = len(p_ext)
            best_path = p_ext
        # Extend backward
        p_ext = greedy_extend(best_path, 'backward', max_steps=100)
        if len(p_ext) > best_len:
            best_len = len(p_ext)
            best_path = p_ext

    # Phase 4: Local improvement polish (try to insert vertices or extend)
    if time_left() > 8.0:
        def local_improve(path):
            """Try to extend the path or find local improvements."""
            path = list(path)
            improved = True
            while improved and time_left() > 3.0:
                improved = False
                L = len(path)
                # Build mask of forbidden vertices (all used + neighbors of internal vertices)
                used_mask = 0
                for v in path:
                    used_mask |= VERTEX_MASKS[v]
                forbidden_mask = used_mask
                for i in range(1, L - 1):
                    forbidden_mask |= NEIGHBOR_MASKS[path[i]]
                
                # Try extending forward
                cur = path[-1]
                avail = NEIGHBOR_MASKS[cur] & ~forbidden_mask
                if avail:
                    candidates = _get_sorted_candidates(avail, forbidden_mask, cur)
                    if candidates:
                        next_v = candidates[0]
                        path.append(next_v)
                        improved = True
                        continue
                
                # Try extending backward
                cur = path[0]
                avail = NEIGHBOR_MASKS[cur] & ~forbidden_mask
                if avail:
                    candidates = _get_sorted_candidates(avail, forbidden_mask, cur)
                    if candidates:
                        next_v = candidates[0]
                        path.insert(0, next_v)
                        improved = True
                        continue
                
                # Try inserting a vertex between consecutive vertices
                # If path[i] and path[i+1] are adjacent, maybe we can insert a vertex
                # between them that is adjacent to both (impossible in hypercube since
                # they differ by 1 bit, so no common neighbor that differs by 1 from both)
                # Instead, try to replace a segment: skip path[i] if path[i-1] and path[i+1]
                # are at distance 2, and find a replacement
                for i in range(1, L - 1):
                    if time_left() < 3.0:
                        break
                    # Check if we can remove path[i] and connect path[i-1] to path[i+1]
                    # This would shorten the path, so only do it if we can then extend more
                    # Skip for now - focus on extensions
                    pass
            
            return path

        p_improved = local_improve(best_path)
        if len(p_improved) > best_len:
            best_len = len(p_improved)
            best_path = p_improved

    # Final: more Warnsdorff attempts if time permits
    if time_left() > 10.0:
        for _ in range(8):
            if time_left() < 5.0:
                break
            sv = random.randint(0, SIZE - 1)
            p = warnsdorff_backtrack(sv, max_depth=300, max_backtracks=300000)
            if len(p) > best_len:
                best_len = len(p)
                best_path = p
            # Also try to extend this new path
            if len(p) >= best_len - 5:
                p_ext = greedy_extend(p, 'forward', max_steps=50)
                if len(p_ext) > best_len:
                    best_len = len(p_ext)
                    best_path = p_ext
                p_ext = greedy_extend(p, 'backward', max_steps=50)
                if len(p_ext) > best_len:
                    best_len = len(p_ext)
                    best_path = p_ext

    # Safety verification
    if best_path and not verify_snake(best_path):
        best_path = [0]
    
    return best_path
