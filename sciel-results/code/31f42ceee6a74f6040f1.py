import time
import numpy as np

N = 9
MAX_TIME = 117.0
NUM_VERTICES = 1 << N

def construct():
    start_time = time.time()
    
    # Precompute neighbors
    neighbors = np.empty((NUM_VERTICES, N), dtype=np.uint32)
    for v in range(NUM_VERTICES):
        for b in range(N):
            neighbors[v, b] = v ^ (1 << b)
    
    best_path = None
    best_len = 0
    
    rng = np.random.default_rng(seed=12345)
    
    def time_ok():
        return time.time() - start_time < MAX_TIME
    
    def bidirectional_grow(start_v, end_v, max_backtrack=3, max_depth=50):
        """
        Grow a snake from both ends simultaneously.
        start_v and end_v are the initial endpoints.
        Returns the full path (start side reversed + end side) if valid.
        """
        # Path is stored as list of vertices from start to end
        # We grow from both ends: left end is path[0], right end is path[-1]
        
        if start_v == end_v:
            return [start_v]
        
        # Check initial validity: start_v and end_v must differ by at least 2 bits
        diff = start_v ^ end_v
        if np.count_nonzero(diff & 1) < 2 and diff != 0:
            # They differ by exactly 1 bit — they'd be adjacent, which is fine 
            # as long as they're not consecutive in a way that violates induced path
            # Actually for a 2-vertex path, they just need to differ by 1 bit
            # But for longer paths, non-consecutive must differ by >= 2
            # With just 2 vertices, they're consecutive, so 1-bit diff is OK
            pass
        
        path = [start_v, end_v]
        used = np.zeros(NUM_VERTICES, dtype=bool)
        used[start_v] = True
        used[end_v] = True
        
        # forbidden[v] = True if v is adjacent to any internal vertex
        # Initially no internal vertices (both are endpoints)
        forbidden = np.zeros(NUM_VERTICES, dtype=bool)
        
        # Track which end we're growing
        # left_end = 0, right_end = 1
        side = 0  # 0 = grow from left (start), 1 = grow from right (end)
        
        # For backtracking, we store states
        # State: (path_copy, used_copy, forbidden_copy, side, backtracks_used)
        # To save memory, we'll do limited backtracking
        
        def get_valid_next(cur, used_arr, forbidden_arr):
            """Get valid next vertices for cur given used and forbidden."""
            nbrs = neighbors[cur]
            mask = (~used_arr[nbrs]) & (~forbidden_arr[nbrs])
            return nbrs[mask]
        
        def score_candidate(n, used_arr, forbidden_arr):
            """Score a candidate by min degree of its valid neighbors (lookahead)."""
            nbrs_n = neighbors[n]
            mask = (~used_arr[nbrs_n]) & (~forbidden_arr[nbrs_n])
            valid_nn = nbrs_n[mask]
            if len(valid_nn) == 0:
                return 0
            # Min degree of valid neighbors
            min_deg = 99
            for nn in valid_nn:
                nbrs_nn = neighbors[nn]
                mask2 = (~used_arr[nbrs_nn]) & (~forbidden_arr[nbrs_nn])
                deg = np.sum(mask2)
                if deg < min_deg:
                    min_deg = deg
            return min_deg
        
        def choose_next(cur, used_arr, forbidden_arr, strategy):
            """Choose next vertex from cur."""
            valid = get_valid_next(cur, used_arr, forbidden_arr)
            if len(valid) == 0:
                return None
            
            if len(valid) == 1:
                return int(valid[0])
            
            if strategy == "min_degree":
                best_score = 99
                best_candidates = []
                for n in valid:
                    s = score_candidate(n, used_arr, forbidden_arr)
                    if s < best_score:
                        best_score = s
                        best_candidates = [n]
                    elif s == best_score:
                        best_candidates.append(n)
                return int(best_candidates[rng.integers(len(best_candidates))])
            else:  # max_degree
                best_score = -1
                best_candidates = []
                for n in valid:
                    s = score_candidate(n, used_arr, forbidden_arr)
                    if s > best_score:
                        best_score = s
                        best_candidates = [n]
                    elif s == best_score:
                        best_candidates.append(n)
                return int(best_candidates[rng.integers(len(best_candidates))])
        
        # Backtracking stack
        # Each entry: (path_snapshot, used_snapshot, forbidden_snapshot, side)
        # We limit backtracking depth to avoid memory issues
        backtrack_stack = []
        backtracks_used = 0
        max_backtracks = max_backtrack
        
        while time_ok():
            if side == 0:
                cur = path[0]
            else:
                cur = path[-1]
            
            next_v = choose_next(cur, used, forbidden, "min_degree")
            
            if next_v is None:
                # Stuck at this end
                # Try the other end
                if side == 0:
                    side = 1
                    # If both ends stuck, try backtracking
                    cur2 = path[-1]
                    next_v2 = choose_next(cur2, used, forbidden, "min_degree")
                    if next_v2 is None:
                        # Both ends stuck
                        if backtracks_used < max_backtracks and backtrack_stack:
                            # Backtrack
                            path_snap, used_snap, forb_snap, side_snap = backtrack_stack.pop()
                            path = path_snap
                            used = used_snap
                            forbidden = forb_snap
                            side = side_snap
                            backtracks_used += 1
                            continue
                        else:
                            break
                else:
                    side = 0
                    cur2 = path[0]
                    next_v2 = choose_next(cur2, used, forbidden, "min_degree")
                    if next_v2 is None:
                        if backtracks_used < max_backtracks and backtrack_stack:
                            path_snap, used_snap, forb_snap, side_snap = backtrack_stack.pop()
                            path = path_snap
                            used = used_snap
                            forbidden = forb_snap
                            side = side_snap
                            backtracks_used += 1
                            continue
                        else:
                            break
                    else:
                        next_v = next_v2
                        side = 0
            else:
                # We have a move
                pass
            
            # Save state for potential backtracking
            if backtracks_used < max_backtracks:
                backtrack_stack.append((path[:], used.copy(), forbidden.copy(), side))
                if len(backtrack_stack) > 10:
                    backtrack_stack.pop(0)
            
            # Execute the move
            if side == 0:
                # Insert at beginning
                path.insert(0, next_v)
                # cur becomes internal
                for nc in neighbors[cur]:
                    forbidden[nc] = True
                used[next_v] = True
            else:
                # Append at end
                path.append(next_v)
                # cur becomes internal
                for nc in neighbors[cur]:
                    forbidden[nc] = True
                used[next_v] = True
            
            # Alternate sides (bias toward shorter side)
            # Left side length: index of start in path
            # Right side length: index of end in path
            # Actually, let's just alternate
            side = 1 - side
        
        return path
    
    def validate_path(path):
        """Check if path is a valid induced path (snake)."""
        if len(path) < 2:
            return len(path) >= 1
        # Check distinct
        if len(set(path)) != len(path):
            return False
        # Check consecutive differ by exactly 1 bit
        for i in range(len(path) - 1):
            diff = path[i] ^ path[i+1]
            if diff == 0 or (diff & (diff - 1)) != 0:
                return False
        # Check non-consecutive differ by at least 2 bits
        # For efficiency, only check a subset if path is long
        n = len(path)
        if n <= 100:
            for i in range(n):
                for j in range(i + 2, n):
                    diff = path[i] ^ path[j]
                    if diff != 0 and (diff & (diff - 1)) == 0:
                        return False
        else:
            # Sample check
            for _ in range(200):
                i = rng.integers(n)
                j = rng.integers(n)
                if abs(i - j) < 2:
                    continue
                diff = path[i] ^ path[j]
                if diff != 0 and (diff & (diff - 1)) == 0:
                    return False
        return True
    
    def try_bidirectional(start_v, end_v):
        """Try bidirectional growth from start_v to end_v."""
        if start_v == end_v:
            return [start_v]
        
        # They should be at distance >= 2 for a non-trivial snake
        # But even distance 1 works for 2-vertex path
        diff = start_v ^ end_v
        pc = bin(diff).count('1')
        
        if pc == 0:
            return [start_v]
        
        # For bidirectional to work well, start and end should be far apart
        # but not too far (to allow growth from both sides)
        # Let's just try it
        path = bidirectional_grow(start_v, end_v, max_backtrack=2, max_depth=50)
        return path
    
    # Main search loop
    while time_ok():
        # Choose strategy for this iteration
        r = rng.random()
        
        if r < 0.6:
            # Bidirectional growth
            start_v = int(rng.integers(NUM_VERTICES))
            # Choose end_v at Hamming distance 4-7 from start_v
            target_dist = int(rng.integers(4, 8))
            # Pick random bits to flip
            bits = rng.choice(N, size=target_dist, replace=False)
            mask = 0
            for b in bits:
                mask |= (1 << b)
            end_v = start_v ^ mask
            
            path = try_bidirectional(start_v, end_v)
            
        elif r < 0.8:
            # Bidirectional with random endpoints at distance 3-5
            start_v = int(rng.integers(NUM_VERTICES))
            target_dist = int(rng.integers(3, 6))
            bits = rng.choice(N, size=target_dist, replace=False)
            mask = 0
            for b in bits:
                mask |= (1 << b)
            end_v = start_v ^ mask
            
            path = try_bidirectional(start_v, end_v)
        else:
            # Single-direction greedy (fallback)
            start_v = int(rng.integers(NUM_VERTICES))
            path = [start_v]
            used = np.zeros(NUM_VERTICES, dtype=bool)
            used[start_v] = True
            forbidden = np.zeros(NUM_VERTICES, dtype=bool)
            cur = start_v
            
            while time_ok():
                nbrs = neighbors[cur]
                valid_mask = (~used[nbrs]) & (~forbidden[nbrs])
                valid_next = nbrs[valid_mask]
                
                if len(valid_next) == 0:
                    break
                
                if len(valid_next) == 1:
                    next_v = int(valid_next[0])
                else:
                    # Min degree strategy
                    best_score = 99
                    best_candidates = []
                    for n in valid_next:
                        nbrs_n = neighbors[n]
                        mask = (~used[nbrs_n]) & (~forbidden[nbrs_n])
                        s = np.sum(mask)
                        if s < best_score:
                            best_score = s
                            best_candidates = [n]
                        elif s == best_score:
                            best_candidates.append(n)
                    next_v = int(best_candidates[rng.integers(len(best_candidates))])
                
                path.append(next_v)
                used[next_v] = True
                for nc in nbrs:
                    forbidden[nc] = True
                cur = next_v
        
        if len(path) > best_len:
            # Validate before accepting
            if validate_path(path):
                best_len = len(path)
                best_path = path
    
    if best_path is None:
        # Fallback: trivial path
        best_path = [0]
    
    return best_path
