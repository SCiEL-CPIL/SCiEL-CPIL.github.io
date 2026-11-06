"""
Snake-in-the-Box solver for Q_9 using Warnsdorff backtracking and splicing.
"""
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
        best_path = [start_vertex]
        best_len = 1
        
        path = [start_vertex]
        mask = VERTEX_MASKS[start_vertex]
        
        stack = []  # Each entry: (vertex, candidates_list, next_idx, mask_before)
        
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
                next_v = candidates[idx]
                stack[-1] = (cur_v, candidates, idx + 1, mask_before)
                
                path.append(next_v)
                new_mask = mask_before | NEIGHBOR_MASKS[cur_v] | VERTEX_MASKS[next_v]
                
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = list(path)
                
                if len(path) >= max_depth:
                    path.pop()
                    continue
                
                next_mask = NEIGHBOR_MASKS[next_v]
                next_avail = next_mask & ~new_mask
                if next_avail:
                    next_candidates = _get_sorted_candidates(next_avail, new_mask, next_v)
                    stack.append((next_v, next_candidates, 0, new_mask))
                else:
                    path.pop()
                    backtracks += 1
            else:
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
            v_avail = NEIGHBOR_MASKS[v] & ~current_mask & ~VERTEX_MASKS[current_vertex]
            score = v_avail.bit_count()
            candidates.append((score, v.bit_count(), v))
        candidates.sort(key=lambda x: (x[0], x[1]))
        return [c[2] for c in candidates]

    def verify_snake(path):
        """Verify validity of the snake using bitmasks for speed."""
        if not path:
            return False
        if len(set(path)) != len(path):
            return False
        
        L = len(path)
        
        for i in range(L - 1):
            diff = path[i] ^ path[i+1]
            if diff == 0 or (diff & (diff - 1)) != 0:
                return False
        
        idx_map = {v: i for i, v in enumerate(path)}
        
        for i, v in enumerate(path):
            for nb in NEIGHBORS[v]:
                if nb in idx_map:
                    j = idx_map[nb]
                    if abs(i - j) > 1:
                        return False
        return True

    def try_splice(path1, path2):
        """
        Try to splice two paths together.
        If an endpoint of path1 is adjacent to an endpoint of path2,
        and the spliced path is valid, return the spliced path.
        """
        for p1_end in [0, -1]:
            for p2_end in [0, -1]:
                if p1_end == 0:
                    p1_seg = path1
                else:
                    p1_seg = path1[::-1]
                
                if p2_end == 0:
                    p2_seg = path2
                else:
                    p2_seg = path2[::-1]
                
                if p1_seg[-1] == p2_seg[0]:
                    continue
                if (p1_seg[-1] ^ p2_seg[0]).bit_count() != 1:
                    continue
                
                combined = p1_seg + p2_seg[1:]
                if verify_snake(combined):
                    return combined
        return None

    best_path = [0]
    best_len = 1
    found_paths = []
    
    # Phase 1: Multiple restarts of Warnsdorff backtracking
    while time_left() > 10.0:
        seed = random.randint(0, SIZE - 1)
        path = warnsdorff_backtrack(seed, max_depth=200, max_backtracks=50000)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
            found_paths.append(path)
        
        # Keep a few best paths for splicing
        if len(found_paths) > 5:
            found_paths.sort(key=len, reverse=True)
            found_paths = found_paths[:5]
    
    # Phase 2: Splicing attempts with remaining time
    while time_left() > 5.0 and len(found_paths) >= 2:
        for i in range(len(found_paths)):
            for j in range(i + 1, len(found_paths)):
                spliced = try_splice(found_paths[i], found_paths[j])
                if spliced is not None:
                    if len(spliced) > best_len:
                        best_len = len(spliced)
                        best_path = spliced
                        found_paths.append(spliced)
                        found_paths.sort(key=len, reverse=True)
                        found_paths = found_paths[:5]
                    break
            if time_left() < 5.0:
                break
    
    return best_path
