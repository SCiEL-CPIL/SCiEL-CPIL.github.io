"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

N = 9
FULL_MASK = (1 << N) - 1
ALL_BITS = [1 << b for b in range(N)]
DEADLINE_FRACTION = 0.85  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _popcount(x):
    return x.bit_count()

def _validate_snake(path):
    """Verify the snake is valid: distinct vertices, consecutive differ by 1 bit,
    non-consecutive differ by at least 2 bits."""
    n = len(path)
    if n == 0:
        return True
    seen = set()
    for v in path:
        if v in seen:
            return False
        seen.add(v)
    for i in range(n - 1):
        if _popcount(path[i] ^ path[i + 1]) != 1:
            return False
    for i in range(n - 1):
        for j in range(i + 2, n):
            if _popcount(path[i] ^ path[j]) == 1:
                return False
    return True

def _is_safe(path, v):
    """Check if vertex v can be appended to path."""
    # v must not already be in path
    for x in path:
        if x == v:
            return False
    # v must not be adjacent to any non-consecutive vertex (i.e., all except last)
    for i in range(len(path) - 1):
        if _popcount(v ^ path[i]) == 1:
            return False
    return True

def _count_safe_moves(path, v):
    """Count how many neighbors of v are safe to visit after appending v."""
    count = 0
    for b in range(N):
        w = v ^ (1 << b)
        # w must not be in path
        in_path = False
        for x in path:
            if x == w:
                in_path = True
                break
        if in_path:
            continue
        # w must not be adjacent to any non-consecutive vertex of (path + [v])
        # Non-consecutive to w in the extended path means all vertices except v
        # (and w itself isn't in path yet)
        safe = True
        for i in range(len(path)):
            if _popcount(w ^ path[i]) == 1:
                safe = False
                break
        if safe:
            count += 1
    return count

def _search_warnsdorff(start, time_limit, best_path):
    """
    Warnsdorff heuristic search: at each step, prefer the candidate with
    the fewest safe future moves. This tends to produce longer snakes.
    """
    path = [start]
    
    while True:
        cur = path[-1]
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if not _is_safe(path, v):
                continue
            candidates.append(v)
        
        if not candidates:
            break
        
        # Score each candidate by number of safe moves it leaves
        scored = []
        for v in candidates:
            s = _count_safe_moves(path, v)
            scored.append((s, v))
        
        # Sort by fewest safe moves (Warnsdorff rule)
        scored.sort(key=lambda x: x[0])
        
        # Take the best (fewest options remaining)
        next_v = scored[0][1]
        path.append(next_v)
        
        if len(path) > len(best_path[0]):
            best_path[0] = list(path)
        
        if time.time() > time_limit:
            break
    
    return path

def _search_warnsdorff_randomized(start, time_limit, best_path, diversity=0.15):
    """
    Warnsdorff with randomization: sometimes pick a non-best candidate
    to explore different regions of the search space.
    """
    path = [start]
    
    while True:
        cur = path[-1]
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if not _is_safe(path, v):
                continue
            candidates.append(v)
        
        if not candidates:
            break
        
        # Score each candidate
        scored = []
        for v in candidates:
            s = _count_safe_moves(path, v)
            scored.append((s, v))
        
        # Sort by fewest safe moves
        scored.sort(key=lambda x: x[0])
        
        # With some probability, pick randomly from top-k for diversity
        k = min(3, len(scored))
        if random.random() < diversity:
            next_v = random.choice([s[1] for s in scored[:k]])
        else:
            next_v = scored[0][1]
        
        path.append(next_v)
        
        if len(path) > len(best_path[0]):
            best_path[0] = list(path)
        
        if time.time() > time_limit:
            break
    
    return path

def _try_extend(path, time_limit, best_path):
    """Try to greedily extend the path using Warnsdorff from its end."""
    cur_path = list(path)
    while True:
        cur = cur_path[-1]
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if not _is_safe(cur_path, v):
                continue
            candidates.append(v)
        
        if not candidates:
            break
        
        scored = []
        for v in candidates:
            s = _count_safe_moves(cur_path, v)
            scored.append((s, v))
        scored.sort(key=lambda x: x[0])
        
        next_v = scored[0][1]
        cur_path.append(next_v)
        
        if len(cur_path) > len(best_path[0]):
            best_path[0] = list(cur_path)
        
        if time.time() > time_limit:
            break
    
    return cur_path

def _local_search_improve(path, time_limit, best_path):
    """
    Local improvement: try removing the last few vertices and re-extending
    with different choices.
    """
    best = list(path)
    
    for trim in range(1, min(8, len(best))):
        if time.time() > time_limit:
            break
        trimmed = list(best[:-trim])
        # Try to re-extend the trimmed path
        extended = _try_extend(trimmed, time_limit, best_path)
        if len(extended) > len(best):
            best = list(extended)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 120 * DEADLINE_FRACTION  # 102 seconds
    
    best_path = [[]]
    
    # Phase 1: Multiple Warnsdorff searches from different starts
    # Start from 0 first (canonical)
    _search_warnsdorff(0, deadline, best_path)
    
    # Try several random starts with randomized Warnsdorff
    random.seed(42)
    num_starts = 50
    for i in range(num_starts):
        if time.time() > deadline:
            break
        start = random.randint(0, (1 << N) - 1)
        _search_warnsdorff_randomized(start, deadline, best_path, diversity=0.1)
    
    # Phase 2: Local improvement on the best found
    if time.time() < deadline:
        improved = _local_search_improve(best_path[0], deadline, best_path)
    
    # Phase 3: Try more random starts if time remains
    if time.time() < deadline:
        for i in range(20):
            if time.time() > deadline:
                break
            start = random.randint(0, (1 << N) - 1)
            _search_warnsdorff(start, deadline, best_path)
    
    # Final validation
    final_path = best_path[0]
    if not _validate_snake(final_path):
        # Fallback: return a known valid short snake
        final_path = [0]
    
    return final_path

# EVOLVE-BLOCK-END
