"""Snake-in-the-box in Q_9: bidirectional growth with Warnsdorff's rule."""
import time
import random
import numpy as np

N = 9
SEED_TIME = 150.0  # evaluator wall-clock kill; keep margin
DEADLINE = SEED_TIME - 20.0  # 130s safety margin
T0 = time.time()

def _elapsed():
    return time.time() - T0

def _deadline_hit():
    return _elapsed() >= DEADLINE

def _is_valid_extension(v, path_arr, used_set, end_index):
    """Check if v can be appended at position end_index (0 or 1).
    end_index=0 means prepend, end_index=1 means append.
    path_arr is numpy array of current path vertices.
    """
    if v in used_set:
        return False
    # v must differ in >=2 bits from all path vertices
    # Consecutive vertices (the one it connects to) differ in exactly 1 bit (guaranteed by construction)
    # All OTHER vertices must differ in >=2 bits
    if len(path_arr) == 0:
        return True
    diffs = np.bitwise_xor(path_arr, v)
    # popcount via lookup
    bc = _popcount(diffs)
    # The vertex it connects to (end vertex) has bc==1; all others must have bc>=2
    # For append: connecting vertex is path_arr[-1]; others are path_arr[:-1]
    # For prepend: connecting vertex is path_arr[0]; others are path_arr[1:]
    if end_index == 1:
        others_bc = bc[:-1] if len(path_arr) > 1 else np.array([], dtype=np.int32)
    else:
        others_bc = bc[1:] if len(path_arr) > 1 else np.array([], dtype=np.int32)
    if len(others_bc) > 0 and np.any(others_bc == 1):
        return False
    return True

# Precompute popcount for all 9-bit values
_POPCOUNT_TABLE = np.array([i.bit_count() for i in range(512)], dtype=np.int32)

def _popcount(arr):
    """Vectorized popcount for array of small integers (< 512)."""
    return _POPCOUNT_TABLE[arr.astype(np.int64)]

def _count_safe_options(v, path_arr, used_set):
    """Count how many neighbors of v are still safe to extend (Warnsdorff heuristic).
    Fewer safe options = more urgent to place this vertex."""
    count = 0
    for b in range(N):
        u = v ^ (1 << b)
        if u in used_set:
            continue
        if len(path_arr) == 0:
            count += 1
            continue
        diffs = np.bitwise_xor(path_arr, u)
        bc = _popcount(diffs)
        if np.any(bc == 1):
            continue
        count += 1
    return count

def _bidirectional_grow(start_vertex, bit_order, rng):
    """Grow a snake bidirectionally from start_vertex using Warnsdorff's rule."""
    path = [start_vertex]
    used = {start_vertex}
    path_arr = np.array(path, dtype=np.int64)
    
    # Alternate growing left and right
    grow_left = False  # False means grow right (append)
    
    while not _deadline_hit():
        # Decide which end to grow
        if grow_left:
            cur = path[0]
            end_index = 0
        else:
            cur = path[-1]
            end_index = 1
        
        # Find all valid neighbors
        candidates = []
        for b in bit_order:
            v = cur ^ (1 << b)
            if v in used:
                continue
            if not _is_valid_extension(v, path_arr, used, end_index):
                continue
            candidates.append(v)
        
        if not candidates:
            break
        
        # Warnsdorff: pick candidate with fewest safe options
        # Shuffle ties randomly for diversity
        scored = []
        for v in candidates:
            opts = _count_safe_options(v, path_arr, used)
            scored.append((opts, rng.random(), v))
        scored.sort(key=lambda x: (x[0], x[1]))
        
        v = scored[0][2]
        
        if end_index == 1:
            path.append(v)
            path_arr = np.append(path_arr, v)
        else:
            path.insert(0, v)
            path_arr = np.insert(path_arr, 0, v)
        used.add(v)
        
        grow_left = not grow_left
    
    return path

def _validate(path):
    """Validate that path is a valid induced path in Q_N."""
    n = len(path)
    if n == 0:
        return False
    seen = set(path)
    if len(seen) != n:
        return False
    for i in range(n):
        if path[i] < 0 or path[i] >= (1 << N):
            return False
    for i in range(n - 1):
        d = (path[i] ^ path[i+1]).bit_count()
        if d != 1:
            return False
    for i in range(n):
        for j in range(i + 2, n):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    return True

def construct():
    global T0
    T0 = time.time()
    
    best_path = None
    best_len = 0
    
    # First: run the original greedy to establish baseline
    path_greedy, used_greedy = [0], {0}
    while not _deadline_hit():
        cur = path_greedy[-1]
        nxt = None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used_greedy:
                continue
            # Check against all previous vertices
            ok = True
            for p in path_greedy[:-1]:
                if (v ^ p).bit_count() == 1:
                    ok = False
                    break
            if ok:
                nxt = v
                break
        if nxt is None:
            break
        path_greedy.append(nxt)
        used_greedy.add(nxt)
    
    if len(path_greedy) > best_len:
        best_len = len(path_greedy)
        best_path = path_greedy[:]
    
    # Now run multiple bidirectional growth attempts with different seeds
    rng = random.Random(42)
    
    # Try different starting vertices and bit orders
    start_vertices = [0, (1 << N) - 1]  # 0 and all-ones
    # Add some random starts
    for _ in range(4):
        start_vertices.append(rng.randint(0, (1 << N) - 1))
    
    attempts = 0
    for sv in start_vertices:
        if _deadline_hit():
            break
        # Try multiple bit orderings
        for trial in range(6):
            if _deadline_hit():
                break
            bit_order = list(range(N))
            rng.shuffle(bit_order)
            
            path = _bidirectional_grow(sv, bit_order, rng)
            if len(path) > best_len:
                # Validate before accepting
                if _validate(path):
                    best_len = len(path)
                    best_path = path[:]
            attempts += 1
    
    # If we have time, do more randomized restarts
    while not _deadline_hit():
        sv = rng.randint(0, (1 << N) - 1)
        bit_order = list(range(N))
        rng.shuffle(bit_order)
        path = _bidirectional_grow(sv, bit_order, rng)
        if len(path) > best_len:
            if _validate(path):
                best_len = len(path)
                best_path = path[:]
        attempts += 1
        if attempts > 2000:  # safety cap
            break
    
    # Final safety: if deadline is very close, return immediately
    if best_path is None:
        return [0]
    
    # Final validation
    if not _validate(best_path):
        # Fallback to greedy
        return path_greedy
    
    return best_path

# Precompute popcount table at module load
_POPCOUNT_TABLE = np.array([i.bit_count() for i in range(512)], dtype=np.int32)
