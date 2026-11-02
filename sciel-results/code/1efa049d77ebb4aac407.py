"""Combined program: structured constructions + difference-sparse AP gap optimization."""
import time
import math
import random

_START = time.time()
_DEADLINE = 148.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START

def _remaining():
    return _DEADLINE - _elapsed()

def _score_set(U):
    """Compute score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    U = sorted(set(U))
    if 0 not in U:
        return -1.0
    n = len(U)
    if n < 2:
        return -1.0
    maxU = max(U)
    if maxU == 0:
        return -1.0
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    ss = len(sumset)
    ds = len(diffset)
    if ss == 0:
        return -1.0
    ratio = ds / ss
    if ratio <= 0:
        return -1.0
    denom = math.log(2 * maxU + 1)
    if denom <= 0:
        return -1.0
    return 1.0 + math.log(ratio) / denom

def _diff_sparse_ap(n, gap):
    """
    Build a difference-sparse set: n elements in an arithmetic progression
    with common difference 'gap'. This keeps |U-U| = 2*gap*(n-1)+1 (linear in n)
    while |U+U| grows quadratically, but the large gap inflates max(U) in the
    denominator, and the ratio |U-U|/|U+U| can be optimized over gap.
    """
    if n <= 1:
        return [0]
    return [0] + [i * gap for i in range(1, n)]

def _diff_sparse_subsample(n, gap, keep_ratio=0.7):
    """
    Difference-sparse: take a subset of the AP {0, gap, 2*gap, ...} to reduce
    sumset collisions while keeping differences spread out.
    """
    if n <= 1:
        return [0]
    # Choose indices from 0..(n-1) with some structure
    indices = [0]
    remaining = list(range(1, n))
    random.shuffle(remaining)
    for i in remaining:
        if len(indices) >= n:
            break
        indices.append(i)
    indices.sort()
    return [i * gap for i in indices]

def _golomb_ruler(n):
    """Build a Golomb ruler of length n (all pairwise distances distinct)."""
    if n <= 1:
        return [0]
    U = [0]
    # Greedy: place each new mark as far as possible without creating duplicate distances
    for i in range(1, n):
        placed = False
        for pos in range(1, 1000):
            # Check if pos creates duplicate distances
            ok = True
            for existing in U:
                d = pos - existing
                for other in U:
                    if other != existing:
                        if abs(pos - other) == abs(existing - other):
                            ok = False
                            break
                        if d == abs(pos - other) and other != existing:
                            # Check against all existing differences
                            pass
                if not ok:
                    break
            if ok:
                U.append(pos)
                placed = True
                break
        if not placed:
            # Fallback: just extend
            U.append(U[-1] + 1)
    return U

def _superincreasing(k):
    """Superincreasing sequence: each element > sum of all previous."""
    U = [0]
    for i in range(1, k):
        s = sum(U)
        U.append(s + 1)
    return U

def _staircase(n_blocks, block_size):
    """Staircase: blocks with growing gaps."""
    U = [0]
    offset = 1
    for i in range(n_blocks):
        for j in range(1, block_size):
            U.append(offset + j)
        offset += block_size * (block_size + 1) + 1
    return sorted(set(U))

def _local_search(U, max_iter=500, time_limit=10.0):
    """Local search with add/remove/shift operations."""
    start = time.time()
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = list(U)
    
    max_val = max(U) if U else 10
    search_range = max(max_val * 2, 50)
    
    for iteration in range(max_iter):
        if time.time() - start > time_limit or _remaining() < 5.0:
            break
        improved = False
        
        # Generate candidate moves
        all_moves = []
        
        # Add
        if len(U) < 25:
            for x in range(1, search_range + 1):
                if x not in U:
                    all_moves.append(('add', x))
        
        # Remove
        for x in U:
            if x != 0:
                all_moves.append(('remove', x))
        
        # Shift
        for x in U:
            if x == 0:
                continue
            for delta in [-5, -3, -2, -1, 1, 2, 3, 5, 8, 13]:
                new_x = x + delta
                if new_x >= 0 and new_x != x and new_x not in U:
                    all_moves.append(('shift', x, new_x))
        
        random.shuffle(all_moves)
        
        for move in all_moves[:30]:
            if time.time() - start > time_limit or _remaining() < 5.0:
                break
            if move[0] == 'add':
                new_U = list(U) + [move[1]]
            elif move[0] == 'remove':
                new_U = [x for x in U if x != move[1]]
            else:
                new_U = [move[2] if x == move[1] else x for x in U]
            
            if 0 not in new_U or len(new_U) < 2:
                continue
            
            s = _score_set(new_U)
            if s > best_score + 1e-12:
                best_score = s
                best_U = new_U
                U = new_U
                search_range = max(max(U) * 2, 50)
                improved = True
                break
        
        if not improved:
            # Random perturbation
            for _ in range(3):
                if time.time() - start > time_limit or _remaining() < 5.0:
                    break
                new_U = list(U)
                for _ in range(random.randint(1, 2)):
                    new_U.append(random.randint(1, search_range))
                if len(new_U) > 3:
                    rem = random.choice([x for x in new_U if x != 0])
                    new_U.remove(rem)
                
                if 0 not in new_U or len(new_U) < 2:
                    continue
                s = _score_set(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U
                    U = new_U
                    improved = True
                    break
    
    return best_U, best_score

def _gap_optimization(n, time_limit=5.0):
    """
    For a fixed n, optimize the gap in a difference-sparse AP.
    Try various gaps and also try subsampled versions.
    """
    start = time.time()
    best_U = None
    best_score = -1.0
    
    # Try various gaps
    for gap in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50, 70, 100]:
        if time.time() - start > time_limit or _remaining() < 3.0:
            break
        U = _diff_sparse_ap(n, gap)
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U
        
        # Also try local search on this
        if _remaining() > 8.0:
            U_ls, s_ls = _local_search(U, max_iter=100, time_limit=min(2.0, time.time() - start + time_limit))
            if s_ls > best_score:
                best_score = s_ls
                best_U = U_ls
    
    # Try subsampled versions with larger gaps
    for gap in [5, 10, 15, 20, 30, 50]:
        if time.time() - start > time_limit or _remaining() < 3.0:
            break
        for keep in [0.5, 0.6, 0.7, 0.8]:
            random.seed(42 + gap + int(keep * 100))
            U = _diff_sparse_subsample(n, gap, keep_ratio=keep)
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U
    
    return best_U, best_score

def construct():
    random.seed(42)
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # Strategy 1: Difference-sparse AP with gap optimization
    for n in range(3, 18):
        if _remaining() < 5.0:
            break
        U, s = _gap_optimization(n, time_limit=min(3.0, _remaining() - 5.0))
        if U and s > best_score:
            best_score = s
            best_U = U
    
    # Strategy 2: Golomb rulers
    for n in range(3, 15):
        if _remaining() < 5.0:
            break
        U = _golomb_ruler(n)
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
        if _remaining() > 10.0:
            U_ls, s_ls = _local_search(U, max_iter=200, time_limit=min(3.0, _remaining() - 5.0))
            if s_ls > best_score:
                best_score = s_ls
                best_U = U_ls
    
    # Strategy 3: Superincreasing sequences
    for k in range(3, 15):
        if _remaining() < 5.0:
            break
        U = _superincreasing(k)
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
        if _remaining() > 10.0:
            U_ls, s_ls = _local_search(U, max_iter=200, time_limit=min(3.0, _remaining() - 5.0))
            if s_ls > best_score:
                best_score = s_ls
                best_U = U_ls
    
    # Strategy 4: Staircase constructions
    for nb in range(2, 7):
        if _remaining() < 5.0:
            break
        for bs in range(2, 7):
            if _remaining() < 5.0:
                break
            U = _staircase(nb, bs)
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = list(U)
            if _remaining() > 10.0:
                U_ls, s_ls = _local_search(U, max_iter=150, time_limit=min(2.0, _remaining() - 5.0))
                if s_ls > best_score:
                    best_score = s_ls
                    best_U = U_ls
    
    # Strategy 5: Structured initial candidates
    initial_candidates = [
        [0, 1, 3],
        [0, 1, 2, 4],
        [0, 1, 3, 6],
        [0, 1, 2, 3, 5],
        [0, 1, 3, 4, 6],
        [0, 1, 2, 4, 5],
        [0, 1, 3, 5, 6],
        [0, 1, 2, 3, 4, 6],
        [0, 1, 3, 4, 5, 7],
        [0, 1, 2, 4, 6, 7],
        [0, 1, 4, 5, 8],
        [0, 1, 3, 7, 8],
        [0, 2, 3, 5, 7],
        [0, 1, 4, 6, 7],
        [0, 1, 2, 4, 5, 7, 8],
        [0, 1, 3, 4, 6, 7, 9],
    ]
    
    for cand in initial_candidates:
        if _remaining() < 5.0:
            break
        s = _score_set(cand)
        if s > best_score:
            best_score = s
            best_U = list(cand)
    
    # Strategy 6: Random constructions with local search
    for trial in range(30):
        if _remaining() < 10.0:
            break
        size = random.randint(3, 14)
        max_val = random.randint(10, 80)
        U = {0}
        while len(U) < size:
            U.add(random.randint(1, max_val))
        U = sorted(U)
        
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U
        
        if _remaining() > 10.0:
            U2, s2 = _local_search(U, max_iter=150, time_limit=min(3.0, _remaining() - 5.0))
            if s2 > best_score:
                best_score = s2
                best_U = U2
    
    # Final polish: intensive local search on best
    if _remaining() > 5.0:
        best_U, best_score = _local_search(best_U, max_iter=500, time_limit=min(_remaining() - 3.0, 15.0))
    
    return best_U
