import random
import time
import itertools

def _score(S):
    """Count distinct d>0 such that a, a+d, a+2d all in S."""
    S = set(S)
    diffs = set()
    lst = sorted(S)
    n = len(lst)
    for i in range(n):
        for j in range(i + 1, n):
            a = lst[i]
            c = lst[j]
            if (a + c) % 2 == 0:
                b = (a + c) // 2
                if b in S:
                    d = (c - a) // 2
                    if d > 0:
                        diffs.add(d)
    return len(diffs)

def _grid_construct(r, c, R, C, offset):
    """Generate a grid set with r rows, c cols."""
    S = set()
    for i in range(r):
        for j in range(c):
            val = i * R + j * C + offset
            if 1 <= val <= 512:
                S.add(val)
    return S

def _local_search(S, max_iters=20000):
    """Improve set via local search (swap with random element in 1..512)."""
    S = set(S)
    current_score = _score(S)
    best_S = set(S)
    best_score = current_score
    lst = sorted(S)
    
    for it in range(max_iters):
        # Pick a random element to replace
        idx = random.randint(0, len(lst) - 1)
        old_val = lst[idx]
        new_val = random.randint(1, 512)
        if new_val in S:
            continue
        # Try the swap
        new_S = set(S)
        new_S.discard(old_val)
        new_S.add(new_val)
        new_score = _score(new_S)
        if new_score > current_score:
            S = new_S
            current_score = new_score
            lst = sorted(S)
            if current_score > best_score:
                best_S = set(S)
                best_score = current_score
        # Occasionally restart from best
        if it > 0 and it % 5000 == 0:
            if random.random() < 0.3:
                S = set(best_S)
                current_score = best_score
                lst = sorted(S)
    
    return best_S, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s budget, leave margin
    
    best_S = set(range(1, 33))
    best_score = _score(best_S)
    
    # Try various grid configurations
    # r * c = 32
    configs = []
    for r in range(2, 33):
        if 32 % r == 0:
            c = 32 // r
            configs.append((r, c))
    
    random.seed(42)
    
    for (r, c) in configs:
        if time.time() > deadline - 10:
            break
        # Try various row spacings and column spacings
        # Row spacing R should be large enough that rows don't overlap
        # Column spacing C should be at least 1
        for R in range(c + 1, min(512 // r + c, 200), 1):
            if time.time() > deadline - 10:
                break
            for C in range(1, min(512 // c + 1, 50), 1):
                if time.time() > deadline - 10:
                    break
                # Max value: (r-1)*R + (c-1)*C + offset
                # We need this <= 512
                # offset >= 1
                max_val_no_offset = (r - 1) * R + (c - 1) * C
                if max_val_no_offset + 1 > 512:
                    continue
                # Try a few offsets
                for offset in [1, 2, 3, 5, 10]:
                    if time.time() > deadline - 10:
                        break
                    S = _grid_construct(r, c, R, C, offset)
                    if len(S) != 32:
                        continue
                    sc = _score(S)
                    if sc > best_score:
                        best_S = set(S)
                        best_score = sc
        
        # Also try with random perturbations of grid
        if time.time() > deadline - 15:
            break
        for _ in range(20):
            if time.time() > deadline - 15:
                break
            R = random.randint(c + 1, min(512 // r + c, 200))
            C = random.randint(1, min(512 // c + 1, 50))
            offset = random.randint(1, 10)
            S = _grid_construct(r, c, R, C, offset)
            if len(S) != 32:
                continue
            sc = _score(S)
            if sc > best_score:
                best_S = set(S)
                best_score = sc
    
    # Local search to improve
    if time.time() < deadline - 5:
        # Run local search on best found
        ls_S, ls_score = _local_search(best_S, max_iters=50000)
        if ls_score > best_score:
            best_S = ls_S
            best_score = ls_score
    
    # Final local search with more iterations if time permits
    if time.time() < deadline - 3:
        remaining_iters = min(50000, int((deadline - 3 - time.time()) * 2000))
        if remaining_iters > 1000:
            ls_S2, ls_score2 = _local_search(best_S, max_iters=remaining_iters)
            if ls_score2 > best_score:
                best_S = ls_S2
                best_score = ls_score2
    
    return list(best_S)
