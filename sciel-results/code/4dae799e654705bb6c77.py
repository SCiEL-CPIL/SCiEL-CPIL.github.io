import math
import time
import random
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 150.0
_SAFETY_MARGIN = 6.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _TIME_LIMIT - _SAFETY_MARGIN - _elapsed()

def _is_golomb(U):
    """Check if U is a Golomb ruler."""
    diffs = set()
    n = len(U)
    for i in range(n):
        for j in range(i+1, n):
            d = U[j] - U[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _score(U):
    """Compute the score."""
    U = sorted(set(int(x) for x in U if x >= 0))
    if len(U) < 2:
        return 0.0
    M = max(U)
    if M == 0:
        return 0.0
    
    # For Golomb rulers, we can use the formula directly if we know it's Golomb
    # But to be safe and general, we compute |U-U| and |U+U|
    # However, if we are searching Golomb rulers, we can just use the formula
    # |U-U| = n(n-1)+1
    # |U+U| = n(n+1)/2
    # But only if it IS a Golomb ruler.
    # If it's not, the score might be different.
    # Given the insight, we focus on Golomb rulers.
    
    n = len(U)
    if _is_golomb(U):
        diff_count = n * (n - 1) + 1
        sum_count = n * (n + 1) // 2
    else:
        # Fallback: exact computation
        sumset = set()
        diffset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
                diffset.add(a - b)
        sum_count = len(sumset)
        diff_count = len(diffset)
        
    if sum_count == 0:
        return 0.0
    ratio = diff_count / sum_count
    if ratio <= 0:
        return 0.0
    denom = math.log(2 * M + 1, 2)
    if denom == 0:
        return 0.0
    return 1.0 + math.log(ratio, 2) / denom

def _optimize_golomb(n, time_budget=5.0):
    """Find a short Golomb ruler with n marks."""
    if n <= 1:
        return [0]
    
    # Initial guess: greedy or known optimal for small n
    # Known optimal lengths for n=1..15
    # 1: 0
    # 2: 1
    # 3: 3
    # 4: 6
    # 5: 11
    # 6: 17
    # 7: 25
    # 8: 34
    # 9: 44
    # 10: 55
    # 11: 72
    # 12: 85
    # 13: 106
    # 14: 129
    # 15: 153
    
    # Start with a random valid Golomb ruler or greedy
    U = [0]
    for i in range(1, n):
        # Try to place next mark
        found = False
        # Search range
        upper = i * i * 4 + 20
        for pos in range(i * i, upper):
            cand = U + [pos]
            if _is_golomb(cand):
                U.append(pos)
                found = True
                break
        if not found:
            # Extend search
            for pos in range(upper, upper + 100):
                cand = U + [pos]
                if _is_golomb(cand):
                    U.append(pos)
                    found = True
                    break
            if not found:
                # Fallback: just add a large number
                U.append(U[-1] + 100)
    
    best_U = U
    best_len = max(U)
    start = time.time()
    rng = random.Random(n)
    
    while time.time() - start < time_budget and _remaining() > 10:
        # Local search: move a mark
        improved = False
        for i in range(1, n):
            orig = U[i]
            # Try moving to nearby positions
            for d in range(-20, 21):
                if d == 0: continue
                nv = orig + d
                if nv < 0: continue
                if nv == 0: continue
                # Check if nv is already in U
                if nv in U: continue
                new_U = U[:i] + [nv] + U[i+1:]
                new_U.sort()
                if _is_golomb(new_U):
                    if max(new_U) < best_len:
                        best_len = max(new_U)
                        U = new_U
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            # Random restart if stuck for a while?
            # Or just break
            if rng.random() < 0.1:
                # Perturb
                i = rng.randint(1, n-1)
                U[i] = U[i] + rng.randint(-10, 10)
                U.sort()
                # Ensure it's still Golomb? If not, revert
                if not _is_golomb(U):
                    U = U[:i] + [U[i] - rng.randint(-10, 10)] + U[i+1:]
                    U.sort()
                    if not _is_golomb(U):
                        # Revert to previous best
                        pass # Hard to revert easily without history
                        # Let's just break if we lose Golomb property
                        break

    return U

def construct():
    global _START_TIME
    _START_TIME = time.time()
    random.seed(42)
    
    best_U = [0, 1, 3]
    best_s = _score(best_U)
    
    # Try various n
    for n in range(3, 16):
        if _remaining() < 15:
            break
        # Allocate time
        alloc = min(5.0, _remaining() - 10)
        if alloc < 1.0:
            break
        U = _optimize_golomb(n, time_budget=alloc)
        s = _score(U)
        if s > best_s:
            best_s = s
            best_U = U
            
    return best_U
