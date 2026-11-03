import time
import itertools
from math import log

def compute_score(U):
    U = sorted(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    ratio = len(diffset) / len(sumset)
    if ratio <= 1:
        return 1.0
    return 1.0 + log(ratio) / log(2 * max_u + 1)

def construct():
    start = time.time()
    deadline = start + 110  # 110s deadline, 40s safety margin
    
    best_U = [0, 1, 3]
    best_score = compute_score(best_U)
    
    # Try structured families
    # Family 1: {0, 1, 3, 7, ..., 2^k - 1}
    for k in range(2, 20):
        if time.time() > deadline:
            break
        U = [2**i - 1 for i in range(k)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Family 2: {0, 1, 3, 9, 27, ...} powers of 3
    for k in range(2, 15):
        if time.time() > deadline:
            break
        U = [3**i for i in range(k)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Family 3: Geometric progression with tunable base b
    # U = {floor(b^k) for k=0..m}, b > 1
    # Use ternary search to maximize score over b for each m
    
    def try_geometric(b, m):
        U = []
        val = 1.0
        for k in range(m + 1):
            U.append(int(val))
            val *= b
        U = sorted(set(U))
        if 0 not in U:
            U = [0] + U
        return U
    
    def score_geometric(b, m):
        U = try_geometric(b, m)
        return compute_score(U)
    
    def ternary_search_b(m, b_lo, b_hi, iters=50):
        """Ternary search over b in [b_lo, b_hi] for fixed m"""
        lo = b_lo
        hi = b_hi
        for _ in range(iters):
            if time.time() > deadline:
                break
            m1 = lo + (hi - lo) / 3.0
            m2 = hi - (hi - lo) / 3.0
            f1 = score_geometric(m1, m)
            f2 = score_geometric(m2, m)
            if f1 < f2:
                lo = m1
            else:
                hi = m2
        b_opt = (lo + hi) / 2.0
        return b_opt, score_geometric(b_opt, m)
    
    # Sweep over m and use ternary search for optimal b
    for m in range(3, 35):
        if time.time() > deadline:
            break
        # Determine reasonable b range based on m
        # For small m, b can be larger; for large m, b should be closer to 1
        b_lo = 1.01
        b_hi = 50.0 if m < 10 else 10.0 if m < 20 else 5.0
        
        b_opt, s_opt = ternary_search_b(m, b_lo, b_hi, iters=40)
        if s_opt > best_score:
            best_score = s_opt
            best_U = try_geometric(b_opt, m)
    
    # Fine-tune: try more granular m values and tighter b ranges
    for m in range(3, 25):
        if time.time() > deadline:
            break
        b_lo = 1.05
        b_hi = 20.0 if m < 10 else 8.0
        b_opt, s_opt = ternary_search_b(m, b_lo, b_hi, iters=60)
        if s_opt > best_score:
            best_score = s_opt
            best_U = try_geometric(b_opt, m)
    
    # Family 4: Local polish - try small perturbations around best geometric
    # Try adding/removing elements near the geometric progression
    if time.time() < deadline:
        U = best_U[:]
        current_score = best_score
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding elements in a range around max(U)
            max_u = max(U) if U else 0
            candidates = list(range(1, min(2*max_u + 20, max_u + 100)))
            # Also try some multiplicative candidates
            for mult in [1.5, 2, 2.5, 3, 4, 5]:
                c = int(max_u * mult)
                if c > 0 and c not in U:
                    candidates.append(c)
            candidates = sorted(set(candidates))
            for c in candidates:
                if c in U:
                    continue
                new_U = sorted(U + [c])
                new_score = compute_score(new_U)
                if new_score > current_score:
                    current_score = new_score
                    U = new_U
                    improved = True
                    break
        
        if current_score > best_score:
            best_score = current_score
            best_U = U
    
    return best_U
