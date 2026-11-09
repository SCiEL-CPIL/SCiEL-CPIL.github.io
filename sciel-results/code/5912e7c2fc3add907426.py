"""
Strategy: Optimize the base sequence of a lacunary (superincreasing) set.
The set U is defined by u_0=0 and u_k = u_{k-1} * b_k for k>=1, where b_k >= 2.
This ensures U is superincreasing, which typically maximizes the ratio |U-U|/|U+U|
for a given size and max element, because sums and differences are unique or nearly unique.
The search operates on the sequence of multipliers b_k, not the raw integers.
"""
# EVOLVE-BLOCK-START
import time
import math
import random

def _score_lacunary(bases):
    """
    Compute score for a lacunary set defined by bases.
    bases = [b_1, b_2, ..., b_n] where u_0=0, u_1=1, u_k = u_{k-1}*b_k.
    Actually, let's define u_0=0. Then u_1, ..., u_n.
    Usually lacunary starts with 1. Let U = {0, u_1, ..., u_n}.
    u_1 = 1.
    u_k = u_{k-1} * b_k.
    """
    if not bases:
        return 0.0, [0]
    
    u = [0, 1]
    for b in bases:
        u.append(u[-1] * b)
    
    U = u
    n = len(U)
    if n < 2:
        return 0.0, U
        
    # Compute |U+U| and |U-U|
    # For superincreasing sequences, sums and differences are often unique.
    # But we must verify.
    # n is small enough (likely < 50) for O(n^2) or O(n^2 log n)
    
    plus = set()
    minus = set()
    
    for i in range(n):
        a = U[i]
        for j in range(i, n):
            plus.add(a + U[j])
        for j in range(n):
            minus.add(a - U[j])
            
    nplus = len(plus)
    nminus = len(minus)
    
    if nplus == 0:
        return 0.0, U
        
    mx = U[-1]
    if mx <= 0:
        return 0.0, U
        
    log_den = math.log(2 * mx + 1)
    if log_den <= 0:
        return 0.0, U
        
    ratio = nminus / nplus
    if ratio <= 0:
        return 0.0, U
        
    val = 1.0 + math.log(ratio) / log_den
    return val, U

def _score_general(U):
    """Fallback score for general sets if needed."""
    U = list(set(int(x) for x in U))
    if 0 not in U:
        U.append(0)
    if not U:
        return 0.0, []
    mx = max(U)
    if mx <= 0:
        return 0.0, sorted(U)
    
    n = len(U)
    plus = set()
    minus = set()
    for i in range(n):
        a = U[i]
        for j in range(i, n):
            plus.add(a + U[j])
        for j in range(n):
            minus.add(a - U[j])
            
    nplus = len(plus)
    nminus = len(minus)
    if nplus == 0:
        return 0.0, sorted(U)
        
    log_den = math.log(2 * mx + 1)
    if log_den <= 0:
        return 0.0, sorted(U)
        
    ratio = nminus / nplus
    if ratio <= 0:
        return 0.0, sorted(U)
        
    val = 1.0 + math.log(ratio) / log_den
    return val, sorted(U)

def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    rng = random.Random(42)
    
    best_val = -1e9
    best_bases = [2, 2, 2] # Default: 0, 1, 2, 4
    best_set = [0, 1, 2, 4]
    
    def consider(bases):
        nonlocal best_val, best_bases, best_set
        if time.time() > deadline:
            return
        try:
            v, s = _score_lacunary(bases)
        except Exception:
            return
        if v > best_val:
            best_val = v
            best_bases = list(bases)
            best_set = s

    # 1. Initial exploration of lacunary base sequences
    # We want to maximize log(ratio)/log(max).
    # For pure geometric (constant base b), ratio is constant (approx 2 for large n? No, for geometric, sums are not unique if base is small, unique if base > 2?
    # If base=2: 1, 2, 4, 8. Sums: 0,1,2,3,4,5,6,7,8,9,10,12,16. Diffs: -7..7.
    # Actually, for base=2, U={0,1,2,4,8...}.
    # Sums of two distinct elements from {1,2,4,8}: 1+2=3, 1+4=5, 1+8=9, 2+4=6, 2+8=10, 4+8=12. Plus doubles 2,4,8,16.
    # It turns out for base=2, the set of sums is quite dense.
    # For base=3: 1, 3, 9, 27. Sums: 1,2,3,4,6,9,10,12,18,27,28,30,54.
    # Generally, larger base -> larger max -> smaller denominator? No, larger max -> larger log(max) -> smaller score if ratio doesn't grow.
    # But ratio |U-U|/|U+U| is bounded by 2 (since |U-U| <= 2|U+U| roughly? No, |U-U| can be up to 2n^2, |U+U| up to n^2. Max ratio is 2?
    # Actually, for symmetric sets, |U-U| = 2|U+U| - |2U|?
    # For superincreasing sets, |U+U| is close to n(n+1)/2 and |U-U| is close to n^2.
    # Ratio approaches 2 as n grows?
    # Let's test: U={0, 1, M}. Sums: 0, 1, M, M+1. (4). Diffs: -M, -1, 0, 1, M. (5). Ratio 5/4 = 1.25.
    # U={0, 1, M, M^2}. Sums: ~10. Diffs: ~16. Ratio 1.6.
    # So we want large n. But max U grows exponentially.
    # Score = 1 + log(ratio)/log(2*max U + 1).
    # If ratio -> 2, log(ratio) -> log(2) = 0.693.
    # We want to MINIMIZE log(2*max U + 1) while keeping n large enough for ratio to be close to 2.
    # This suggests using small bases to keep max U small, but small bases might reduce the ratio (more collisions).
    # Base 2 is the smallest integer base > 1.
    # Let's check if base 2 is optimal for small n.
    
    # Search over base sequences
    # Start with constant bases
    for n in range(2, 30):
        for b in [2, 3, 4, 5]:
            consider([b] * n)
            
    # Random base sequences
    for _ in range(5000):
        if time.time() > deadline:
            break
        n = rng.randint(2, 20)
        bases = [rng.choice([2, 3, 4, 5, 6]) for _ in range(n)]
        consider(bases)
        
    # Local search on bases
    cur = list(best_bases)
    improved = True
    it = 0
    while improved and time.time() < deadline - 2.0 and it < 1000:
        improved = False
        it += 1
        
        # 1. Change a base
        if len(cur) > 0:
            idx = rng.randint(0, len(cur) - 1)
            old_b = cur[idx]
            # Try nearby bases
            new_b = old_b + rng.choice([-2, -1, 1, 2])
            if new_b < 2:
                new_b = 2
            if new_b > 20:
                new_b = 20
            if new_b != old_b:
                cur[idx] = new_b
                consider(cur)
                if best_bases == cur:
                    improved = True
                else:
                    cur[idx] = old_b
        else:
            # Start a new sequence
            n = rng.randint(2, 10)
            cur = [rng.choice([2, 3, 4]) for _ in range(n)]
            consider(cur)
            if best_bases == cur:
                improved = True
            else:
                cur = list(best_bases)
                
        # 2. Add a base
        if len(cur) < 30:
            cur_new = cur + [rng.choice([2, 3, 4, 5])]
            consider(cur_new)
            if best_bases == cur_new:
                cur = cur_new
                improved = True
            else:
                # revert
                pass # cur is still old best if not improved, but we modified cur? 
                # We should be careful. Let's use a temp list.
                # Actually, consider() updates best_bases if better.
                # If not better, cur is unchanged in this block? No, I did cur_new = cur + ...
                # So cur is safe.
                
        # 3. Remove a base
        if len(cur) > 2:
            idx = rng.randint(0, len(cur) - 1)
            cur_new = cur[:idx] + cur[idx+1:]
            consider(cur_new)
            if best_bases == cur_new:
                cur = cur_new
                improved = True
            else:
                pass
                
        # 4. Swap two bases
        if len(cur) > 1:
            i = rng.randint(0, len(cur) - 1)
            j = rng.randint(0, len(cur) - 1)
            if i != j:
                cur_new = cur[:]
                cur_new[i], cur_new[j] = cur_new[j], cur_new[i]
                consider(cur_new)
                if best_bases == cur_new:
                    cur = cur_new
                    improved = True
        else:
            cur = list(best_bases)

    # 5. Hybrid: Start with lacunary, then perturb the last few elements?
    # No, let's stick to the base representation for consistency.
    # But maybe we can allow the first few elements to be arbitrary?
    # The lacunary assumption is strong. Let's see if we can relax it.
    # What if we use a "mixed" base?
    # Or simply, run a local search on the SET itself but constrained to be "near lacunary"?
    # Given the time, let's do a final pass of general local search on the best set found.
    
    cur_set = list(best_set)
    improved = True
    it2 = 0
    while improved and time.time() < deadline - 1.0 and it2 < 500:
        improved = False
        it2 += 1
        mx = max(cur_set) if cur_set else 0
        n_cur = len(cur_set)
        
        # Generate candidates
        cands = set()
        for c in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]:
            cands.add(c)
        cands.add(mx + 1)
        cands.add(mx * 2)
        cands.add(mx * 3)
        cands.add(mx // 2)
        cands.add(mx // 3)
        
        for _ in range(20):
            cands.add(rng.randint(1, max(2, mx * 2)))
            
        cands = [c for c in cands if c >= 0 and c not in cur_set]
        rng.shuffle(cands)
        
        for c in cands:
            U = cur_set + [c]
            v, s = _score_general(U)
            if v > best_val + 1e-12:
                best_val = v
                best_set = s
                cur_set = s
                improved = True
                break
        
        if not improved:
            # Try removal
            if len(cur_set) > 2:
                for idx in range(1, len(cur_set)):
                    U = [x for x in cur_set if x != cur_set[idx]]
                    v, s = _score_general(U)
                    if v > best_val + 1e-12:
                        best_val = v
                        best_set = s
                        cur_set = s
                        improved = True
                        break

    return best_set
# EVOLVE-BLOCK-END
