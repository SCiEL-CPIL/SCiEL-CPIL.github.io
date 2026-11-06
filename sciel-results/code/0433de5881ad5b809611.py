"""Program to find optimal A, B for pairwise-coprime sumset maximization."""
# EVOLVE-BLOCK-START
import math
import time
import random
from itertools import combinations, product

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _pairwise_coprime(vals):
    vs = list(set(vals))
    n = len(vs)
    for i in range(n):
        for j in range(i+1, n):
            if _gcd(vs[i], vs[j]) != 1:
                return False
    return True

def _eval(A, B):
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    if len(sums) <= 1:
        return len(sums)
    if _pairwise_coprime(list(sums)):
        return len(sums)
    return 0

def _is_valid(A, B):
    """Check if A and B are valid: 3 distinct integers in [1,128]."""
    if len(A) != 3 or len(B) != 3:
        return False
    if len(set(A)) != 3 or len(set(B)) != 3:
        return False
    if any(x < 1 or x > 128 for x in A) or any(x < 1 or x > 128 for x in B):
        return False
    return True

def _search():
    """Search for optimal A, B using difference-structure CSP approach."""
    deadline = time.time() + 75  # 75s budget, leaving 15s safety margin
    
    # Precompute sieve for primality up to 256
    is_prime = [True] * 257
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(256)) + 1):
        if is_prime[i]:
            for j in range(i*i, 257, i):
                is_prime[j] = False
    
    # Precompute GCD table for all pairs up to 256
    # gcd_table[i][j] = gcd(i, j) for i,j in [2, 256]
    max_s = 256
    gcd_table = [[0] * (max_s + 1) for _ in range(max_s + 1)]
    for i in range(2, max_s + 1):
        for j in range(i, max_s + 1):
            gcd_table[i][j] = gcd_table[j][i] = _gcd(i, j)
    
    def _check_pairwise_coprime_fast(sums_list):
        """Fast pairwise coprime check using precomputed gcd table."""
        for i in range(len(sums_list)):
            for j in range(i + 1, len(sums_list)):
                if gcd_table[sums_list[i]][sums_list[j]] != 1:
                    return False
        return True
    
    def _check_all_distinct_and_coprime(sums_list):
        """Check all sums are distinct and pairwise coprime."""
        if len(set(sums_list)) != len(sums_list):
            return False
        return _check_pairwise_coprime_fast(sums_list)
    
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = _eval(best_A, best_B)
    
    # Phase 1: Difference-structure CSP
    # For a 3x3 grid: A = [a1, a2, a3], B = [b1, b2, b3]
    # Let da = a2 - a1 > 0, db = a3 - a2 > 0 (so a3 - a1 = da + db)
    # Let dc = b2 - b1 > 0, dd = b3 - b2 > 0 (so b3 - b1 = dc + dd)
    # Sums: s[i][j] = a[i] + b[j]
    # s[0][0] = a1 + b1 = base
    # s[0][1] = base + dc
    # s[0][2] = base + dc + dd
    # s[1][0] = base + da
    # s[1][1] = base + da + dc
    # s[1][2] = base + da + dc + dd
    # s[2][0] = base + da + db
    # s[2][1] = base + da + db + dc
    # s[2][2] = base + da + db + dc + dd
    
    # The 9 offsets from base are:
    # 0, dc, dc+dd, da, da+dc, da+dc+dd, da+db, da+db+dc, da+db+dc+dd
    
    # We need:
    # 1. All 9 sums distinct (guaranteed if da,db,dc,dd > 0)
    # 2. All 9 sums pairwise coprime
    # 3. a1, a2, a3, b1, b2, b3 all in [1, 128]
    #    => a1 >= 1, a1+da+db <= 128
    #    => b1 >= 1, b1+dc+dd <= 128
    #    => base = a1+b1 >= 2, base + da+db+dc+dd <= 256
    
    # Enumerate difference patterns
    # da, db, dc, dd are positive integers
    # Constraints: da+db <= 127 (a range), dc+dd <= 127 (b range)
    # base >= 2, base <= 256 - (da+db+dc+dd)
    
    # For efficiency, limit the range of differences
    # Max useful difference: if da+db > 127, no valid a1 exists
    # Similarly for dc+dd
    
    found_optimal = False
    
    # Try smaller differences first (more likely to find valid configs)
    max_diff = 60  # limit each individual difference
    
    for da in range(1, min(max_diff, 127) + 1):
        if found_optimal or time.time() > deadline:
            break
        for db in range(1, min(max_diff, 127 - da + 1) + 1):
            if found_optimal or time.time() > deadline:
                break
            for dc in range(1, min(max_diff, 127) + 1):
                if found_optimal or time.time() > deadline:
                    break
                for dd in range(1, min(max_diff, 127 - dc + 1) + 1):
                    if found_optimal or time.time() > deadline:
                        break
                    
                    # Compute the 9 offsets
                    offsets = [
                        0, dc, dc + dd,
                        da, da + dc, da + dc + dd,
                        da + db, da + db + dc, da + db + dc + dd
                    ]
                    
                    max_offset = max(offsets)
                    # base must satisfy:
                    # base >= 2
                    # base + max_offset <= 256
                    # a1 = base - b1 >= 1, a1 + da + db <= 128
                    # b1 >= 1, b1 + dc + dd <= 128
                    # So: base - b1 >= 1 => b1 <= base - 1
                    #     base - b1 + da + db <= 128 => b1 >= base + da + db - 128
                    #     b1 >= 1
                    #     b1 <= 128 - dc - dd
                    #     b1 <= base - 1
                    #     b1 >= base + da + db - 128
                    #     b1 >= 1
                    # Also base >= 2, base + max_offset <= 256
                    
                    min_base = 2
                    max_base = 256 - max_offset
                    
                    if min_base > max_base:
                        continue
                    
                    # For each valid base, check if there exists b1 such that:
                    # max(1, base + da + db - 128) <= b1 <= min(base - 1, 128 - dc - dd)
                    # and all 9 sums are pairwise coprime
                    
                    # Precompute: for a given set of offsets, check coprimality
                    # The 9 sums are: base + offset[0], base + offset[1], ..., base + offset[8]
                    
                    # To speed up, we can check coprimality of the 9 values
                    # for each candidate base
                    
                    b1_min_global = 1
                    b1_max_global = 128 - dc - dd
                    
                    if b1_min_global > b1_max_global:
                        continue
                    
                    # Iterate over possible base values
                    # For each base, compute valid b1 range and check coprimality
                    for base in range(min_base, max_base + 1):
                        if found_optimal or time.time() > deadline:
                            break
                        
                        b1_lo = max(b1_min_global, base + da + db - 128)
                        b1_hi = min(b1_max_global, base - 1)
                        
                        if b1_lo > b1_hi:
                            continue
                        
                        # Check coprimality of the 9 sums
                        sums_9 = [base + off for off in offsets]
                        
                        if not _check_all_distinct_and_coprime(sums_9):
                            continue
                        
                        # Found a valid configuration!
                        # Pick b1 = b1_lo (any valid b1 works)
                        b1 = b1_lo
                        a1 = base - b1
                        a2 = a1 + da
                        a3 = a1 + da + db
                        b2 = b1 + dc
                        b3 = b1 + dc + dd
                        
                        A = [a1, a2, a3]
                        B = [b1, b2, b3]
                        
                        # Verify
                        if not _is_valid(A, B):
                            continue
                        
                        score = _eval(A, B)
                        if score > best_score:
                            best_score = score
                            best_A = A[:]
                            best_B = B[:]
                        
                        if best_score >= 9:
                            found_optimal = True
                            break
    
    # Phase 2: If we didn't find 9, try broader search with random + local improvement
    if not found_optimal:
        rng = random.Random(42)
        
        # Random search
        for _ in range(50000):
            if time.time() > deadline:
                break
            A = list(rng.sample(range(1, 129), 3))
            B = list(rng.sample(range(1, 129), 3))
            score = _eval(A, B)
            if score > best_score:
                best_score = score
                best_A = A[:]
                best_B = B[:]
                if best_score >= 9:
                    return best_A, best_B
        
        # Local search
        for _ in range(100000):
            if time.time() > deadline:
                break
            if rng.random() < 0.5:
                idx = rng.randint(0, 2)
                new_val = rng.randint(1, 128)
                if new_val not in best_A:
                    old = best_A[idx]
                    best_A[idx] = new_val
                    score = _eval(best_A, best_B)
                    if score >= best_score:
                        best_score = score
                        if best_score >= 9:
                            return best_A, best_B
                    else:
                        best_A[idx] = old
            else:
                idx = rng.randint(0, 2)
                new_val = rng.randint(1, 128)
                if new_val not in best_B:
                    old = best_B[idx]
                    best_B[idx] = new_val
                    score = _eval(best_A, best_B)
                    if score >= best_score:
                        best_score = score
                        if best_score >= 9:
                            return best_A, best_B
                    else:
                        best_B[idx] = old
    
    return best_A, best_B

# Pre-compute the best answer
_best_result = _search()

def construct():
    return _best_result
# EVOLVE-BLOCK-END
