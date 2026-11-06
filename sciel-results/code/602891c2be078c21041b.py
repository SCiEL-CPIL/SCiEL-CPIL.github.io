"""Combined program: SPF-based fast coprime check + structured search for max |A+B|."""
import time
import math
import random
import itertools

_DEADLINE = 85.0  # safety margin before 90s kill


def _build_spf(max_n):
    """Build smallest prime factor table."""
    spf = list(range(max_n + 1))
    for i in range(2, int(max_n ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, max_n + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf


def _build_pf_cache(spf, max_n):
    """Build prime factor set cache for all numbers 1..max_n."""
    pf = [set() for _ in range(max_n + 1)]
    for n in range(2, max_n + 1):
        x = n
        while x > 1:
            p = spf[x]
            pf[n].add(p)
            while x % p == 0:
                x //= p
    return pf


def _pairwise_coprime_fast(vals, pf_cache):
    """Check pairwise coprimality using precomputed prime factor sets."""
    n = len(vals)
    for i in range(n):
        fi = pf_cache[vals[i]]
        for j in range(i + 1, n):
            if fi & pf_cache[vals[j]]:
                return False
    return True


def _sumset(A, B):
    return set(a + b for a in A for b in B)


def _valid_solution(A, B, pf_cache):
    s = _sumset(A, B)
    return len(s), _pairwise_coprime_fast(list(s), pf_cache)


def construct():
    start_time = time.time()
    
    # Build SPF and prime factor cache
    max_sum = 256
    spf = _build_spf(max_sum)
    pf_cache = _build_pf_cache(spf, max_sum)
    
    # Seed solution
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    s_seed = _sumset(best_A, best_B)
    if _pairwise_coprime_fast(list(s_seed), pf_cache):
        best_score = len(s_seed)
    
    # Phase 1: Structured search with parity separation
    # A odd, B even (or vice versa) to avoid trivial gcd=2
    odd_vals = list(range(1, 129, 2))
    even_vals = list(range(2, 129, 2))
    
    # Limit search ranges for tractability
    A_search = odd_vals[:40]   # 1 to 79
    B_search = even_vals[:40]  # 2 to 80
    
    found_9 = False
    
    for A in itertools.combinations(A_search, 3):
        if time.time() - start_time > _DEADLINE - 5:
            break
        a1, a2, a3 = A
        
        # Precompute valid B values for this A
        valid_b = []
        for b in B_search:
            s1 = a1 + b
            s2 = a2 + b
            s3 = a3 + b
            if s1 > max_sum or s2 > max_sum or s3 > max_sum:
                continue
            # Quick check: the three sums for this b must be pairwise coprime
            if pf_cache[s1] & pf_cache[s2] or pf_cache[s1] & pf_cache[s3] or pf_cache[s2] & pf_cache[s3]:
                continue
            valid_b.append((b, s1, s2, s3))
        
        if len(valid_b) < 3:
            continue
        
        # Limit candidates for performance
        if len(valid_b) > 25:
            valid_b = valid_b[:25]
        
        n = len(valid_b)
        for i in range(n):
            if time.time() - start_time > _DEADLINE - 3:
                break
            b1, s1_1, s2_1, s3_1 = valid_b[i]
            sums1 = [s1_1, s2_1, s3_1]
            pf1 = pf_cache[s1_1] | pf_cache[s2_1] | pf_cache[s3_1]
            
            for j in range(i + 1, n):
                b2, s1_2, s2_2, s3_2 = valid_b[j]
                # Check new sums against existing
                if (pf1 & pf_cache[s1_2]) or (pf1 & pf_cache[s2_2]) or (pf1 & pf_cache[s3_2]):
                    continue
                if (pf_cache[s1_2] & pf_cache[s2_2]) or (pf_cache[s1_2] & pf_cache[s3_2]) or (pf_cache[s2_2] & pf_cache[s3_2]):
                    continue
                
                pf12 = pf1 | pf_cache[s1_2] | pf_cache[s2_2] | pf_cache[s3_2]
                sums12 = sums1 + [s1_2, s2_2, s3_2]
                
                for k in range(j + 1, n):
                    b3, s1_3, s2_3, s3_3 = valid_b[k]
                    # Check new sums against all existing
                    if (pf12 & pf_cache[s1_3]) or (pf12 & pf_cache[s2_3]) or (pf12 & pf_cache[s3_3]):
                        continue
                    if (pf_cache[s1_3] & pf_cache[s2_3]) or (pf_cache[s1_3] & pf_cache[s3_3]) or (pf_cache[s2_3] & pf_cache[s3_3]):
                        continue
                    
                    all_sums = sums12 + [s1_3, s2_3, s3_3]
                    distinct = set(all_sums)
                    if len(distinct) <= best_score:
                        continue
                    
                    # Final check: all 9 (or fewer distinct) sums pairwise coprime
                    if _pairwise_coprime_fast(list(distinct), pf_cache):
                        if len(distinct) > best_score:
                            best_score = len(distinct)
                            best_A = list(A)
                            best_B = [b1, b2, b3]
                            if best_score == 9:
                                found_9 = True
                                break
                if found_9:
                    break
            if found_9:
                break
        if found_9:
            break
    
    if found_9:
        return best_A, best_B
    
    # Phase 2: Broader random search with both parities
    rng = random.Random(42)
    all_vals = list(range(1, 129))
    
    for _ in range(500000):
        if time.time() - start_time > _DEADLINE - 2:
            break
        A = sorted(rng.sample(all_vals, 3))
        B = sorted(rng.sample(all_vals, 3))
        s = _sumset(A, B)
        if len(s) > best_score and _pairwise_coprime_fast(list(s), pf_cache):
            best_score = len(s)
            best_A, best_B = A, B
            if best_score == 9:
                return best_A, best_B
    
    # Verify and return
    s = _sumset(best_A, best_B)
    if _pairwise_coprime_fast(list(s), pf_cache):
        return best_A, best_B
    
    return [1, 7, 13], [4, 10, 16]
