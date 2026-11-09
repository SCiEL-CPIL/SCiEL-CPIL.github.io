"""Pairwise-coprime sumset maximizer: prime-rectangle search + DFS over prime structures."""
import time
from math import gcd
from itertools import combinations

# ---- precompute primes up to 256 ----
def _sieve(n):
    is_p = [True] * (n + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_p[i]:
            for j in range(i * i, n + 1, i):
                is_p[j] = False
    return [i for i in range(2, n + 1) if is_p[i]]

PRIMES = _sieve(256)
PRIME_SET = set(PRIMES)

def _pfactors(n):
    fs = set()
    for p in PRIMES:
        if p * p > n:
            break
        if n % p == 0:
            fs.add(p)
            while n % p == 0:
                n //= p
    if n > 1:
        fs.add(n)
    return fs

# prime-factor set for each possible sum 2..256
PF = {}
for s in range(2, 257):
    PF[s] = _pfactors(s)

def _pairwise_coprime(vals):
    seen = set()
    for v in vals:
        for p in PF[v]:
            if p in seen:
                return False
            seen.add(p)
    return True

def _valid_set(lst):
    if len(set(lst)) != 3:
        return False
    for x in lst:
        if x < 1 or x > 128:
            return False
    return True

def _sums(A, B):
    return [a + b for a in A for b in B]

def _distinct_count(A, B):
    return len(set(_sums(A, B)))

def _is_valid(A, B):
    if not _valid_set(A) or not _valid_set(B):
        return False
    return _pairwise_coprime(set(_sums(A, B)))

def _is_prime(n):
    return n in PRIME_SET

def _all_sums_prime(A, B):
    """Check if all 9 sums are distinct primes."""
    sums = [a + b for a in A for b in B]
    if len(set(sums)) != 9:
        return False
    return all(_is_prime(s) for s in sums)

def construct():
    t0 = time.time()
    deadline = t0 + 55.0   # safety margin before 60s

    best = ([1, 7, 13], [4, 10, 16])
    best_score = 0
    if _is_valid(best[0], best[1]):
        best_score = _distinct_count(best[0], best[1])

    # Phase 0: Direct search for prime rectangles (all 9 sums are distinct primes)
    # This is the "magic rectangle" approach: search over additive structures
    # where a_i + b_j = p_ij for distinct primes p_ij.
    #
    # Key structure: if A = [a0, a1, a2] and B = [b0, b1, b2],
    # the 9 sums form a grid. We can parameterize by differences.
    # Let A = [x, x+d1, x+d2] and B = [y, y+e1, y+e2].
    # Sums: x+y, x+y+e1, x+y+e2, x+d1+y, x+d1+y+e1, ...
    # For all to be prime, we need x+y, x+y+e1, x+y+e2, x+d1+y, etc. all prime.
    #
    # Simpler: enumerate small A, B and check directly.
    
    small_range = list(range(1, 40))
    found_prime_rect = False
    
    # Try structured A sets: [1, p, q] where p, q are small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # For each candidate A, find B such that all 9 sums are distinct primes
    for i in range(len(small_primes)):
        if time.time() > deadline:
            break
        p = small_primes[i]
        for j in range(i + 1, len(small_primes)):
            if time.time() > deadline:
                break
            q = small_primes[j]
            if p + q > 128:
                break
            A = [1, p, q]
            
            # Find B such that all 9 sums are distinct primes
            valid_b = []
            for b in small_range:
                sums = [1 + b, p + b, q + b]
                if all(_is_prime(s) for s in sums):
                    valid_b.append(b)
            
            if len(valid_b) < 3:
                continue
            
            # Check all pairs/triples of valid_b
            for b1 in valid_b:
                if time.time() > deadline:
                    break
                sums1 = [1 + b1, p + b1, q + b1]
                for b2 in valid_b:
                    if b2 <= b1:
                        continue
                    if time.time() > deadline:
                        break
                    sums2 = [1 + b2, p + b2, q + b2]
                    # Check combined sums are all distinct primes
                    combined = sums1 + sums2
                    if len(set(combined)) != 6:
                        continue
                    # Check pairwise coprime (all prime, so just need distinct)
                    for b3 in valid_b:
                        if b3 <= b2:
                            continue
                        if time.time() > deadline:
                            break
                        sums3 = [1 + b3, p + b3, q + b3]
                        all_sums = sums1 + sums2 + sums3
                        if len(set(all_sums)) == 9 and all(_is_prime(s) for s in all_sums):
                            B = [b1, b2, b3]
                            score = _distinct_count(A, B)
                            if score > best_score:
                                best_score = score
                                best = (A[:], B[:])
                                found_prime_rect = True
                                if best_score == 9:
                                    break
                        if best_score == 9:
                            break
                    if best_score == 9:
                        break
                if best_score == 9:
                    break
            if best_score == 9:
                break
        if best_score == 9:
            break
    
    # Phase 1: DFS over prime-structured A sets
    if best_score < 9 and time.time() < deadline:
        all_vals = list(range(1, 129))
        
        # Generate candidate A sets with prime structure
        candidate_A_sets = []
        
        # Type 1: A = [1, p, q] where p, q are small primes
        for i in range(len(small_primes)):
            for j in range(i + 1, len(small_primes)):
                p, q = small_primes[i], small_primes[j]
                if p + q <= 128:
                    candidate_A_sets.append([1, p, q])
        
        # Type 2: A = [p, q, r] all small primes
        for combo in combinations(small_primes[:15], 3):
            if max(combo) <= 128:
                candidate_A_sets.append(list(combo))
        
        # Type 3: A includes 1 and small coprime composites
        for x in range(2, 25):
            for y in range(x + 1, 25):
                if gcd(x, y) == 1:
                    candidate_A_sets.append([1, x, y])
        
        # Deduplicate
        seen_A = set()
        unique_A_sets = []
        for A in candidate_A_sets:
            key = tuple(sorted(A))
            if key not in seen_A:
                seen_A.add(key)
                unique_A_sets.append(A)
        
        # For each candidate A, find best B using DFS
        for A in unique_A_sets:
            if time.time() > deadline:
                break
            if not _valid_set(A):
                continue
            
            # Find all valid b values
            cand_b = []
            for b in all_vals:
                sums = [a + b for a in A]
                if _pairwise_coprime(sums):
                    cand_b.append(b)
            
            if len(cand_b) < 3:
                continue
            
            # Precompute sums for each b
            b_sums = {b: [a + b for a in A] for b in cand_b}
            
            # DFS to find best B of size 3
            best_for_A = 0
            best_B_for_A = None
            
            def dfs(chosen, sum_set, used_primes, depth):
                nonlocal best_for_A, best_B_for_A
                if depth == 3:
                    if len(sum_set) > best_for_A:
                        best_for_A = len(sum_set)
                        best_B_for_A = list(chosen)
                    return
                
                # Upper bound pruning
                if len(sum_set) + (3 - depth) * 3 <= best_for_A:
                    return
                
                start = chosen[-1] + 1 if chosen else 0
                for i in range(start, len(cand_b)):
                    b = cand_b[i]
                    new_sums = b_sums[b]
                    new_sum_set = set(sum_set)
                    new_sum_set.update(new_sums)
                    
                    # Check pairwise coprime
                    ok = True
                    new_used = set(used_primes)
                    for s in new_sums:
                        for p in PF[s]:
                            if p in new_used:
                                ok = False
                                break
                            new_used.add(p)
                        if not ok:
                            break
                    if not ok:
                        continue
                    
                    chosen.append(b)
                    dfs(chosen, new_sum_set, new_used, depth + 1)
                    chosen.pop()
                    
                    if best_for_A == 9:
                        return
                    if time.time() > deadline:
                        return
            
            dfs([], set(), set(), 0)
            
            if best_for_A > best_score:
                best_score = best_for_A
                best = (A[:], best_B_for_A if best_B_for_A else best[1])
                if best_score == 9:
                    break
    
    # Phase 2: Broader random search if not found 9
    if best_score < 9 and time.time() < deadline - 5:
        import random
        random.seed(42)
        
        for _ in range(30000):
            if time.time() > deadline:
                break
            
            strategy = random.randint(0, 3)
            if strategy == 0:
                A = random.sample(range(1, 35), 3)
                B = random.sample(range(1, 35), 3)
            elif strategy == 1:
                A = random.sample(range(1, 25), 3)
                B = random.sample(range(1, 50), 3)
            elif strategy == 2:
                A = [1] + random.sample(range(2, 45), 2)
                random.shuffle(A)
                B = random.sample(range(1, 45), 3)
            else:
                A = random.sample(range(1, 129), 3)
                B = random.sample(range(1, 129), 3)
            
            if _is_valid(A, B):
                s = _distinct_count(A, B)
                if s > best_score:
                    best_score = s
                    best = (A[:], B[:])
                    if best_score == 9:
                        break
    
    return [list(best[0]), list(best[1])]
