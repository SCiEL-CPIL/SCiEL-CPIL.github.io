"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    def is_prime_power_or_one(n):
        if n == 1:
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if n % p == 0:
                m = n
                while m % p == 0:
                    m //= p
                return m == 1
        return True
    
    # Precompute atomic sums (1 and prime powers) in [2, 256]
    atomic_sums = set()
    for n in range(2, 257):
        if is_prime_power_or_one(n):
            atomic_sums.add(n)
    
    # Precompute gcd compatibility for all possible sums
    # compat[s1][s2] = 1 if gcd(s1,s2)==1 else 0
    max_sum = 256
    compat = {}
    for s1 in range(2, max_sum + 1):
        compat[s1] = {}
        for s2 in range(2, max_sum + 1):
            if s1 != s2:
                compat[s1][s2] = 1 if math.gcd(s1, s2) == 1 else 0
            else:
                compat[s1][s2] = 1
    
    def evaluate_fast(A, B):
        """Fast evaluation using precomputed compatibility"""
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        slist = list(sums)
        for i in range(len(slist)):
            for j in range(i + 1, len(slist)):
                if compat[slist[i]][slist[j]] != 1:
                    return -1
        return len(slist)
    
    # Score each value in [1,128] by how many prime-power sums it can form
    all_vals = list(range(1, 129))
    val_score = {}
    for v in all_vals:
        count = 0
        for w in all_vals:
            if (v + w) in atomic_sums:
                count += 1
        val_score[v] = count
    
    sorted_vals = sorted(all_vals, key=lambda x: val_score[x], reverse=True)
    
    # Constructive approach: try to build A and B that produce many distinct coprime sums
    # Key insight: if we choose A = {a1, a2, a3} and B = {b1, b2, b3},
    # the 9 sums are a_i + b_j. We want these to be distinct and pairwise coprime.
    # 
    # Strategy: pick a "base" structure. For example, if we use small offsets,
    # the sums will be close together, making it hard for them to be coprime.
    # Better: use values that spread out the sums.
    
    # Precompute for each pair (a, b) the set of sums they create with a full A,B
    # Actually, let's do a smarter search:
    # 1. Pick A first (3 values)
    # 2. For each candidate B value, check if adding it to B creates valid coprime sums
    
    best_score = 0
    best_A = None
    best_B = None
    
    # Try some hand-crafted good starting points
    # These are based on the idea that prime sums are more likely coprime
    # A = {1, 2, 4}, B = {1, 3, 5} gives sums: 2,3,4,4,5,6,5,7,8 -> distinct: {2,3,4,5,6,7,8}
    # But gcd(2,4)=2, gcd(2,6)=2, etc. Not pairwise coprime.
    
    # Let's try: A = {1, 3, 5}, B = {1, 2, 4}
    # Sums: 2,3,5, 4,5,7, 6,7,9 -> distinct: {2,3,4,5,6,7,9}
    # gcd(2,4)=2 -> fail
    
    # Try: A = {1, 2, 3}, B = {1, 4, 7}
    # Sums: 2,5,8, 3,6,9, 4,7,10 -> distinct: {2,3,4,5,6,7,8,9,10}
    # gcd(2,4)=2 -> fail
    
    # The key is to avoid even numbers in the sumset, or have at most one even.
    # Since we need 9 sums and they must be pairwise coprime, we can have at most 1 even number.
    # So at least 8 sums must be odd.
    
    # For a+b to be odd, one of a,b must be odd and the other even.
    # If A has k odd and (3-k) even, and B has m odd and (3-m) even,
    # then odd sums = k*(3-m) + (3-k)*m
    
    # To maximize odd sums, we want k=3, m=0 or k=0, m=3 (all of one parity)
    # Then odd sums = 3*3 = 9 if all A same parity and all B opposite parity.
    
    # So: A all odd, B all even (or vice versa) gives all 9 sums odd.
    # Then we need 9 distinct odd numbers in [2,256] that are pairwise coprime.
    # Odd numbers that are pairwise coprime: primes, prime powers, 1
    # But sums are at least 1+2=3, so no 1.
    
    # Let's search for A (3 distinct odd in [1,128]) and B (3 distinct even in [2,128])
    # such that all 9 sums are distinct and pairwise coprime.
    
    odd_vals = [v for v in range(1, 129, 2)]  # 1,3,5,...,127
    even_vals = [v for v in range(2, 129, 2)]  # 2,4,6,...,128
    
    # Score odd values by how many prime-power sums they form with even values
    odd_score = {}
    for v in odd_vals:
        count = 0
        for w in even_vals:
            if (v + w) in atomic_sums:
                count += 1
        odd_score[v] = count
    
    even_score = {}
    for v in even_vals:
        count = 0
        for w in odd_vals:
            if (v + w) in atomic_sums:
                count += 1
        even_score[v] = count
    
    sorted_odds = sorted(odd_vals, key=lambda x: odd_score[x], reverse=True)
    sorted_evens = sorted(even_vals, key=lambda x: even_score[x], reverse=True)
    
    # Phase 1: Systematic search over top candidates
    random.seed(42)
    
    # Try all combinations of top 10 odds and top 10 evens (10 choose 3)^2 = 120^2 = 14400
    # That's fast enough
    top_odds = sorted_odds[:12]
    top_evens = sorted_evens[:12]
    
    for i in range(len(top_odds)):
        if time.time() > deadline - 10:
            break
        for j in range(i + 1, len(top_odds)):
            for k in range(j + 1, len(top_odds)):
                A = [top_odds[i], top_odds[j], top_odds[k]]
                for x in range(len(top_evens)):
                    if time.time() > deadline - 10:
                        break
                    for y in range(x + 1, len(top_evens)):
                        for z in range(y + 1, len(top_evens)):
                            B = [top_evens[x], top_evens[y], top_evens[z]]
                            score = evaluate_fast(A, B)
                            if score > best_score:
                                best_score = score
                                best_A = A[:]
                                best_B = B[:]
                                if best_score >= 9:
                                    return [best_A, best_B]
    
    # Phase 2: Random search with local improvement
    for _ in range(2000):
        if time.time() > deadline - 10:
            break
        
        # Random A from odds, B from evens
        A = random.sample(odd_vals, 3)
        B = random.sample(even_vals, 3)
        score = evaluate_fast(A, B)
        
        if score > best_score:
            best_score = score
            best_A = A[:]
            best_B = B[:]
            if best_score >= 9:
                return [best_A, best_B]
        
        # Local search
        if score > 0:
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                # Try modifying A
                for idx in range(3):
                    for new_val in sorted_odds[:30]:
                        if new_val in A:
                            continue
                        old_val = A[idx]
                        A[idx] = new_val
                        new_score = evaluate_fast(A, B)
                        if new_score > score:
                            score = new_score
                            improved = True
                            if score > best_score:
                                best_score = score
                                best_A = A[:]
                                best_B = B[:]
                                if best_score >= 9:
                                    return [best_A, best_B]
                            break
                        else:
                            A[idx] = old_val
                # Try modifying B
                for idx in range(3):
                    for new_val in sorted_evens[:30]:
                        if new_val in B:
                            continue
                        old_val = B[idx]
                        B[idx] = new_val
                        new_score = evaluate_fast(A, B)
                        if new_score > score:
                            score = new_score
                            improved = True
                            if score > best_score:
                                best_score = score
                                best_A = A[:]
                                best_B = B[:]
                                if best_score >= 9:
                                    return [best_A, best_B]
                            break
                        else:
                            B[idx] = old_val
    
    # Phase 3: Try mixed parity (not all same parity) in case we can do better
    for _ in range(1000):
        if time.time() > deadline - 5:
            break
        A = random.sample(all_vals, 3)
        B = random.sample(all_vals, 3)
        score = evaluate_fast(A, B)
        if score > best_score:
            best_score = score
            best_A = A[:]
            best_B = B[:]
            if best_score >= 9:
                return [best_A, best_B]
        
        if score > 0:
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                idx = random.randint(0, 2)
                which = random.choice([0, 1])
                if which == 0:
                    old_val = A[idx]
                    new_val = random.choice(sorted_vals[:50])
                    if new_val in A:
                        continue
                    A[idx] = new_val
                    new_score = evaluate_fast(A, B)
                    if new_score > score:
                        score = new_score
                        if score > best_score:
                            best_score = score
                            best_A = A[:]
                            best_B = B[:]
                            if best_score >= 9:
                                return [best_A, best_B]
                    else:
                        A[idx] = old_val
                else:
                    old_val = B[idx]
                    new_val = random.choice(sorted_vals[:50])
                    if new_val in B:
                        continue
                    B[idx] = new_val
                    new_score = evaluate_fast(A, B)
                    if new_score > score:
                        score = new_score
                        if score > best_score:
                            best_score = score
                            best_A = A[:]
                            best_B = B[:]
                            if best_score >= 9:
                                return [best_A, best_B]
                    else:
                        B[idx] = old_val
    
    if best_A is not None and best_B is not None:
        return [best_A, best_B]
    else:
        # Fallback: try a known good configuration
        # A = {1, 3, 7}, B = {2, 4, 8}
        # Sums: 3,5,9, 5,7,11, 9,11,15 -> distinct: {3,5,7,9,11,15}
        # gcd(3,9)=3 -> fail
        # A = {1, 3, 7}, B = {2, 4, 10}
        # Sums: 3,5,11, 5,7,13, 9,11,17 -> distinct: {3,5,7,9,11,13,17}
        # gcd(3,9)=3 -> fail
        # A = {1, 5, 7}, B = {2, 4, 10}
        # Sums: 3,5,11, 7,9,15, 9,11,17 -> distinct: {3,5,7,9,11,15,17}
        # gcd(3,9)=3 -> fail
        # A = {1, 5, 7}, B = {2, 6, 10}
        # Sums: 3,7,11, 7,11,15, 9,13,17 -> distinct: {3,7,9,11,13,15,17}
        # gcd(3,9)=3 -> fail
        # A = {1, 5, 11}, B = {2, 6, 10}
        # Sums: 3,7,11, 7,11,15, 13,17,21 -> distinct: {3,7,11,13,15,17,21}
        # gcd(3,21)=3 -> fail
        # A = {1, 5, 11}, B = {2, 6, 14}
        # Sums: 3,7,15, 7,11,19, 13,17,25 -> distinct: {3,7,11,13,15,17,19,25}
        # gcd(3,15)=3 -> fail
        # A = {1, 5, 11}, B = {2, 8, 14}
        # Sums: 3,9,15, 7,13,19, 13,19,25 -> distinct: {3,7,9,13,15,19,25}
        # gcd(3,9)=3 -> fail
        # A = {1, 7, 11}, B = {2, 6, 10}
        # Sums: 3,7,11, 9,13,17, 13,17,21 -> distinct: {3,7,9,11,13,17,21}
        # gcd(3,9)=3 -> fail
        # A = {1, 7, 11}, B = {2, 8, 14}
        # Sums: 3,9,15, 9,15,21, 13,19,25 -> distinct: {3,9,13,15,19,21,25}
        # gcd(3,9)=3 -> fail
        # A = {1, 7, 13}, B = {2, 6, 10}
        # Sums: 3,7,11, 9,13,17, 15,19,23 -> distinct: {3,7,9,11,13,15,17,19,23}
        # gcd(3,9)=3 -> fail
        # A = {1, 7, 13}, B = {2, 8, 14}
        # Sums: 3,9,15, 9,15,21, 15,21,27 -> distinct: {3,9,15,21,27}
        # All divisible by 3 -> fail
        # A = {1, 7, 13}, B = {4, 6, 10}
        # Sums: 5,7,11, 11,13,17, 17,19,23 -> distinct: {5,7,11,13,17,19,23}
        # Check: gcd(5,7)=1, gcd(5,11)=1, gcd(5,13)=1, gcd(5,17)=1, gcd(5,19)=1, gcd(5,23)=1
        # gcd(7,11)=1, gcd(7,13)=1, gcd(7,17)=1, gcd(7,19)=1, gcd(7,23)=1
        # gcd(11,13)=1, gcd(11,17)=1, gcd(11,19)=1, gcd(11,23)=1
        # gcd(13,17)=1, gcd(13,19)=1, gcd(13,23)=1
        # gcd(17,19)=1, gcd(17,23)=1
        # gcd(19,23)=1
        # All pairwise coprime! 7 distinct sums.
        return [[1, 7, 13], [4, 6, 10]]
# EVOLVE-BLOCK-END
