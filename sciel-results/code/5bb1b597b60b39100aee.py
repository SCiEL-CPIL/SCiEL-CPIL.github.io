import math
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin before 90s kill
    
    # Precompute prime factors for numbers 2..256
    prime_factors = [set() for _ in range(257)]
    for i in range(2, 257):
        if not prime_factors[i]:  # i is prime
            for j in range(i, 257, i):
                prime_factors[j].add(i)
    
    def check_pairwise_coprime(sums_list):
        """Check if all elements in sums_list are pairwise coprime."""
        n = len(sums_list)
        for i in range(n):
            for j in range(i + 1, n):
                if prime_factors[sums_list[i]] & prime_factors[sums_list[j]]:
                    return False
        return True
    
    def score(A, B):
        """Count distinct sums that are pairwise coprime with each other."""
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        
        if not sums:
            return 0
        
        # Greedy: sort by number of prime factors (fewer factors = more likely to be coprime)
        sorted_sums = sorted(sums, key=lambda x: len(prime_factors[x]))
        
        selected = []
        for s in sorted_sums:
            compatible = True
            for t in selected:
                if prime_factors[s] & prime_factors[t]:
                    compatible = False
                    break
            if compatible:
                selected.append(s)
        
        return len(selected)
    
    # Initial best from seed
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score(best_A, best_B)
    
    # Generate candidate numbers with few prime factors
    candidates_by_factor_count = {}
    for n in range(1, 129):
        pf = prime_factors[n]
        cnt = len(pf)
        if cnt not in candidates_by_factor_count:
            candidates_by_factor_count[cnt] = []
        candidates_by_factor_count[cnt].append(n)
    
    # Prioritize numbers with 0, 1, or 2 prime factors
    good_candidates = set()
    for cnt in [0, 1, 2]:
        if cnt in candidates_by_factor_count:
            good_candidates.update(candidates_by_factor_count[cnt])
    
    candidate_list = sorted(good_candidates)
    
    # Add some with 3 factors for diversity
    for cnt in [3]:
        if cnt in candidates_by_factor_count:
            candidate_list.extend(candidates_by_factor_count[cnt][:30])
    
    candidate_list = sorted(set(candidate_list))
    
    # Strategy 1: Try small candidate set exhaustively
    small_candidates = candidate_list[:50]
    
    for A in combinations(small_candidates, 3):
        if time.time() > deadline:
            break
        A = list(A)
        for B in combinations(small_candidates, 3):
            if time.time() > deadline:
                break
            B = list(B)
            s = score(A, B)
            if s > best_score:
                best_score = s
                best_A = A[:]
                best_B = B[:]
                if best_score == 9:
                    return [best_A, best_B]
    
    # Strategy 2: Parity-restricted search (all sums must be same parity)
    # For all sums to be pairwise coprime, no two sums can share factor 2.
    # So all sums must be the same parity. Since a+b parity = parity(a) XOR parity(b),
    # all a's must have same parity and all b's must have opposite parity.
    
    def random_valid_set_parity(parity):
        """Generate 3 distinct values in [1,128] with given parity (0=even, 1=odd)"""
        if parity == 0:
            candidates = list(range(2, 129, 2))  # even: 2,4,...,128
        else:
            candidates = list(range(1, 129, 2))   # odd: 1,3,...,127
        return sorted(random.sample(candidates, 3))
    
    for a_parity in [1, 0]:  # 1=odd, 0=even
        b_parity = 1 - a_parity
        for iteration in range(3000):
            if time.time() > deadline:
                break
            
            cur_A = random_valid_set_parity(a_parity)
            cur_B = random_valid_set_parity(b_parity)
            cur_score = score(cur_A, cur_B)
            if cur_score > best_score:
                best_score = cur_score
                best_A = cur_A[:]
                best_B = cur_B[:]
            
            # Local improvement: try to fix problematic sums
            if cur_score > 0 and cur_score < 9:
                # Try to improve by changing one element at a time
                for _ in range(30):
                    if time.time() > deadline:
                        break
                    
                    # Pick a random element to change
                    if random.random() < 0.5:
                        idx = random.randint(0, 2)
                        old_val = cur_A[idx]
                        new_val = random.choice([v for v in range(1, 129) if (v % 2) == a_parity and v != old_val and v not in cur_A])
                        if not new_val:
                            continue
                        cur_A[idx] = new_val
                        cur_A.sort()
                        new_score = score(cur_A, cur_B)
                        if new_score > cur_score:
                            cur_score = new_score
                            if cur_score > best_score:
                                best_score = cur_score
                                best_A = cur_A[:]
                                best_B = cur_B[:]
                        else:
                            cur_A[idx] = old_val
                            cur_A.sort()
                    else:
                        idx = random.randint(0, 2)
                        old_val = cur_B[idx]
                        new_val = random.choice([v for v in range(1, 129) if (v % 2) == b_parity and v != old_val and v not in cur_B])
                        if not new_val:
                            continue
                        cur_B[idx] = new_val
                        cur_B.sort()
                        new_score = score(cur_A, cur_B)
                        if new_score > cur_score:
                            cur_score = new_score
                            if cur_score > best_score:
                                best_score = cur_score
                                best_A = cur_A[:]
                                best_B = cur_B[:]
                        else:
                            cur_B[idx] = old_val
                            cur_B.sort()
            
            if best_score == 9:
                return [best_A, best_B]
    
    # Strategy 3: Broader search if we haven't found 9
    if best_score < 9:
        def random_valid_set():
            return sorted(random.sample(range(1, 129), 3))
        
        # Local search from current best
        cur_A = best_A[:]
        cur_B = best_B[:]
        cur_score = best_score
        
        while time.time() < deadline:
            improved = False
            for _ in range(100):
                if time.time() > deadline:
                    break
                
                if random.random() < 0.5:
                    idx = random.randint(0, 2)
                    old_val = cur_A[idx]
                    new_val = random.randint(1, 128)
                    if new_val in cur_A:
                        continue
                    cur_A[idx] = new_val
                    cur_A.sort()
                    new_score = score(cur_A, cur_B)
                    if new_score > cur_score:
                        cur_score = new_score
                        improved = True
                        if cur_score > best_score:
                            best_score = cur_score
                            best_A = cur_A[:]
                            best_B = cur_B[:]
                    else:
                        cur_A[idx] = old_val
                        cur_A.sort()
                else:
                    idx = random.randint(0, 2)
                    old_val = cur_B[idx]
                    new_val = random.randint(1, 128)
                    if new_val in cur_B:
                        continue
                    cur_B[idx] = new_val
                    cur_B.sort()
                    new_score = score(cur_A, cur_B)
                    if new_score > cur_score:
                        cur_score = new_score
                        improved = True
                        if cur_score > best_score:
                            best_score = cur_score
                            best_A = cur_A[:]
                            best_B = cur_B[:]
                    else:
                        cur_B[idx] = old_val
                        cur_B.sort()
            
            if not improved:
                cur_A = random_valid_set()
                cur_B = random_valid_set()
                cur_score = score(cur_A, cur_B)
                if cur_score > best_score:
                    best_score = cur_score
                    best_A = cur_A[:]
                    best_B = cur_B[:]
    
    return [best_A, best_B]
