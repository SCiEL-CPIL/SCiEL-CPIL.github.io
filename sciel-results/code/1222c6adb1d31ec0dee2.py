import math
from itertools import combinations

def construct():
    best_score = 0
    best_A = []
    best_B = []
    
    # Candidate B sets based on directive: small consecutive primes, and other small sets
    # We'll include a mix of small numbers to increase the chance of finding valid A.
    # If B is small, sums are small, easier to be coprime?
    # Actually, large sums are more likely to be prime (and thus coprime to other primes).
    # But small B allows A to be larger? No, A is bounded by 128.
    # Let's try B sets with small elements and B sets with larger elements.
    
    # Generate candidate B sets
    b_candidates = []
    
    # 1. Small consecutive integers
    for start in range(1, 10):
        b_candidates.append([start, start+1, start+2])
        
    # 2. Small primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    for i in range(len(primes)-2):
        b_candidates.append([primes[i], primes[i+1], primes[i+2]])
        
    # 3. Random B sets
    import random
    random.seed(42)
    for _ in range(50):
        b_candidates.append(random.sample(range(1, 129), 3))
        
    # Remove duplicates and sort
    b_candidates = [sorted(t) for t in set(tuple(b) for b in b_candidates)]
    
    # Iterate over B candidates
    for B in b_candidates:
        # Iterate over all A subsets of size 3 from [1, 128]
        # This is C(128, 3) ~ 340k iterations per B.
        # With ~100 B candidates, that's 34 million iterations.
        # Might be too slow in Python if not optimized.
        # Let's limit B candidates or optimize A search.
        
        # Optimization: We can break early if we find score 9.
        # Also, we can be smarter about A.
        # But let's try a subset of B candidates first.
        
        if len(b_candidates) > 20:
             # Only use first 20 diverse B's to keep time low?
             # Or just iterate all if fast enough.
             pass

    # Let's refine the loop to be efficient.
    # We will iterate B candidates, and for each, iterate A.
    # To speed up A iteration, we can pre-compute properties?
    # Or just rely on the fact that we might find a solution quickly.
    
    # Let's restrict B candidates to a smaller, high-quality list.
    # The directive suggests "consecutive primes".
    # Let's prioritize those.
    
    high_priority_B = [
        [2, 3, 5],
        [3, 5, 7],
        [5, 7, 11],
        [1, 2, 3],
        [1, 2, 4],
        [1, 3, 4],
        [2, 3, 4],
        [1, 1, 1] # Invalid, distinct required.
    ]
    
    # Add some random ones
    for _ in range(20):
        high_priority_B.append(random.sample(range(1, 129), 3))
        
    # Deduplicate
    seen = set()
    unique_Bs = []
    for b in high_priority_B:
        if len(set(b)) == 3:
            t = tuple(sorted(b))
            if t not in seen:
                seen.add(t)
                unique_Bs.append(list(t))
                
    for B in unique_Bs:
        # Search A
        for A in combinations(range(1, 129), 3):
            A_list = list(A)
            sums = set()
            for a in A_list:
                for b in B:
                    sums.add(a + b)
            
            if len(sums) < 9:
                # Cannot beat 9, but maybe we don't have 9 yet?
                # If we haven't found 9, a score of 8 is better than 5.
                # But if we are looking for max, we can skip if we already have 9.
                if best_score == 9:
                    continue
                # If best_score is low, we might accept <9.
                # But pairwise coprime check is the bottleneck.
                pass
            
            distinct_sums = list(sums)
            n = len(distinct_sums)
            
            # Check pairwise coprime
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    if math.gcd(distinct_sums[i], distinct_sums[j]) != 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                if n > best_score:
                    best_score = n
                    best_A = A_list
                    best_B = B
                    if best_score == 9:
                        return [best_A, best_B]
                        
    # Fallback to seed if no better found
    if best_score < 5:
        best_A = [1, 7, 13]
        best_B = [4, 10, 16]
        best_score = 5
        
    return [best_A, best_B]
