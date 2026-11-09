import itertools
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    def pairwise_coprime(values):
        """Check if all distinct values are pairwise coprime."""
        vals = sorted(set(values))
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                if math.gcd(vals[i], vals[j]) != 1:
                    return False
        return True
    
    best_count = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Precompute coprime pairs for efficiency
    # We'll search over combinations of A and B
    
    # Strategy: iterate over A sets, then for each A, iterate over B sets
    # But 128 choose 3 is ~350k, so 350k^2 is too many
    # Instead, use a smarter approach: fix structure and search
    
    # Better approach: 
    # 1. Generate candidate A sets (spread out values)
    # 2. For each A, generate candidate B sets and check
    
    # To keep it fast, limit the search space intelligently
    # Use values that are likely to produce coprime sums
    
    # Precompute: for each pair (a, b), is a+b coprime with other sums?
    # This is complex, so let's do a structured search
    
    # Approach: 
    # - Try A sets with values spread across [1,128]
    # - For each A, try B sets and count distinct coprime sums
    
    # To make this tractable, we'll use a two-phase approach:
    # Phase 1: Random sampling with local improvement
    # Phase 2: Systematic search over promising regions
    
    import random
    random.seed(42)
    
    def score_pair(A, B):
        """Return (is_valid, count) for the pair (A, B)."""
        sums = [a + b for a in A for b in B]
        distinct_sums = sorted(set(sums))
        if len(distinct_sums) != len(set(sums)):
            # There are duplicates, but that's fine - we just count distinct
            pass
        # Check pairwise coprime
        for i in range(len(distinct_sums)):
            for j in range(i + 1, len(distinct_sums)):
                if math.gcd(distinct_sums[i], distinct_sums[j]) != 1:
                    return (False, 0)
        return (True, len(distinct_sums))
    
    # Phase 1: Random sampling
    max_iter = 500000
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        A = sorted(random.sample(range(1, 129), 3))
        B = sorted(random.sample(range(1, 129), 3))
        
        valid, count = score_pair(A, B)
        if valid and count > best_count:
            best_count = count
            best_A = A
            best_B = B
            if best_count == 9:
                # Maximum possible
                break
    
    # Phase 2: Local search from best found
    if best_count < 9 and time.time() < deadline:
        # Try to improve by local modifications
        for _ in range(200000):
            if time.time() > deadline:
                break
            
            # Pick a random element from A or B and perturb it
            if random.random() < 0.5:
                idx = random.randint(0, 2)
                old_val = best_A[idx]
                # Try nearby values
                for delta in [-2, -1, 1, 2, -5, -3, 3, 5, -10, -7, 7, 10]:
                    new_val = old_val + delta
                    if 1 <= new_val <= 128 and new_val not in best_A:
                        new_A = best_A[:idx] + [new_val] + best_A[idx+1:]
                        new_A = sorted(new_A)
                        valid, count = score_pair(new_A, best_B)
                        if valid and count > best_count:
                            best_count = count
                            best_A = new_A
                            break
            else:
                idx = random.randint(0, 2)
                old_val = best_B[idx]
                for delta in [-2, -1, 1, 2, -5, -3, 3, 5, -10, -7, 7, 10]:
                    new_val = old_val + delta
                    if 1 <= new_val <= 128 and new_val not in best_B:
                        new_B = best_B[:idx] + [new_val] + best_B[idx+1:]
                        new_B = sorted(new_B)
                        valid, count = score_pair(best_A, new_B)
                        if valid and count > best_count:
                            best_count = count
                            best_B = new_B
                            break
    
    return [best_A, best_B]
