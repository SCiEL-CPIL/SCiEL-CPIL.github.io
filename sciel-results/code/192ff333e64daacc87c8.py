import time
import random
import sys

def construct():
    """
    Finds a set of distinct integers in 1..256 maximizing the size,
    such that all non-empty subset sums are distinct and no two distinct
    subset sums divide one another.
    """
    start_time = time.time()
    time_limit = 78  # Keep 12s margin for evaluator overhead
    
    def compute_sums(S):
        """Compute all subset sums of S (including 0)."""
        sums = {0}
        for x in S:
            sums = sums | {s + x for s in sums}
        return sums
    
    def check_divisibility_sieve(sums_set):
        """
        Fast divisibility check using sieve approach.
        For each sum u, mark all multiples 2u, 3u, ... and check if any is in sums_set.
        """
        if len(sums_set) <= 1:
            return True
        
        sorted_sums = sorted(sums_set)
        max_val = sorted_sums[-1]
        
        # Use a boolean array for fast membership
        present = [False] * (max_val + 1)
        for s in sums_set:
            present[s] = True
        
        for u in sorted_sums:
            if u == 0:
                continue
            # Check multiples of u
            for mult in range(2 * u, max_val + 1, u):
                if present[mult]:
                    return False
        return True
    
    def can_add_fast(S, x, current_sums):
        """
        Check if we can add x to S.
        Returns (bool, new_sums) if valid.
        """
        # Generate new sums
        new_sums = {x + s for s in current_sums}
        
        # Check distinctness: new_sums should not overlap with current_sums
        if new_sums & current_sums:
            return False, None
        
        # Combined sums
        combined = current_sums | new_sums
        non_empty = combined - {0}
        
        # Check divisibility using sieve
        if not check_divisibility_sieve(non_empty):
            return False, None
        
        return True, combined
    
    def greedy_build(candidate_order, current_sums=None, S=None):
        """Greedy build using a given order of candidates."""
        if S is None:
            S = []
        if current_sums is None:
            current_sums = {0}
        
        for x in candidate_order:
            if time.time() - start_time > time_limit:
                break
            valid, new_sums = can_add_fast(S, x, current_sums)
            if valid:
                S.append(x)
                current_sums = new_sums
        return S, current_sums
    
    best_set = []
    best_size = 0
    
    random.seed(42)
    
    # Multiple restart strategies
    num_restarts = 120
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        if restart < 30:
            # Strategy 1: Larger numbers first with some randomness
            candidates = list(range(256, 0, -1))
            random.shuffle(candidates[:80])
            random.shuffle(candidates[80:])
        elif restart < 60:
            # Strategy 2: Fully random
            candidates = list(range(1, 257))
            random.shuffle(candidates)
        elif restart < 90:
            # Strategy 3: Weighted toward larger numbers
            candidates = list(range(1, 257))
            candidates.sort(key=lambda x: -x + random.gauss(0, 50))
        else:
            # Strategy 4: Superincreasing bias - prefer numbers > sum of selected so far
            candidates = list(range(1, 257))
            random.shuffle(candidates)
        
        S, _ = greedy_build(candidates)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
    
    # Local search: try to improve by adding elements
    if best_size >= 2:
        base_set = best_set[:]
        
        for _ in range(300):
            if time.time() - start_time > time_limit:
                break
            
            S = base_set[:]
            sums_set = compute_sums(S)
            improved = False
            
            # Try adding each candidate not in S, prefer larger
            for x in range(256, 0, -1):
                if x in S:
                    continue
                valid, new_sums = can_add_fast(S, x, sums_set)
                if valid:
                    S.append(x)
                    sums_set = new_sums
                    improved = True
                    # Continue trying to add more
                    for x2 in range(256, 0, -1):
                        if x2 in S:
                            continue
                        valid2, new_sums2 = can_add_fast(S, x2, sums_set)
                        if valid2:
                            S.append(x2)
                            sums_set = new_sums2
                    break
            
            if improved:
                base_set = S
                if len(base_set) > best_size:
                    best_size = len(base_set)
                    best_set = base_set[:]
            else:
                break
    
    # Final verification
    if best_set:
        final_sums = compute_sums(best_set)
        non_empty = final_sums - {0}
        if len(non_empty) != (1 << len(best_set)) - 1:
            # Duplicates found, fallback
            best_set = []
        elif not check_divisibility_sieve(non_empty):
            # Divisibility violation, fallback
            best_set = []
    
    # Ensure we return a valid set
    if not best_set or len(best_set) < 1:
        return [256]
    
    return [int(x) for x in best_set]

if __name__ == "__main__":
    res = construct()
    print(res)
