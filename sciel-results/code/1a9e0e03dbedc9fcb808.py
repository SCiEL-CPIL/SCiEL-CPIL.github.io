import time
import random

def construct():
    """
    Finds a set of distinct integers in 1..256 maximizing the size,
    such that all non-empty subset sums are distinct and no two distinct
    subset sums divide one another.
    """
    start_time = time.time()
    time_limit = 58  # Keep 2s margin
    
    best_set = [31, 32, 34]
    best_size = len(best_set)
    
    def compute_sums(S):
        """Compute all subset sums of S (including 0)."""
        sums = {0}
        for x in S:
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            sums = sums | new_sums
        return sums

    def is_antichain(sums_set):
        """
        Check if sums_set (excluding 0) is an antichain under divisibility.
        Optimized: For each u, check if any multiple of u is in the set.
        """
        if not sums_set:
            return True
        
        # We need to check distinct non-zero sums
        # Note: sums_set should already contain only distinct values.
        # We iterate through sorted sums to potentially prune? 
        # Actually, checking multiples is independent of order.
        
        # To speed up, we can convert to a list and sort.
        # But checking `if 2*u in sums_set` is O(1) average.
        
        # One optimization: if u > max_sum // 2, no multiple of u (other than u itself) 
        # can be in the set (since max_sum < 2*u). 
        # So we only need to check u <= max_sum // 2.
        
        max_sum = max(sums_set)
        
        # We can also skip u if u is 1? 
        # If 1 is in sums_set, it divides everything. 
        # So if 1 in sums_set, and len(sums_set) > 1, return False.
        if 1 in sums_set and len(sums_set) > 1:
            return False
            
        for u in sums_set:
            if u == 1:
                continue # handled above
            # Check multiples 2u, 3u, ... up to max_sum
            # Step by u
            # Start from 2*u
            multiple = 2 * u
            while multiple <= max_sum:
                if multiple in sums_set:
                    return False
                multiple += u
        return True

    def check_validity_incremental(S, x, sums_S):
        """
        Checks if S U {x} is valid, given S is valid.
        Returns (True, new_sums) if valid, (False, None) otherwise.
        """
        if x in S:
            return False, None
            
        # Generate new sums: x + s for s in sums_S
        # sums_S contains 0.
        new_sums = {x + s for s in sums_S}
        
        # Check distinctness: sums_S and new_sums must be disjoint
        if sums_S & new_sums:
            return False, None
        
        all_sums = sums_S | new_sums
        non_empty_all = all_sums - {0}
        
        # Check divisibility antichain
        if not is_antichain(non_empty_all):
            return False, None
        
        return True, all_sums

    # Search Strategy
    # We want to maximize size.
    # Larger numbers are generally safer for divisibility constraints 
    # (fewer multiples within the range).
    # However, we must ensure distinct subset sums.
    # Superincreasing sequences guarantee distinct sums.
    # Let's try to build superincreasing sequences with large bases.
    
    random.seed(42)
    
    num_restarts = 2000
    for attempt in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        S = []
        sums_S = {0}
        
        # Create candidate list
        # Bias towards larger numbers?
        # Or random?
        # Let's try a mix.
        
        candidates = list(range(1, 257))
        
        # Strategy 1: Descending (largest first)
        # Strategy 2: Random
        # Strategy 3: Random but filter out numbers that are multiples of existing sums?
        
        if attempt % 3 == 0:
            candidates.sort(reverse=True)
        elif attempt % 3 == 1:
            random.shuffle(candidates)
        else:
            # Sort by random key, but with a bias towards larger numbers
            # e.g., key = random.random() * x
            candidates.sort(key=lambda x: random.random() * x, reverse=True)
            
        for x in candidates:
            if time.time() - start_time > time_limit:
                break
            
            # Quick pre-check: 
            # If x is such that x divides any existing sum, or x + sum divides something?
            # The incremental check handles this.
            
            valid, new_sums = check_validity_incremental(S, x, sums_S)
            if valid:
                S.append(x)
                sums_S = new_sums
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S[:]
                
                # If we have a very large set, checks are slower.
                # But with the optimized antichain check, it should be fine up to size 10-12.
    
    # Final verification
    if best_set:
        final_sums = compute_sums(best_set)
        non_empty_final = final_sums - {0}
        if len(non_empty_final) != (1 << len(best_set)) - 1:
            # Distinctness failed? Should not happen if incremental check was correct.
            best_set = [31, 32, 34]
            best_size = 3
        elif not is_antichain(non_empty_final):
            # Antichain failed?
            best_set = [31, 32, 34]
            best_size = 3
    
    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(res)
