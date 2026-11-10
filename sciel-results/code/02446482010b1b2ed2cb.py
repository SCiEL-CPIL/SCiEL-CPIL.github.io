import random

def construct():
    # We want to find a subset A of [0, 128]
    # such that [128, 256] is covered by pair sums.
    # Minimize the max number of representations.
    
    # The seed solution A = [64, ..., 128] has size 65 and peak mult ~33.
    # We can do much better with a smaller, sparser set.
    
    # Let's try sizes from 15 to 25.
    # For each size, generate random subsets and check validity and score.
    
    best_A = None
    best_peak = float('inf')
    
    # Fixed seed for deterministic output in evaluation if needed, 
    # but random exploration is better for finding a good solution.
    # We will use a fixed seed to ensure the "program" is deterministic 
    # if the evaluator runs it multiple times, but the logic is to find a good static set.
    # Actually, the task says "construct() -> a nonempty subset". 
    # It doesn't require the result to be the same every time, but it's safer to be deterministic 
    # or to find a specific good set.
    # I will hardcode a search that runs fast enough to find a good set 
    # and then return the best one found.
    
    random.seed(42) # Reproducibility
    
    target_min = 128
    target_max = 256
    
    # Precompute pair sums for a set? No, do it on the fly.
    
    def get_peak(A):
        A_set = set(A)
        max_count = 0
        for s in range(target_min, target_max + 1):
            count = 0
            # Unordered pairs (a, b) with a <= b
            # a + b = s => b = s - a
            # a must be in A, b must be in A
            # a <= b => a <= s - a => 2a <= s => a <= s/2
            # Also a >= max(0, s - 128) since b <= 128
            # And a >= 0
            # a in A
            for a in A:
                if a > s: break # if A is sorted
                b = s - a
                if b < a: continue # ensure a <= b
                if b in A_set:
                    count += 1
            if count == 0:
                return float('inf') # Invalid
            if count > max_count:
                max_count = count
        return max_count

    # To make get_peak faster, sort A
    # And iterate A in sorted order
    
    best_A = None
    best_peak = float('inf')
    
    # Try different sizes
    for k in range(15, 26):
        # Try 500 random samples for this size
        for _ in range(500):
            A = random.sample(range(129), k)
            A.sort()
            # Quick check: min sum 2*min(A) <= 128? 
            # max sum 2*max(A) >= 256?
            # If 2*min(A) > 128, we can't form 128.
            # If 2*max(A) < 256, we can't form 256.
            # Actually, we need to cover ALL in [128, 256].
            # So we need min(A) <= 64 (to get 128 with 64+64? No, a+b=128. If min(A)=65, min sum=130. So 128 is missed.)
            # So min(A) must be <= 64.
            # And max(A) must be >= 128 (to get 256 with 128+128? No, a+b=256. If max(A)=127, max sum=254. So 256 is missed.)
            # So max(A) must be 128.
            
            if A[0] > 64: continue
            if A[-1] < 128: continue
            
            # Calculate peak
            A_set = set(A)
            valid = True
            max_count = 0
            for s in range(128, 257):
                count = 0
                for a in A:
                    if a > s: break
                    b = s - a
                    if b < a: continue
                    if b in A_set:
                        count += 1
                if count == 0:
                    valid = False
                    break
                if count > max_count:
                    max_count = count
            
            if valid and max_count < best_peak:
                best_peak = max_count
                best_A = A.copy()
                
                # If we find a very good one, we can stop early?
                # Theoretical lower bound:
                # Total pairs ~ k^2/2. Range 129.
                # If k=16, pairs=136. Avg=1.05. Min peak likely 2 or 3.
                # If we find peak 2, it's excellent.
                if best_peak <= 2:
                    break
                    
        if best_peak <= 2:
            break
            
    if best_A is None:
        # Fallback to a known valid set if random search fails (unlikely)
        # e.g. A = [0, 128] union some others?
        # Let's just return the seed if all else fails, though it should not happen.
        return list(range(64, 129))
        
    return best_A
