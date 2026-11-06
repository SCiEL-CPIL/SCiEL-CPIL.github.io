import time
import heapq
import numpy as np

def construct():
    """
    Constructs a subset A of [0, 128] such that every integer in [128, 256]
    is a sum of two elements of A. Minimizes the largest representation multiplicity.
    
    Strategy:
    1. Deterministic Greedy Construction:
       - Start with an empty set (or forced elements like 128).
       - Maintain a priority queue of candidate elements based on how much
         they reduce the "deficit" (missing sums or high multiplicity sums).
       - Specifically, prioritize elements that cover the currently most
         under-represented sums in [128, 256].
    2. Greedy Removal (Pruning):
       - Once a valid set is formed, attempt to remove elements one by one
         (starting with those contributing least to the peak or most to low
         multiplicity sums) if the set remains valid and the peak does not increase.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # Target sums: 128 to 256 inclusive
    target_min = 128
    target_max = 256
    targets = np.arange(target_min, target_max + 1)
    
    # We need to track counts of representations for each target sum.
    # A representation is an unordered pair {a, b} with a, b in A, a+b = s.
    # If a == b, it's one pair. If a != b, {a, b} is one pair.
    # Note: The task says "Pair representations are unordered and permit equal summands."
    # So for sum s, the count is the number of pairs {a, b} with a <= b, a+b=s, a,b in A.
    
    # Helper to calculate counts for a given set A (as a boolean array or set)
    def get_counts(A_set):
        """
        Returns an array of counts for sums 0..256.
        count[s] = number of unordered pairs {a,b} in A such that a+b=s.
        """
        if not A_set:
            return np.zeros(257, dtype=np.int32)
        
        # Use convolution for speed, then adjust for unordered.
        # Let v be indicator vector of A.
        # conv = v * v gives ordered counts N_ord[s].
        # N_ord[s] = 2 * N_unord_distinct[s] + N_diag[s]
        # where N_unord_distinct[s] is count of pairs {a,b} with a<b, a+b=s
        # and N_diag[s] is 1 if s is even and s/2 in A, else 0.
        # We want N_unord_total[s] = N_unord_distinct[s] + N_diag[s]
        
        # From N_ord:
        # N_ord[s] = 2 * N_unord_distinct[s] + N_diag[s]
        # N_unord_distinct[s] = (N_ord[s] - N_diag[s]) / 2
        # N_unord_total[s] = (N_ord[s] - N_diag[s]) / 2 + N_diag[s] = (N_ord[s] + N_diag[s]) / 2
        
        A_bool = np.zeros(129, dtype=np.int32)
        for a in A_set:
            if 0 <= a <= 128:
                A_bool[a] = 1
                
        conv = np.convolve(A_bool, A_bool, mode='full')
        
        # conv has length 257.
        # Determine diag presence for each sum s.
        # s is even, and s/2 is in A.
        # s ranges 0..256.
        # s/2 ranges 0..128.
        
        # Vectorized calculation of diag term
        # Create array of sums 0..256
        sums = np.arange(257)
        even_mask = (sums % 2 == 0)
        mids = sums[even_mask] // 2
        # mids are in [0, 128]
        # Check if A_bool[mid] is 1
        diag_terms = np.zeros(257, dtype=np.int32)
        diag_terms[even_mask] = A_bool[mids]
        
        counts = (conv + diag_terms) // 2
        return counts

    # Initial A: Must cover 256, so 128 is essential (128+128).
    # Also, to cover 128, we need pairs summing to 128.
    # Let's start with A = {128}.
    A = {128}
    
    # Current counts
    counts = get_counts(A)
    
    # Check validity: all targets must have count > 0.
    def is_valid(counts):
        return np.all(counts[target_min:target_max+1] > 0)
    
    # We will greedily add elements.
    # Priority function: For a candidate element x not in A,
    # calculate the new counts if x is added.
    # Score = reduction in "deficit".
    # Deficit can be defined as:
    # 1. Number of uncovered sums (count == 0)
    # 2. Weighted by how "hard" they are to cover?
    # Let's use a composite score:
    # Primary: Minimize number of uncovered sums.
    # Secondary: Minimize the peak multiplicity among covered sums?
    # Or simpler: Minimize the sum of (max_count - count[s]) for uncovered/low counts?
    
    # Let's try: Score = - (number of newly covered sums) * 1000 - (reduction in peak)
    # Actually, just covering is most important first.
    
    # To make it fast, we can precompute which sums each candidate x helps.
    # If we add x, it forms pairs {x, y} for all y in A.
    # It also forms {x, x}.
    # So it affects sums: x+y for y in A, and 2x.
    
    # We can maintain a "potential" score for each candidate x in [0, 128].
    # However, as A changes, the scores change.
    # Since |A| is small initially, we can recompute scores for all candidates
    # in each step? 129 candidates * 129 ops = ~16k ops. Very fast.
    
    candidates = list(range(129))
    
    # We will run until valid or time runs out.
    # Max size of A is 129.
    
    while not is_valid(counts) and time.time() < deadline - 10:
        best_x = -1
        best_score = -float('inf')
        
        # Current peak
        current_peak = np.max(counts[target_min:target_max+1])
        
        for x in candidates:
            if x in A:
                continue
                
            # Calculate impact of adding x
            # New pairs: {x, y} for y in A, and {x, x}
            # Sums affected: x + y for y in A, and 2x.
            
            # We need to update counts for these specific sums.
            # Since we only care about [128, 256], we filter.
            
            # Let's simulate the count change.
            # Instead of full convolution, just increment the relevant sums.
            
            # Create a temporary copy of counts? No, too slow if we do it for every x.
            # Let's calculate the "gain" analytically.
            
            # For each sum s in [128, 256]:
            # new_count[s] = counts[s] + (1 if (x in A and s-x in A and x <= s-x) else 0) ... wait, unordered logic is tricky.
            
            # Simpler:
            # Let's just compute the new counts for the affected sums only?
            # No, the peak might be elsewhere.
            
            # Let's stick to the full convolution but optimize the loop.
            # Actually, 129 candidates * (convolution cost) is fine.
            # Convolution of 129-length vectors is very fast in NumPy.
            
            # But doing it inside the loop is O(N^2) overall. 129^2 * 129 steps?
            # 129 steps max. 129 * 129 * 129 ~ 2 million ops. Fine.
            
            A_temp = A | {x}
            counts_temp = get_counts(A_temp)
            
            # Score:
            # 1. Number of sums in [128, 256] that become covered (were 0, now > 0)
            covered_before = np.sum(counts[target_min:target_max+1] == 0)
            covered_after = np.sum(counts_temp[target_min:target_max+1] == 0)
            gain_covered = covered_before - covered_after
            
            # 2. Peak reduction?
            peak_after = np.max(counts_temp[target_min:target_max+1])
            
            # We want to maximize gain_covered.
            # If tie, minimize peak_after.
            
            score = gain_covered * 1000 - peak_after
            
            if score > best_score:
                best_score = score
                best_x = x
                
        if best_x == -1:
            break
            
        A.add(best_x)
        counts = get_counts(A)
        
        # Optional: If we are valid, we can stop adding.
        if is_valid(counts):
            break
            
    if not is_valid(counts):
        # Fallback: return a dense set
        return list(range(64, 129))
        
    # --- Phase 2: Greedy Removal ---
    # Now A is valid. Try to remove elements to reduce size/peak.
    # Iterate:
    # 1. For each x in A, check if A\{x} is valid.
    # 2. If valid, calculate new peak.
    # 3. If new_peak <= current_peak, remove x (prefer removal if peak decreases or stays same).
    # 4. Repeat until no more removals possible.
    
    current_peak = np.max(counts[target_min:target_max+1])
    
    while time.time() < deadline - 5:
        improved = False
        # Sort A by some heuristic? E.g., elements that contribute to the peak most?
        # Or just iterate all.
        
        # To be deterministic, iterate in sorted order or fixed order.
        # Let's try removing elements that are "least useful".
        # An element x is "least useful" if removing it doesn't break validity
        # and results in the lowest peak.
        
        best_removable = -1
        best_new_peak = current_peak
        
        for x in sorted(A):
            if len(A) <= 1:
                break
            A_temp = A - {x}
            counts_temp = get_counts(A_temp)
            
            if is_valid(counts_temp):
                peak_temp = np.max(counts_temp[target_min:target_max+1])
                # We want to remove if peak_temp <= current_peak
                # Prefer lower peak.
                if peak_temp < best_new_peak or (peak_temp == best_new_peak and best_removable == -1):
                    best_new_peak = peak_temp
                    best_removable = x
                    
        if best_removable != -1:
            A.remove(best_removable)
            counts = get_counts(A)
            current_peak = best_new_peak
            improved = True
            
        if not improved:
            break
            
    return list(A)

# EVOLVE-BLOCK-END
