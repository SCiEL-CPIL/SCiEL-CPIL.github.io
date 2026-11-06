"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 138.0
    N = 1000
    fft_size = 4096
    # Precompute parity masks for fast rep-count extraction
    even_sums = np.arange(2, 2*N+1, 2)
    odd_sums = np.arange(3, 2*N+1, 2)
    
    # Single-pass validation: returns (is_valid, max_rep, counts_array)
    # counts[s] = number of representations a<=b with a+b=s
    def validate_counts(indicator):
        """Returns (valid, max_rep, counts) where counts[s] = # reps for sum s."""
        f = np.zeros(fft_size)
        f[1:N+1] = indicator[1:N+1].astype(float)
        F = np.fft.fft(f)
        conv = np.fft.ifft(F * F).real
        
        # For each sum s, compute unordered rep count
        # rep(s) = (conv[s] + (1 if s even and s//2 in set else 0)) / 2
        # Vectorized:
        all_sums = np.arange(2, 2*N+1)
        # mid index for even sums
        mids = all_sums // 2
        # indicator of whether mid is in set and mid in [1,N]
        mid_valid = (mids >= 1) & (mids <= N)
        mid_in_set = np.zeros(2*N+1, dtype=np.float64)
        mid_in_set[1:N+1] = indicator[1:N+1].astype(float)
        diag = np.where(mid_valid, mid_in_set[mids], 0.0)
        
        reps = (conv[2:2*N+1] + diag) / 2.0
        max_rep = np.max(reps)
        valid = max_rep <= 2.0 + 1e-9
        return valid, max_rep, reps
    
    # Fast incremental check: given current counts and a new element x,
    # compute new counts and check validity
    def add_element_check(counts, indicator, x):
        """Try adding x. Returns (valid, new_counts)."""
        # New sums introduced by x: x+y for y in set, plus 2*x
        # We need to check if any sum exceeds 2
        # Efficient: iterate over current set members
        new_counts = counts.copy()
        # Find all y in set
        y_values = np.where(indicator[1:N+1] > 0)[0] + 1
        for y in y_values:
            s = x + y
            if s <= 2*N:
                new_counts[s-2] += 1  # counts indexed from 2
                if new_counts[s-2] > 2.0 + 1e-9:
                    return False, None
        # Check 2*x
        s = 2*x
        if s <= 2*N:
            new_counts[s-2] += 1
            if new_counts[s-2] > 2.0 + 1e-9:
                return False, None
        return True, new_counts
    
    # Greedy baseline: sequential
    chosen = []
    counts = np.zeros(2*N-1, dtype=np.float64)  # counts[s] for s=2..2N
    indicator = np.zeros(N+1, dtype=np.uint8)
    
    for x in range(1, N+1):
        valid, new_counts = add_element_check(counts, indicator, x)
        if valid:
            chosen.append(x)
            counts = new_counts
            indicator[x] = 1
    
    best_set = list(chosen)
    best_size = len(chosen)
    best_counts = counts.copy()
    best_indicator = indicator.copy()
    
    np.random.seed(42)
    
    # Multiple greedy restarts with different orderings
    for attempt in range(80):
        if time.time() > deadline:
            break
        
        candidates = list(range(1, N+1))
        np.random.shuffle(candidates)
        
        indicator = np.zeros(N+1, dtype=np.uint8)
        counts = np.zeros(2*N-1, dtype=np.float64)
        chosen = []
        
        for x in candidates:
            if time.time() > deadline:
                break
            valid, new_counts = add_element_check(counts, indicator, x)
            if valid:
                chosen.append(x)
                counts = new_counts
                indicator[x] = 1
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = list(chosen)
            best_counts = counts
            best_indicator = indicator
    
    # Local search: try to add elements, or swap to gain size
    if time.time() < deadline - 3:
        # First, try pure additions
        for _ in range(500):
            if time.time() > deadline:
                break
            # Try adding a random element
            x = np.random.randint(1, N+1)
            if best_indicator[x] == 0:
                valid, new_counts = add_element_check(best_counts, best_indicator, x)
                if valid:
                    best_set.append(x)
                    best_size += 1
                    best_counts = new_counts
                    best_indicator[x] = 1
            else:
                # Try swap: remove one element, add this one
                idx = np.random.randint(len(best_set))
                removed = best_set[idx]
                # Remove from counts
                temp_indicator = best_indicator.copy()
                temp_indicator[removed] = 0
                # Recompute counts after removal
                f = np.zeros(fft_size)
                f[1:N+1] = temp_indicator[1:N+1].astype(float)
                F = np.fft.fft(f)
                conv = np.fft.ifft(F * F).real
                all_sums = np.arange(2, 2*N+1)
                mids = all_sums // 2
                mid_valid = (mids >= 1) & (mids <= N)
                mid_in_set = np.zeros(2*N+1, dtype=np.float64)
                mid_in_set[1:N+1] = temp_indicator[1:N+1].astype(float)
                diag = np.where(mid_valid, mid_in_set[mids], 0.0)
                temp_counts = (conv[2:2*N+1] + diag) / 2.0
                
                valid, new_counts = add_element_check(temp_counts, temp_indicator, x)
                if valid:
                    new_set = best_set[:idx] + best_set[idx+1:] + [x]
                    best_set = new_set
                    best_size = len(best_set)
                    best_counts = new_counts
                    best_indicator = temp_indicator
                    best_indicator[x] = 1
                # If not valid, keep original
    
    # Final polish: try all remaining elements for addition
    if time.time() < deadline - 1:
        for x in range(1, N+1):
            if time.time() > deadline:
                break
            if best_indicator[x] == 0:
                valid, new_counts = add_element_check(best_counts, best_indicator, x)
                if valid:
                    best_set.append(x)
                    best_size += 1
                    best_counts = new_counts
                    best_indicator[x] = 1
    
    return best_set
# EVOLVE-BLOCK-END
