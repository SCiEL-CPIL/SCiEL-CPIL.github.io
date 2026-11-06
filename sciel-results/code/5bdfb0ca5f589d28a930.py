import time
import numpy as np

def construct():
    """
    Constructs a subset A of [0, 128] such that every integer in [128, 256]
    is a sum of two elements of A. Minimizes the largest representation multiplicity.
    """
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit
    
    target_min = 128
    target_max = 256
    N = 129  # Range [0, 128]
    
    # Initial state: A = [64, ..., 128] guarantees coverage
    A_mask = np.zeros(N, dtype=np.int8)
    A_mask[64:129] = 1
    
    # Precompute: for each sum s in [128, 256], which indices i satisfy i in A and (s-i) in A
    # We'll use a bitset approach: represent A as a Python int bitmask for fast intersection
    
    def mask_to_int(mask):
        """Convert numpy mask to Python int for bit operations."""
        bits = 0
        for i in range(N):
            if mask[i]:
                bits |= (1 << i)
        return bits
    
    def get_counts_fast(mask, mask_int=None):
        """
        Returns (peak_count, uncovered_count, counts_array) using bitset operations.
        For each sum s in [128,256], count unordered pairs (i,j) with i<=j, i+j=s, i,j in A.
        """
        if mask_int is None:
            mask_int = mask_to_int(mask)
        
        counts = np.zeros(129, dtype=np.int32)  # for sums 128..256
        
        for s in range(target_min, target_max + 1):
            # Count pairs (i, j) with i <= j, i + j = s, i in A, j in A
            # i ranges from max(0, s-128) to min(128, s//2)
            i_lo = max(0, s - 128)
            i_hi = min(128, s // 2)
            
            cnt = 0
            for i in range(i_lo, i_hi + 1):
                j = s - i
                if mask[i] and mask[j]:
                    cnt += 1
            counts[s - target_min] = cnt
        
        peak = int(np.max(counts))
        uncovered = int(np.sum(counts == 0))
        
        return peak, uncovered, counts
    
    def get_smooth_peak(counts, beta=2.0):
        """Soft surrogate for peak: sum of exp(beta * (c - max_c)) or sum of c^beta."""
        # Use sum of squares as a smooth surrogate for peak
        return float(np.sum(counts.astype(np.float64) ** 2))
    
    # Initial metrics
    current_peak, current_uncovered, current_counts = get_counts_fast(A_mask)
    current_smooth = get_smooth_peak(current_counts)
    current_obj = current_peak * 1000 + current_uncovered * 10 + current_smooth
    
    best_A_mask = A_mask.copy()
    best_obj = current_obj
    best_peak = current_peak
    best_uncovered = current_uncovered
    best_counts = current_counts.copy()
    
    rng = np.random.default_rng(42)
    
    # Simulated Annealing
    T = 5.0
    cooling_rate = 0.998
    step = 0
    
    # Precompute candidate indices
    all_indices = np.arange(N)
    
    while time.time() < deadline and T > 0.005:
        step += 1
        
        # Choose move type
        move_type = rng.random()
        new_mask = A_mask.copy()
        move_valid = True
        
        if move_type < 0.25:
            # Add an element
            candidates = np.where(new_mask == 0)[0]
            if len(candidates) > 0:
                idx = int(rng.choice(candidates))
                new_mask[idx] = 1
            else:
                continue
        elif move_type < 0.5:
            # Remove an element
            candidates = np.where(new_mask == 1)[0]
            if len(candidates) > 1:
                idx = int(rng.choice(candidates))
                new_mask[idx] = 0
            else:
                continue
        elif move_type < 0.75:
            # Swap: Remove one, Add one
            candidates_in = np.where(new_mask == 1)[0]
            candidates_out = np.where(new_mask == 0)[0]
            if len(candidates_in) > 1 and len(candidates_out) > 0:
                idx_in = int(rng.choice(candidates_in))
                idx_out = int(rng.choice(candidates_out))
                new_mask[idx_in] = 0
                new_mask[idx_out] = 1
            else:
                continue
        else:
            # Targeted move: add an element that helps cover an uncovered sum
            # Find uncovered sums
            uncovered_sums = np.where(current_counts == 0)[0] + target_min
            if len(uncovered_sums) > 0:
                # Pick a random uncovered sum
                s = int(rng.choice(uncovered_sums))
                # Find which single additions could help cover s
                # For sum s, we need i in A and j = s-i in A
                # Currently uncovered means no such pair exists
                # Try adding a random element that could form a pair
                i_lo = max(0, s - 128)
                i_hi = min(128, s)
                possible = np.arange(i_lo, i_hi + 1)
                # For each i, check if adding i would create a pair for s
                # i in new A and (s-i) in current A, or i in current A and (s-i) in new A
                helpful = []
                for i in possible:
                    j = s - i
                    if 0 <= j < N:
                        if (new_mask[i] == 0 and new_mask[j] == 1) or (new_mask[i] == 1 and new_mask[j] == 0):
                            # Adding the missing one would help
                            if new_mask[i] == 0:
                                helpful.append(i)
                            else:
                                helpful.append(j)
                if helpful:
                    idx = int(rng.choice(helpful))
                    if new_mask[idx] == 0:
                        new_mask[idx] = 1
                    else:
                        continue
                else:
                    # Fallback: random add
                    candidates = np.where(new_mask == 0)[0]
                    if len(candidates) > 0:
                        idx = int(rng.choice(candidates))
                        new_mask[idx] = 1
                    else:
                        continue
            else:
                # No uncovered sums, do a random swap to reduce peak
                candidates_in = np.where(new_mask == 1)[0]
                candidates_out = np.where(new_mask == 0)[0]
                if len(candidates_in) > 1 and len(candidates_out) > 0:
                    idx_in = int(rng.choice(candidates_in))
                    idx_out = int(rng.choice(candidates_out))
                    new_mask[idx_in] = 0
                    new_mask[idx_out] = 1
                else:
                    continue
        
        # Evaluate new state
        new_peak, new_uncovered, new_counts = get_counts_fast(new_mask)
        new_smooth = get_smooth_peak(new_counts)
        new_obj = new_peak * 1000 + new_uncovered * 10 + new_smooth
        
        # Acceptance criterion
        delta_obj = new_obj - current_obj
        
        if delta_obj < 0:
            # Improvement
            A_mask = new_mask
            current_peak, current_uncovered, current_counts = new_peak, new_uncovered, new_counts
            current_smooth = new_smooth
            current_obj = new_obj
            
            if current_obj < best_obj:
                best_obj = current_obj
                best_peak = current_peak
                best_uncovered = current_uncovered
                best_counts = current_counts.copy()
                best_A_mask = A_mask.copy()
        else:
            # Worse, accept with probability exp(-delta/T)
            if rng.random() < np.exp(-delta_obj / T):
                A_mask = new_mask
                current_peak, current_uncovered, current_counts = new_peak, new_uncovered, new_counts
                current_smooth = new_smooth
                current_obj = new_obj
        
        T *= cooling_rate
        
        # Early stop if perfect
        if best_uncovered == 0 and best_peak == 1:
            break
    
    # Final polish: local search to try to reduce peak further
    if time.time() < deadline - 5:
        polish_deadline = time.time() + (deadline - time.time()) * 0.8
        current_mask = best_A_mask.copy()
        current_peak, current_uncovered, current_counts = get_counts_fast(current_mask)
        current_smooth = get_smooth_peak(current_counts)
        current_obj = current_peak * 1000 + current_uncovered * 10 + current_smooth
        
        T_polish = 0.5
        cooling_polish = 0.995
        
        while time.time() < polish_deadline and T_polish > 0.001:
            # Random single-element swap
            candidates_in = np.where(current_mask == 1)[0]
            candidates_out = np.where(current_mask == 0)[0]
            if len(candidates_in) > 0 and len(candidates_out) > 0:
                idx_in = int(rng.choice(candidates_in))
                idx_out = int(rng.choice(candidates_out))
                
                new_mask = current_mask.copy()
                new_mask[idx_in] = 0
                new_mask[idx_out] = 1
                
                new_peak, new_uncovered, new_counts = get_counts_fast(new_mask)
                new_smooth = get_smooth_peak(new_counts)
                new_obj = new_peak * 1000 + new_uncovered * 10 + new_smooth
                
                delta = new_obj - current_obj
                if delta < 0 or (rng.random() < np.exp(-delta / T_polish)):
                    current_mask = new_mask
                    current_peak, current_uncovered, current_counts = new_peak, new_uncovered, new_counts
                    current_smooth = new_smooth
                    current_obj = new_obj
                    
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_peak = current_peak
                        best_uncovered = current_uncovered
                        best_counts = current_counts.copy()
                        best_A_mask = current_mask.copy()
            
            T_polish *= cooling_polish
    
    # Return the best set found
    best_A_list = np.where(best_A_mask == 1)[0].tolist()
    
    # Ensure non-empty
    if not best_A_list:
        return [128]
        
    return best_A_list

# EVOLVE-BLOCK-END
