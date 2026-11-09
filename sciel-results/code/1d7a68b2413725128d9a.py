import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin for 90s limit
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128

    def evaluate(A):
        """
        Evaluates the set A using convolution for speed.
        Returns (peak, is_covered).
        peak: max multiplicity in [128, 256]
        is_covered: True if all sums in [128, 256] have count > 0
        """
        if not A:
            return float('inf'), False
            
        A_arr = np.array(list(A), dtype=np.int32)
        
        # Create indicator vector for A
        indicator = np.zeros(MAX_VAL + 1, dtype=np.int32)
        indicator[A_arr] = 1
        
        # Convolution gives count of ordered pairs (a, b) with a + b = s
        conv = np.convolve(indicator, indicator, mode='full')
        
        # Extract counts for target range [128, 256]
        # conv index s corresponds to sum s
        target_sums = np.arange(N_LOW, N_HIGH + 1)
        conv_target = conv[target_sums]
        
        # Adjust for unordered pairs
        # For sum s:
        # If s is even, mid = s/2. Unordered count = (conv[s] + indicator[mid]) / 2
        # If s is odd, Unordered count = conv[s] / 2
        # We can vectorize this:
        # mask_even = (target_sums % 2 == 0)
        # For even sums, we need to add indicator[s//2] to conv[s] before dividing by 2.
        
        counts_ordered = conv_target.astype(np.float64)
        
        # Identify even sums
        even_mask = (target_sums % 2 == 0)
        
        if np.any(even_mask):
            even_sums = target_sums[even_mask]
            mids = even_sums // 2
            # Ensure mids are within bounds [0, MAX_VAL]
            valid_mid_mask = (mids >= 0) & (mids <= MAX_VAL)
            # Only adjust if mid is valid
            # For invalid mids (which shouldn't happen for 128..256 given 0..128 range? 
            # Min sum 0+0=0, Max 128+128=256. 
            # For s=128, mid=64. For s=256, mid=128. All mids are in [64, 128], which is in [0, 128].
            # So valid_mid_mask should be all True.
            
            # Add indicator[mid] to conv[s] for even sums
            # We need to map mids back to indicator
            # indicator is 1D array.
            # We create a temporary array for the adjustment
            adj = np.zeros_like(counts_ordered)
            # Get indicator values at mid positions
            # mids are indices into indicator
            mid_indicators = indicator[mids]
            adj[even_mask] = mid_indicators
            
            counts_ordered += adj
        
        counts = counts_ordered / 2.0
        
        # Due to floating point, we might have 0.5 etc, but since counts are integers in reality,
        # and conv is integer, (conv+ind)/2 is integer.
        # Let's cast to int to be safe for comparison, but keep float for peak if needed?
        # Peak is integer.
        peak = int(np.max(counts))
        is_covered = bool(np.all(counts > 0))
        
        return peak, is_covered

    best_peak = float('inf')
    best_A = None

    # Strategy: Prune-Down
    # 1. Start with a valid set (e.g., all 0..128)
    # 2. Try to remove elements.
    # 3. If removal breaks coverage, try swaps.
    
    # We will run a few restarts with different initial "dense" sets
    # A full set [0, 128] has high multiplicity. 
    # A random subset of size ~60-80 might be valid and have lower peak.
    
    restarts = 10
    
    for r in range(restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialize A
        # Start with a random subset of size ~ 80 to ensure validity but not too redundant
        # Or start with full set and prune? Full set pruning is safe.
        # Let's try starting with a random valid set.
        # To ensure validity, we can start with full set and remove random elements until it's "small enough" or use a heuristic.
        
        # Heuristic: Start with all elements.
        A = set(range(0, 129))
        
        # Optional: Randomly remove some initially to jump to a different region?
        # Let's stick to systematic pruning for the first few restarts, then random init.
        
        if r > 3:
             # Random initialization
             # Pick a random subset of size 70
             # Check if valid. If not, add random elements until valid.
             A = set(random.sample(range(0, 129), 70))
             current_peak, is_covered = evaluate(A)
             attempts = 0
             while not is_covered and attempts < 100:
                 # Add random element
                 A.add(random.randint(0, 128))
                 current_peak, is_covered = evaluate(A)
                 attempts += 1
             if not is_covered:
                 continue # Skip this restart if we couldn't make it valid quickly
        
        current_peak, is_covered = evaluate(A)
        
        if not is_covered:
            continue
            
        if current_peak < best_peak:
            best_peak = current_peak
            best_A = list(A)
            
        # Pruning Phase
        # Try to remove elements.
        # We want to remove elements that are "least useful" for coverage or "most redundant" for peak.
        # Simple greedy: Try removing each element. If coverage remains, check new peak.
        # Accept removal if it doesn't increase peak, or if it increases peak only slightly?
        # Actually, we want to MINIMIZE peak. 
        # If removing x keeps coverage and reduces peak, DO IT.
        # If removing x keeps coverage and maintains peak, DO IT (to reduce size, which might help later swaps).
        # If removing x breaks coverage, SKIP.
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Sort elements to make process deterministic-ish or random?
            # Random order helps escape local minima in removal order
            elements_to_try = list(A)
            random.shuffle(elements_to_try)
            
            for x in elements_to_try:
                if time.time() > deadline - 10:
                    break
                    
                new_A = A.copy()
                new_A.discard(x)
                
                if not new_A:
                    continue
                    
                new_peak, is_covered = evaluate(new_A)
                
                if is_covered:
                    if new_peak < current_peak:
                        # Strict improvement in peak
                        A = new_A
                        current_peak = new_peak
                        improved = True
                        if current_peak < best_peak:
                            best_peak = current_peak
                            best_A = list(A)
                        break # Restart loop with new A
                    elif new_peak == current_peak:
                        # Same peak, smaller set. Accept to allow further optimization?
                        # Smaller set might have more "flexibility" for swaps to reduce peak later.
                        A = new_A
                        # current_peak remains same
                        improved = True
                        if current_peak < best_peak:
                            best_peak = current_peak
                            best_A = list(A)
                        break # Restart loop with new A
                    # If new_peak > current_peak, we don't accept removal (we want min peak)
            
        # Swap Phase
        # Now A is "minimal" in size (no single removal possible without breaking coverage or increasing peak).
        # Try swaps to reduce peak.
        T = 50.0 # Temperature for simulated annealing-like acceptance of swaps
        for step in range(200): # Limit steps per restart
            if time.time() > deadline - 10:
                break
                
            if not A:
                break
                
            x = random.choice(list(A))
            y = random.randint(0, 128)
            if x == y:
                continue
            if y in A:
                continue
                
            new_A = A.copy()
            new_A.discard(x)
            new_A.add(y)
            
            new_peak, is_covered = evaluate(new_A)
            
            if is_covered:
                delta = new_peak - current_peak
                if delta < 0:
                    # Improvement
                    A = new_A
                    current_peak = new_peak
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = list(A)
                else:
                    # Accept worse with prob exp(-delta/T)
                    if random.random() < np.exp(-delta / T):
                        A = new_A
                        current_peak = new_peak
            else:
                # Not covered, reject
                pass
                
            T *= 0.95

    if best_A is None:
        best_A = list(range(0, 129))

    return best_A
