import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 128
    target_min = 128
    target_max = 256
    target_size = target_max - target_min + 1  # 129 targets
    
    # FFT-based count computation
    # We need counts of unordered pairs (a,b) with a<=b, a+b in [128,256]
    # Using convolution: if vec is indicator of A, conv = vec * vec gives ordered pair counts
    # For unordered: count_unordered[s] = (conv[s] + delta(s/2 in A and s even)) / 2
    
    def compute_counts_fft(A_set):
        """Compute unordered pair representation counts for sums in [target_min, target_max]."""
        if not A_set:
            return np.zeros(target_size, dtype=int)
        
        # Create indicator vector
        vec = np.zeros(n + 1, dtype=float)
        for a in A_set:
            vec[a] = 1.0
        
        # Convolution gives ordered pair counts
        conv = np.convolve(vec, vec, mode='full')
        
        # Convert to unordered counts
        counts = np.zeros(target_size, dtype=int)
        for idx, s in enumerate(range(target_min, target_max + 1)):
            ordered = conv[s]
            if s % 2 == 0:
                half = s // 2
                if 0 <= half <= n and half in A_set:
                    counts[idx] = (ordered + 1) // 2
                else:
                    counts[idx] = ordered // 2
            else:
                counts[idx] = ordered // 2
        
        return counts
    
    def get_peak_and_missing(counts):
        """Get peak multiplicity and number of missing targets."""
        if len(counts) == 0:
            return float('inf'), target_size
        missing = np.sum(counts == 0)
        peak = np.max(counts)
        return peak, missing
    
    # Initial set: use a structured approach
    # Start with [64, 128] which covers [128, 256] with all sums having multiple representations
    best_A = set(range(64, 129))
    best_counts = compute_counts_fft(best_A)
    best_peak, best_missing = get_peak_and_missing(best_counts)
    
    # Greedy peeling: try to remove elements while maintaining coverage
    def greedy_peel(A_init, max_time):
        """Greedily remove elements to minimize peak while maintaining full coverage."""
        A = set(A_init)
        counts = compute_counts_fft(A)
        peak, missing = get_peak_and_missing(counts)
        
        if missing > 0:
            return A, peak
        
        best_local_A = A.copy()
        best_local_peak = peak
        local_start = time.time()
        
        while time.time() < max_time:
            # Find the sum with maximum count
            max_idx = np.argmax(counts)
            max_val = counts[max_idx]
            
            if max_val <= 1:
                break  # Cannot reduce further
            
            # Try removing each element, prioritize those that appear in high-count sums
            A_list = sorted(A)
            improved = False
            best_candidate = None
            best_candidate_peak = peak
            best_candidate_size = len(A)
            
            for x in A_list:
                new_A = A.copy()
                new_A.discard(x)
                if not new_A:
                    continue
                
                new_counts = compute_counts_fft(new_A)
                new_peak, new_missing = get_peak_and_missing(new_counts)
                
                if new_missing > 0:
                    continue
                
                # Accept if: lower peak, or same peak but smaller size
                if new_peak < best_candidate_peak:
                    best_candidate = new_A
                    best_candidate_peak = new_peak
                    best_candidate_size = len(new_A)
                elif new_peak == best_candidate_peak and len(new_A) < best_candidate_size:
                    best_candidate = new_A
                    best_candidate_peak = new_peak
                    best_candidate_size = len(new_A)
            
            if best_candidate is not None:
                A = best_candidate
                counts = compute_counts_fft(A)
                peak, missing = get_peak_and_missing(counts)
                
                if peak < best_local_peak:
                    best_local_peak = peak
                    best_local_A = A.copy()
            else:
                break
        
        return best_local_A, best_local_peak
    
    # Run greedy peeling
    peeled_A, peeled_peak = greedy_peel(best_A, deadline - 10.0)
    
    if peeled_peak < best_peak:
        best_A = peeled_A
        best_peak = peeled_peak
        best_counts = compute_counts_fft(best_A)
    
    # Local search: try swaps to further reduce peak
    if time.time() < deadline - 5.0:
        current_A = best_A.copy()
        current_counts = compute_counts_fft(current_A)
        current_peak, current_missing = get_peak_and_missing(current_counts)
        
        rng = random.Random(42)
        no_improve_count = 0
        max_no_improve = 30
        
        while time.time() < deadline - 5.0 and no_improve_count < max_no_improve:
            improved = False
            in_A = list(current_A)
            not_in_A = [x for x in range(n + 1) if x not in current_A]
            
            # Try multiple random swaps
            for _ in range(100):
                if not in_A or not not_in_A:
                    break
                
                x = rng.choice(in_A)
                y = rng.choice(not_in_A)
                
                new_A = current_A.copy()
                new_A.discard(x)
                new_A.add(y)
                
                new_counts = compute_counts_fft(new_A)
                new_peak, new_missing = get_peak_and_missing(new_counts)
                
                if new_missing > 0:
                    continue
                
                if new_peak < current_peak:
                    current_A = new_A
                    current_counts = new_counts
                    current_peak = new_peak
                    
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = current_A.copy()
                        best_counts = current_counts.copy()
                    
                    improved = True
                    no_improve_count = 0
                    break
            
            if not improved:
                no_improve_count += 1
    
    # Final validation
    final_counts = compute_counts_fft(best_A)
    final_peak, final_missing = get_peak_and_missing(final_counts)
    
    if final_missing > 0:
        # Fallback to known valid set
        return list(range(64, 129))
    
    return list(best_A)
