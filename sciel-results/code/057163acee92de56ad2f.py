import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin for 60s limit
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1  # 129 sums
    MAX_VAL = 128
    
    # Precompute FFT size for convolution
    conv_size = 129 + 129 - 1  # 257
    fft_n = 1
    while fft_n < conv_size:
        fft_n *= 2
    
    def calc_reps(A_set):
        """Compute unordered pair representation counts for sums 128..256 using FFT."""
        ind = np.zeros(129, dtype=np.float64)
        for a in A_set:
            ind[a] = 1.0
        
        f = np.fft.rfft(ind, n=fft_n)
        conv = np.fft.irfft(f * f, n=fft_n)[:conv_size].round().astype(np.int64)
        
        ordered = conv[N_LOW:N_HIGH+1]  # length 129
        
        reps = np.zeros(RANGE_SIZE, dtype=np.int64)
        for idx in range(RANGE_SIZE):
            s = N_LOW + idx
            if s % 2 == 0:
                half = s // 2
                diag = 1.0 if (0 <= half <= 128 and ind[half] > 0) else 0.0
                reps[idx] = (ordered[idx] + diag) // 2
            else:
                reps[idx] = ordered[idx] // 2
        
        return reps
    
    def is_covered(reps):
        return np.all(reps > 0)
    
    def get_peak(reps):
        return int(np.max(reps))
    
    # Phase 1: Greedy construction - build A to cover [128,256] with minimal peak
    # Strategy: start with required elements, then add elements that best balance coverage
    best_A = None
    best_peak = float('inf')
    
    # Start with elements needed for extreme sums
    # 128 = 0+128, 256 = 128+128
    required = {0, 128}
    
    # Try multiple random seeds for diversity
    for seed in range(20):
        if time.time() > deadline - 35:
            break
        
        random.seed(seed)
        A = set(required)
        
        # Track which sums are covered
        reps = calc_reps(A)
        
        # Greedily add elements to cover uncovered sums and balance multiplicity
        for step in range(60):
            if time.time() > deadline - 35:
                break
            
            # Find uncovered sums
            uncovered = np.where(reps == 0)[0]
            
            if len(uncovered) == 0:
                break
            
            # For each uncovered sum, find candidate elements that would help
            # Score each candidate by how many uncovered sums it helps and how much it increases peak
            best_candidate = None
            best_score = float('-inf')
            
            # Sample candidates for efficiency
            candidates = list(range(0, 129))
            random.shuffle(candidates)
            candidates = candidates[:50]
            
            for c in candidates:
                if c in A:
                    continue
                
                # Compute new reps if we add c
                new_A = set(A)
                new_A.add(c)
                new_reps = calc_reps(new_A)
                
                if not is_covered(new_reps):
                    # Count how many previously uncovered are now covered
                    new_uncovered = np.where(new_reps == 0)[0]
                    newly_covered = len(uncovered) - len(new_uncovered)
                else:
                    newly_covered = len(uncovered)
                
                new_peak = get_peak(new_reps)
                
                # Score: prefer covering more sums, penalize high peak
                score = newly_covered * 10 - new_peak
                
                if score > best_score:
                    best_score = score
                    best_candidate = c
            
            if best_candidate is not None:
                A.add(best_candidate)
                reps = calc_reps(A)
            else:
                break
        
        # Check if we have a valid solution
        if is_covered(reps):
            peak = get_peak(reps)
            if peak < best_peak:
                best_peak = peak
                best_A = list(A)
    
    if best_A is None:
        # Fallback: use a structured set
        return list(range(64, 129))
    
    current_A = set(best_A)
    current_reps = calc_reps(current_A)
    current_peak = get_peak(current_reps)
    
    # Phase 2: Local search with swaps to reduce peak
    T = 0.5
    cooling_rate = 0.9995
    min_T = 0.05
    stuck_count = 0
    
    while time.time() < deadline - 15 and T > min_T:
        # Swap move: remove one element, add another
        if len(current_A) == 0:
            break
        
        # Focus on removing elements that contribute to peak sums
        peak_sums = np.where(current_reps == current_peak)[0]
        
        if len(peak_sums) > 0:
            peak_sum_vals = [N_LOW + idx for idx in peak_sums[:5]]
            candidate_removals = []
            for s in peak_sum_vals:
                for a in current_A:
                    b = s - a
                    if b in current_A and a <= b:  # Avoid duplicates
                        candidate_removals.append(a)
                        break
            if candidate_removals:
                item_to_remove = random.choice(candidate_removals)
            else:
                item_to_remove = random.choice(list(current_A))
        else:
            item_to_remove = random.choice(list(current_A))
        
        new_item = random.randint(0, 128)
        
        if new_item == item_to_remove:
            T *= cooling_rate
            continue
            
        new_A = set(current_A)
        new_A.remove(item_to_remove)
        new_A.add(new_item)
        
        new_reps = calc_reps(new_A)
        
        if is_covered(new_reps):
            new_peak = get_peak(new_reps)
            delta_peak = new_peak - current_peak
            
            if delta_peak < 0:
                current_A = new_A
                current_reps = new_reps
                current_peak = new_peak
                stuck_count = 0
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_A = list(current_A)
            else:
                prob = np.exp(-delta_peak / max(T, 0.01))
                if random.random() < prob:
                    current_A = new_A
                    current_reps = new_reps
                    current_peak = new_peak
                    stuck_count = 0
                else:
                    stuck_count += 1
        else:
            stuck_count += 1
        
        T *= cooling_rate
        
        if stuck_count > 100:
            # Perturb
            for _ in range(2):
                if len(current_A) > 0:
                    item = random.choice(list(current_A))
                    current_A.remove(item)
                    new_item = random.randint(0, 128)
                    current_A.add(new_item)
            current_reps = calc_reps(current_A)
            if is_covered(current_reps):
                current_peak = get_peak(current_reps)
            else:
                current_A = set(best_A)
                current_reps = calc_reps(current_A)
                current_peak = get_peak(current_reps)
            stuck_count = 0
            T = max(T, 0.2)
    
    # Phase 3: Greedy peeling - remove elements while maintaining coverage
    if best_A is not None:
        final_A = set(best_A)
        final_reps = calc_reps(final_A)
        if is_covered(final_reps):
            # Try to remove elements one by one, prioritizing those that don't increase peak
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                # Sort by contribution to peak sums
                contribs = []
                for a in final_A:
                    # Count how many peak sums this element participates in
                    count = 0
                    for idx in np.where(final_reps == get_peak(final_reps))[0]:
                        s = N_LOW + idx
                        b = s - a
                        if b in final_A:
                            count += 1
                    contribs.append((count, a))
                
                # Try removing elements with lowest contribution first
                contribs.sort()
                for count, a in contribs:
                    if len(final_A) <= 15:
                        break
                    test_A = set(final_A)
                    test_A.remove(a)
                    test_reps = calc_reps(test_A)
                    if is_covered(test_reps):
                        test_peak = get_peak(test_reps)
                        if test_peak <= get_peak(final_reps):
                            final_A = test_A
                            final_reps = test_reps
                            changed = True
                            break
            
            if get_peak(final_reps) < best_peak:
                best_peak = get_peak(final_reps)
                best_A = list(final_A)
    
    if best_A is None:
        return list(range(64, 129))
        
    return best_A
