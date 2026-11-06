import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin from 90s limit
    
    n = 128
    target_low = 128
    target_high = 256
    target_size = target_high - target_low + 1  # 129 targets
    
    # Fast multiplicity calculation using NumPy
    def get_multiplicities(A_set):
        A_arr = np.array(sorted(A_set), dtype=np.int32)
        # Compute all pair sums (i <= j)
        # Use broadcasting for efficiency
        a, b = np.meshgrid(A_arr, A_arr, indexing='ij')
        sums = a + b
        # Only consider i <= j to avoid double counting
        mask = a <= b
        pair_sums = sums[mask]
        # Count occurrences in target range
        in_range = (pair_sums >= target_low) & (pair_sums <= target_high)
        valid_sums = pair_sums[in_range]
        if len(valid_sums) == 0:
            return np.zeros(target_size, dtype=np.int32)
        # Use bincount to count
        counts = np.bincount(valid_sums - target_low, minlength=target_size)
        return counts
    
    def get_peak(mults):
        return int(np.max(mults))
    
    def is_covered(mults):
        return bool(np.all(mults > 0))
    
    def calc_score(A_set):
        m = get_multiplicities(A_set)
        if not is_covered(m):
            return (999, 0)  # Invalid
        peak = int(np.max(m))
        sq_sum = int(np.sum(m * m))
        return (peak, sq_sum)
    
    # Phase 1: Start with a smaller initial set and try to build up
    # Try different initial configurations
    
    best_A = None
    best_score = (999, 0)
    
    # Strategy: Start with various random subsets and optimize
    random.seed(42)
    
    # Initial candidate sets to try
    initial_sets = []
    
    # Contiguous block [64, 128]
    initial_sets.append(set(range(64, 129)))
    
    # Two intervals: lower and upper parts
    # A = [0, 32] ∪ [96, 128] -> sums cover [0,64] ∪ [96,192] ∪ [192,256]
    # Not quite right. Let me think...
    # We need sums in [128, 256]
    # If A has elements in [0, 64] and [64, 128]:
    #   [0,64] + [0,64] = [0, 128]
    #   [0,64] + [64,128] = [64, 192]
    #   [64,128] + [64,128] = [128, 256]
    # So [64,128] + [64,128] covers [128,256]
    
    # Try smaller sets by removing from the contiguous block
    # First, let's do greedy removal from [64, 128]
    
    A = set(range(64, 129))
    
    # Greedy removal
    while True:
        if time.time() > deadline - 30:
            break
            
        current_mults = get_multiplicities(A)
        if not is_covered(current_mults):
            break
            
        current_peak = get_peak(current_mults)
        current_sq = int(np.sum(current_mults * current_mults))
        current_score = (current_peak, current_sq)
        
        best_candidate = None
        best_candidate_score = current_score
        
        # Try removing each element
        A_list = list(A)
        random.shuffle(A_list)
        
        for a in A_list:
            if time.time() > deadline - 30:
                break
                
            A_temp = A - {a}
            if not A_temp:
                continue
                
            mults_temp = get_multiplicities(A_temp)
            if not is_covered(mults_temp):
                continue
            
            peak_temp = get_peak(mults_temp)
            sq_temp = int(np.sum(mults_temp * mults_temp))
            score_temp = (peak_temp, sq_temp)
            
            if score_temp < best_candidate_score:
                best_candidate_score = score_temp
                best_candidate = a
                
        if best_candidate is None:
            break
            
        A.remove(best_candidate)
    
    best_A = set(A)
    best_score = calc_score(A)
    
    # Phase 2: Simulated Annealing / Local Search with swaps
    # Try to improve by swapping elements
    
    def try_improve(A_set, max_time):
        A = set(A_set)
        current_mults = get_multiplicities(A)
        if not is_covered(current_mults):
            return None, (999, 0)
        
        current_peak = get_peak(current_mults)
        current_sq = int(np.sum(current_mults * current_mults))
        current_score = (current_peak, current_sq)
        
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < max_time and no_improve_count < max_no_improve:
            improved = False
            A_list = list(A)
            random.shuffle(A_list)
            
            for a_out in A_list:
                if time.time() > max_time:
                    break
                    
                # Try adding random elements
                candidates = [x for x in range(0, 129) if x not in A]
                random.shuffle(candidates)
                
                # Limit candidates for speed
                for a_in in candidates[:50]:
                    if time.time() > max_time:
                        break
                        
                    A_temp = A - {a_out}
                    A_temp.add(a_in)
                    
                    mults_temp = get_multiplicities(A_temp)
                    if not is_covered(mults_temp):
                        continue
                    
                    peak_temp = get_peak(mults_temp)
                    sq_temp = int(np.sum(mults_temp * mults_temp))
                    score_temp = (peak_temp, sq_temp)
                    
                    if score_temp < current_score:
                        A = A_temp
                        current_score = score_temp
                        current_mults = mults_temp
                        improved = True
                        no_improve_count = 0
                        break
                else:
                    continue
                break
            
            if not improved:
                no_improve_count += 1
        
        return A, current_score
    
    # Try multiple random restarts
    num_restarts = 10
    restart_time = (deadline - 5 - time.time()) / num_restarts if deadline - 5 > time.time() else 1
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Create a perturbed version of best_A
        if best_A is None:
            A_restart = set(range(64, 129))
        else:
            A_restart = set(best_A)
            # Perturb: swap 1-3 random elements
            if A_restart:
                num_swaps = random.randint(1, 3)
                for _ in range(num_swaps):
                    if not A_restart:
                        break
                    a_out = random.choice(list(A_restart))
                    candidates = [x for x in range(0, 129) if x not in A_restart]
                    if not candidates:
                        break
                    a_in = random.choice(candidates)
                    A_restart.remove(a_out)
                    A_restart.add(a_in)
        
        # Greedy removal from this starting point
        A_greedy = set(A_restart)
        while True:
            if time.time() > deadline - 10:
                break
            current_mults = get_multiplicities(A_greedy)
            if not is_covered(current_mults):
                break
            current_peak = get_peak(current_mults)
            current_sq = int(np.sum(current_mults * current_mults))
            current_score = (current_peak, current_sq)
            
            best_candidate = None
            best_candidate_score = current_score
            
            A_list = list(A_greedy)
            random.shuffle(A_list)
            
            for a in A_list:
                if time.time() > deadline - 10:
                    break
                A_temp = A_greedy - {a}
                if not A_temp:
                    continue
                mults_temp = get_multiplicities(A_temp)
                if not is_covered(mults_temp):
                    continue
                peak_temp = get_peak(mults_temp)
                sq_temp = int(np.sum(mults_temp * mults_temp))
                score_temp = (peak_temp, sq_temp)
                if score_temp < best_candidate_score:
                    best_candidate_score = score_temp
                    best_candidate = a
            
            if best_candidate is None:
                break
            A_greedy.remove(best_candidate)
        
        # Local search
        A_improved, score_improved = try_improve(A_greedy, deadline - 5)
        
        if A_improved is not None and score_improved < best_score:
            best_A = A_improved
            best_score = score_improved
    
    # Final local search on the best found
    if best_A is not None:
        A_final, score_final = try_improve(best_A, deadline - 2)
        if A_final is not None and score_final < best_score:
            best_A = A_final
            best_score = score_final
    
    if best_A is None:
        # Fallback: return the contiguous block
        return list(range(64, 129))
    
    return list(best_A)
