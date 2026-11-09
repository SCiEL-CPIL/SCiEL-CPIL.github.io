import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128
    
    def calc_reps(A):
        # Fast convolution-based representation counting
        A_list = list(A)
        # Create indicator array
        arr = np.zeros(MAX_VAL + 1, dtype=np.int32)
        for a in A_list:
            arr[a] = 1
        
        # Convolution gives ordered pair counts; adjust for unordered with equal summands
        conv = np.convolve(arr, arr, mode='full')
        # conv[s] = number of ordered pairs (i,j) with i+j=s
        # For unordered pairs: if i != j, count is conv[s]/2; if i==j (s even, i=s/2), add 1
        reps = np.zeros(RANGE_SIZE, dtype=np.int32)
        for idx in range(RANGE_SIZE):
            s = N_LOW + idx
            if s % 2 == 0:
                half = s // 2
                if 0 <= half <= MAX_VAL:
                    # Ordered count includes (half, half) once
                    # Unordered: (conv[s] - 1) / 2 + 1 = (conv[s] + 1) / 2
                    reps[idx] = (conv[s] + 1) // 2
                else:
                    reps[idx] = conv[s] // 2
            else:
                # No equal summands possible
                reps[idx] = conv[s] // 2
        return reps.tolist()

    def is_covered(reps):
        return all(r > 0 for r in reps)

    def get_peak(reps):
        return max(reps)

    # Phase 1: Better initialization using structured approach
    # Start with a dense set that's likely to cover well
    # Use a mix of small and large values
    A = set()
    
    # Add anchors
    A.add(0)
    A.add(128)
    
    # Add a structured base to ensure good coverage
    # Use every k-th value in certain ranges
    for v in range(0, 65, 4):
        A.add(v)
    for v in range(64, 129, 4):
        A.add(v)
    
    # Ensure coverage with greedy additions
    max_greedy_iters = 200
    for _ in range(max_greedy_iters):
        if time.time() > deadline:
            break
        reps = calc_reps(A)
        if is_covered(reps):
            break
        
        missing_indices = [i for i, r in enumerate(reps) if r == 0]
        if not missing_indices:
            break
            
        # Pick the most "constrained" missing sum (one with fewest candidates)
        best_idx = None
        best_candidate_count = float('inf')
        for idx in missing_indices:
            S = N_LOW + idx
            candidates = []
            for z in A:
                y = S - z
                if 0 <= y <= MAX_VAL and y not in A:
                    candidates.append(y)
            if len(candidates) < best_candidate_count:
                best_candidate_count = len(candidates)
                best_idx = idx
            if best_candidate_count == 1:
                break
        
        if best_idx is None:
            break
            
        S = N_LOW + best_idx
        candidates = []
        for z in A:
            y = S - z
            if 0 <= y <= MAX_VAL and y not in A:
                candidates.append(y)
        
        if candidates:
            # Prefer candidates that help with other missing sums
            y = random.choice(candidates)
            A.add(y)
        else:
            non_A = [v for v in range(MAX_VAL + 1) if v not in A]
            if non_A:
                A.add(random.choice(non_A))
            else:
                break
    
    if not is_covered(calc_reps(A)):
        # Fallback: use a known-good dense set
        A = set(range(0, 129, 2))
        if not is_covered(calc_reps(A)):
            A = set(range(64, 129))
    
    # Quick local search to improve initial solution
    for _ in range(50):
        if time.time() > deadline:
            break
        improved = False
        for a in list(A):
            if time.time() > deadline:
                break
            new_A = A - {a}
            if not new_A:
                continue
            new_reps = calc_reps(new_A)
            if is_covered(new_reps):
                new_peak = max(new_reps)
                old_peak = max(calc_reps(A))
                if new_peak <= old_peak:
                    A = new_A
                    improved = True
        if not improved:
            break

    # Phase 2: Optimization with Simulated Annealing (faster with incremental updates)
    best_A = set(A)
    best_reps = calc_reps(best_A)
    best_peak = get_peak(best_reps)
    
    current_A = set(best_A)
    current_reps = calc_reps(current_A)
    current_peak = get_peak(current_reps)
    
    T = 5.0
    T_min = 0.1
    cooling = 0.998
    
    # Precompute for faster evaluation
    def quick_swap_eval(A_set, remove_val, add_val):
        """Fast evaluation of swapping remove_val with add_val in A_set"""
        # This is still O(n) but we can early exit
        new_A = A_set - {remove_val} | {add_val}
        reps = calc_reps(new_A)
        if not all(r > 0 for r in reps):
            return None, None
        return reps, max(reps)
    
    while time.time() < deadline and T > T_min:
        if not current_A:
            break
            
        a = random.choice(list(current_A))
        
        # Find b not in A
        b = random.randint(0, MAX_VAL)
        attempts = 0
        while b in current_A and attempts < 10:
            b = random.randint(0, MAX_VAL)
            attempts += 1
        if b in current_A:
            T *= cooling
            continue
        
        # Perform swap
        new_reps, new_peak = quick_swap_eval(current_A, a, b)
        
        if new_reps is None:
            T *= cooling
            continue
            
        delta = new_peak - current_peak
        
        accept = False
        if delta < 0:
            accept = True
        elif delta == 0:
            accept = random.random() < 0.3
        else:
            if T > 0:
                accept = random.random() < math.exp(-delta / T)
        
        if accept:
            current_A = current_A - {a} | {b}
            current_reps = new_reps
            current_peak = new_peak
            
            if current_peak < best_peak:
                best_peak = current_peak
                best_A = set(current_A)
                best_reps = current_reps
        
        T *= cooling
        
    # Final Polish: Greedy Removals and Swaps
    improved = True
    polish_iterations = 0
    while improved and time.time() < deadline and polish_iterations < 10:
        improved = False
        polish_iterations += 1
        
        # Try removals
        for a in list(best_A):
            if time.time() > deadline:
                break
            new_A = best_A - {a}
            if not new_A:
                continue
            new_reps = calc_reps(new_A)
            if is_covered(new_reps):
                new_peak = get_peak(new_reps)
                if new_peak <= best_peak:
                    best_A = new_A
                    best_reps = new_reps
                    best_peak = new_peak
                    improved = True
        
        # Try swaps to reduce peak
        if time.time() < deadline:
            for a in list(best_A):
                if time.time() > deadline:
                    break
                # Try replacing a with values that might help
                for b in range(0, MAX_VAL + 1, 3):  # Sample every 3rd value
                    if b in best_A:
                        continue
                    new_A = best_A - {a} | {b}
                    new_reps = calc_reps(new_A)
                    if is_covered(new_reps):
                        new_peak = get_peak(new_reps)
                        if new_peak < best_peak:
                            best_A = new_A
                            best_reps = new_reps
                            best_peak = new_peak
                            improved = True
                            break
                if time.time() > deadline:
                    break

    return list(best_A)

if __name__ == "__main__":
    t0 = time.time()
    A = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, |A|={len(A)}")
    reps = [0]*129
    for i in range(len(A)):
        for j in range(i, len(A)):
            s = A[i] + A[j]
            if 128 <= s <= 256:
                reps[s-128] += 1
    print(f"Covered: {all(r>0 for r in reps)}, Peak: {max(reps)}")
