import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128
    
    def calc_reps(A):
        """Vectorized representation calculation using numpy"""
        A_arr = np.array(list(A))
        n = len(A_arr)
        # Compute all pair sums (i <= j for unordered pairs)
        # Use broadcasting for efficiency
        i_idx = np.repeat(np.arange(n), np.arange(n, 0, -1))
        j_idx = np.tile(np.arange(n), np.concatenate([np.arange(n, 0, -1), [1]]))
        # Actually, let's use a simpler approach: compute upper triangle
        i_idx, j_idx = np.triu_indices(n, k=0)
        sums = A_arr[i_idx] + A_arr[j_idx]
        # Filter to target range
        mask = (sums >= N_LOW) & (sums <= N_HIGH)
        filtered_sums = sums[mask] - N_LOW
        # Count occurrences
        reps = np.bincount(filtered_sums, minlength=RANGE_SIZE)
        return reps
    
    def is_covered(reps):
        return np.all(reps > 0)
    
    def get_peak(reps):
        return int(np.max(reps))
    
    # Phase 1: Better initialization
    # Start with a random subset that likely covers the range
    # Strategy: sample elements to ensure good spread
    
    def random_covering_set():
        """Generate a random set that covers [128, 256]"""
        # Start with anchors
        A = {0, 128}
        # Add random elements until covered
        max_attempts = 500
        for _ in range(max_attempts):
            if is_covered(calc_reps(A)):
                break
            # Add a random element
            if len(A) < MAX_VAL + 1:
                A.add(random.randint(0, MAX_VAL))
        return A
    
    # Try a few random initializations and pick the best
    best_init = None
    best_init_peak = float('inf')
    
    for _ in range(5):
        A = random_covering_set()
        if is_covered(calc_reps(A)):
            peak = get_peak(calc_reps(A))
            if peak < best_init_peak:
                best_init_peak = peak
                best_init = A
                if peak <= 10:
                    break
    
    if best_init is None:
        # Fallback: use the original greedy approach
        A = {0, 128}
        while not is_covered(calc_reps(A)) and len(A) < MAX_VAL + 1:
            A.add(random.randint(0, MAX_VAL))
        best_init = A
        best_init_peak = get_peak(calc_reps(best_init))
    
    best_A = set(best_init)
    best_reps = calc_reps(best_A)
    best_peak = best_init_peak
    
    # Phase 2: Simulated annealing
    # Temperature schedule
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.999
    
    # Precompute some useful arrays
    all_vals = np.arange(MAX_VAL + 1)
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
        
        # Try a random move
        move_type = random.random()
        new_A = set(best_A)
        
        if move_type < 0.3:
            # Removal: try to remove an element
            if len(new_A) > 1:
                a = random.choice(list(new_A))
                new_A.remove(a)
        elif move_type < 0.6:
            # Single swap: remove a, add b
            if len(new_A) > 0:
                a = random.choice(list(new_A))
                non_A = [v for v in range(MAX_VAL + 1) if v not in best_A]
                if non_A:
                    b = random.choice(non_A)
                    new_A.remove(a)
                    new_A.add(b)
        elif move_type < 0.8:
            # Double swap: remove a1, a2, add b1, b2
            if len(new_A) >= 2:
                a1, a2 = random.sample(list(new_A), 2)
                non_A = [v for v in range(MAX_VAL + 1) if v not in best_A]
                if len(non_A) >= 2:
                    b1, b2 = random.sample(non_A, 2)
                    new_A.remove(a1)
                    new_A.remove(a2)
                    new_A.add(b1)
                    new_A.add(b2)
        else:
            # Perturbation: add a random element (to escape local minima)
            non_A = [v for v in range(MAX_VAL + 1) if v not in best_A]
            if non_A:
                b = random.choice(non_A)
                new_A.add(b)
        
        if not new_A:
            new_A = {0, 128}
        
        new_reps = calc_reps(new_A)
        
        if not is_covered(new_reps):
            # Reject uncovered sets
            T *= cooling_rate
            continue
        
        new_peak = get_peak(new_reps)
        delta = new_peak - best_peak
        
        if delta < 0:
            # Accept improvement
            best_A = new_A
            best_reps = new_reps
            best_peak = new_peak
        else:
            # Accept with probability exp(-delta/T)
            if random.random() < np.exp(-delta / T):
                best_A = new_A
                best_reps = new_reps
                best_peak = new_peak
        
        # Cool down
        T *= cooling_rate
        
        # Check if we've reached a very good solution
        if best_peak <= 3:
            break
    
    # Phase 3: Final local search refinement
    # Try to remove elements if possible
    for a in list(best_A):
        if time.time() > deadline - 2:
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
    
    # Try single swaps to further reduce peak
    for a in list(best_A):
        if time.time() > deadline - 1:
            break
        non_A = [v for v in range(MAX_VAL + 1) if v not in best_A]
        for b in non_A:
            if time.time() > deadline - 1:
                break
            new_A = best_A - {a}
            new_A.add(b)
            new_reps = calc_reps(new_A)
            if is_covered(new_reps):
                new_peak = get_peak(new_reps)
                if new_peak < best_peak:
                    best_A = new_A
                    best_reps = new_reps
                    best_peak = new_peak
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
