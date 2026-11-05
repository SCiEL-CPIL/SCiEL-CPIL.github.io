import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 128
    min_sum = 128
    max_sum = 256
    target_sums = range(min_sum, max_sum + 1)
    fft_size = 256  # Next power of 2 >= 2*128+1
    
    # Precomputed indices for diagonal check
    even_sums = np.arange(0, 257, 2)
    halves = even_sums // 2
    
    def get_counts(in_A):
        """
        in_A: numpy array of shape (129,) with 0/1 values.
        Returns: numpy array of shape (257,) with unordered pair counts.
        """
        v = np.zeros(fft_size, dtype=np.float64)
        v[:129] = in_A
        
        fft_v = np.fft.fft(v)
        conv = np.fft.ifft(fft_v * fft_v).real
        
        # Round to nearest integer to handle floating point errors
        c_ord = np.round(conv[:257]).astype(np.int32)
        
        # Diagonal correction
        # diag_full[s] is 1 if s is even and s/2 is in A, else 0
        diag_flags = in_A[halves].astype(np.int32)
        diag_full = np.zeros(257, dtype=np.int32)
        diag_full[even_sums] = diag_flags
        
        # Unordered count = (Ordered + Diagonal) / 2
        return (c_ord + diag_full) // 2

    def evaluate(in_A):
        """
        Returns (is_valid, peak_multiplicity)
        """
        counts = get_counts(in_A)
        # Check coverage: all counts in [128, 256] must be > 0
        target_counts = counts[min_sum:max_sum+1]
        if np.any(target_counts == 0):
            return False, float('inf')
        peak = np.max(target_counts)
        return True, peak

    best_peak = float('inf')
    best_in_A = None
    
    # 1. Generate Initial Candidates
    candidates = []
    
    # Seed: range(64, 129)
    seed_A = np.zeros(129, dtype=np.int32)
    seed_A[64:129] = 1
    candidates.append(seed_A.copy())
    
    # Random sets
    random.seed(42)
    for _ in range(20):
        size = random.randint(20, 50)
        idx = random.sample(range(129), size)
        arr = np.zeros(129, dtype=np.int32)
        arr[idx] = 1
        candidates.append(arr)
        
    # Structured: Sparse APs
    for step in range(2, 15):
        arr = np.zeros(129, dtype=np.int32)
        for i in range(0, 129, step):
            arr[i] = 1
        candidates.append(arr)
        
    # Union of two sparse APs
    for step1 in range(2, 10):
        for step2 in range(2, 10):
            if step1 == step2: continue
            arr = np.zeros(129, dtype=np.int32)
            for i in range(0, 129, step1):
                arr[i] = 1
            for i in range(1, 129, step2):
                arr[i] = 1
            candidates.append(arr)

    # Evaluate all candidates
    for cand in candidates:
        valid, peak = evaluate(cand)
        if valid and peak < best_peak:
            best_peak = peak
            best_in_A = cand.copy()

    if best_in_A is None:
        # Fallback to seed if no valid solution found (should not happen)
        return list(range(64, 129))

    # 2. Simulated Annealing
    current_in_A = best_in_A.copy()
    current_valid, current_peak = evaluate(current_in_A)
    
    def calc_energy(in_A, valid, peak):
        if valid:
            return float(peak)
        else:
            counts = get_counts(in_A)
            uncovered = np.sum(counts[min_sum:max_sum+1] == 0)
            return 100.0 + float(uncovered)

    current_energy = calc_energy(current_in_A, current_valid, current_peak)
    
    best_in_A = current_in_A.copy()
    best_peak_sa = current_peak if current_valid else float('inf')
    
    T = 5.0
    cooling_rate = 0.9995
    min_T = 0.001
    
    # Main SA Loop
    while time.time() < deadline - 1.0 and T > min_T:
        idx = random.randint(0, 128)
        
        # Flip bit
        old_val = current_in_A[idx]
        current_in_A[idx] = 1 - old_val
        
        valid, peak = evaluate(current_in_A)
        new_energy = calc_energy(current_in_A, valid, peak)
        
        delta = new_energy - current_energy
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            # Accept move
            current_energy = new_energy
            if valid and peak < best_peak_sa:
                best_peak_sa = peak
                best_in_A = current_in_A.copy()
        else:
            # Reject move, revert
            current_in_A[idx] = old_val
            
        T *= cooling_rate
        
    # Update global best if SA found something better
    if best_peak_sa < best_peak:
        best_peak = best_peak_sa
        best_in_A = best_in_A

    # Convert numpy array to list of indices
    return list(np.where(best_in_A == 1)[0])

