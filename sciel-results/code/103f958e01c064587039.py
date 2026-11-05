import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    n = 128
    min_sum = 128
    max_sum = 256
    target_sums = np.arange(min_sum, max_sum + 1)
    
    def compute_peak_vec(v):
        """
        v: binary vector of length 129, v[i]=1 if i in A.
        Compute peak multiplicity over [128,256] for unordered pair sums.
        Returns (peak, uncovered_count).
        """
        # conv[s] = sum_i v[i]*v[s-i] = ordered pair count for sum s
        conv = np.convolve(v, v)
        # conv has length 257, indices 0..256
        
        # For target sums 128..256:
        # unordered_count[s] = (conv[s] + (v[s//2] if s%2==0 else 0)) // 2
        # But we need integer arithmetic. conv[s] and diag are integers.
        
        target_conv = conv[128:257]  # length 129
        # diag: for s in 128..256, diag[s] = v[s//2] if s even else 0
        s_vals = np.arange(128, 257)
        even_mask = (s_vals % 2 == 0)
        diag = np.zeros(129, dtype=np.int32)
        diag_indices = s_vals[even_mask] // 2
        diag[even_mask] = v[diag_indices]
        
        # unordered count
        counts = (target_conv + diag) // 2
        
        # Check coverage
        uncovered = np.sum(counts == 0)
        if uncovered > 0:
            return float('inf'), int(uncovered)
        
        peak = int(np.max(counts))
        return peak, 0

    def fitness_vec(v):
        peak, uncovered = compute_peak_vec(v)
        if uncovered > 0:
            return 10000 + uncovered * 100 + peak
        return peak

    def set_to_vec(A_set):
        v = np.zeros(129, dtype=np.int32)
        for x in A_set:
            v[x] = 1
        return v

    def vec_to_set(v):
        return set(np.nonzero(v)[0].tolist())

    # Phase 1: Generate diverse initial candidates as binary vectors
    best_v = None
    best_fit = float('inf')
    
    num_restarts = 30
    
    for r in range(num_restarts):
        if time.time() > deadline - 8:
            break
            
        # Generate initial binary vector
        if r % 3 == 0:
            # Random subset
            size = random.randint(18, 28)
            elems = random.sample(range(0, 129), size)
            v = np.zeros(129, dtype=np.int32)
            for x in elems:
                v[x] = 1
        elif r % 3 == 1:
            # Force 0 and 128
            v = np.zeros(129, dtype=np.int32)
            v[0] = 1
            v[128] = 1
            rest = random.sample(range(1, 128), random.randint(16, 24))
            for x in rest:
                v[x] = 1
        else:
            # Two blocks
            v = np.zeros(129, dtype=np.int32)
            low = random.sample(range(0, 30), random.randint(8, 12))
            high = random.sample(range(100, 129), random.randint(8, 12))
            for x in low + high:
                v[x] = 1
        
        current_v = v.copy()
        current_fit = fitness_vec(current_v)
        
        if current_fit < best_fit:
            best_fit = current_fit
            best_v = current_v.copy()
        
        # Simulated Annealing on binary vector
        T = 15.0
        cooling = 0.997
        min_T = 0.05
        
        while T > min_T and time.time() < deadline - 4:
            # Generate neighbor: flip 1-3 random bits
            new_v = current_v.copy()
            num_flips = random.choice([1, 1, 1, 2, 2, 3])
            for _ in range(num_flips):
                idx = random.randint(0, 128)
                new_v[idx] = 1 - new_v[idx]
            
            # Ensure non-empty
            if np.sum(new_v) == 0:
                new_v[random.randint(0, 128)] = 1
            
            new_fit = fitness_vec(new_v)
            
            # Acceptance
            delta = new_fit - current_fit
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_v = new_v
                current_fit = new_fit
                if current_fit < best_fit:
                    best_fit = current_fit
                    best_v = current_v.copy()
            
            T *= cooling
            
            # Tighten if good
            if current_fit < 5:
                T = max(min_T, T * 0.98)

    if best_v is None:
        # Fallback: return a valid covering set
        A = list(range(64, 129))
        return A
    
    # Final hill climb on best vector
    current_v = best_v.copy()
    current_fit = fitness_vec(current_v)
    
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        for _ in range(2000):
            if time.time() > deadline - 1:
                break
            new_v = current_v.copy()
            idx = random.randint(0, 128)
            new_v[idx] = 1 - new_v[idx]
            if np.sum(new_v) == 0:
                new_v[random.randint(0, 128)] = 1
            
            new_fit = fitness_vec(new_v)
            if new_fit < current_fit:
                current_v = new_v
                current_fit = new_fit
                improved = True
                if current_fit < best_fit:
                    best_fit = current_fit
                    best_v = current_v.copy()
    
    return sorted(vec_to_set(best_v))
