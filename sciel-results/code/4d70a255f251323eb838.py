import time
import math
import numpy as np

def compute_score(U):
    U = sorted(set(U))
    if not U or U[0] != 0:
        return -1
    max_u = max(U)
    if max_u == 0:
        return 1.0
    # Compute U+U
    sum_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
    # Compute U-U
    diff_set = set()
    for a in U:
        for b in U:
            diff_set.add(a - b)
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return -1
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return -1
    return 1 + math.log(ratio) / denom

def compute_score_fast(U):
    """Compute score using FFT for larger sets, falling back to naive for small sets."""
    U = sorted(set(U))
    if not U or U[0] != 0:
        return -1
    max_u = max(U)
    if max_u == 0:
        return 1.0
    n = len(U)
    
    if n <= 50:
        # Naive computation for small sets
        sum_set = set()
        for a in U:
            for b in U:
                sum_set.add(a + b)
        diff_set = set()
        for a in U:
            for b in U:
                diff_set.add(a - b)
        ratio = len(diff_set) / len(sum_set)
        if ratio <= 0:
            return -1
        denom = math.log(2 * max_u + 1)
        if denom <= 0:
            return -1
        return 1 + math.log(ratio) / denom
    
    # FFT-based computation for larger sets
    # For sumset: convolve indicator of U with itself
    # For diffset: convolve indicator of U with reversed indicator
    
    # Sumset: U+U = {a+b : a,b in U}
    # Create indicator array
    size = max_u + 1
    # Pad to power of 2 for efficient FFT
    fft_size = 1
    while fft_size < 2 * size:
        fft_size *= 2
    
    ind = np.zeros(fft_size)
    for u in U:
        ind[u] = 1.0
    
    # Convolution for sumset
    fft_ind = np.fft.fft(ind)
    conv_sum = np.fft.ifft(fft_ind * fft_ind).real
    # Count non-zero entries in valid range [0, 2*max_u]
    sum_count = int(np.sum(conv_sum[:2*max_u+1] > 0.5))
    
    # Diffset: U-U = {a-b : a,b in U}
    # This is convolution of U with reversed U
    # Reverse indicator: ind_rev[i] = ind[size-1-i] for i in range(size)
    ind_rev = np.zeros(fft_size)
    for u in U:
        ind_rev[size - 1 - u] = 1.0
    
    fft_rev = np.fft.fft(ind_rev)
    conv_diff = np.fft.ifft(fft_ind * fft_rev).real
    # The diffset values range from -max_u to max_u
    # In the convolution, value (a-b) appears at index (a + (size-1-b)) = a - b + size - 1
    # So diff value d appears at index d + size - 1
    # We need indices from size-1-max_u to size-1+max_u
    start_idx = size - 1 - max_u
    end_idx = size - 1 + max_u + 1
    if start_idx < 0:
        start_idx = 0
    if end_idx > fft_size:
        end_idx = fft_size
    diff_count = int(np.sum(conv_diff[start_idx:end_idx] > 0.5))
    
    ratio = diff_count / sum_count
    if ratio <= 0:
        return -1
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return -1
    return 1 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_U = [0, 1, 3]
    best_score = compute_score(best_U)
    
    def try_set(U):
        nonlocal best_U, best_score
        U = sorted(set(U))
        if not U or U[0] != 0:
            return
        s = compute_score_fast(U)
        if s > best_score:
            best_score = s
            best_U = U
    
    def time_remaining():
        return deadline - time.time()
    
    # Construction 1: Blocks with growing gaps (Golomb-ruler-like)
    for k in range(1, 20):
        if time_remaining() < 5:
            break
        U = set()
        for i in range(k + 1):
            base = 0
            for j in range(i):
                base += 2 * (3 ** j) + 1
            U.add(base)
            U.add(base + 1)
        try_set(list(U))
    
    # Construction 2: Pairs with cubic gaps
    for k in range(1, 15):
        if time_remaining() < 5:
            break
        U = set()
        pos = 0
        for i in range(k + 1):
            U.add(pos)
            U.add(pos + 1)
            pos += 2 + 2 * (3 ** i)
        try_set(list(U))
    
    # Construction 3: Sidon-like sets with perturbation
    for n in range(2, 25):
        if time_remaining() < 5:
            break
        for d in range(1, 8):
            if time_remaining() < 5:
                break
            U = [i * d for i in range(n)]
            for variant in range(3):
                Uv = list(U)
                for idx in range(1, n):
                    if (idx + variant) % 3 == 0:
                        Uv[idx] += 1
                try_set(Uv)
    
    # Local search with simulated annealing
    def local_search(U, max_iters=1000):
        nonlocal best_U, best_score
        U = sorted(set(U))
        if not U or U[0] != 0:
            return
        current_score = compute_score_fast(U)
        T = 0.01  # Initial temperature
        T_min = 1e-6
        cooling = 0.999
        
        for it in range(max_iters):
            if time_remaining() < 2:
                break
            
            improved = False
            max_u = max(U)
            n = len(U)
            
            # Generate candidates for addition
            candidates = set()
            for u in U:
                for delta in range(-4, 5):
                    c = u + delta
                    if c >= 0 and c not in U:
                        candidates.add(c)
            # Add elements above max
            for c in range(max_u + 1, max_u + 6):
                candidates.add(c)
            
            # Try adding a random candidate
            if candidates:
                c = min(candidates)  # Start with smallest, could randomize
                U_new = sorted(set(U + [c]))
                s_new = compute_score_fast(U_new)
                delta_s = s_new - current_score
                if delta_s > 0 or (T > T_min and np.random.random() < math.exp(delta_s / T)):
                    U = U_new
                    current_score = s_new
                    improved = True
                    if s_new > best_score:
                        best_score = s_new
                        best_U = list(U)
            
            # Try removing a random element (not 0)
            if not improved and len(U) > 2:
                idx = np.random.randint(1, len(U))
                U_new = sorted(set(U[:idx] + U[idx+1:]))
                if U_new and U_new[0] == 0:
                    s_new = compute_score_fast(U_new)
                    delta_s = s_new - current_score
                    if delta_s > 0 or (T > T_min and np.random.random() < math.exp(delta_s / T)):
                        U = U_new
                        current_score = s_new
                        improved = True
                        if s_new > best_score:
                            best_score = s_new
                            best_U = list(U)
            
            # Try swapping: remove one, add another
            if not improved and len(U) > 2:
                remove_idx = np.random.randint(1, len(U))
                candidates_add = set()
                for u in U:
                    for delta in range(-5, 6):
                        c = u + delta
                        if c >= 0 and c not in U:
                            candidates_add.add(c)
                for c in range(max_u + 1, max_u + 8):
                    candidates_add.add(c)
                if candidates_add:
                    c = min(candidates_add)
                    U_temp = list(U)
                    U_temp.pop(remove_idx)
                    U_new = sorted(set(U_temp + [c]))
                    if U_new and U_new[0] == 0:
                        s_new = compute_score_fast(U_new)
                        delta_s = s_new - current_score
                        if delta_s > 0 or (T > T_min and np.random.random() < math.exp(delta_s / T)):
                            U = U_new
                            current_score = s_new
                            improved = True
                            if s_new > best_score:
                                best_score = s_new
                                best_U = list(U)
            
            if not improved:
                # Cool down
                T *= cooling
            
            # If we've been stuck, try a bigger perturbation
            if it % 100 == 0 and T < 0.005:
                # Random restart from current best with perturbation
                if np.random.random() < 0.1:
                    U = list(best_U)
                    # Add a random element
                    max_u2 = max(U)
                    new_elem = np.random.randint(0, max_u2 + 10)
                    U = sorted(set(U + [new_elem]))
                    current_score = compute_score_fast(U)
                    T = 0.005
        
        return U
    
    # Run local search from several starting points
    starts = [
        [0, 1, 3],
        [0, 1, 3, 4],
        [0, 1, 3, 4, 9, 10, 12, 13],
        [0, 1, 2, 5, 6, 9, 10, 11],
        [0, 1, 4, 6, 7, 10, 12],
    ]
    
    for start in starts:
        if time_remaining() < 5:
            break
        U = list(start)
        if 0 not in U:
            U.append(0)
        U = sorted(set(U))
        local_search(U, max_iters=800)
    
    # Final: try structured sets and local search
    for n in range(2, 15):
        if time_remaining() < 5:
            break
        U = []
        for i in range(n):
            U.append(i * n)
            U.append(i * n + 1)
        try_set(U)
        if time_remaining() > 10:
            local_search(U, max_iters=300)
    
    return best_U
