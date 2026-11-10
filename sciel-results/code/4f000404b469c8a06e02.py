import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    MIN_SUM = 128
    MAX_SUM = 256
    NUM_SUMS = MAX_SUM - MIN_SUM + 1
    
    # Fast multiplicity calculation using convolution
    def calc_mult_fast(A):
        """Compute representation multiplicities using FFT-based convolution."""
        vec = np.zeros(257, dtype=np.float64)
        for a in A:
            vec[a] = 1.0
        # Convolution gives ordered pairs; we need unordered with self-pairs
        conv = np.convolve(vec, vec, mode='full')
        # conv[s] counts ordered pairs (a,b) with a+b=s
        # For unordered: M[s] = (conv[s] - diag[s]) / 2 + diag[s] where diag[s]=1 if s even and s/2 in A
        M = np.zeros(NUM_SUMS, dtype=np.float64)
        for s in range(MIN_SUM, MAX_SUM + 1):
            idx = s - MIN_SUM
            ordered = conv[s]
            # Subtract self-pairs counted twice in ordered, add back once
            if s % 2 == 0:
                half = s // 2
                if 0 <= half <= 128 and vec[half] > 0:
                    ordered -= 1  # Remove one of the two (a,a) ordered counts
            M[idx] = ordered / 2.0
        return M.astype(np.int32)
    
    # Incremental update helpers
    def remove_element(M, A, x):
        """Remove element x from A and update M incrementally."""
        M_new = M.copy()
        # Remove (x, x)
        s = 2 * x
        if MIN_SUM <= s <= MAX_SUM:
            M_new[s - MIN_SUM] -= 1
        # Remove (a, x) for a in A, a != x
        for a in A:
            if a == x:
                continue
            s = a + x
            if MIN_SUM <= s <= MAX_SUM:
                M_new[s - MIN_SUM] -= 1
        return M_new
    
    def add_element(M, A, y):
        """Add element y to A and update M incrementally."""
        M_new = M.copy()
        # Add (y, y)
        s = 2 * y
        if MIN_SUM <= s <= MAX_SUM:
            M_new[s - MIN_SUM] += 1
        # Add (a, y) for a in A
        for a in A:
            s = a + y
            if MIN_SUM <= s <= MAX_SUM:
                M_new[s - MIN_SUM] += 1
        return M_new
    
    def swap_element(M, A, x, y):
        """Swap x for y in A and update M incrementally."""
        M_new = M.copy()
        # Remove x
        s = 2 * x
        if MIN_SUM <= s <= MAX_SUM:
            M_new[s - MIN_SUM] -= 1
        for a in A:
            if a == x:
                continue
            s = a + x
            if MIN_SUM <= s <= MAX_SUM:
                M_new[s - MIN_SUM] -= 1
        # Add y
        s = 2 * y
        if MIN_SUM <= s <= MAX_SUM:
            M_new[s - MIN_SUM] += 1
        for a in A:
            if a == x:
                continue
            s = a + y
            if MIN_SUM <= s <= MAX_SUM:
                M_new[s - MIN_SUM] += 1
        return M_new
    
    # Initial set: dense interval [64, 128]
    A = set(range(64, 129))
    M = calc_mult_fast(A)
    
    if np.min(M) < 1:
        A = set(range(0, 129))
        M = calc_mult_fast(A)
    
    current_score = int(np.max(M))
    best_score = current_score
    best_A = set(A)
    
    # Simulated Annealing
    T = 5.0
    cooling = 0.999
    min_T = 0.05
    
    ALL_ELEMENTS = set(range(0, 129))
    
    step_count = 0
    while time.time() < deadline:
        step_count += 1
        
        # Choose move type
        r = random.random()
        
        if r < 0.45:
            # Remove
            if len(A) <= 12:
                T *= cooling
                continue
            x = random.choice(list(A))
            
            M_new = remove_element(M, A, x)
            
            if np.min(M_new) >= 1:
                score_new = int(np.max(M_new))
                delta = score_new - current_score
                
                if delta < 0 or (delta == 0 and random.random() < 0.5) or (delta > 0 and random.random() < np.exp(-delta / T)):
                    A.remove(x)
                    M = M_new
                    current_score = score_new
                    if score_new < best_score:
                        best_score = score_new
                        best_A = set(A)
                        
        elif r < 0.85:
            # Swap
            if len(A) == 0:
                T *= cooling
                continue
            x = random.choice(list(A))
            candidates = [y for y in ALL_ELEMENTS if y not in A]
            if not candidates:
                T *= cooling
                continue
            y = random.choice(candidates)
            
            M_new = swap_element(M, A, x, y)
            
            if np.min(M_new) >= 1:
                score_new = int(np.max(M_new))
                delta = score_new - current_score
                
                if delta < 0 or (delta == 0 and random.random() < 0.5) or (delta > 0 and random.random() < np.exp(-delta / T)):
                    A.remove(x)
                    A.add(y)
                    M = M_new
                    current_score = score_new
                    if score_new < best_score:
                        best_score = score_new
                        best_A = set(A)
                        
        else:
            # Add
            candidates = [y for y in ALL_ELEMENTS if y not in A]
            if not candidates:
                T *= cooling
                continue
            y = random.choice(candidates)
            
            M_new = add_element(M, A, y)
            
            score_new = int(np.max(M_new))
            delta = score_new - current_score
            
            if delta < 0 or (delta == 0 and random.random() < 0.5) or (delta > 0 and random.random() < np.exp(-delta / T)):
                A.add(y)
                M = M_new
                current_score = score_new
        
        T *= cooling
        if T < min_T:
            T = min_T
        
        # Periodic restart to escape local minima
        if step_count % 5000 == 0 and T <= min_T:
            T = 3.0
    
    # Final polish: try to reduce peak further with targeted removals
    if time.time() < deadline - 5:
        # Try removing elements that don't break coverage
        for _ in range(200):
            if time.time() > deadline - 2:
                break
            improved = False
            for x in list(best_A):
                if len(best_A) <= 12:
                    break
                # Check if removing x is valid
                M_test = remove_element(M, best_A, x)
                if np.min(M_test) >= 1:
                    score_test = int(np.max(M_test))
                    if score_test < best_score:
                        best_A.remove(x)
                        M = M_test
                        best_score = score_test
                        improved = True
                        break
            if not improved:
                break
    
    return sorted(list(best_A))
