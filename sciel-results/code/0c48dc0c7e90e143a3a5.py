import time
import random
import numpy as np

def construct():
    deadline = time.time() + 82
    rng = random.Random(42)
    
    def compute_m_numpy(A_set):
        """Compute sum multiplicity using numpy convolution for speed."""
        if not A_set:
            return np.zeros(257, dtype=np.int32)
        A = np.array(sorted(A_set), dtype=np.int32)
        indicator = np.zeros(129, dtype=np.float64)
        indicator[A] = 1.0
        conv = np.convolve(indicator, indicator, mode='full')
        # conv[s] = number of ordered pairs (a,b) with a+b=s, a,b in A
        # Unordered pairs with self-pairs: m[s] = (conv[s] + I(s/2 in A, s even)) / 2
        m = np.zeros(257, dtype=np.int32)
        for s in range(128, 257):
            c = conv[s]
            if s % 2 == 0:
                half = s // 2
                if 0 <= half <= 128 and indicator[half] > 0:
                    c += 1
            m[s] = int(c) // 2
        return m
    
    def evaluate_numpy(A_set):
        """Evaluate using numpy, returns (peak, valid)."""
        if not A_set:
            return float('inf'), False
        m = compute_m_numpy(A_set)
        peak = 0
        for t in range(128, 257):
            if m[t] == 0:
                return float('inf'), False
            if m[t] > peak:
                peak = int(m[t])
        return peak, True
    
    def evaluate_quick(A_set):
        """Faster evaluation with early exit for small sets."""
        if not A_set:
            return float('inf'), False
        A = sorted(A_set)
        n = len(A)
        if n < 1:
            return float('inf'), False
        m = [0] * 257
        for i in range(n):
            a = A[i]
            for j in range(i, n):
                s = a + A[j]
                if s > 256:
                    break
                m[s] += 1
        peak = 0
        for t in range(128, 257):
            if m[t] == 0:
                return float('inf'), False
            if m[t] > peak:
                peak = m[t]
        return peak, True
    
    def local_search(A_set, initial_peak, end_time):
        A = set(A_set)
        best_peak = initial_peak
        current_peak = initial_peak
        stall_count = 0
        best_A = set(A)
        
        while time.time() < end_time:
            improved = False
            
            # Try single moves
            for attempt in range(40):
                if time.time() > end_time:
                    break
                op = rng.choice(['remove', 'add', 'swap', 'swap'])
                candidate = A.copy()
                
                if op == 'remove' and len(A) > 1:
                    el = rng.choice(list(A))
                    candidate.discard(el)
                elif op == 'add':
                    available = [i for i in range(129) if i not in A]
                    if not available:
                        continue
                    new_el = rng.choice(available)
                    candidate.add(new_el)
                elif op == 'swap' and len(A) > 1:
                    el = rng.choice(list(A))
                    candidate.discard(el)
                    avail2 = [i for i in range(129) if i not in candidate]
                    if not avail2:
                        continue
                    new_el = rng.choice(avail2)
                    candidate.add(new_el)
                else:
                    continue
                
                # Use numpy for larger sets, quick for smaller
                if len(candidate) > 30:
                    peak_c, valid_c = evaluate_numpy(candidate)
                else:
                    peak_c, valid_c = evaluate_quick(candidate)
                
                if valid_c and peak_c < current_peak:
                    A = candidate
                    current_peak = peak_c
                    if peak_c < best_peak:
                        best_peak = peak_c
                        best_A = set(A)
                    improved = True
                    stall_count = 0
                    break
            
            if not improved:
                stall_count += 1
                # Multi-move perturbation
                candidate = A.copy()
                k_remove = rng.randint(1, 3)
                k_add = rng.randint(1, 3)
                
                if len(candidate) > k_remove + 1:
                    to_remove = rng.sample(list(candidate), k_remove)
                    for el in to_remove:
                        candidate.discard(el)
                
                available = [i for i in range(129) if i not in candidate]
                if available:
                    k_add_actual = min(k_add, len(available))
                    to_add = rng.sample(available, k_add_actual)
                    for el in to_add:
                        candidate.add(el)
                
                if len(candidate) > 30:
                    peak_c, valid_c = evaluate_numpy(candidate)
                else:
                    peak_c, valid_c = evaluate_quick(candidate)
                
                if valid_c and peak_c < current_peak:
                    A = candidate
                    current_peak = peak_c
                    if peak_c < best_peak:
                        best_peak = peak_c
                        best_A = set(A)
                    improved = True
                    stall_count = 0
                
                if not improved:
                    if stall_count > 8:
                        break
        
        return best_A, best_peak
    
    best_A = None
    best_peak = float('inf')
    
    # Diverse initializations
    inits = [
        set(range(64, 129)),
        set(range(0, 32)) | set(range(96, 129)),
        set(range(0, 24)) | set(range(80, 129)),
        set(range(16, 48)) | set(range(80, 112)),
        set(range(0, 16)) | set(range(32, 48)) | set(range(96, 129)),
        set(range(0, 16)) | set(range(48, 64)) | set(range(96, 129)),
        set(range(32, 64)) | set(range(64, 96)),
        set(range(0, 20)) | set(range(60, 80)) | set(range(100, 129)),
        set(range(0, 12)) | set(range(24, 36)) | set(range(48, 60)) | set(range(96, 129)),
        set(range(0, 12)) | set(range(36, 48)) | set(range(72, 84)) | set(range(108, 129)),
        set(range(0, 20)) | set(range(40, 60)) | set(range(90, 129)),
        set(range(10, 50)) | set(range(80, 120)),
        set(range(20, 70)) | set(range(60, 110)),
    ]
    
    # Random initializations of various sizes
    for k in [25, 30, 35, 40, 45, 50, 55, 60, 64]:
        inits.append(set(rng.sample(range(129), k)))
    
    # Evaluate all inits first to find best starting point
    init_results = []
    for init_A in inits:
        if time.time() > deadline - 5:
            break
        peak, valid = evaluate_numpy(init_A)
        if valid:
            init_results.append((peak, init_A))
    
    # Sort by peak and process best first
    init_results.sort(key=lambda x: x[0])
    
    for peak, init_A in init_results:
        if time.time() > deadline - 5:
            break
        if peak < best_peak:
            best_peak = peak
            best_A = set(init_A)
        
        # Local search with time limit
        remaining = deadline - 3 - time.time()
        if remaining > 2:
            # Allocate more time to better initial solutions
            alloc = min(remaining, max(4, 10 - peak))
            A_local, peak_local = local_search(init_A, peak, time.time() + alloc)
            if peak_local < best_peak:
                best_peak = peak_local
                best_A = set(A_local)
    
    # Final polish if time permits
    if best_A is not None and time.time() < deadline - 4:
        remaining = deadline - 4 - time.time()
        if remaining > 2:
            A_final, peak_final = local_search(best_A, best_peak, time.time() + remaining)
            if peak_final < best_peak:
                best_peak = peak_final
                best_A = set(A_final)
    
    if best_A is None:
        return list(range(64, 129))
    
    return list(best_A)
