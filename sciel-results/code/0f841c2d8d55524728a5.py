import time
import random
import numpy as np

def construct():
    deadline = time.time() + 80
    rng = random.Random(42)
    
    def compute_m_fast(A_set):
        """Compute sum multiplicity using numpy for speed."""
        if not A_set:
            return np.zeros(257, dtype=np.int32)
        A = np.array(sorted(A_set), dtype=np.int32)
        n = len(A)
        # Use convolution: create indicator vector
        indicator = np.zeros(129, dtype=np.float64)
        indicator[A] = 1.0
        # Convolution gives ordered pairs; we need unordered with self-pairs
        # conv[i] = sum_{a+b=i} indicator[a]*indicator[b]
        # For unordered pairs with equal summands allowed:
        # m[s] = (conv[s] + (1 if s even and s/2 in A else 0)) / 2
        conv = np.convolve(indicator, indicator, mode='full')
        # conv has length 2*129-1 = 257, indices 0..256
        m = np.zeros(257, dtype=np.int32)
        for s in range(128, 257):
            if s < len(conv):
                c = conv[s]
                # Adjust for unordered pairs
                if s % 2 == 0:
                    half = s // 2
                    if 0 <= half <= 128 and half in A_set:
                        c += 1  # Add the self-pair (half, half) which was counted once in ordered
                m[s] = int(c) // 2
        return m
    
    def evaluate(A_set):
        if not A_set:
            return float('inf'), False
        m = compute_m_fast(A_set)
        peak = 0
        for t in range(128, 257):
            if m[t] == 0:
                return float('inf'), False
            if m[t] > peak:
                peak = m[t]
        return peak, True
    
    def evaluate_quick(A_set):
        """Faster evaluation that early-exits."""
        if not A_set:
            return float('inf'), False
        A = sorted(A_set)
        n = len(A)
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
        
        while time.time() < end_time:
            improved = False
            # Try targeted moves: focus on elements contributing to peak
            for attempt in range(30):
                if time.time() > end_time:
                    break
                op = rng.choice(['remove', 'add', 'swap', 'swap'])
                candidate = A.copy()
                
                if op == 'remove' and len(A) > 1:
                    # Prefer removing elements that appear in many high-multiplicity sums
                    el = rng.choice(list(A))
                    candidate.discard(el)
                elif op == 'add':
                    available = [i for i in range(129) if i not in A]
                    if not available:
                        continue
                    # Prefer adding elements that help cover low-multiplicity targets
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
                
                peak_c, valid_c = evaluate_quick(candidate)
                if valid_c and peak_c < current_peak:
                    A = candidate
                    current_peak = peak_c
                    if peak_c < best_peak:
                        best_peak = peak_c
                    improved = True
                    stall_count = 0
                    break
            
            if not improved:
                stall_count += 1
                # Try multi-move perturbation
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
                
                peak_c, valid_c = evaluate_quick(candidate)
                if valid_c and peak_c < current_peak:
                    A = candidate
                    current_peak = peak_c
                    if peak_c < best_peak:
                        best_peak = peak_c
                    improved = True
                    stall_count = 0
                
                if not improved and stall_count > 5:
                    break
                elif not improved:
                    # Reset stall counter occasionally
                    if stall_count > 15:
                        break
        
        return A, best_peak
    
    best_A = None
    best_peak = float('inf')
    
    # Carefully crafted initializations
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
    ]
    
    # Add some random initializations
    for k in [30, 40, 50, 60, 64]:
        inits.append(set(rng.sample(range(129), k)))
    
    for init_A in inits:
        if time.time() > deadline - 5:
            break
        peak, valid = evaluate_quick(init_A)
        if not valid:
            continue
        if peak < best_peak:
            best_peak = peak
            best_A = set(init_A)
        
        # Local search with time limit
        remaining = deadline - 3 - time.time()
        if remaining > 2:
            alloc = min(remaining, 6)
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
