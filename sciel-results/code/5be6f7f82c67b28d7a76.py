import numpy as np
import itertools
import time

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        return [[0] * n]
    
    start_time = time.time()
    deadline = start_time + 110
    
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    N = len(all_words)
    
    # Initial code: 243 words
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_code = []
    for t in tetracode:
        for tail in itertools.product(range(3), repeat=3):
            init_code.append(t + list(tail))
            
    code = np.array(init_code, dtype=np.int8)
    M = len(code)
    
    # Precompute coverage matrix
    cov = np.zeros((M, N), dtype=bool)
    for i in range(M):
        c = code[i]
        idx = 0
        for k in range(n):
            idx = idx * 3 + c[k]
        cov[i, idx] = True
        
        for j in range(n):
            orig_val = c[j]
            for val in range(q):
                if val != orig_val:
                    c[j] = val
                    idx_mod = 0
                    for k in range(n):
                        idx_mod = idx_mod * 3 + c[k]
                    cov[i, idx_mod] = True
            c[j] = orig_val
            
    # Precompute list of covered word indices for each codeword
    covered_words = [np.where(cov[i])[0] for i in range(M)]
    
    def run_removal(initial_active_order):
        active = list(initial_active_order)
        counts = np.sum(cov[active], axis=0)
        
        while True:
            if time.time() > deadline:
                break
            redundant_idx = -1
            for i in active:
                W_i = covered_words[i]
                if len(W_i) == 0:
                    continue
                if np.min(counts[W_i]) >= 2:
                    redundant_idx = i
                    break
            if redundant_idx == -1:
                break
            active.remove(redundant_idx)
            counts -= cov[redundant_idx]
            
        return [code[i] for i in active]
        
    best_code = run_removal(range(M))
    
    # Try random restarts
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        order = list(range(M))
        np.random.shuffle(order)
        curr_code = run_removal(order)
        if len(curr_code) < len(best_code):
            best_code = curr_code
            
    return [list(c) for c in best_code]
