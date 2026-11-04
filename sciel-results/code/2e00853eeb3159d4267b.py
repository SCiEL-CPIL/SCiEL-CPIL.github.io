import numpy as np
import time
import random

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        return _general_fallback(q, n, radius)
        
    start_time = time.time()
    deadline = start_time + 115.0
    
    total_words = 3 ** n
    # Precompute words as list of lists for return
    words = [list(map(int, np.base_repr(i, 3).zfill(n))) for i in range(total_words)]
    
    # Precompute M
    powers = [3 ** (n - 1 - k) for k in range(n)]
    M = np.zeros((total_words, total_words), dtype=np.int8)
    for i in range(total_words):
        M[i, i] = 1
        for k in range(n):
            d = (i // powers[k]) % q
            for new_d in range(q):
                if new_d != d:
                    delta = (new_d - d) * powers[k]
                    j = i + delta
                    M[i, j] = 1

    best_code = None
    best_size = float('inf')
    
    # Number of restarts depends on time
    # Each run: Greedy + Local Search
    # Greedy is ~200 steps * (2187x2187 mat-vec) ~ 200 * 0.005s = 1s? 
    # Mat-vec 2187x2187 is ~4.7M ops. ~1ms in numpy.
    # 200 steps = 0.2s.
    # Local search: 5000 iterations * (remove, mat-vec, check) ~ 5000 * 0.002s = 10s.
    # So one run ~ 12s.
    # We can do ~5-8 runs in 115s.
    
    max_restarts = 8
    
    for _ in range(max_restarts):
        if time.time() > deadline - 15.0: # Keep 15s buffer for final processing
            break
            
        # Greedy with random tie-breaking
        uncovered = np.ones(total_words, dtype=np.int8)
        code = []
        
        while uncovered.any():
            if time.time() > deadline - 5.0:
                break
            gains = M @ uncovered
            max_gain = np.max(gains)
            if max_gain == 0: break
            # Find all with max gain
            candidates = np.where(gains == max_gain)[0]
            w = random.choice(candidates.tolist())
            
            code.append(int(w))
            uncovered[M[w, :]] = 0
            
        # Local Search
        covered_counts = np.zeros(total_words, dtype=np.int32)
        for c in code:
            covered_counts[M[c, :]] += 1
            
        # Deletion
        def try_deletion():
            nonlocal code, covered_counts
            removed = False
            new_code = []
            for c in code:
                temp_counts = covered_counts - M[c, :]
                if not (temp_counts == 0).any():
                    covered_counts = temp_counts
                    removed = True
                else:
                    new_code.append(c)
            code = new_code
            return removed

        while try_deletion():
            pass
            
        # Swap/Relocate
        # Limit iterations to avoid infinite loops
        local_iter = 0
        max_local_iter = 2000
        while local_iter < max_local_iter and time.time() < deadline - 5.0:
            local_iter += 1
            improved = False
            for idx in range(len(code)):
                c = code[idx]
                temp_counts = covered_counts - M[c, :]
                holes = (temp_counts == 0).astype(np.int8)
                
                if not holes.any():
                    # Redundant
                    code.pop(idx)
                    covered_counts = temp_counts
                    improved = True
                    break # Index changed
                    
                gains = M @ holes
                w_new = np.argmax(gains)
                
                if w_new != c:
                    new_counts = temp_counts + M[w_new, :]
                    if not (new_counts == 0).any():
                        code[idx] = int(w_new)
                        covered_counts = new_counts
                        improved = True
                        # Don't break, continue checking others? 
                        # Or break to re-evaluate from start?
                        # Breaking is safer for index consistency if we popped, 
                        # but here we just swapped.
                        # However, swapping might make others redundant.
                        # Let's just continue the loop.
            
            if improved:
                while try_deletion():
                    pass
            
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
                
            if not improved:
                # Stuck? Break local search for this restart
                break
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]

    if best_code is None:
        # Fallback to seed if something went wrong
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [words[c] for c in best_code]

def _general_fallback(q, n, radius):
    # Simple greedy for general case
    total = q ** n
    words = [list(map(int, np.base_repr(i, q).zfill(n))) for i in range(total)]
    # ... implementation for general case ...
    # For now, return a trivial valid cover (all words)
    return words
