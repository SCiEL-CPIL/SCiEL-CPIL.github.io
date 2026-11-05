"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time

def construct(q, n, radius):
    # Deadline guard: 110s margin out of 150s limit
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Baseline: Direct sum of tetracode (for q=3, n>=4) or generic fallback
    # For q=3, n=6, radius=1, baseline is 9 * 3^(6-4) = 9 * 9 = 81 words.
    # Best known is 73. We want to beat 81.
    
    # If the space is small enough, we can use a greedy set cover approach.
    # Total words = q^n. For q=3, n=6, total=729.
    # Sphere size for radius 1: 1 + n*(q-1) = 1 + 6*2 = 13.
    # Greedy steps ~ K. Ops ~ K * Total * Sphere.
    # If K~73, Ops ~ 73 * 729 * 13 ~ 690k. Very fast in NumPy.
    
    total_words = q ** n
    
    # Check if greedy is feasible
    # If total_words is large, greedy might be too slow.
    # For q=3, n=10, total=59049. Sphere=21.
    # If K~300, Ops ~ 300 * 59049 * 21 ~ 370 million.
    # NumPy vectorized ops can handle ~100M simple ops/sec. 370M might take ~4s.
    # But we also have overhead. Let's set a threshold.
    # If total_words <= 10000, greedy is safe.
    # If larger, we might still try it if we have time, but let's be conservative.
    # Actually, for n=6, it's tiny. Let's just do it for n <= 8 or total <= 10000.
    
    if total_words <= 10000 and radius == 1:
        # NumPy Greedy Set Cover with Random Restarts
        # Precompute neighbors for all words
        sphere_size = 1 + n * (q - 1)
        
        # Initialize neighbors array
        neighbors = np.zeros((total_words, sphere_size), dtype=np.int32)
        
        # Precompute powers of q
        q_powers = [q ** (n - 1 - i) for i in range(n)]
        
        for i in range(total_words):
            nbrs = [i]
            for pos in range(n):
                pow_q = q_powers[pos]
                current_val = (i // pow_q) % q
                for new_val in range(q):
                    if new_val != current_val:
                        delta = (new_val - current_val) * pow_q
                        nbrs.append(i + delta)
            neighbors[i] = nbrs
        
        best_code_indices = None
        best_size = float('inf')
        
        # Run multiple greedy instances with random tie-breaking
        num_restarts = 50
        for r in range(num_restarts):
            if time.time() > deadline - 5.0: # Leave time for final check
                break
            
            covered = np.zeros(total_words, dtype=bool)
            code_indices = []
            
            # Greedy loop
            while True:
                # Calculate gain for each candidate: number of uncovered neighbors
                gains = (~covered)[neighbors].sum(axis=1)
                
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                
                # Get all candidates with max gain
                candidates = np.where(gains == max_gain)[0]
                
                # Pick one randomly
                best_idx = candidates[np.random.randint(len(candidates))]
                
                code_indices.append(best_idx)
                covered[neighbors[best_idx]] = True
            
            if covered.all():
                if len(code_indices) < best_size:
                    best_size = len(code_indices)
                    best_code_indices = code_indices[:]
        
        if best_code_indices is not None:
            # Convert indices back to words
            result_words = []
            for idx in best_code_indices:
                w = []
                temp = idx
                for _ in range(n):
                    w.append(temp % q)
                    temp //= q
                w.reverse()
                result_words.append(w)
            return result_words

    # Fallback: Direct Sum (Baseline)
    # This is the current program's logic.
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        base_n = 4
        base_code = tetracode
    else:
        # Generic fallback: just use all-zero and unit vectors? No, that's not covering.
        # For generic q, n, radius, a simple covering code is hard.
        # But the task is specifically K_3(6,1).
        # If q!=3 or n!=6, we don't have a specific baseline.
        # Let's try a simple greedy for small cases if baseline doesn't apply.
        # Or just return a dummy list? No, must be valid.
        # For now, assume q=3, n=6 is the primary case.
        # If not, try a simple construction.
        # If q=3, n=6, we are handled above.
        
        # If we get here, we didn't use greedy.
        # Let's try to use the baseline logic if possible.
        if q == 3 and n >= 4:
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             base_n = 4
             base_code = tetracode
        else:
             # Very generic fallback: all words (valid but huge)
             return [list(w) for w in itertools.product(range(q), repeat=n)]
        
        if n < base_n:
             return [list(range(n))]
             
        tails = list(itertools.product(range(q), repeat=n - base_n))
        return [h + list(tail) for h in base_code for tail in tails]

# EVOLVE-BLOCK-END
