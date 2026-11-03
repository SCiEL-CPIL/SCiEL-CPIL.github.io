import numpy as np
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = 100.0  # Safety margin for 120s limit

    N = q ** n
    
    # Precompute neighbors
    # We need to map integer index to base-q digits and back
    # For efficiency, we can precompute the digit representation
    digits = np.zeros((N, n), dtype=int)
    for i in range(N):
        x = i
        for j in range(n - 1, -1, -1):
            digits[i, j] = x % q
            x //= q
            
    neighbors = np.zeros((N, 1 + n * (q - 1)), dtype=int)
    for i in range(N):
        nbrs = [i]
        for pos in range(n):
            orig_val = digits[i, pos]
            for new_val in range(q):
                if new_val != orig_val:
                    # Calculate new index
                    # index = sum(d_j * q^(n-1-j))
                    # delta = (new_val - orig_val) * q^(n-1-pos)
                    diff = new_val - orig_val
                    new_idx = i + diff * (q ** (n - 1 - pos))
                    nbrs.append(new_idx)
        neighbors[i] = nbrs
        
    def prune(code):
        if not code:
            return []
        cover_count = np.zeros(N, dtype=int)
        for c in code:
            cover_count[neighbors[c]] += 1
        
        final_code = []
        # Try to remove in reverse order of addition
        for c in reversed(code):
            cover_count[neighbors[c]] -= 1
            if np.min(cover_count) >= 1:
                pass
            else:
                cover_count[neighbors[c]] += 1
                final_code.append(c)
        return final_code

    def greedy_cover(noise_scale=0.0, rng=None):
        uncovered = np.ones(N, dtype=bool)
        uncovered_count = N
        code = []
        
        while uncovered_count > 0:
            # Calculate gain
            gains = np.sum(uncovered[neighbors], axis=1)
            
            if noise_scale > 0 and rng is not None:
                gains = gains + rng.normal(0, noise_scale, size=N)
            
            best_idx = np.argmax(gains)
            
            nbrs = neighbors[best_idx]
            newly_covered = np.sum(uncovered[nbrs])
            uncovered[nbrs] = False
            uncovered_count -= newly_covered
            
            code.append(best_idx)
            
            if time.time() - start_time > deadline - 5:
                break
                
        return prune(code)

    best_code = None
    best_size = float('inf')
    
    # Deterministic run
    code = greedy_cover(noise_scale=0.0)
    if len(code) < best_size:
        best_size = len(code)
        best_code = code
        
    # Randomized runs
    seed = 0
    while time.time() - start_time < deadline - 5:
        rng = np.random.default_rng(seed)
        # Vary noise scale
        noise = 0.5 + (seed % 5) * 0.1
        code = greedy_cover(noise_scale=noise, rng=rng)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
        seed += 1
        
    if best_code is None:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    # Convert indices to words
    # Use the precomputed digits
    return [digits[c].tolist() for c in best_code]
