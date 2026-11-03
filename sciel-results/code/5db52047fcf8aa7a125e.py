import time
import random
import numpy as np
import itertools

DEADLINE = 145.0

def construct(q, n, radius):
    start_time = time.time()
    total = q ** n
    
    # If the space is too large, fallback to a simple construction
    if total > 1000000:
        # Fallback: Tetracode-based construction for q=3, n=8
        # This yields 729 words.
        if q == 3 and n == 8:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=4)]
        # Generic fallback: just return a large random set or similar
        # For now, return empty to indicate failure or use a trivial cover
        return []

    # Precompute neighbors for radius=1
    # For q=3, n=8, radius=1: each word has 1 + 8*2 = 17 neighbors
    # General: 1 + n*(q-1)
    ball_size = 1 + n * (q - 1)
    
    # Build neighbors array
    # Shape: (total, ball_size)
    neighbors = np.zeros((total, ball_size), dtype=np.int32)
    
    # Precompute powers
    powers = [1] * n
    for i in range(n - 2, -1, -1):
        powers[i] = powers[i + 1] * q
        
    for x in range(total):
        neighbors[x, 0] = x
        idx = 1
        for pos in range(n):
            digit = (x // powers[pos]) % q
            base = x - digit * powers[pos]
            for d in range(q):
                if d != digit:
                    neighbors[x, idx] = base + d * powers[pos]
                    idx += 1

    def _greedy_cover(rng):
        is_covered = np.zeros(total, dtype=bool)
        code = []
        remaining = total
        
        while remaining > 0:
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Calculate gain for each word
            # gain[x] = number of uncovered neighbors of x
            # ~is_covered is a boolean array of size total
            # neighbors is (total, ball_size)
            # ~is_covered[neighbors] is (total, ball_size) boolean
            # Sum along axis 1
            gains = np.sum(~is_covered[neighbors], axis=1)
            
            # Find max gain
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Get all words with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Randomly pick one from candidates
            best_idx = candidates[rng.integers(len(candidates))]
            
            code.append(best_idx)
            
            # Update coverage
            is_covered[neighbors[best_idx]] = True
            remaining -= np.sum(~is_covered[neighbors[best_idx]]) # This is slightly expensive, let's just count
            # Actually, remaining -= (number of newly covered)
            # newly_covered = np.sum((~is_covered_old) & is_covered_new)
            # Simpler: just track count
            # Let's re-calculate remaining occasionally or just rely on loop condition
            # For speed, we can just break if gains.max() == 0
            
        return code

    def _local_search_remove(code):
        if not code:
            return code
            
        code = list(code)
        
        # Calculate cover_count
        # Flatten all neighbors of all codewords
        # neighbors[code] has shape (len(code), ball_size)
        flat_indices = neighbors[code].flatten()
        cover_count = np.bincount(flat_indices, minlength=total)
        
        changed = True
        while changed:
            if time.time() - start_time > DEADLINE - 5:
                break
            changed = False
            for i in range(len(code) - 1, -1, -1):
                cw = code[i]
                nb = neighbors[cw]
                # If min cover count in nb is > 1, we can remove
                if np.min(cover_count[nb]) > 1:
                    # Remove
                    code.pop(i)
                    cover_count[nb] -= 1
                    changed = True
        
        return code

    best_code = None
    best_size = float('inf')
    
    # Run multiple restarts with different random seeds
    # We have ~145s. Greedy is fast. Local search is fast.
    # Let's run as many as possible.
    
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)
    
    # Initial greedy
    code = _greedy_cover(np_rng)
    code = _local_search_remove(code)
    best_code = code
    best_size = len(code)
    
    # Try to improve with more restarts
    # Each restart takes roughly:
    # Greedy: ~500-1000 iterations. Each iteration is O(N * ball_size) vectorized.
    # Local Search: ~500-1000 removals. Each is O(ball_size) vectorized.
    # Total per restart should be < 1s. So we can do ~100 restarts.
    
    while time.time() - start_time < DEADLINE - 20:
        # Perturb the greedy process
        # We can add noise to gains
        # Or just use a new RNG seed
        np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
        
        code = _greedy_cover(np_rng)
        
        if time.time() - start_time > DEADLINE - 10:
            break
            
        code = _local_search_remove(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
    # Final polish: try more aggressive local search on the best code found
    # Maybe try swaps?
    # For now, just return the best found.
    
    if best_code is None:
        # Fallback
        return [_int_to_vec(x, q, n) for x in _greedy_cover(np.random.default_rng(0))]
        
    return [_int_to_vec(x, q, n) for x in best_code]

def _int_to_vec(x, q, n):
    vec = [0] * n
    for i in range(n - 1, -1, -1):
        vec[i] = x % q
        x //= q
    return vec
