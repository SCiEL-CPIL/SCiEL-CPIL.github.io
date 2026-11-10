import numpy as np
import time
import itertools

def construct(q, n, radius):
    """
    Construct a ternary covering code of length n with radius r.
    For q=3, n=8, r=1, we aim to find a code with fewer than 486 codewords.
    """
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        N = q ** n
        neighbors = np.zeros((N, 1 + 2 * n), dtype=np.int32)
        for w in range(N):
            idx = 0
            neighbors[w, idx] = w
            idx += 1
            temp = w
            digits = [0] * n
            for i in range(n):
                digits[i] = temp % q
                temp //= q
            for i in range(n):
                d = digits[i]
                for new_d in range(q):
                    if new_d != d:
                        w1 = w + (new_d - d) * (q ** i)
                        neighbors[w, idx] = w1
                        idx += 1
        uncovered = np.ones(N, dtype=np.uint8)
        code = []
        while uncovered.sum() > 0:
            gains = uncovered[neighbors].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            w = candidates[np.random.randint(len(candidates))]
            code.append(w)
            uncovered[neighbors[w]] = 0
        return [[int(c) for c in np.base_repr(w, base=q).zfill(n)][::-1] for w in code]

    N = 3 ** 8  # 6561
    deadline = time.time() + 115.0  # Safety margin

    # Precompute neighbors for all N words
    neighbors = np.zeros((N, 1 + 2 * n), dtype=np.int32)
    powers = [3 ** i for i in range(n)]
    
    for w in range(N):
        neighbors[w, 0] = w
        idx = 1
        temp = w
        for i in range(n):
            d = temp % 3
            temp //= 3
            w1 = w + ((d + 1) % 3 - d) * powers[i]
            neighbors[w, idx] = w1
            idx += 1
            w2 = w + ((d + 2) % 3 - d) * powers[i]
            neighbors[w, idx] = w2
            idx += 1

    # Seed solution: Tetracode direct sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_code_ints = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            w_int = 0
            for i in range(n):
                w_int += word[i] * powers[i]
            seed_code_ints.append(w_int)
    
    best_code = seed_code_ints[:]
    best_size = len(best_code)
    
    rng = np.random.default_rng(42)

    def local_search(code):
        """
        Attempts to minimize the code size by removing or swapping codewords.
        """
        if not code:
            return code
        
        # cover_count[w] is the number of codewords covering w
        cover_count = np.zeros(N, dtype=np.int32)
        for c in code:
            cover_count[neighbors[c]] += 1
        
        code_list = code[:]
        improved = True
        passes = 0
        max_passes = 100 # Safety limit
        
        while improved and passes < max_passes and time.time() < deadline:
            improved = False
            passes += 1
            
            # Shuffle indices to randomize the order of attempts
            indices = np.arange(len(code_list))
            rng.shuffle(indices)
            
            for idx in indices:
                if idx >= len(code_list):
                    continue
                
                c = code_list[idx]
                
                # 1. Attempt Removal
                cover_count[neighbors[c]] -= 1
                
                if np.min(cover_count) > 0:
                    # Removal is valid
                    code_list.pop(idx)
                    improved = True
                    # Restart pass to handle index shifts safely
                    break
                
                # 2. Attempt Swap
                # If removal was invalid, there are holes
                holes = (cover_count == 0)
                hole_count = np.sum(holes)
                
                if hole_count > 0:
                    # Find candidate v that covers the most holes
                    gains = holes[neighbors].sum(axis=1)
                    max_gain = np.max(gains)
                    
                    if max_gain == hole_count:
                        candidates = np.where(gains == max_gain)[0]
                        # Filter out c itself? If c is in candidates, gain would be 0 
                        # because holes are defined by c's absence, and c covers its neighbors.
                        # Wait, if we removed c, holes are neighbors of c that are NOT covered by others.
                        # c covers its neighbors. So c would cover all holes?
                        # No, holes are words w where cover_count[w]==0.
                        # If we put c back, cover_count[neighbors[c]] increases.
                        # So c covers all holes?
                        # Yes, by definition, the holes are caused by removing c.
                        # So c is always a valid swap (it reverts the change).
                        # We want a *different* word if possible, or just any word.
                        # If we pick c, we revert. 
                        # We should prefer a word that is not c.
                        
                        # Find candidates != c
                        valid_candidates = [v for v in candidates if v != c]
                        
                        if valid_candidates:
                            v = valid_candidates[rng.integers(len(valid_candidates))]
                        else:
                            # Only c works, so we revert
                            cover_count[neighbors[c]] += 1
                            continue
                        
                        # Apply swap
                        cover_count[neighbors[v]] += 1
                        code_list[idx] = v
                        improved = True
                        # Restart pass
                        break
                    else:
                        # No single word covers all holes, revert removal
                        cover_count[neighbors[c]] += 1
                else:
                    # No holes? This implies removal was valid, which was handled above.
                    # But just in case, revert if we got here
                    cover_count[neighbors[c]] += 1

        return code_list

    # Main Search Loop
    while time.time() < deadline:
        # 1. Greedy Construction
        uncovered = np.ones(N, dtype=np.uint8)
        code = []
        
        while uncovered.sum() > 0:
            gains = uncovered[neighbors].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            w = rng.choice(candidates)
            code.append(w)
            uncovered[neighbors[w]] = 0
        
        # 2. Local Search Optimization
        optimized_code = local_search(code)
        
        if len(optimized_code) < best_size:
            best_size = len(optimized_code)
            best_code = optimized_code[:]
            
            # If we reach the lower bound, we can stop
            if best_size <= 403:
                break

    # Convert integer codewords back to lists of integers
    final_code = []
    for w in best_code:
        word = []
        temp = w
        for _ in range(n):
            word.append(temp % 3)
            temp //= 3
        final_code.append(word)
        
    return final_code
