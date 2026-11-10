import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Handle general case fallback if not the specific task
    if not (q == 3 and n == 8 and radius == 1):
        # Simple greedy fallback for robustness
        N = q ** n
        if N > 100000:
            # Too large, return all words (trivial upper bound)
            return [[i % q for i in range(n)] for i in range(N)]
        
        powers = [q ** i for i in range(n)]
        neighbors = []
        for w in range(N):
            ball = {w}
            digits = [0] * n
            temp = w
            for i in range(n):
                digits[i] = temp % q
                temp //= q
            for pos in range(n):
                original = digits[pos]
                for val in range(q):
                    if val == original:
                        continue
                    digits[pos] = val
                    new_w = 0
                    mult = 1
                    for i in range(n):
                        new_w += digits[i] * mult
                        mult *= q
                    ball.add(new_w)
                digits[pos] = original
            neighbors.append(list(ball))
        
        uncovered = set(range(N))
        code = []
        while uncovered and time.time() < deadline:
            best_cand = -1
            best_count = -1
            for c in uncovered:
                count = sum(1 for nb in neighbors[c] if nb in uncovered)
                if count > best_count:
                    best_count = count
                    best_cand = c
            if best_cand == -1:
                break
            code.append(best_cand)
            for nb in neighbors[best_cand]:
                if nb in uncovered:
                    uncovered.remove(nb)
        if uncovered:
            code.extend(uncovered)
            
        result = []
        for w in code:
            digits = []
            temp = w
            for _ in range(n):
                digits.append(temp % q)
                temp //= q
            result.append(digits)
        return result

    # Specific implementation for q=3, n=8, radius=1
    N = q ** n  # 6561
    powers = [q ** i for i in range(n)]
    neigh_size = 1 + n * (q - 1)  # 17
    
    # Precompute neighbors as a 2D numpy array for fast access
    neighbors = np.zeros((N, neigh_size), dtype=np.int32)
    for w in range(N):
        neighbors[w, 0] = w
        idx = 1
        for i in range(n):
            d = (w // powers[i]) % q
            for delta in range(1, q):
                new_d = (d + delta) % q
                diff = new_d - d
                new_w = w + diff * powers[i]
                neighbors[w, idx] = new_w
                idx += 1

    def compute_coverage(code_set):
        """Compute coverage count for each word given a set of codewords."""
        cov = np.zeros(N, dtype=np.int32)
        # Vectorized addition is tricky with sets, so we iterate.
        # However, 486 * 17 operations is very small.
        for c in code_set:
            cov[neighbors[c, :]] += 1
        return cov

    def greedy_init(rng, randomize=False):
        """Greedy initialization."""
        uncovered = np.ones(N, dtype=np.bool_)
        code = set()
        
        while np.any(uncovered) and time.time() < deadline - 5:
            # Gain is number of uncovered neighbors
            # We can compute this by summing the boolean mask of uncovered over the neighbor rows
            # neighbors is shape (N, 17). uncovered is shape (N,).
            # We want sum of uncovered[neighbors[i]] for each i.
            # This is effectively a matrix-vector product if we treat uncovered as 0/1.
            gains = neighbors @ uncovered.astype(np.int32)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            candidates = np.where(gains == max_gain)[0]
            
            if randomize and len(candidates) > 1:
                # Pick from top candidates randomly to diversify restarts
                k = min(5, len(candidates))
                w = rng.choice(candidates[:k])
            else:
                w = candidates[0]
                
            code.add(w)
            uncovered[neighbors[w, :]] = False
            
        return code

    def local_search(code_set, rng, deadline):
        """
        Local search: 
        1. Reduce: Remove redundant codewords.
        2. Swap: If stuck, try to swap a codeword to enable further reduction.
        """
        code_list = list(code_set)
        
        while time.time() < deadline - 2:
            # 1. Compute Coverage
            cov = compute_coverage(code_set)
            
            # 2. Deletion Pass (Reduction)
            # A codeword c is redundant if min(cov[neighbors[c]]) > 1
            # We check all codewords.
            redundant_found = False
            # Iterate over a copy because we modify the set
            for c in list(code_set):
                if np.min(cov[neighbors[c, :]]) > 1:
                    code_set.remove(c)
                    # Update coverage incrementally
                    cov[neighbors[c, :]] -= 1
                    redundant_found = True
            
            if redundant_found:
                continue # Coverage changed, restart reduction loop
            
            # 3. Swap Pass
            # No single codeword is redundant. Try to swap one to create a redundancy.
            improved = False
            c_list = list(code_set)
            rng.shuffle(c_list)
            
            # Try a limited number of swaps to avoid spending too much time here
            max_swaps_to_try = min(len(c_list), 50)
            
            for c in c_list[:max_swaps_to_try]:
                if time.time() > deadline - 2:
                    break
                
                c_neighbors = neighbors[c, :]
                # Essential words: covered ONLY by c
                essential_mask = (cov[c_neighbors] == 1)
                essential_words = c_neighbors[essential_mask]
                
                if len(essential_words) == 0:
                    # Should have been caught in deletion pass, but just in case
                    code_set.remove(c)
                    cov[neighbors[c, :]] -= 1
                    improved = True
                    break
                
                # Find a candidate d that covers all essential_words
                # d must be in intersection of N(ew) for all ew in essential_words
                if len(essential_words) == 1:
                    ew = essential_words[0]
                    candidates = neighbors[ew, :].tolist()
                else:
                    # Intersection of neighbor sets
                    cand_set = set(neighbors[essential_words[0], :].tolist())
                    for ew in essential_words[1:]:
                        cand_set.intersection_update(neighbors[ew, :].tolist())
                        if not cand_set:
                            break
                    candidates = list(cand_set)
                
                if not candidates:
                    continue
                
                # Filter out existing codewords
                valid_candidates = [d for d in candidates if d not in code_set]
                
                if not valid_candidates:
                    continue
                
                # Pick a random valid candidate
                d = rng.choice(valid_candidates)
                
                # Perform swap
                code_set.remove(c)
                code_set.add(d)
                cov[neighbors[c, :]] -= 1
                cov[neighbors[d, :]] += 1
                improved = True
                break # Restart reduction pass with new set
            
            if not improved:
                # No swap found. We are in a local minimum.
                # To escape, we could try a more aggressive move (e.g., double swap) 
                # or just break and restart from a different initialization.
                # Given the time constraint, breaking to allow a new restart is efficient.
                break

        return code_set

    best_code = None
    best_size = float('inf')
    
    # Run multiple restarts
    # Each restart: Greedy Init -> Local Search
    # We estimate ~1s per restart. We have ~110s. So ~50-100 restarts possible.
    # However, local search can be faster if it converges quickly.
    
    for i in range(200):
        if time.time() > deadline - 10:
            break
            
        rng = np.random.default_rng(12345 + i)
        
        # Greedy Init with randomization to explore different basins
        code = greedy_init(rng, randomize=(i > 0))
        
        # Local Search
        code = local_search(code, rng, deadline)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
            
        # If we hit the known best, we can stop early or try to improve further
        if best_size <= 486:
            # Try to improve a bit more if time permits
            if best_size < 486:
                break
            else:
                # Continue searching for < 486
                pass

    if best_code is None:
        # Fallback to a known structure if search fails (should not happen)
        # Tetracode based construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        best_code = set()
        import itertools
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                word = h + list(tail)
                val = 0
                for i, d in enumerate(word):
                    val += d * (q ** i)
                best_code.add(val)

    final_code = []
    for w in best_code:
        word = []
        temp = w
        for _ in range(n):
            word.append(temp % q)
            temp //= q
        final_code.append(word)
    
    return final_code
