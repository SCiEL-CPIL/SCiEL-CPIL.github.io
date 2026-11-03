"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = 110.0  # seconds, safety margin before 120s limit

    # For the specific case: q=3, n=8, radius=1
    # Start with the tetracode direct sum construction (486 words)
    if q == 3 and n == 8 and radius == 1:
        # Tetracode: 9 words of length 4, perfect code of covering radius 1
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        
        # Direct sum: 9 * 3^4 = 486 words
        code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=4)]
        
        # Try to improve via local search: swap operations
        # A swap: remove one codeword, add another, maintaining coverage
        # We use a coverage count array to track how many codewords cover each word
        
        # Precompute: for each codeword, the set of words it covers (itself + all at distance 1)
        # Space size: 3^8 = 6561
        space_size = 3 ** n
        
        # Convert code to numpy for speed
        code_arr = np.array(code, dtype=np.int8)
        
        # Coverage count: for each word in space, how many codewords are within distance <= radius
        # We'll use an array indexed by the integer representation of each word
        def word_to_int(w, q, n):
            val = 0
            for i in range(n):
                val = val * q + w[i]
            return val
        
        def int_to_word(val, q, n):
            w = [0] * n
            for i in range(n - 1, -1, -1):
                w[i] = val % q
                val //= q
            return w
        
        # Build coverage counts
        coverage = np.zeros(space_size, dtype=np.int32)
        
        # For each codeword, increment coverage for itself and all neighbors at distance 1
        for cw in code_arr:
            cw_int = 0
            for i in range(n):
                cw_int = cw_int * q + cw[i]
            coverage[cw_int] += 1
            # Neighbors at distance 1
            for pos in range(n):
                for new_val in range(q):
                    if new_val != cw[pos]:
                        nb = cw.copy()
                        nb[pos] = new_val
                        nb_int = 0
                        for i in range(n):
                            nb_int = nb_int * q + nb[i]
                        coverage[nb_int] += 1
        
        # Check initial validity
        assert np.all(coverage >= 1), "Initial code is not valid"
        
        best_code = code[:]
        best_size = len(code)
        
        # Local search: try to remove codewords by swapping
        # Strategy: try to find a codeword that can be removed, then find a replacement
        # that covers the newly uncovered words while potentially allowing further removals
        
        # Simulated annealing approach:
        # State: current code (set of codewords)
        # Move: swap one codeword for another (keep same size)
        # Or: try to remove a codeword if coverage remains >= 1 for all words
        # We'll use a greedy approach first, then simulated annealing
        
        # Greedy removal: repeatedly try to remove a codeword if coverage stays >= 1
        removed = 0
        for _ in range(50):  # Try many passes
            if time.time() - start_time > deadline - 20:
                break
            progress = False
            # Shuffle order for randomness
            indices = list(range(len(code_arr)))
            random.shuffle(indices)
            for idx in indices:
                if time.time() - start_time > deadline - 20:
                    break
                cw = code_arr[idx]
                # Check if removing this codeword leaves all words covered
                # Decrement coverage for this codeword's sphere
                cw_int = 0
                for i in range(n):
                    cw_int = cw_int * q + cw[i]
                
                # Temporarily decrement
                coverage[cw_int] -= 1
                min_cov = 1
                for pos in range(n):
                    for new_val in range(q):
                        if new_val != cw[pos]:
                            nb = cw.copy()
                            nb[pos] = new_val
                            nb_int = 0
                            for i in range(n):
                                nb_int = nb_int * q + nb[i]
                            coverage[nb_int] -= 1
                            if coverage[nb_int] < min_cov:
                                min_cov = coverage[nb_int]
                
                if min_cov >= 1:
                    # Safe to remove
                    code_arr = np.delete(code_arr, idx, axis=0)
                    removed += 1
                    progress = True
                else:
                    # Restore coverage
                    coverage[cw_int] += 1
                    for pos in range(n):
                        for new_val in range(q):
                            if new_val != cw[pos]:
                                nb = cw.copy()
                                nb[pos] = new_val
                                nb_int = 0
                                for i in range(n):
                                    nb_int = nb_int * q + nb[i]
                                coverage[nb_int] += 1
                
                if time.time() - start_time > deadline - 20:
                    break
            
            if not progress:
                break
        
        if removed > 0:
            best_code = [list(row) for row in code_arr]
            best_size = len(code_arr)
        
        # Simulated annealing: try swaps to enable further removals
        if time.time() - start_time < deadline - 10:
            # Current state
            current_code = code_arr.copy()
            current_coverage = coverage.copy()
            current_size = len(current_code)
            
            # Temperature schedule
            T = 100.0
            T_min = 0.1
            cooling = 0.95
            max_iters = 5000
            
            for it in range(max_iters):
                if time.time() - start_time > deadline - 5:
                    break
                if T < T_min:
                    break
                
                # Pick a random codeword to potentially swap
                idx = random.randint(0, current_size - 1)
                cw = current_code[idx]
                
                # Generate a random replacement codeword
                new_cw = np.array([random.randint(0, q - 1) for _ in range(n)], dtype=np.int8)
                
                if np.array_equal(cw, new_cw):
                    continue
                
                # Compute change in coverage
                # Remove cw, add new_cw
                # Check if after swap, all words still covered
                # And if we can then remove some codeword
                
                cw_int = 0
                for i in range(n):
                    cw_int = cw_int * q + cw[i]
                
                new_cw_int = 0
                for i in range(n):
                    new_cw_int = new_cw_int * q + new_cw[i]
                
                # Get spheres
                def get_sphere(cw_arr):
                    sphere = [0] * (n * (q - 1) + 1)
                    sphere[0] = 0  # placeholder
                    vals = []
                    v = 0
                    for i in range(n):
                        v = v * q + cw_arr[i]
                    vals.append(v)
                    for pos in range(n):
                        for new_val in range(q):
                            if new_val != cw_arr[pos]:
                                nb = cw_arr.copy()
                                nb[pos] = new_val
                                nv = 0
                                for i in range(n):
                                    nv = nv * q + nb[i]
                                vals.append(nv)
                    return vals
                
                cw_sphere = get_sphere(cw)
                new_cw_sphere = get_sphere(new_cw)
                
                # Check if swap maintains coverage
                # Words only covered by cw (not by new_cw) must still be covered by others
                # This is complex; simpler to just do the swap and check
                
                # Perform swap
                current_code[idx] = new_cw
                # Update coverage
                for v in cw_sphere:
                    current_coverage[v] -= 1
                for v in new_cw_sphere:
                    current_coverage[v] += 1
                
                # Check validity
                if np.min(current_coverage) < 1:
                    # Revert
                    current_code[idx] = cw
                    for v in cw_sphere:
                        current_coverage[v] += 1
                    for v in new_cw_sphere:
                        current_coverage[v] -= 1
                    continue
                
                # Now try to remove a codeword
                # Find a codeword whose removal keeps coverage >= 1
                removed_idx = -1
                for j in range(current_size):
                    if j == idx:
                        continue
                    cj = current_code[j]
                    cj_int = 0
                    for i in range(n):
                        cj_int = cj_int * q + cj[i]
                    
                    cj_sphere = get_sphere(cj)
                    
                    # Check if removing cj keeps coverage >= 1
                    can_remove = True
                    for v in cj_sphere:
                        if current_coverage[v] <= 1:
                            can_remove = False
                            break
                    
                    if can_remove:
                        removed_idx = j
                        break
                
                if removed_idx >= 0:
                    # Perform removal
                    cj = current_code[removed_idx]
                    cj_sphere = get_sphere(cj)
                    for v in cj_sphere:
                        current_coverage[v] -= 1
                    current_code = np.delete(current_code, removed_idx, axis=0)
                    current_size -= 1
                    
                    if current_size < best_size:
                        best_code = [list(row) for row in current_code]
                        best_size = current_size
                else:
                    # No removal possible; decide whether to keep the swap
                    # Accept with some probability (simulated annealing)
                    # Energy change: 0 (same size), so always accept for exploration
                    pass  # Keep the swap
                
                T *= cooling
            
            # Final greedy removal pass
            for _ in range(20):
                if time.time() - start_time > deadline - 2:
                    break
                progress = False
                indices = list(range(len(current_code)))
                random.shuffle(indices)
                for idx in indices:
                    if time.time() - start_time > deadline - 2:
                        break
                    cw = current_code[idx]
                    cw_int = 0
                    for i in range(n):
                        cw_int = cw_int * q + cw[i]
                    
                    cw_sphere = get_sphere(cw)
                    
                    can_remove = True
                    for v in cw_sphere:
                        if current_coverage[v] <= 1:
                            can_remove = False
                            break
                    
                    if can_remove:
                        for v in cw_sphere:
                            current_coverage[v] -= 1
                        current_code = np.delete(current_code, idx, axis=0)
                        progress = True
                
                if not progress:
                    break
            
            if len(current_code) < best_size:
                best_code = [list(row) for row in current_code]
                best_size = len(current_code)
        
        return best_code
    
    # General case: greedy set cover
    # This is a fallback for other (q, n, radius) values
    space_size = q ** n
    
    def word_to_int(w, q, n):
        val = 0
        for i in range(n):
            val = val * q + w[i]
        return val
    
    def int_to_word(val, q, n):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = val % q
            val //= q
        return w
    
    # Generate all words
    all_words = [int_to_word(i, q, n) for i in range(space_size)]
    
    # For each word, compute its sphere (words within distance <= radius)
    # This is expensive for large spaces, so we'll be careful
    
    # Greedy: repeatedly pick the word that covers the most uncovered words
    uncovered = set(range(space_size))
    code = []
    
    # Precompute spheres for all words (expensive but necessary for correctness)
    # For small spaces this is fine
    if space_size > 100000:
        # Too large; return a trivial upper bound
        # Just return all words (not optimal but valid)
        return all_words
    
    spheres = []
    for i in range(space_size):
        w = all_words[i]
        sphere = set()
        for j in range(space_size):
            # Compute Hamming distance
            dist = sum(1 for k in range(n) if w[k] != all_words[j][k])
            if dist <= radius:
                sphere.add(j)
        spheres.append(sphere)
    
    while uncovered and len(uncovered) > 0:
        if time.time() - start_time > deadline - 5:
            break
        # Find the word that covers the most uncovered words
        best_idx = -1
        best_count = -1
        for i in range(space_size):
            count = len(spheres[i] & uncovered)
            if count > best_count:
                best_count = count
                best_idx = i
            if time.time() - start_time > deadline - 5:
                break
        if best_idx < 0:
            break
        code.append(all_words[best_idx])
        uncovered -= spheres[best_idx]
    
    return code
# EVOLVE-BLOCK-END
