"""Combined program for ternary covering code K_3(7,1)."""
import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    # Fallback for non-specific cases
    def fallback():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(c) for c in itertools.product(range(q), repeat=n)]
    
    if q != 3 or n != 7 or radius != 1:
        return fallback()
        
    total_words = q ** n  # 2187
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for 120s limit
    
    # Precompute neighbors and cover bits
    # Map word to index
    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx
        
    def idx_to_word(idx):
        w = []
        for _ in range(n):
            w.append(idx % q)
            idx //= q
        return w[::-1]
        
    all_words = [idx_to_word(i) for i in range(total_words)]
    
    neighbors = []
    cover_bits = []
    
    # Precompute powers for fast index calculation
    powers = [1, 3, 9, 27, 81, 243, 729]
    
    for i in range(total_words):
        # Decode word to digits
        digits = [0]*7
        tmp = i
        for pos in range(7):
            digits[pos] = tmp % 3
            tmp //= 3
            
        nbs = [i]
        bit = 1 << i
        
        for pos in range(7):
            orig_val = digits[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                diff = new_val - orig_val
                nb_idx = i + diff * powers[pos]
                nbs.append(nb_idx)
                bit |= (1 << nb_idx)
        neighbors.append(nbs)
        cover_bits.append(bit)
        
    # Convert neighbors to numpy arrays for fast min operations
    neighbors_arr = np.array(neighbors, dtype=int)
    
    def greedy_cover(randomize=False):
        uncovered = (1 << total_words) - 1
        code = []
        remaining = list(range(total_words))
        while uncovered:
            best_idx = -1
            max_cov = -1
            # If randomize, we can shuffle remaining or pick randomly from top-k
            # For speed, we just iterate.
            for c in remaining:
                cov = (uncovered & cover_bits[c]).bit_count()
                if cov > max_cov:
                    max_cov = cov
                    best_idx = c
                    if randomize and max_cov > 0:
                        # Optional: break early if we find a good one?
                        # No, greedy needs best.
                        pass
            
            if best_idx == -1:
                break
                
            code.append(best_idx)
            uncovered &= ~cover_bits[best_idx]
            remaining.remove(best_idx)
            
        return code

    def local_search(code):
        code_set = set(code)
        # Use Python lists for counts to avoid NumPy overhead in hot loop
        cover_count = [0] * total_words
        for c in code:
            for nb in neighbors[c]:
                cover_count[nb] += 1
                
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Pass 1: Removals
            # Iterate in random order to break symmetry
            current_list = list(code_set)
            random.shuffle(current_list)
            
            for c in current_list:
                if c not in code_set:
                    continue
                # Check if safe to remove
                safe = True
                for nb in neighbors[c]:
                    if cover_count[nb] <= 1:
                        safe = False
                        break
                if safe:
                    for nb in neighbors[c]:
                        cover_count[nb] -= 1
                    code_set.remove(c)
                    improved = True
                    
            if improved:
                continue
                
            # Pass 2: Swaps
            # Try to find a valid swap to perturb the state
            # We limit attempts to avoid wasting time on dead ends
            # Increased attempts due to speedup from pure Python lists
            for _ in range(200):
                if not code_set:
                    break
                c = random.choice(list(code_set))
                # Get critical words (covered only by c)
                crit = [w for w in neighbors[c] if cover_count[w] == 1]
                if not crit:
                    # Should have been removed, but maybe due to order?
                    # Force removal
                    for nb in neighbors[c]:
                        cover_count[nb] -= 1
                    code_set.remove(c)
                    improved = True
                    break
                    
                # Try random x
                x = random.randint(0, total_words - 1)
                if x in code_set:
                    continue
                    
                # Check if x covers all critical words of c
                valid = True
                for w in crit:
                    if not (cover_bits[x] >> w) & 1:
                        valid = False
                        break
                        
                if valid:
                    # Perform swap
                    for nb in neighbors[c]:
                        cover_count[nb] -= 1
                    for nb in neighbors[x]:
                        cover_count[nb] += 1
                    code_set.remove(c)
                    code_set.add(x)
                    improved = True
                    break
                    
        return list(code_set)

    def lp_relaxation_rounding():
        """
        LP-Relaxation with Rounding using vertex-transitivity.
        The dual of the covering code problem is the maximum packing problem.
        We use a simple iterative rounding scheme that exploits symmetry.
        """
        # Start with uniform distribution over all words
        # This is the LP relaxation solution due to vertex-transitivity
        # Then round by selecting words with highest "weight"
        
        # Simple approach: use greedy with a bias toward well-spread solutions
        # But first, let's try a simple LP-based initialization
        
        # The LP relaxation for covering is:
        # min sum x_i
        # s.t. sum_{j: d(i,j) <= 1} x_j >= 1 for all i
        # x_i >= 0
        
        # Due to vertex-transitivity, the optimal LP solution is x_i = 1/15 for all i
        # (since each word covers 15 words: itself + 14 neighbors)
        # But we need integer solutions.
        
        # Strategy: Use a random subset of words that is "well-spread"
        # We can do this by:
        # 1. Start with all words
        # 2. Iteratively remove words that are "redundant" given the current selection
        # 3. Use a probabilistic approach based on the LP solution
        
        # Simpler: Use a random permutation and take the first k words that cover everything
        # where k is estimated from the LP bound
        
        # LP bound: 2187 / 15 = 145.8, so we need at least 146 words
        # Let's try to find a solution with around 180-190 words
        
        # Use a randomized greedy with a different strategy
        uncovered = (1 << total_words) - 1
        code = []
        remaining = list(range(total_words))
        random.shuffle(remaining)
        
        while uncovered:
            best_idx = -1
            max_cov = -1
            # Pick the word that covers the most uncovered words
            # But add some randomness to break ties
            candidates = []
            for c in remaining:
                cov = (uncovered & cover_bits[c]).bit_count()
                if cov > max_cov:
                    max_cov = cov
                    candidates = [c]
                elif cov == max_cov:
                    candidates.append(c)
                    
            if not candidates:
                break
                
            best_idx = random.choice(candidates)
            code.append(best_idx)
            uncovered &= ~cover_bits[best_idx]
            remaining.remove(best_idx)
            
        return code

    best_code = None
    
    # Run multiple rounds
    # First, a deterministic greedy
    cand = greedy_cover(randomize=False)
    cand = local_search(cand)
    best_code = cand
    
    # Then, LP-relaxation-based initialization
    if time.time() < deadline:
        cand = lp_relaxation_rounding()
        cand = local_search(cand)
        if best_code is None or len(cand) < len(best_code):
            best_code = cand
    
    # Then, randomized restarts
    while time.time() < deadline:
        # Randomized greedy
        # To make it different, we can shuffle the order of evaluation in greedy
        # Or just run local search on the same start? No, need diversity.
        # Let's implement a slightly randomized greedy
        def greedy_cover_random():
            uncovered = (1 << total_words) - 1
            code = []
            remaining = list(range(total_words))
            while uncovered:
                # Pick a random subset of candidates? Or just pick best with some noise?
                # Simple: Pick best, but if multiple have same score, pick random.
                best_idx = -1
                max_cov = -1
                candidates = []
                for c in remaining:
                    cov = (uncovered & cover_bits[c]).bit_count()
                    if cov > max_cov:
                        max_cov = cov
                        candidates = [c]
                    elif cov == max_cov:
                        candidates.append(c)
                        
                if not candidates:
                    break
                    
                best_idx = random.choice(candidates)
                code.append(best_idx)
                uncovered &= ~cover_bits[best_idx]
                remaining.remove(best_idx)
            return code

        cand = greedy_cover_random()
        cand = local_search(cand)
        
        if best_code is None or len(cand) < len(best_code):
            best_code = cand
            
    if best_code is None:
        best_code = greedy_cover(randomize=False)
        best_code = local_search(best_code)
        
    return [all_words[i] for i in best_code]
