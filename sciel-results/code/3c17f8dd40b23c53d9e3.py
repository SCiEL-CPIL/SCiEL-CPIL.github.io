"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np
from collections import defaultdict

def construct(q, n, radius):
    # Fallback for non-specific cases or if optimization fails
    def fallback():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    if q != 3 or n != 7 or radius != 1:
        return fallback()
        
    total_words = q ** n
    # Precompute neighbors for all words
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
        
    # Generate all words
    all_words = [idx_to_word(i) for i in range(total_words)]
    
    # Precompute neighbors (distance <= radius)
    # For radius 1, neighbors are the word itself + 1 word change in each of n positions
    neighbors = []
    for i in range(total_words):
        w = all_words[i]
        nbs = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w_copy = w[:]
                w_copy[pos] = new_val
                nbs.append(word_to_idx(w_copy))
        neighbors.append(nbs)
        
    neighbors_arr = np.array(neighbors)
    
    # Algebraic approach: use ternary Hamming code structure
    # The [7,4,3] ternary Hamming code has 3^4 = 81 codewords
    # We'll create a base covering code and exploit symmetries
    
    # Generate the ternary Hamming code [7,4,3]
    # Generator matrix (4x7)
    G = np.array([
        [1, 0, 0, 0, 1, 1, 2],
        [0, 1, 0, 0, 1, 2, 1],
        [0, 0, 1, 0, 2, 1, 1],
        [0, 0, 0, 1, 1, 2, 2]
    ], dtype=np.int64)
    
    # Generate all 81 codewords of the Hamming code
    hamming_codewords = []
    for msg in itertools.product(range(3), repeat=4):
        cw = np.array(msg, dtype=np.int64) @ G % 3
        hamming_codewords.append(cw.tolist())
    
    # Create parity check matrix H (3x7) for the Hamming code
    # Columns are all non-zero vectors in GF(3)^3, each appearing once
    # There are 3^3 - 1 = 26 non-zero vectors, but we need 7 columns
    # Actually, for ternary Hamming [7,4,3], we use 7 specific columns
    H = np.array([
        [1, 0, 0, 0, 1, 1, 2],
        [0, 1, 0, 0, 1, 2, 1],
        [0, 0, 1, 0, 2, 1, 1]
    ], dtype=np.int64)
    
    # For each word, compute its syndrome (H @ word^T mod 3)
    # Words with the same syndrome are in the same coset
    syndromes = []
    for w in all_words:
        w_arr = np.array(w, dtype=np.int64)
        syn = (H @ w_arr) % 3
        syndromes.append(tuple(syn.tolist()))
    
    # Group words by syndrome
    syndrome_to_words = defaultdict(list)
    for i, syn in enumerate(syndromes):
        syndrome_to_words[syn].append(i)
    
    # There are 27 possible syndromes (3^3)
    # Each coset has 2187/27 = 81 words (matching the Hamming code size)
    
    # Strategy: For each syndrome class, pick a representative
    # Then use local search to optimize the choice of representatives
    
    # Initial greedy: pick one word from each syndrome class
    base_code = []
    for syn in syndrome_to_words:
        words_in_class = syndrome_to_words[syn]
        # Pick the first one (or use greedy)
        base_code.append(words_in_class[0])
    
    # Now we have 27 codewords, one per syndrome class
    # But this may not cover all words within distance 1
    # We need to add more codewords to ensure coverage
    
    # Better approach: Use the Hamming code structure to create a covering code
    # The idea is to select codewords such that their distance-1 neighborhoods cover all 2187 words
    
    # Let's use a constructive approach:
    # Start with the zero word, then greedily add words that cover the most uncovered words
    
    def greedy_cover_algebraic():
        """Greedy cover using algebraic structure for initialization."""
        uncovered = np.ones(total_words, dtype=bool)
        code = []
        
        # First, add one word from each syndrome class to ensure we have coverage
        # across all cosets
        for syn in list(syndrome_to_words.keys()):
            words_in_class = syndrome_to_words[syn]
            # Find the word in this class that covers the most uncovered words
            best_word = None
            best_count = -1
            for w_idx in words_in_class:
                count = uncovered[neighbors_arr[w_idx]].sum()
                if count > best_count:
                    best_count = count
                    best_word = w_idx
            
            if best_word is not None:
                code.append(best_word)
                uncovered[neighbors_arr[best_word]] = False
        
        # Continue greedy until all covered
        while uncovered.any():
            scores = uncovered[neighbors_arr].sum(axis=1)
            idx = np.argmax(scores)
            code.append(idx)
            uncovered[neighbors_arr[idx]] = False
        
        return code
    
    # Run the algebraic greedy
    alg_code = greedy_cover_algebraic()
    
    # Now apply local search to improve
    # But first, let's also try the original greedy for comparison
    def greedy_cover(randomize=False):
        uncovered = np.ones(total_words, dtype=bool)
        code = []
        while uncovered.any():
            scores = uncovered[neighbors_arr].sum(axis=1)
            
            if randomize:
                max_score = scores.max()
                candidates = np.where(scores == max_score)[0]
                idx = np.random.choice(candidates)
            else:
                idx = np.argmax(scores)
                
            code.append(idx)
            uncovered[neighbors_arr[idx]] = False
            
        return code
    
    def greedy_cover(randomize=False):
        uncovered = np.ones(total_words, dtype=bool)
        code = []
        while uncovered.any():
            # Count how many uncovered words each candidate covers
            # neighbors_arr is (total_words, radius_size)
            # uncovered[neighbors_arr[i]] gives boolean vector of covered status for candidate i
            # sum gives count
            scores = uncovered[neighbors_arr].sum(axis=1)
            
            if randomize:
                # Get max score
                max_score = scores.max()
                # Get indices with max score
                candidates = np.where(scores == max_score)[0]
                # Pick one randomly
                idx = np.random.choice(candidates)
            else:
                idx = np.argmax(scores)
                
            code.append(idx)
            # Update uncovered
            uncovered[neighbors_arr[idx]] = False
            
        return code

    def reduce_code(code):
        code_set = set(code)
        # Count coverage for each word
        cover_count = np.zeros(total_words, dtype=int)
        for c in code:
            cover_count[neighbors_arr[c]] += 1
            
        changed = True
        while changed:
            changed = False
            # Iterate over a copy of the list
            for c in list(code_set):
                # Check if all neighbors of c are covered by at least 2 codewords
                # If min cover count > 1, we can remove c
                min_cov = cover_count[neighbors_arr[c]].min()
                if min_cov > 1:
                    cover_count[neighbors_arr[c]] -= 1
                    code_set.remove(c)
                    changed = True
        return list(code_set)
    
    def simulated_annealing(code, max_time=30.0):
        """Simulated annealing to refine the code size.
        
        Moves: remove a codeword (if coverage stays valid) or swap a codeword.
        Objective: minimize code size while maintaining full coverage.
        """
        code_set = set(code)
        cover_count = np.zeros(total_words, dtype=int)
        for c in code_set:
            cover_count[neighbors_arr[c]] += 1
        
        # Ensure initial coverage is valid
        if cover_count.min() < 1:
            return list(code_set)
        
        current_size = len(code_set)
        best_size = current_size
        best_set = set(code_set)
        
        start_sa = time.time()
        deadline_sa = start_sa + max_time
        
        # Temperature schedule
        T0 = 1.0
        T_min = 0.01
        cooling = 0.95
        
        T = T0
        rng = np.random.default_rng(42)
        
        while T > T_min and time.time() < deadline_sa:
            # Try a move: either remove a codeword or swap
            move_type = rng.random()
            
            if move_type < 0.3:
                # Try to remove a random codeword
                if len(code_set) == 0:
                    T *= cooling
                    continue
                c = rng.choice(list(code_set))
                nbs = neighbors_arr[c]
                # Check if removing c would leave any word uncovered
                would_uncover = (cover_count[nbs] == 1).any()
                if would_uncover:
                    # Can't remove without breaking coverage
                    T *= cooling
                    continue
                else:
                    # Safe to remove
                    cover_count[nbs] -= 1
                    code_set.remove(c)
                    current_size -= 1
                    if current_size < best_size:
                        best_size = current_size
                        best_set = set(code_set)
                    T *= cooling
            elif move_type < 0.7:
                # Try to swap: remove one codeword, add another
                if len(code_set) < 2:
                    T *= cooling
                    continue
                c_remove = rng.choice(list(code_set))
                # Pick a random candidate to add
                c_add = rng.integers(0, total_words)
                if c_add in code_set:
                    T *= cooling
                    continue
                
                nbs_remove = neighbors_arr[c_remove]
                nbs_add = neighbors_arr[c_add]
                
                # Check if removing c_remove and adding c_add maintains coverage
                temp_cover = cover_count.copy()
                temp_cover[nbs_remove] -= 1
                temp_cover[nbs_add] += 1
                
                if temp_cover.min() < 1:
                    # This swap breaks coverage
                    uncovered = (temp_cover < 1).sum()
                    # Accept with probability exp(-uncovered / T)
                    if rng.random() < np.exp(-uncovered / max(T, 1e-10)):
                        cover_count = temp_cover
                        code_set.remove(c_remove)
                        code_set.add(c_add)
                        # Size stays the same
                    T *= cooling
                else:
                    # Valid swap, size unchanged
                    cover_count = temp_cover
                    code_set.remove(c_remove)
                    code_set.add(c_add)
                    T *= cooling
            else:
                # Try to add a codeword that helps most (then we might be able to remove others)
                T *= cooling
        
        # Final reduction
        final_set = set(best_set)
        final_cover = np.zeros(total_words, dtype=int)
        for c in final_set:
            final_cover[neighbors_arr[c]] += 1
        
        changed = True
        while changed:
            changed = False
            for c in list(final_set):
                if final_cover[neighbors_arr[c]].min() > 1:
                    final_cover[neighbors_arr[c]] -= 1
                    final_set.remove(c)
                    changed = True
        
        return list(final_set)
    
    # Apply symmetry: generate all affine transformations of the space
    # and take the union of transformed codes
    def apply_affine_transform(code, permutation, shifts):
        """Apply an affine transformation to a set of codewords.
        
        Args:
            code: list of codeword indices
            permutation: list of position permutations (length 7)
            shifts: list of shifts per position (length 7)
        
        Returns:
            list of transformed codeword indices
        """
        transformed = []
        for idx in code:
            w = all_words[idx]
            new_w = [(w[permutation[i]] + shifts[i]) % 3 for i in range(7)]
            transformed.append(word_to_idx(new_w))
        return transformed
    
    # Generate a set of useful affine transformations
    # Identity
    identity_perm = list(range(7))
    identity_shifts = [0] * 7
    
    # Some position permutations (we'll use a subset to keep it manageable)
    # All cyclic shifts
    cyclic_perms = []
    for shift in range(7):
        perm = [(i + shift) % 7 for i in range(7)]
        cyclic_perms.append(perm)
    
    # Some reversal-like permutations
    reversal_perm = [6, 5, 4, 3, 2, 1, 0]
    half_turn_perm = [3, 4, 5, 6, 0, 1, 2]
    
    all_perms = cyclic_perms + [reversal_perm, half_turn_perm]
    # Remove duplicates
    unique_perms = []
    seen = set()
    for p in all_perms:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            unique_perms.append(p)
    
    # Generate transformed codes and take union
    def expand_with_symmetry(code):
        """Expand a code using affine symmetries and take the union."""
        all_codewords = set()
        
        for perm in unique_perms:
            for shifts in itertools.product(range(3), repeat=7):
                # Only use shifts that are "simple" to keep the set manageable
                # Use shifts where most are 0
                non_zero = sum(1 for s in shifts if s != 0)
                if non_zero > 2:
                    continue
                
                transformed = apply_affine_transform(code, perm, shifts)
                all_codewords.update(transformed)
        
        return list(all_codewords)
    
    best_code = None
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Strategy:
    # 1. Try algebraic greedy
    # 2. Try standard greedy
    # 3. Apply local search to both
    # 4. Optionally expand with symmetry for the best result
    
    num_rounds = 0
    while time.time() < deadline:
        num_rounds += 1
        
        # Alternate between algebraic and standard greedy
        if num_rounds % 3 == 1:
            cand = greedy_cover_algebraic()
        elif num_rounds % 3 == 2:
            cand = greedy_cover(randomize=False)
        else:
            cand = greedy_cover(randomize=True)
            
        cand = reduce_code(cand)
        
        # Apply simulated annealing with time limit
        sa_time = min(5.0, max(1.0, deadline - time.time() - 5.0))
        if sa_time > 1.0:
            cand_sa = simulated_annealing(cand, max_time=sa_time)
            if len(cand_sa) < len(cand):
                cand = cand_sa
        
        if best_code is None or len(cand) < len(best_code):
            best_code = cand
            # If we have a good result, try symmetry expansion
            if len(cand) <= 195 and time.time() < deadline - 20:
                expanded = expand_with_symmetry(cand)
                expanded_reduced = reduce_code(expanded)
                if len(expanded_reduced) < len(best_code):
                    best_code = expanded_reduced
                    # Polish the expanded result
                    if time.time() < deadline - 10:
                        polish_time = min(10.0, deadline - time.time() - 5.0)
                        if polish_time > 3.0:
                            cand_polish = simulated_annealing(best_code, max_time=polish_time)
                            if len(cand_polish) < len(best_code):
                                best_code = cand_polish
    
    if best_code is None:
        best_code = greedy_cover(randomize=False)
        best_code = reduce_code(best_code)
        
    # Convert indices back to words
    return [all_words[i] for i in best_code]
# EVOLVE-BLOCK-END
