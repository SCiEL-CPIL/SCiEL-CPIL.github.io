import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    """
    Construct a ternary covering code K_3(6,1).
    Returns a list of codewords (lists of integers).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Fallback for non-standard parameters
    if q != 3 or n != 6 or radius != 1:
        if n <= 4:
            return [list(t) for t in itertools.product(range(q), repeat=n)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    total_words = 3 ** n  # 729
    
    # --- 1. Precompute Data Structures ---
    
    # Helper: Convert index to word tuple
    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[n - 1 - i] = temp % 3
            temp //= 3
        return tuple(w)
    
    # Helper: Convert word tuple to index
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * 3 + val
        return idx

    # Precompute spheres (Hamming balls of radius 1)
    # spheres[i] is a list of indices
    spheres = [None] * total_words
    for idx in range(total_words):
        w = list(idx_to_word(idx))
        sphere_set = {idx}
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                sphere_set.add(word_to_idx(w))
                w[pos] = orig_val
        spheres[idx] = list(sphere_set)
    
    # Precompute coverers: for each word w, which codewords cover it
    # coverers[w] = list of c such that w in spheres[c]
    coverers = [None] * total_words
    for w in range(total_words):
        coverers[w] = []
    
    for c in range(total_words):
        for w in spheres[c]:
            coverers[w].append(c)
    
    # Convert coverers to sets for fast membership testing
    coverers_set = [set(c) for c in coverers]
    
    # --- 2. Initial Code Construction ---
    # Direct sum of tetracode (9 words, len 4) and all words on remaining n-4 coords
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_codewords.append(h + list(tail))
    
    initial_indices = [word_to_idx(cw) for cw in initial_codewords]
    
    # --- 3. Helper Functions for Search ---
    
    def compute_counts(code_indices):
        counts = np.zeros(total_words, dtype=np.int32)
        for idx in code_indices:
            counts[spheres[idx]] += 1
        return counts

    def reduce_code(code_indices, counts=None):
        """
        Greedily remove redundant codewords.
        Returns (new_code_indices, new_counts)
        """
        if counts is None:
            counts = compute_counts(code_indices)
        
        code_set = set(code_indices)
        changed = True
        while changed:
            changed = False
            for idx in list(code_set):
                # Check if all neighbors are covered by >1 codeword
                # If min(counts[spheres[idx]]) > 1, then idx is redundant
                if np.min(counts[spheres[idx]]) > 1:
                    code_set.remove(idx)
                    counts[spheres[idx]] -= 1
                    changed = True
        
        return list(code_set), counts

    def is_valid_swap(c_remove, c_add, code_set, counts):
        """
        Check if replacing c_remove with c_add keeps the code valid.
        c_remove must be in code_set, c_add must not be in code_set.
        """
        if c_remove not in code_set or c_add in code_set:
            return False
        
        # Words that become uncovered if we remove c_remove
        # These are words in Sphere(c_remove) that have count == 1
        # We need Sphere(c_add) to cover all these words.
        
        sphere_remove = spheres[c_remove]
        sphere_add = spheres[c_add]
        
        # Create a set for fast lookup of sphere_add
        sphere_add_set = set(sphere_add)
        
        for w in sphere_remove:
            if counts[w] == 1:
                if w not in sphere_add_set:
                    return False
        return True

    # --- 4. Main Search Loop ---
    
    rng = random.Random(42)
    
    # Start with initial code
    current_code, current_counts = reduce_code(initial_indices)
    best_code = current_code[:]
    best_counts = current_counts.copy()
    best_len = len(best_code)
    
    # If we already hit 73, we can try to improve, but 73 is the target.
    # We will run ILS.
    
    all_indices = list(range(total_words))
    
    # Strategy:
    # 1. Reduce current code.
    # 2. Try to find a beneficial swap.
    # 3. If no beneficial swap, perturb.
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        # 1. Reduce
        current_code, current_counts = reduce_code(current_code, current_counts)
        
        if len(current_code) < best_len:
            best_len = len(current_code)
            best_code = current_code[:]
            best_counts = current_counts.copy()
            no_improve_count = 0
            if best_len <= 71: # Theoretical lower bound is 71, best known 73. If we hit 71, we are done.
                break
        else:
            no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            # Perturb
            # Pick a random codeword to remove and a random one to add
            if current_code:
                c_remove = rng.choice(current_code)
                # Pick a random c_add not in current_code
                # To avoid O(N) selection, we can pick random indices until we find one not in set
                current_set = set(current_code)
                while True:
                    c_add = rng.randint(0, total_words - 1)
                    if c_add not in current_set:
                        break
                
                # Check if swap is valid
                if is_valid_swap(c_remove, c_add, current_set, current_counts):
                    # Perform swap
                    current_set.remove(c_remove)
                    current_set.add(c_add)
                    # Update counts
                    current_counts[spheres[c_remove]] -= 1
                    current_counts[spheres[c_add]] += 1
                    current_code = list(current_set)
                    no_improve_count = 0
                else:
                    # If invalid, try a few more random swaps before giving up on perturbation
                    found_valid = False
                    for _ in range(100):
                        c_remove = rng.choice(current_code)
                        while True:
                            c_add = rng.randint(0, total_words - 1)
                            if c_add not in current_set:
                                break
                        if is_valid_swap(c_remove, c_add, current_set, current_counts):
                            current_set.remove(c_remove)
                            current_set.add(c_add)
                            current_counts[spheres[c_remove]] -= 1
                            current_counts[spheres[c_add]] += 1
                            current_code = list(current_set)
                            found_valid = True
                            no_improve_count = 0
                            break
                    if not found_valid:
                        # Reset to best code to avoid drifting too far
                        current_code = best_code[:]
                        current_counts = best_counts.copy()
                        no_improve_count = 0
            continue

        # 2. Try to find a beneficial swap
        # A swap is "beneficial" if it allows further reduction.
        # We iterate through codewords in current_code.
        # For each, we try to find a c_add that makes the code valid.
        # Then we check if the new code can be reduced to a smaller size.
        
        # To save time, we only try a subset of swaps per iteration?
        # Or we do one full pass of swaps.
        
        improved = False
        current_set = set(current_code)
        
        # Randomize order of removal candidates
        candidates = current_code[:]
        rng.shuffle(candidates)
        
        for c_remove in candidates:
            if c_remove not in current_set:
                continue
            
            # Find U: words in Sphere(c_remove) with count 1
            # If U is empty, it's redundant, but we just reduced, so U should not be empty.
            # However, counts might have changed if we did a swap earlier in this loop? 
            # We are doing sequential swaps in this loop.
            
            # We need to check validity against the *current* state.
            # is_valid_swap uses current_counts.
            
            # We want to find a c_add such that:
            # 1. Swap is valid.
            # 2. After swap, we can reduce the code to size < best_len (or < current_len).
            
            # To avoid O(N^2) with reduction checks (which are O(N * SphereSize)),
            # we can just perform the swap if valid, then try to reduce.
            # If reduction improves, keep it. If not, revert.
            
            # Optimization: Only try c_add that cover "critical" words.
            # Critical words are those in Sphere(c_remove) with count 1.
            # Let U = { w in Sphere(c_remove) | counts[w] == 1 }
            
            critical_words = [w for w in spheres[c_remove] if current_counts[w] == 1]
            
            if not critical_words:
                # This shouldn't happen after reduction, but if it does, we can just remove it.
                # But we are in swap mode.
                continue

            # We need c_add such that U subseteq Sphere(c_add).
            # Instead of checking all 729, we can check only those c_add that cover the first critical word.
            # Precomputation: For each word w, list of codewords that cover it?
            # This is static!
            # coverers[w] = [c for c in 0..728 if w in spheres[c]]
            
            # Let's precompute this once.
            # coverers is a list of lists.
            
            # We'll do this precomputation outside the main loop.
            
            # For now, just iterate all c_add not in current_set.
            # This is 729 iterations.
            # Inside, check validity (13 ops).
            # If valid, perform swap, reduce, check size.
            
            # To make it faster, we can break early if we find a good swap.
            
            # Let's pick a random subset of c_add to try?
            # No, let's try all, but break if we find an improvement.
            
            # Use coverers to only consider c_add that cover all critical words
            # Start with candidates from the first critical word
            candidate_c_adds = coverers[critical_words[0]]
            
            # Filter to only those that cover all critical words
            valid_candidates = []
            for c_add in candidate_c_adds:
                if c_add in current_set:
                    continue
                # Check if c_add covers all critical words
                valid = True
                for w in critical_words:
                    if w not in coverers_set[c_add]:  # This is wrong, should be checking if c_add covers w
                        valid = False
                        break
                if valid:
                    valid_candidates.append(c_add)
            
            # Actually, let's use a simpler approach: iterate through all potential c_add
            # but use the coverers to quickly filter
            for c_add in all_indices:
                if c_add in current_set:
                    continue
                
                # Quick check: Does c_add cover all critical words?
                # Use precomputed coverers for efficiency
                valid = True
                for w in critical_words:
                    if c_add not in coverers_set[w]:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # It is a valid swap.
                # Perform it temporarily
                current_set.remove(c_remove)
                current_set.add(c_add)
                
                # Update counts
                current_counts[spheres[c_remove]] -= 1
                current_counts[spheres[c_add]] += 1
                
                # Try to reduce
                new_code, new_counts = reduce_code(list(current_set), current_counts)
                
                if len(new_code) < len(current_code):
                    # Accept
                    current_code = new_code
                    current_counts = new_counts
                    current_set = set(current_code)
                    improved = True
                    break
                else:
                    # Revert
                    current_set.remove(c_add)
                    current_set.add(c_remove)
                    current_counts[spheres[c_add]] -= 1
                    current_counts[spheres[c_remove]] += 1
            
            if improved:
                break
            if time.time() > deadline - 5.0:
                break

    # Final check: ensure best_code is valid and converted to lists
    # Verify coverage
    final_counts = np.zeros(total_words, dtype=np.int32)
    for idx in best_code:
        final_counts[spheres[idx]] += 1
    if np.any(final_counts == 0):
        # Fallback to initial if something went wrong
        return [list(idx_to_word(idx)) for idx in initial_indices]
    
    return [list(idx_to_word(idx)) for idx in best_code]
