"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(q, n, radius):
    """Construct a covering code of length n over alphabet {0..q-1} with covering radius.
    
    Uses greedy set cover with removal-based optimization for the ternary case.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback: direct sum construction for other parameters
        # For general case, use a simple greedy approach
        return _greedy_construct_general(q, n, radius, deadline)
    
    # Specialized for K_3(6,1)
    return _construct_k3_6_1(deadline)


def _greedy_construct_general(q, n, radius, deadline):
    """General greedy set cover for covering codes."""
    all_words = list(itertools.product(range(q), repeat=n))
    total = q ** n
    
    # Precompute neighborhoods
    def hamming_dist(a, b):
        return sum(1 for i in range(n) if a[i] != b[i])
    
    # Greedy: pick codeword that covers most uncovered words
    uncovered = set(all_words)
    codewords = []
    
    while uncovered and time.time() < deadline - 5:
        best_word = None
        best_count = -1
        for w in uncovered:
            count = 0
            for d in range(radius + 1):
                # Count words at distance d from w that are uncovered
                # This is expensive, so use sampling for large spaces
                pass
            # Simplified: just count neighbors
            for v in uncovered:
                if hamming_dist(w, v) <= radius:
                    count += 1
            if count > best_count:
                best_count = count
                best_word = w
        
        if best_word is None:
            break
        
        codewords.append(best_word)
        # Remove covered words
        to_remove = set()
        for v in uncovered:
            if hamming_dist(best_word, v) <= radius:
                to_remove.add(v)
        uncovered -= to_remove
    
    return [list(w) for w in codewords]


def _construct_k3_6_1(deadline):
    """Specialized construction for K_3(6,1) using bitset-based greedy removal and swap local search."""
    
    # 1. Precompute all 729 words and map them to indices
    all_words = list(itertools.product(range(3), repeat=6))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N_WORDS = len(all_words)
    
    # 2. Precompute coverage bitmask for every possible word in the space
    # mask[w] is an integer where bit i is 1 if all_words[i] is within distance 1 of w
    # We need this for any potential codeword, so we compute for all 729 words
    cover_masks = [0] * N_WORDS
    for idx, w in enumerate(all_words):
        mask = 0
        # The word itself
        mask |= (1 << idx)
        # Words at distance 1
        for pos in range(6):
            for val in range(3):
                if w[pos] == val:
                    continue
                new_word = list(w)
                new_word[pos] = val
                new_idx = word_to_idx[tuple(new_word)]
                mask |= (1 << new_idx)
        cover_masks[idx] = mask
    
    FULL_MASK = (1 << N_WORDS) - 1
    
    # 3. Initialize with direct sum construction (81 codewords)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    # Convert to indices
    current_indices = [word_to_idx[tuple(cw)] for cw in initial_codewords]
    
    # 4. Greedy Removal Loop
    # We maintain a list of active indices
    active = list(current_indices)
    
    def is_valid_cover(indices):
        """Check if the set of indices covers all words."""
        union = 0
        for idx in indices:
            union |= cover_masks[idx]
        # Check if union is all 1s (N_WORDS bits)
        return (union == FULL_MASK)
    
    def try_remove(active_list):
        """Attempt to remove redundant codewords."""
        removed_any = True
        while removed_any and time.time() < deadline - 20:
            removed_any = False
            current_union = 0
            for idx in active_list:
                current_union |= cover_masks[idx]
            
            new_active = []
            for idx in active_list:
                # Check if removing idx leaves a valid cover
                # new_union = current_union without cover_masks[idx]?
                # No, we need to recalculate union of the rest.
                union_others = 0
                for jdx in active_list:
                    if jdx != idx:
                        union_others |= cover_masks[jdx]
                
                if union_others == FULL_MASK:
                    # We can remove idx
                    removed_any = True
                    continue
                else:
                    new_active.append(idx)
            
            if removed_any:
                active_list = new_active
        
        return active_list

    active = try_remove(active)
    
    # 5. Swap-based Local Search
    # Try to replace a codeword with another word to potentially enable further removals.
    # We iterate through active codewords and try swapping them with any word in the space.
    # If the new set is a valid cover, we accept it (size remains same, but structure changes).
    # Then we try to remove again.
    
    best_size = len(active)
    best_active = list(active)
    
    # We will run multiple rounds of swap/remove
    # To improve efficiency, we precompute U_except for the current set once per round
    # and try all swaps for each position.
    
    for round_num in range(200): # Increased rounds to allow more search within time limit
        if time.time() > deadline - 10:
            break
        
        improved = False
        # Shuffle active to break symmetry
        random.shuffle(active)
        
        # Precompute U_except for each position in active
        # U_except[i] is the union of cover_masks of all active[j] where j != i
        U_except = [0] * len(active)
        for i_idx in range(len(active)):
            u = 0
            for j_idx in range(len(active)):
                if i_idx != j_idx:
                    u |= cover_masks[active[j_idx]]
            U_except[i_idx] = u
        
        # Try swaps
        # For each i, find a k such that U_except[i] | cover_masks[k] == FULL_MASK
        # This is equivalent to cover_masks[k] covering the missing bits of U_except[i]
        
        for i_idx in range(len(active)):
            if time.time() > deadline - 10:
                break
            
            u_except = U_except[i_idx]
            missing = FULL_MASK ^ u_except
            
            if missing == 0:
                # Can remove active[i_idx]
                active.pop(i_idx)
                improved = True
                break # Restart loop with smaller active
            
            # Find a k that covers missing
            # Optimization: Iterate through all k? 729 checks.
            # We can also check if any existing active[j] (j != i) can cover missing?
            # If so, we could potentially remove i? No, that's handled by missing==0.
            # We need a NEW word k (or existing one if we are just moving? No, swap means replace).
            # If k is already in active, we are just moving it? No, active is a list of indices.
            # If k is in active, we are replacing active[i] with active[j]?
            # If k is in active, then U_except[i] already includes cover_masks[k].
            # So if cover_masks[k] covers missing, then U_except[i] | cover_masks[k] would have been FULL_MASK?
            # Wait. U_except[i] = Union_{j!=i} cover_masks[j].
            # If k is in active and k != i, then cover_masks[k] is part of U_except[i].
            # So U_except[i] | cover_masks[k] = U_except[i].
            # So if missing != 0, we cannot cover it with an existing codeword (unless my logic is flawed).
            # Actually, if cover_masks[k] covers missing, and k is in active (j!=i), then missing is covered by U_except[i].
            # So missing & cover_masks[k] == missing implies missing & U_except[i] == missing.
            # But missing = FULL_MASK ^ U_except[i]. So missing & U_except[i] == 0.
            # So missing == 0. Contradiction.
            # So k must NOT be in active (or k == i, but we are replacing i).
            # So we can just iterate k in range(N_WORDS) and skip if k in active?
            # Or just iterate all, and if k in active, it won't work anyway (except if k==i, but we skip i).
            
            found_swap = False
            for k in range(N_WORDS):
                if k == active[i_idx]:
                    continue
                # Check if k covers missing
                if (cover_masks[k] & missing) == missing:
                    # Check if k is already in active?
                    # If k is in active, it's a no-op or duplicate?
                    # If k is in active, we are effectively removing i and keeping k?
                    # No, we are replacing i with k. If k is already there, we have duplicate?
                    # The problem says codewords must be distinct.
                    # So we should ensure k is not in active.
                    if k in active:
                        continue
                    
                    active[i_idx] = k
                    found_swap = True
                    improved = True
                    break
            
            if found_swap:
                break # Restart removal/swap loop with new structure
        
        if len(active) < best_size:
            best_size = len(active)
            best_active = list(active)
        
        if not improved:
            # No swaps found.
            # We could try random restarts or perturbations here?
            # For now, break if we are stuck?
            # Or continue to try different shuffles?
            # Shuffling doesn't change the set, so U_except is the same.
            # So if no swap found in one pass, no swap will be found in another pass with the same set.
            # Unless we allow "move" swaps? i.e. active[i] -> k where k is in active?
            # That doesn't change the set.
            # So if no new word k covers the missing bits, we are stuck with this set size.
            # To escape, we might need to accept a temporary increase in size? (Simulated Annealing style)
            # Or just accept that we found a local minimum.
            
            # Let's try a "2-opt" move: Remove 2, Add 1?
            # Or Remove 1, Add 2?
            # This is more complex.
            
            # For now, let's just break if no improvement.
            # But maybe we can try random initializations?
            # No, we want to stick to the design family.
            
            # Let's try to perturb: Pick a random active[i], remove it, add a random k?
            # This might break coverage.
            # We need to check validity.
            
            # Let's stick to the current logic for now.
            pass

    # Final removal pass
    best_active = try_remove(best_active)
    
    # Convert back to words
    result = [list(all_words[idx]) for idx in best_active]
    
    # Verify
    if not is_valid_cover(best_active):
        # Fallback to initial
        result = [list(all_words[idx]) for idx in current_indices]
        
    return result
# EVOLVE-BLOCK-END
