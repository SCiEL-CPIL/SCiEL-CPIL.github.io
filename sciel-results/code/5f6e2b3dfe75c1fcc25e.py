"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Optimized for q=3, n=6, radius=1 using bitset operations for fast coverage checks.
    """
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: generic greedy
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute neighbors
        neighbors = {}
        for w in all_words:
            nb = set()
            for v in all_words:
                dist = sum(1 for i in range(n) if w[i] != v[i])
                if dist <= radius:
                    nb.add(v)
            neighbors[w] = nb
        
        uncovered = set(all_words)
        code = []
        
        while uncovered:
            best_word = None
            max_covered = -1
            for w in all_words:
                if w in code:
                    continue
                covered_count = len(neighbors[w] & uncovered)
                if covered_count > max_covered:
                    max_covered = covered_count
                    best_word = w
            
            if best_word is None:
                break
                
            code.append(best_word)
            uncovered -= neighbors[best_word]
            
        return [list(c) for c in code]

    # Specific optimization for q=3, n=6, radius=1
    total_words = 3 ** n
    all_words = list(itertools.product(range(3), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors as bitmasks
    # Each word maps to an integer where bit i is set if word i is in the neighborhood
    neighbors_mask = [0] * total_words
    for i, w in enumerate(all_words):
        mask = 1 << i
        for j in range(n):
            for val in range(3):
                if val == w[j]:
                    continue
                new_w = list(w)
                new_w[j] = val
                new_w = tuple(new_w)
                idx = word_to_idx[new_w]
                mask |= (1 << idx)
        neighbors_mask[i] = mask

    # Start with tetracode-based construction (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code_indices = set()
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = tuple(h + list(tail))
            initial_code_indices.add(word_to_idx[w])
    
    code_indices = initial_code_indices
    full_coverage = (1 << total_words) - 1
    
    def get_coverage_mask(code_set):
        mask = 0
        for idx in code_set:
            mask |= neighbors_mask[idx]
        return mask

    current_mask = get_coverage_mask(code_indices)
    
    # Local Search Phase 1: Remove redundant codewords
    improved = True
    while improved:
        improved = False
        for idx in list(code_indices):
            new_mask = current_mask & ~neighbors_mask[idx]
            # Check if coverage is still full
            if new_mask == full_coverage:
                code_indices.remove(idx)
                current_mask = new_mask
                improved = True
                break # Restart to check for more removals

    # Local Search Phase 2: Swap to reduce size or improve structure
    # We try to swap a codeword with another to maintain coverage.
    # Since we already removed all single-removal candidates, we need swaps that might allow further removals later.
    # Strategy: Try swapping codeword A with B. If coverage remains full, accept.
    # To find good swaps, we can look for codewords with low "unique coverage" (words covered only by them).
    
    # Precompute unique coverage for each codeword? 
    # Let's just do a random-ish local search with bitsets.
    
    start_time = time.time()
    time_limit = 100.0 # Leave buffer for overhead
    
    while time.time() - start_time < time_limit:
        improved = False
        code_list = list(code_indices)
        # Shuffle for diversity? Or just iterate.
        # Let's iterate and try to swap with words that are NOT in code.
        
        # To speed up, we can precompute the set of indices NOT in code
        # But that's 729 - len(code).
        
        for idx_remove in code_list:
            # Coverage provided by this codeword that is NOT provided by others
            # This is expensive to compute exactly every time.
            # Approximation: Just try swapping with random candidates or all candidates?
            # 729 candidates * 73 codewords = ~50k checks per full pass.
            # Each check is O(1) bitwise ops.
            # 50k * 100 passes = 5M ops. Very fast.
            
            # We want to find an idx_add such that (current_mask ^ neighbors_mask[idx_remove] | neighbors_mask[idx_add]) == full_coverage
            # Actually, we need: (current_mask & ~neighbors_mask[idx_remove]) | neighbors_mask[idx_add] == full_coverage
            
            base_mask = current_mask & ~neighbors_mask[idx_remove]
            
            # We need to find idx_add such that base_mask | neighbors_mask[idx_add] == full_coverage
            # This is equivalent to: neighbors_mask[idx_add] must cover all bits in (full_coverage ^ base_mask)
            # Let missing = full_coverage ^ base_mask
            # We need neighbors_mask[idx_add] & missing == missing
            
            missing = full_coverage ^ base_mask
            if missing == 0:
                # We can just remove idx_remove!
                code_indices.remove(idx_remove)
                current_mask = base_mask
                improved = True
                break
            
            # Find any idx_add not in code_indices that covers 'missing'
            # Iterate through all possible adds
            found = False
            for idx_add in range(total_words):
                if idx_add in code_indices:
                    continue
                if (neighbors_mask[idx_add] & missing) == missing:
                    # Valid swap
                    code_indices.remove(idx_remove)
                    code_indices.add(idx_add)
                    current_mask = base_mask | neighbors_mask[idx_add]
                    improved = True
                    found = True
                    break
            
            if found:
                break # Restart outer loop
        
        if not improved:
            # If no simple swap worked, try a 2-swap or random perturbation?
            # Or just break if we are stuck for a while?
            # Given the small size, if we are stuck at 73, we are likely optimal or near-optimal.
            # The best known is 73. If we hit 73, we should stop.
            if len(code_indices) <= 73:
                break
            
            # Try a more complex move: remove 2, add 2?
            # This is O(N^4) which is 729^4 = 2e11, too slow.
            # Instead, let's try random restarts if stuck?
            # Or just accept the current result if we've been stuck.
            
            # Let's try one more pass with a different order or randomization
            # For now, if we can't improve, break to avoid wasting time.
            # The seed usually gets to 74 or 75. 73 is the target.
            # If we are at 74, we might need a deeper search.
            
            # Let's try a "kick": remove a random word, then try to repair by adding words?
            # This is complex. 
            # Let's just break if no improvement. The bitset speed allows many iterations.
            break

    result = [list(all_words[idx]) for idx in code_indices]
    return result

# EVOLVE-BLOCK-END
