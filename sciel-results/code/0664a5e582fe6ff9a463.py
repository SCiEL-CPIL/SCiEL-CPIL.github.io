"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Optimized for q=3, n=6, radius=1 using fast local search with numpy coverage counts.
    """
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: simple greedy
        # This is a basic implementation to ensure the function doesn't crash
        all_words = list(itertools.product(range(q), repeat=n))
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
    
    # Precompute neighbors as indices
    neighbors = []
    for w in all_words:
        nb = [word_to_idx[w]]
        for i in range(n):
            for val in range(3):
                if val == w[i]:
                    continue
                new_w = list(w)
                new_w[i] = val
                new_w = tuple(new_w)
                nb.append(word_to_idx[new_w])
        neighbors.append(nb)
        
    # Start with the tetracode-based construction (9 * 3^2 = 81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_code.append(tuple(h + list(tail)))
            
    code_set = set(initial_code)
    code_indices = [word_to_idx[c] for c in code_set]
    
    # Initialize coverage count array
    coverage_count = np.zeros(total_words, dtype=np.int8)
    for idx in code_indices:
        coverage_count[neighbors[idx]] += 1
        
    # Local Search
    improved = True
    while improved:
        improved = False
        # Try removing
        for idx in list(code_indices):
            # Tentative removal
            coverage_count[neighbors[idx]] -= 1
            if np.min(coverage_count) > 0:
                # Valid removal
                code_set.discard(all_words[idx])
                code_indices.remove(idx)
                improved = True
                break
            else:
                # Revert
                coverage_count[neighbors[idx]] += 1
        
        if improved:
            continue
            
        # Try swapping
        code_indices_copy = list(code_indices)
        for idx_remove in code_indices_copy:
            # Tentative removal
            coverage_count[neighbors[idx_remove]] -= 1
            
            found_swap = False
            for idx_add in range(total_words):
                if idx_add in code_set:
                    continue
                
                # Tentative addition
                coverage_count[neighbors[idx_add]] += 1
                
                if np.min(coverage_count) > 0:
                    # Valid swap
                    code_set.discard(all_words[idx_remove])
                    code_indices.remove(idx_remove)
                    code_set.add(all_words[idx_add])
                    code_indices.append(idx_add)
                    improved = True
                    found_swap = True
                    break
                else:
                    # Revert addition
                    coverage_count[neighbors[idx_add]] -= 1
            
            if found_swap:
                break
            else:
                # Revert removal
                coverage_count[neighbors[idx_remove]] += 1
                
    result = [list(all_words[idx]) for idx in code_indices]
    return result

# EVOLVE-BLOCK-END
