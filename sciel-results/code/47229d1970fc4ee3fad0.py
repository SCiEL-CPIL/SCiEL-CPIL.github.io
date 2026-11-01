"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
from scipy.optimize import minimize

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Optimized for q=3, n=6, radius=1 using efficient local search with coverage counts.
    """
    if q == 3 and n == 6 and radius == 1:
        total_words = 3 ** n
        
        # Map tuples to integers for faster processing
        # 0..728
        all_words = list(itertools.product(range(3), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute neighbors as list of indices
        neighbors = [[] for _ in range(total_words)]
        for i, w in enumerate(all_words):
            nb = [i]
            for pos in range(n):
                orig_val = w[pos]
                for new_val in range(3):
                    if new_val == orig_val:
                        continue
                    # Create new word tuple
                    new_w = list(w)
                    new_w[pos] = new_val
                    new_w = tuple(new_w)
                    nb.append(word_to_idx[new_w])
            neighbors[i] = nb

        # Initial code: Tetracode * 3^2
        # Tetracode: [a, b, (a+b)%3, (a+2b)%3]
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_code_indices = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=2):
                w = tuple(h + list(tail))
                initial_code_indices.append(word_to_idx[w])
        
        code_set = set(initial_code_indices)
        
        # Coverage counts
        cover_count = np.zeros(total_words, dtype=int)
        for c in code_set:
            for nb in neighbors[c]:
                cover_count[nb] += 1
        
        def is_valid_cover(counts):
            return np.all(counts > 0)
        
        # Local Search Phase 1: Remove redundant codewords
        improved = True
        while improved:
            improved = False
            # Shuffle order to avoid bias? Or just iterate.
            # Iterating list(code_set) is fine.
            for c in list(code_set):
                # Tentatively remove c
                for nb in neighbors[c]:
                    cover_count[nb] -= 1
                if is_valid_cover(cover_count):
                    code_set.remove(c)
                    improved = True
                    break
                else:
                    # Restore
                    for nb in neighbors[c]:
                        cover_count[nb] += 1
        
        # Local Search Phase 2: Swap
        # Try to replace a codeword with another to allow further removals or just to improve "density"
        # Since we already removed all individually redundant ones, we need to look for swaps that enable removal of *another* codeword.
        # Or simply: Try replacing c1 with c2. If valid, keep it.
        # This is expensive if done naively. 
        # Strategy: For each c1 in code, for each c2 not in code, check if (code - c1 + c2) is valid.
        # If valid, we might not have reduced size, but we changed the set. 
        # To reduce size, we need a sequence: Remove c1 (invalid), Add c2 (valid?), Remove c3 (valid?).
        # Simpler heuristic: 
        # 1. Try to remove any codeword. (Done)
        # 2. Try swapping c1 with c2. If the new set is valid, accept it. 
        #    Does this help reduce size? Not directly. But it might lead to a state where a removal is possible.
        
        # Let's try a more aggressive local search:
        # Repeat:
        #   a. Try to remove any c. If success, restart.
        #   b. Try to swap c1 with c2. If success, restart.
        
        improved = True
        while improved:
            improved = False
            
            # Try removals again (in case swaps opened up removals)
            for c in list(code_set):
                for nb in neighbors[c]:
                    cover_count[nb] -= 1
                if is_valid_cover(cover_count):
                    code_set.remove(c)
                    improved = True
                    break
                else:
                    for nb in neighbors[c]:
                        cover_count[nb] += 1
            if improved:
                continue

            # Try swaps
            code_list = list(code_set)
            for i, c1 in enumerate(code_list):
                # For each c1, try swapping with c2
                # To speed up, we can pre-filter c2s? 
                # Just iterate all 729 words.
                for c2 in range(total_words):
                    if c2 in code_set:
                        continue
                    
                    # Apply swap: -c1 +c2
                    for nb in neighbors[c1]:
                        cover_count[nb] -= 1
                    for nb in neighbors[c2]:
                        cover_count[nb] += 1
                    
                    if is_valid_cover(cover_count):
                        # Accept swap
                        code_set.remove(c1)
                        code_set.add(c2)
                        improved = True
                        break
                    
                    # Revert
                    for nb in neighbors[c2]:
                        cover_count[nb] -= 1
                    for nb in neighbors[c1]:
                        cover_count[nb] += 1
                
                if improved:
                    break
        
        # Convert back to list of lists
        result = [list(all_words[c]) for c in code_set]
        return result

    # Generic fallback for other parameters
    # Simple greedy algorithm
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

# EVOLVE-BLOCK-END
