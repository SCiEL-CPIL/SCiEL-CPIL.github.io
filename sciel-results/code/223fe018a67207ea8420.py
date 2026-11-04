"""Improved program for ternary covering code K_3(6,1)."""
# EVOLVE-BLOCK-START
import itertools


def construct(q, n, radius):
    # Approach: Best-Fit Greedy Set Cover using bitmasks, followed by redundancy removal.
    # This is a generic, robust approach for small covering code problems.
    
    total_words = q ** n
    
    # Generate all words and map to indices
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_index = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage bitmask for each candidate codeword
    # mask[c] has bit i set if word i is within Hamming distance <= radius of c
    candidate_masks = [0] * total_words
    
    for c_idx, c in enumerate(all_words):
        mask = 0
        for w_idx, w in enumerate(all_words):
            dist = sum(1 for k in range(n) if c[k] != w[k])
            if dist <= radius:
                mask |= (1 << w_idx)
        candidate_masks[c_idx] = mask
        
    full_mask = (1 << total_words) - 1
    
    # 1. Best-Fit Greedy Set Cover
    covered_mask = 0
    selected_indices = []
    available = set(range(total_words))
    
    while covered_mask != full_mask and available:
        best_idx = -1
        best_new_count = 0
        
        for idx in available:
            new_coverage = candidate_masks[idx] & ~covered_mask
            count = new_coverage.bit_count()
            if count > best_new_count:
                best_new_count = count
                best_idx = idx
                
        if best_idx == -1 or best_new_count == 0:
            break
            
        selected_indices.append(best_idx)
        covered_mask |= candidate_masks[best_idx]
        available.remove(best_idx)
        
    # 2. Redundancy Removal
    changed = True
    while changed:
        changed = False
        for i in range(len(selected_indices)):
            idx_to_remove = selected_indices[i]
            
            # Check if all words covered by idx_to_remove are covered by others
            others_mask = 0
            for j in range(len(selected_indices)):
                if j != i:
                    others_mask |= candidate_masks[selected_indices[j]]
                    
            unique_part = candidate_masks[idx_to_remove] & ~others_mask
            if unique_part == 0:
                selected_indices.pop(i)
                changed = True
                break # Restart loop
            
    # Final verification
    final_mask = 0
    for idx in selected_indices:
        final_mask |= candidate_masks[idx]
        
    if final_mask != full_mask:
        # Fallback: return a valid large code (e.g. all words) if greedy failed (unlikely)
        # Or a known construction. Here we just return the greedy result if it's valid, 
        # but if it's invalid, we must return something valid.
        # Since greedy covers everything, this shouldn't happen.
        # If it does, return all words (valid but bad score).
        return [list(w) for w in all_words]

    return [list(all_words[idx]) for idx in selected_indices]
# EVOLVE-BLOCK-END
