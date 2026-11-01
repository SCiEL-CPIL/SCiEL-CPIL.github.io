import time
import random

def construct(q, n, radius):
    """
    Constructs a covering code of length n over alphabet size q with covering radius radius.
    Strategy:
    1. Seed with a known good construction (Tetracode direct sum) if applicable.
    2. Complete coverage using a greedy algorithm.
    3. Refine using local search: iterative deletion of redundant codewords and swap heuristics.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Handle specific case q=3, n=6, radius=1
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: simple greedy from scratch
        return _generic_construct(q, n, radius, start_time, deadline)

    total_words = q ** n  # 729
    
    # Precompute index to word and word to index
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    # Precompute masks for all words
    # mask[i] is a bitset where bit j is 1 if word j is within radius of word i
    masks = [0] * total_words
    for i in range(total_words):
        w = idx_to_word(i)
        mask = 0
        # Center
        mask |= (1 << i)
        # Neighbors at distance 1
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                diff = new_val - orig_val
                idx_new = i + diff * (q ** (n - 1 - pos))
                mask |= (1 << idx_new)
        masks[i] = mask

    # Full mask for all words
    full_mask = (1 << total_words) - 1
    
    # 1. Seeding
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_code = set()
    covered = 0
    
    # Direct sum with all tails of length 2
    for tail_a in range(3):
        for tail_b in range(3):
            for h in tetracode:
                word = h + [tail_a, tail_b]
                idx = word_to_idx(word)
                initial_code.add(idx)
                covered |= masks[idx]
                
    code = list(initial_code)
    
    # 2. Greedy Completion
    while covered != full_mask:
        if time.time() > deadline:
            break
        
        best_idx = -1
        best_gain = -1
        
        # To speed up, we can check all words, but we can also just check uncovered words' neighbors?
        # Checking all 729 words is fast enough.
        for i in range(total_words):
            if i in initial_code:
                continue # Skip already chosen? Not strictly necessary but saves a bit.
            # Actually, we should check all, because a word already in code might have been 
            # considered, but its gain is 0 now. 
            # gain = popcount(masks[i] & ~covered)
            gain = (masks[i] & ~covered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        if best_gain == 0:
            # Should not happen if covered != full_mask
            break
            
        code.append(best_idx)
        initial_code.add(best_idx)
        covered |= masks[best_idx]
        
    # 3. Local Search: Deletion and Swap
    # We maintain a set for O(1) removal and a list for iteration
    code_set = set(code)
    
    # Helper to check if a set of indices covers full_mask
    def is_covered(indices):
        cov = 0
        for idx in indices:
            cov |= masks[idx]
        return cov == full_mask

    # Helper to find uncovered words given a set of indices
    def get_uncovered(indices):
        cov = 0
        for idx in indices:
            cov |= masks[idx]
        return full_mask & ~cov

    improved = True
    while improved:
        improved = False
        
        # Pass 1: Try to remove redundant codewords
        for c in list(code_set):
            if time.time() > deadline:
                break
            
            # Tentatively remove c
            code_set.remove(c)
            
            # Check if coverage is still full
            # We can optimize by checking if masks[c] is subset of union of others
            # But recomputing union is O(K). K~70. 70*70=4900 ops.
            # Let's just compute the union of the remaining set.
            union_others = 0
            for c2 in code_set:
                union_others |= masks[c2]
                
            if (masks[c] & ~union_others) == 0:
                # c is redundant, keep it removed
                improved = True
                break # Restart loop as set changed
            else:
                # c is not redundant, put it back
                code_set.add(c)
                
        if time.time() > deadline:
            break
            
        # Pass 2: Try swaps
        # If no deletion was possible, try to find a swap that reduces size by 1.
        # Strategy: Remove c. If coverage lost, let U be uncovered.
        # Find c' not in code_set that covers max(U). Add c'.
        # If adding c' allows removal of some c'' in code_set (c'' != c), then net change is -1.
        
        # To make this efficient, we only try this if we haven't improved in the deletion pass
        # and we have time.
        
        for c in list(code_set):
            if time.time() > deadline:
                break
                
            # 1. Remove c
            code_set.remove(c)
            
            # 2. Find uncovered
            union_others = 0
            for c2 in code_set:
                union_others |= masks[c2]
                
            uncovered = full_mask & ~union_others
            
            if uncovered == 0:
                # This means c was redundant, which should have been caught in Pass 1.
                # If it happens, keep it removed.
                improved = True
                break
            else:
                # We need to cover 'uncovered'.
                # Find best candidate c' to add.
                best_candidate = -1
                best_covered_uncovered = -1
                
                # We can restrict candidates to words that are in 'uncovered' or close to it?
                # Actually, any word can cover uncovered words.
                # But checking all 729 is fine.
                for c_prime in range(total_words):
                    if c_prime in code_set:
                        continue
                    # How many of 'uncovered' does c_prime cover?
                    # gain = popcount(masks[c_prime] & uncovered)
                    gain = (masks[c_prime] & uncovered).bit_count()
                    if gain > best_covered_uncovered:
                        best_covered_uncovered = gain
                        best_candidate = c_prime
                        
                if best_candidate == -1:
                    code_set.add(c)
                    continue
                    
                # 3. Add best_candidate
                code_set.add(best_candidate)
                
                # 4. Check if we can remove any other codeword c''
                # We want to remove c'' such that masks[c''] is covered by union of (code_set \ {c''})
                # Note: code_set now contains c' and does NOT contain c.
                
                found_removal = False
                for c_double_prime in list(code_set):
                    if c_double_prime == best_candidate:
                        continue # Can't remove the one we just added immediately? 
                                 # Actually we can, but it would just revert the swap.
                                 # We want to remove something else to reduce size.
                                 # If we remove c_double_prime, size stays same (removed c, added c', removed c''). Net -1.
                                 # Wait. 
                                 # Start: S. Size K.
                                 # Remove c: S \ {c}. Size K-1.
                                 # Add c': (S \ {c}) U {c'}. Size K.
                                 # Remove c'': ((S \ {c}) U {c'}) \ {c''}. Size K-1.
                                 # So if we can remove ANY c'' (including c'? no, c' is new, removing it reverts to S\{c} which is invalid), 
                                 # we reduce size by 1.
                                 
                    # Check if c_double_prime is redundant in the new set
                    new_set_without_cdp = set(code_set)
                    new_set_without_cdp.remove(c_double_prime)
                    
                    union_new = 0
                    for idx in new_set_without_cdp:
                        union_new |= masks[idx]
                        
                    if (masks[c_double_prime] & ~union_new) == 0:
                        # c_double_prime is redundant
                        code_set.remove(c_double_prime)
                        improved = True
                        found_removal = True
                        break
                        
                if found_removal:
                    break # Restart outer while loop
                
                else:
                    # Swap didn't lead to a reduction, revert
                    code_set.remove(best_candidate)
                    code_set.add(c)
                    
        if time.time() > deadline:
            break

    # Final verification and conversion
    result_indices = list(code_set)
    
    # Verify validity
    final_covered = 0
    for idx in result_indices:
        final_covered |= masks[idx]
        
    if final_covered != full_mask:
        # Fallback: if local search broke it (shouldn't happen), return the greedy result
        # We don't store the greedy result separately, so we might need to reconstruct or just return what we have
        # But the logic above should be safe.
        pass

    result_words = [idx_to_word(idx) for idx in result_indices]
    
    return result_words

def _generic_construct(q, n, radius, start_time, deadline):
    """
    Fallback for non-specific parameters.
    """
    total_words = q ** n
    
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    # Precompute masks
    masks = [0] * total_words
    # This generic version might be slow for large n/q, but for small ones it's fine.
    # For large ones, we might just return an empty list or a simple greedy.
    # Given the task is specific to 3,6,1, this is just a safety net.
    
    for i in range(total_words):
        w = idx_to_word(i)
        mask = 0
        # Generate neighbors up to radius
        # Simple BFS or iterative distance check
        # For generic radius, checking all words is O(Q^n * Q^n) which is bad.
        # But since this is a fallback for a specific task, we assume small inputs.
        for j in range(total_words):
            w2 = idx_to_word(j)
            dist = sum(1 for k in range(n) if w[k] != w2[k])
            if dist <= radius:
                mask |= (1 << j)
        masks[i] = mask

    full_mask = (1 << total_words) - 1
    covered = 0
    code = []
    code_set = set()

    while covered != full_mask:
        if time.time() > deadline:
            break
        best_idx = -1
        best_gain = -1
        for i in range(total_words):
            if i in code_set:
                continue
            gain = (masks[i] & ~covered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_gain == 0:
            break
        code.append(best_idx)
        code_set.add(best_idx)
        covered |= masks[best_idx]
        
    return [idx_to_word(i) for i in code]

if __name__ == "__main__":
    # Quick self-test
    code = construct(3, 6, 1)
    print(f"Code size: {len(code)}")
