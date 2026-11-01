"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(n):
    if n == 1:
        return "1"
    
    word = [1]
    
    for m in range(2, n + 1):
        # 1. Extract all unique permutations of 1..m-1 from the current word
        perms = []
        seen = set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # We have all (m-1)! permutations. We need to order them to maximize overlap.
        # Greedy strategy:
        # Start with the permutation that matches the suffix of the current 'word' best.
        # Then, for each next permutation, pick the one that matches the suffix of the 
        # 'new_word' (being built) best.
        
        remaining_perms = set(perms)
        
        # Determine the first permutation
        # The suffix of the current word (length m-1) is a permutation.
        # Let's find it.
        suffix_perm = tuple(word[-(m-1):])
        
        if suffix_perm in remaining_perms:
            current_p = suffix_perm
            remaining_perms.remove(current_p)
        else:
            # Fallback: pick any, or the one with max overlap with suffix
            # Actually, suffix_perm must be in perms because word is a superperm of 1..m-1.
            # Wait, is the suffix of a superpermutation always a permutation of 1..m-1?
            # Yes, any window of length m-1 in a superpermutation of 1..m-1 is a permutation.
            # So suffix_perm is always in perms.
            pass
            
        new_word = list(word)
        
        # Helper to append a block with max overlap
        def append_block(new_w, p, m_val):
            block = list(p) + [m_val] + list(p)
            # Find max overlap
            max_overlap = 0
            min_len = min(len(new_w), len(block))
            for o in range(min_len, -1, -1):
                if new_w[len(new_w) - o:] == block[:o]:
                    max_overlap = o
                    break
            new_w.extend(block[max_overlap:])
            return new_w

        # Append the first block
        new_word = append_block(new_word, current_p, m)
        
        # Greedily append the rest
        while remaining_perms:
            # Find the suffix of new_word of length m-1
            # It's a permutation of 1..m-1
            current_suffix = tuple(new_word[-(m-1):])
            
            best_p = None
            best_overlap = -1
            
            for p in remaining_perms:
                block = list(p) + [m] + list(p)
                # Calculate overlap with new_word
                min_len = min(len(new_word), len(block))
                for o in range(min_len, -1, -1):
                    if new_word[len(new_word) - o:] == block[:o]:
                        if o > best_overlap:
                            best_overlap = o
                            best_p = p
                        break
                        
            if best_p is None:
                # Should not happen
                best_p = next(iter(remaining_perms))
                
            remaining_perms.remove(best_p)
            new_word = append_block(new_word, best_p, m)
            
        word = new_word
        
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
