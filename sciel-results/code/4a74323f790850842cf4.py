        new_word = []
        for idx in range(N):
            p_idx = path[idx]
            p_cur = perms[p_idx]
            block_cur = list(p_cur) + [m] + list(p_cur)
            
            # Overlap with new_word
            # If new_word is empty, overlap 0
            if not new_word:
                overlap = 0
            else:
                # Max possible overlap is len(block_cur) or len(new_word)
                # But we know the theoretical max is W[path[idx-1]][path[idx]]
                # Is it always achievable?
                # Yes, because new_word ends with the previous block,
                # which ends with p_prev.
                # So suffix of new_word is suffix of p_prev.
                # Prefix of block_cur is prefix of p_cur.
                # So overlap is W[p_prev][p_cur].
                # EXCEPT if the overlap extends into the part BEFORE p_prev?
                # No, p_prev is a permutation, so its suffixes are unique?
                # Not necessarily.
                # But W is defined as max suffix of p_prev matching prefix of p_cur.
                # Since new_word ends with p_prev, any suffix of new_word that is a suffix of p_prev
                # is available.
                # Could a LONGER suffix of new_word match?
                # That would require matching into the 'm' or previous part.
                # But p_cur is a perm of 1..m-1. 'm' is not in p_cur.
                # So the match cannot cross the 'm'.
                # So the max overlap is indeed W[p_prev][p_cur].
                
                if idx == 0:
                    overlap = 0
                else:
                    p_prev = path[idx-1]
                    overlap = W[p_prev][p_idx]
            
            new_word += block_cur[overlap:]
