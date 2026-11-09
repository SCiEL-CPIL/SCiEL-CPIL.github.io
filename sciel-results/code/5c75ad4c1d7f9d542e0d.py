"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    if q == 3 and n == 6 and radius == 1:
        total_words = q ** n
        
        def word_to_idx(word):
            idx = 0
            for v in word:
                idx = idx * q + v
            return idx

        def idx_to_word(idx):
            word = [0] * n
            for i in range(n - 1, -1, -1):
                word[i] = idx % q
                idx //= q
            return word

        all_words = list(itertools.product(range(q), repeat=n))
        
        # Precompute neighbors mask
        neighbors_mask = np.zeros((total_words, total_words), dtype=np.uint8)
        for i, word in enumerate(all_words):
            neighbors_mask[i, i] = 1
            for pos in range(n):
                orig_val = word[pos]
                for new_val in range(q):
                    if new_val != orig_val:
                        new_word = list(word)
                        new_word[pos] = new_val
                        j = word_to_idx(new_word)
                        neighbors_mask[i, j] = 1

        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        base_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        code_indices = [word_to_idx(c) for c in base_code]
        
        counts = np.zeros(total_words, dtype=np.int32)
        for idx in code_indices:
            counts += neighbors_mask[idx]
        
        rng = np.random.default_rng(42)
        
        # Precompute inverse neighbors? Or just use mask.
        # To find candidates for S_only:
        # We need intersection of columns of neighbors_mask for rows in S_only.
        # neighbors_mask.T is (total_words, total_words) where col i is neighbors of i?
        # neighbors_mask[i, j] = 1 if i covers j.
        # So neighbors_mask[:, w] is the list of codewords that cover w.
        # We want c' such that c' covers all w in S_only.
        # So c' must be in intersection of neighbors_mask[:, w] for w in S_only.
        
        while time.time() < deadline - 5.0:
            # Try Removal
            i = rng.integers(0, len(code_indices))
            c = code_indices[i]
            
            # Check removal
            # Words that become uncovered are those with counts[w] == 1 and neighbors_mask[c, w] == 1
            # Let's find them
            # mask_c = neighbors_mask[c]
            # unique_covered = (counts == 1) & mask_c
            # if not np.any(unique_covered):
            #     # Valid removal
            #     counts -= mask_c
            #     code_indices.pop(i)
            #     continue
            
            # Optimized check:
            # min(counts - mask_c) > 0
            # This is O(729).
            
            if np.min(counts - neighbors_mask[c]) > 0:
                counts -= neighbors_mask[c]
                code_indices.pop(i)
                continue
            
            # Try Swap
            # Find S_only
            s_only_mask = (counts == 1) & neighbors_mask[c]
            if not np.any(s_only_mask):
                # This shouldn't happen if removal check failed, but just in case
                continue
            
            s_only_indices = np.where(s_only_mask)[0]
            
            # Find candidates c' that cover all s_only_indices
            # Start with all possible c'
            candidates = np.ones(total_words, dtype=bool)
            for w in s_only_indices:
                candidates &= neighbors_mask[:, w]
                
            # Exclude current code
            for cw in code_indices:
                candidates[cw] = False
                
            if np.any(candidates):
                # Pick a random candidate
                cand_indices = np.where(candidates)[0]
                c_prime = cand_indices[rng.integers(0, len(cand_indices))]
                
                # Perform swap
                # We know it's valid by construction (covers S_only, and other counts are safe)
                # But let's double check with min > 0 for safety
                new_counts = counts - neighbors_mask[c] + neighbors_mask[c_prime]
                if np.min(new_counts) > 0:
                    counts = new_counts
                    code_indices[i] = c_prime
                # Else revert (should not happen)
            else:
                # No candidate found for this c. Try another c?
                # The loop will pick a new random i next time.
                pass
        
        result = [idx_to_word(idx) for idx in code_indices]
        return result
    else:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
# EVOLVE-BLOCK-END
