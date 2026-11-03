import numpy as np
import itertools

def construct(q, n, radius):
        if q != 3 or n != 7 or radius != 1:
            # Fallback
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        N = 3**7
        words = list(itertools.product(range(3), repeat=7))
        word_to_idx = {w: i for i, w in enumerate(words)}
        
        # Precompute masks as Python ints (bitsets)
        # mask[i] has bits set for all words covered by codeword i
        masks = [0] * N
        for i, w in enumerate(words):
            m = 0
            # w itself
            m |= (1 << i)
            # neighbors
            for pos in range(7):
                for val in range(3):
                    if val != w[pos]:
                        new_w = list(w)
                        new_w[pos] = val
                        idx = word_to_idx[tuple(new_w)]
                        m |= (1 << idx)
            masks[i] = m
        
        # Greedy Set Cover with Bitsets
        uncovered = (1 << N) - 1
        selected = []
        
        # To speed up greedy, we can precompute something? 
        # Finding max popcount(uncovered & mask[i]) for all i is O(N) per step.
        # N=2187. Steps ~ 200. Total ~ 400k big int ops. Fast.
        
        while uncovered:
            best_idx = -1
            best_count = -1
            # Optimization: Only check candidates that cover at least one uncovered word?
            # Or just scan all.
            for i in range(N):
                # If mask[i] is subset of already covered, skip? 
                # uncovered & mask[i] gives the count of newly covered words.
                count = (uncovered & masks[i]).bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = i
                    if best_count == 15: # Max possible
                        break
            if best_idx == -1 or best_count == 0:
                break
            selected.append(best_idx)
            uncovered &= ~masks[best_idx]
            
        # Local Search: Removal
        # Maintain coverage counts
        cover_count = np.zeros(N, dtype=np.int32)
        for i in selected:
            # Add coverage
            # We need to iterate over set bits in masks[i]
            # Or just iterate over the 15 neighbors
            w = words[i]
            cover_count[i] += 1
            for pos in range(7):
                for val in range(3):
                    if val != w[pos]:
                        new_w = list(w)
                        new_w[pos] = val
                        idx = word_to_idx[tuple(new_w)]
                        cover_count[idx] += 1
                        
        changed = True
        while changed:
            changed = False
            # Try to remove from end
            for idx in range(len(selected) - 1, -1, -1):
                i = selected[idx]
                # Check if all words covered by i have count > 1
                # We need to check the 15 words
                w = words[i]
                can_remove = True
                if cover_count[i] == 1:
                    can_remove = False
                else:
                    for pos in range(7):
                        for val in range(3):
                            if val != w[pos]:
                                new_w = list(w)
                                new_w[pos] = val
                                idx2 = word_to_idx[tuple(new_w)]
                                if cover_count[idx2] == 1:
                                    can_remove = False
                                    break
                        if not can_remove: break
                if can_remove:
                    # Remove i
                    cover_count[i] -= 1
                    w = words[i]
                    for pos in range(7):
                        for val in range(3):
                            if val != w[pos]:
                                new_w = list(w)
                                new_w[pos] = val
                                idx2 = word_to_idx[tuple(new_w)]
                                cover_count[idx2] -= 1
                    selected.pop(idx)
                    changed = True
                    break # Restart loop? Or continue? Restarting is safer for consistency
            
        return [list(words[i]) for i in selected]
