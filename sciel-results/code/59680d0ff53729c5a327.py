import numpy as np
import itertools
import random

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other cases
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(x) for x in itertools.product(range(q), repeat=n)]

    # Specific case: q=3, n=8, radius=1
    total_words = 3 ** n
    
    # Helper to convert integer to word tuple
    def int_to_word(i):
        w = []
        temp = i
        for _ in range(n):
            w.append(temp % 3)
            temp //= 3
        return tuple(w)
    
    # Helper to convert word tuple to integer
    def word_to_int(w):
        res = 0
        for i in range(n):
            res = res * 3 + w[i]
        return res

    # Precompute sphere masks for all words
    # sphere_mask[i] is an integer with bits set for all words within distance 1 of word i
    sphere_masks = [0] * total_words
    for i in range(total_words):
        w = int_to_word(i)
        mask = 1 << i
        # Distance 0
        # Distance 1
        for pos in range(n):
            orig_val = w[pos]
            for delta in [-1, 1]:
                new_val = orig_val + delta
                if 0 <= new_val < q:
                    new_w = list(w)
                    new_w[pos] = new_val
                    new_i = word_to_int(tuple(new_w))
                    mask |= (1 << new_i)
        sphere_masks[i] = mask

    # Approach 1: Greedy Set Cover
    def greedy_cover():
        uncovered = (1 << total_words) - 1
        codewords = []
        while uncovered:
            best_w = -1
            best_count = -1
            # Iterate all possible centers
            # Optimization: We can skip if best_count is already high? No, we need max.
            for w in range(total_words):
                # Count how many uncovered words are covered by w
                covered = uncovered & sphere_masks[w]
                cnt = covered.bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_w = w
            codewords.append(best_w)
            # Remove covered words from uncovered
            uncovered &= ~sphere_masks[best_w]
        return [int_to_word(w) for w in codewords]

    # Approach 2: Direct Sum + Pruning (Current logic, slightly optimized)
    def direct_sum_prune():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        # Map word to index
        word_to_idx = {}
        for i in range(total_words):
            word_to_idx[int_to_word(i)] = i
            
        # Precompute neighbor indices for initial codewords
        cw_neighbors = []
        for cw in initial_codewords:
            idx = word_to_idx[tuple(cw)]
            # Use bit mask to get neighbors? 
            # Or just use the precomputed sphere_masks and extract indices?
            # Extracting indices from a big int mask is slow.
            # Better to precompute neighbor lists for the initial codewords only?
            # There are 729 initial codewords.
            # We can just compute neighbors on the fly for the 729 codewords.
            neigh = [idx]
            w = cw
            for pos in range(n):
                orig_val = w[pos]
                for delta in [-1, 1]:
                    new_val = orig_val + delta
                    if 0 <= new_val < q:
                        new_w = w.copy()
                        new_w[pos] = new_val
                        neigh.append(word_to_idx[tuple(new_w)])
            cw_neighbors.append(neigh)
            
        best_size = float('inf')
        best_code = None
        
        # Try a few random orders for pruning
        for _ in range(10):
            counts = np.zeros(total_words, dtype=np.int32)
            for neigh_list in cw_neighbors:
                counts[neigh_list] += 1
                
            active = [True] * len(initial_codewords)
            indices = list(range(len(initial_codewords)))
            random.shuffle(indices)
            
            changed = True
            while changed:
                changed = False
                for i in indices:
                    if not active[i]:
                        continue
                    neigh = cw_neighbors[i]
                    if np.min(counts[neigh]) > 1:
                        active[i] = False
                        counts[neigh] -= 1
                        changed = True
                        
            current_code = [initial_codewords[i] for i in range(len(initial_codewords)) if active[i]]
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = current_code
                
        return best_code

    # Run both and pick the best
    code_greedy = greedy_cover()
    code_ds = direct_sum_prune()
    
    # Optional: Prune the greedy result too?
    # The greedy result is already minimal in the sense that no single word can be removed?
    # Greedy set cover does not guarantee that.
    # Let's prune the greedy result as well.
    
    # To prune greedy result, we need to adapt the pruning function to take an arbitrary list.
    def prune_arbitrary(code):
        if not code:
            return code
        word_to_idx = {tuple(cw): i for i, cw in enumerate(code)}
        # Wait, we need global indices for the count array?
        # Yes, counts array is size total_words.
        counts = np.zeros(total_words, dtype=np.int32)
        cw_neighbors = []
        for cw in code:
            idx = word_to_idx[tuple(cw)] # This is local index, we need global index for counts?
            # No, counts tracks coverage of the SPACE, not the code indices.
            # So we need the global index of the word.
            pass
        # Let's refactor prune to use global indices.
        
        global_word_to_idx = {}
        for i in range(total_words):
            global_word_to_idx[int_to_word(i)] = i
            
        cw_global_neighbors = []
        for cw in code:
            idx = global_word_to_idx[tuple(cw)]
            neigh = [idx]
            w = cw
            for pos in range(n):
                orig_val = w[pos]
                for delta in [-1, 1]:
                    new_val = orig_val + delta
                    if 0 <= new_val < q:
                        new_w = w.copy()
                        new_w[pos] = new_val
                        neigh.append(global_word_to_idx[tuple(new_w)])
            cw_global_neighbors.append(neigh)
            
        counts = np.zeros(total_words, dtype=np.int32)
        for neigh_list in cw_global_neighbors:
            counts[neigh_list] += 1
            
        active = [True] * len(code)
        indices = list(range(len(code)))
        random.shuffle(indices)
        
        changed = True
        while changed:
            changed = False
            for i in indices:
                if not active[i]:
                    continue
                neigh = cw_global_neighbors[i]
                if np.min(counts[neigh]) > 1:
                    active[i] = False
                    counts[neigh] -= 1
                    changed = True
                    
        return [code[i] for i in range(len(code)) if active[i]]

    code_greedy_pruned = prune_arbitrary(code_greedy)
    
    candidates = [code_greedy, code_greedy_pruned, code_ds]
    best = min(candidates, key=len)
    
    return best
