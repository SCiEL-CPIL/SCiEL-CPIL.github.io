import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    import itertools
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n
    ball_size = 1 + 2 * n
    
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                balls[w, idx] = w + (new_digit - digit) * weight
                idx += 1
                
    # Invert balls: for each word, which codewords cover it?
    # word_to_codewords[w] is a list of codewords c such that w in balls[c]
    word_to_codewords = [[] for _ in range(total)]
    for c in range(total):
        for w in balls[c]:
            word_to_codewords[w].append(c)
    # Convert to numpy arrays for faster intersection? Or keep lists.
    # Lists are fine for small sizes (avg 17/6561 * 6561 = 17 neighbors? No, ball size is 17, so each word is in 17 balls. So word_to_codewords[w] has 17 elements.)
    # Wait, each ball has 17 words. Each word is in how many balls?
    # Symmetry: 17.
    # So word_to_codewords[w] is a list of 17 ints.

    best_size = total
    best_set = None
    
    np.random.seed(42)
    
    while time.time() < deadline - 10.0:
        # 1. Greedy
        selected = []
        uncovered = np.ones(total, dtype=bool)
        
        while np.any(uncovered):
            counts = uncovered[balls].sum(axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            uncovered[balls[best_idx]] = False
            
            if time.time() > deadline - 15.0:
                break
        
        if np.any(uncovered):
            continue
            
        selected = np.array(selected, dtype=np.int32)
        
        # 2. Pruning
        cover_count = np.zeros(total, dtype=np.int32)
        cover_count[balls[selected]] += 1
        
        changed = True
        while changed and time.time() < deadline - 5.0:
            mins = np.min(cover_count[balls[selected]], axis=1)
            removable_mask = mins > 1
            
            if not np.any(removable_mask):
                break
                
            remove_idx = selected[removable_mask]
            selected = selected[~removable_mask]
            
            cover_count[balls[remove_idx]] -= 1
            changed = True
            
        # 3. Local Search
        # Use word_to_codewords to find valid swaps efficiently
        
        def try_swaps(selected, cover_count, max_iters=200):
            # Helper to get unique words for a codeword c
            # unique_words are those w in balls[c] where cover_count[w] == 1
            # And w is covered ONLY by c?
            # No, cover_count[w] == 1 means w is covered by exactly 1 codeword in `selected`.
            # Since c covers w, that codeword must be c.
            
            for _ in range(max_iters):
                if time.time() > deadline - 5.0:
                    break
                if len(selected) == 0:
                    break
                
                # Prune first to ensure we are at a local minimum w.r.t removals
                mins = np.min(cover_count[balls[selected]], axis=1)
                removable_mask = mins > 1
                if np.any(removable_mask):
                    remove_idx = selected[removable_mask]
                    selected = selected[~removable_mask]
                    cover_count[balls[remove_idx]] -= 1
                    continue
                
                # Try 1-for-1 swap
                # Pick a random codeword c from selected
                c_idx = np.random.randint(len(selected))
                c = selected[c_idx]
                
                # Find unique words of c
                # words covered by c: balls[c]
                # unique if cover_count[balls[c]] == 1
                c_words = balls[c]
                unique_mask = cover_count[c_words] == 1
                unique_words = c_words[unique_mask]
                
                if len(unique_words) == 0:
                    # Should have been pruned, but maybe cover_count is stale?
                    # Force prune
                    continue
                
                # Find candidates c' that cover ALL unique_words
                # Start with candidates covering the first unique word
                first_w = unique_words[0]
                candidates = set(word_to_codewords[first_w])
                
                for w in unique_words[1:]:
                    if not candidates:
                        break
                    candidates &= set(word_to_codewords[w])
                
                if not candidates:
                    continue
                
                # Exclude c itself (swap with self is no-op)
                candidates.discard(c)
                
                if not candidates:
                    continue
                
                # Pick a random candidate
                c_prime = list(candidates)[np.random.randint(len(candidates))]
                
                # Perform swap: Remove c, Add c_prime
                # Check if swap is valid?
                # By construction, c' covers all unique words of c.
                # Non-unique words of c are covered by others, so removing c is safe for them.
                # Adding c' covers unique words.
                # So the union of balls is preserved.
                # Thus, the code remains a valid cover.
                
                cover_count[balls[c]] -= 1
                cover_count[balls[c_prime]] += 1
                selected[c_idx] = c_prime
                
                # Prune again
                changed = True
                while changed and time.time() < deadline - 5.0:
                    mins = np.min(cover_count[balls[selected]], axis=1)
                    removable_mask = mins > 1
                    if not np.any(removable_mask):
                        break
                    remove_idx = selected[removable_mask]
                    selected = selected[~removable_mask]
                    cover_count[balls[remove_idx]] -= 1
                    changed = True
                    
            return selected, cover_count

        selected, cover_count = try_swaps(selected, cover_count)
        
        # Try 2-for-1 swaps
        # Pick two codewords c1, c2.
        # Union of their unique words U.
        # Find c3 covering U.
        # If found, replace c1, c2 with c3.
        
        def try_2_for_1(selected, cover_count, max_iters=50):
            for _ in range(max_iters):
                if time.time() > deadline - 5.0:
                    break
                if len(selected) < 2:
                    break
                
                # Prune first
                mins = np.min(cover_count[balls[selected]], axis=1)
                removable_mask = mins > 1
                if np.any(removable_mask):
                    remove_idx = selected[removable_mask]
                    selected = selected[~removable_mask]
                    cover_count[balls[remove_idx]] -= 1
                    continue
                
                # Pick two distinct codewords
                idx1, idx2 = np.random.choice(len(selected), 2, replace=False)
                c1 = selected[idx1]
                c2 = selected[idx2]
                
                # Get unique words for c1 and c2
                # Note: A word might be unique to c1, or c2, or neither (covered by >=2).
                # We need to cover ALL words that become uncovered if we remove c1 and c2.
                # These are words w such that:
                # (w in B(c1) and cover_count[w] == 1) OR (w in B(c2) and cover_count[w] == 1)
                # Wait, if w is in B(c1) and B(c2), and cover_count[w] == 2, removing both makes it 0.
                # So we must cover words w where:
                # w in B(c1) union B(c2)
                # AND (cover_count[w] == 1 and w in B(c1)) OR (cover_count[w] == 1 and w in B(c2)) OR (cover_count[w] == 2 and w in B(c1) and w in B(c2))
                # Simpler:
                # Words that will be uncovered:
                # 1. Unique to c1: w in B(c1), cover_count[w]==1
                # 2. Unique to c2: w in B(c2), cover_count[w]==1
                # 3. Shared but only by c1,c2: w in B(c1) and w in B(c2) and cover_count[w]==2
                
                words_c1 = balls[c1]
                words_c2 = balls[c2]
                
                unique_c1 = words_c1[cover_count[words_c1] == 1]
                unique_c2 = words_c2[cover_count[words_c2] == 1]
                
                # Shared words with count 2
                # Intersection of words_c1 and words_c2
                # Since balls are small, use set
                set_c1 = set(words_c1.tolist())
                set_c2 = set(words_c2.tolist())
                shared = list(set_c1 & set_c2)
                shared_arr = np.array(shared, dtype=np.int32)
                shared_unique = shared_arr[cover_count[shared_arr] == 2]
                
                all_to_cover = np.concatenate([unique_c1, unique_c2, shared_unique])
                
                if len(all_to_cover) == 0:
                    # Removing c1 and c2 leaves no uncovered words?
                    # Then they are redundant. Should have been pruned.
                    continue
                
                # Find c3 covering all_to_cover
                first_w = all_to_cover[0]
                candidates = set(word_to_codewords[first_w])
                
                for w in all_to_cover[1:]:
                    if not candidates:
                        break
                    candidates &= set(word_to_codewords[w])
                
                if not candidates:
                    continue
                
                # Exclude c1 and c2
                candidates.discard(c1)
                candidates.discard(c2)
                
                if not candidates:
                    continue
                
                c3 = list(candidates)[np.random.randint(len(candidates))]
                
                # Perform swap: Remove c1, c2. Add c3.
                cover_count[balls[c1]] -= 1
                cover_count[balls[c2]] -= 1
                cover_count[balls[c3]] += 1
                
                # Update selected
                # Remove c2 and c1 (indices might shift)
                # Safer: create new list
                new_selected = []
                for i, c in enumerate(selected):
                    if i != idx1 and i != idx2:
                        new_selected.append(c)
                new_selected.append(c3)
                selected = np.array(new_selected, dtype=np.int32)
                
                # Prune
                changed = True
                while changed and time.time() < deadline - 5.0:
                    mins = np.min(cover_count[balls[selected]], axis=1)
                    removable_mask = mins > 1
                    if not np.any(removable_mask):
                        break
                    remove_idx = selected[removable_mask]
                    selected = selected[~removable_mask]
                    cover_count[balls[remove_idx]] -= 1
                    changed = True
                    
            return selected, cover_count

        selected, cover_count = try_2_for_1(selected, cover_count)

        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            if best_size <= 403:
                break

    if best_set is None:
        best_set = np.arange(total, dtype=np.int32)
        
    result = []
    for w in best_set:
        digits = []
        temp = int(w)
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
