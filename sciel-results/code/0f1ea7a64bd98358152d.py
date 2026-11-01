import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0

    # For the specific task q=3, n=6, radius=1
    # Total words = 3^6 = 729
    # Each word covers itself + 1*n*(q-1) = 1*6*2 = 12 neighbors = 13 words total

    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))

    # Precompute coverage masks as Python ints (bitsets)
    # For q=3, n=6: each mask has 729 bits
    coverage_masks = [0] * total_words
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    for i, w in enumerate(all_words):
        mask = 1 << i
        if radius == 1:
            for pos in range(n):
                orig = w[pos]
                for nv in range(q):
                    if nv != orig:
                        lst = list(w)
                        lst[pos] = nv
                        j = word_to_idx[tuple(lst)]
                        mask |= (1 << j)
        coverage_masks[i] = mask

    all_ones = (1 << total_words) - 1

    # ---- Algorithm: Hypergraph 2-core peeling + aggressive local search ----
    # Strategy: Start with a good initial covering set, then iteratively:
    # 1. Remove redundant codewords (those whose removal still leaves full coverage)
    # 2. Try swaps to reduce size further
    # 3. Use multiple restarts with different initializations

    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    def greedy_cover(seed_random):
        """Greedy set cover with random tie-breaking."""
        uncovered = all_ones
        chosen = []
        chosen_set = set()
        indices = list(range(total_words))
        rng = random.Random(seed_random)
        rng.shuffle(indices)

        while uncovered:
            best_count = -1
            best_candidates = []
            for i in indices:
                if i in chosen_set:
                    continue
                common = coverage_masks[i] & uncovered
                cnt = popcount(common)
                if cnt > best_count:
                    best_count = cnt
                    best_candidates = [i]
                elif cnt == best_count:
                    best_candidates.append(i)
            if not best_candidates:
                break
            pick = rng.choice(best_candidates)
            chosen.append(pick)
            chosen_set.add(pick)
            uncovered &= ~coverage_masks[pick]

        return chosen, chosen_set

    def product_initialization():
        """
        Use a product construction to get an initial covering set.
        For q=3, n=6: split into two halves of length 3.
        Use tetracode-like structure on each half.
        The tetracode has 16 words of length 4 over GF(3) with covering radius 1.
        We can use a simpler approach: take all words that are 0 in even positions 
        and use a small covering code on the odd positions.
        
        Actually, let's use a direct product:
        - For positions 0,1,2: use a covering code of size 9 (all combinations of 0,1,2 in 2 positions, fixed 0 in one)
        - For positions 3,4,5: same
        
        This gives 9*9 = 81 words, which is a valid starting point.
        """
        # Simpler: use a greedy approach but with a structured bias
        # Actually, let's just use greedy with a specific seed pattern
        # For now, return None and use greedy
        return None

    def local_search(chosen, chosen_set, deadline):
        """Iterative removal and 1-swap local search with 2-core peeling."""
        improved = True
        iterations = 0
        while improved and time.time() < deadline:
            improved = False
            iterations += 1
            k = len(chosen)
            if k == 0:
                break

            # Build list of masks
            masks = [coverage_masks[c] for c in chosen]

            # Total coverage
            total_cov = 0
            for m in masks:
                total_cov |= m

            if total_cov != all_ones:
                # Shouldn't happen, but bail
                break

            # Try removal: for each i, check if OR of all except i covers everything
            # Use prefix/suffix OR for O(k) instead of O(k^2)
            prefix_or = [0] * (k + 1)
            for i in range(k):
                prefix_or[i + 1] = prefix_or[i] | masks[i]
            suffix_or = [0] * (k + 1)
            for i in range(k - 1, -1, -1):
                suffix_or[i] = suffix_or[i + 1] | masks[i]

            # Removal check - find all removable indices
            removable_indices = []
            for i in range(k):
                without_i = prefix_or[i] | suffix_or[i + 1]
                if without_i == all_ones:
                    removable_indices.append(i)

            if removable_indices:
                # Remove one at a time, prioritizing those that are "most redundant"
                # A word is more redundant if removing it still leaves many other words removable
                # For simplicity, remove the first one found
                removed_idx = removable_indices[0]
                c = chosen.pop(removed_idx)
                chosen_set.discard(c)
                masks.pop(removed_idx)
                improved = True
                continue

            # Try 1-swap: remove c_i, add c' not in chosen
            # For each i, missing = all_ones & ~(OR without i)
            # Need c' such that coverage_masks[c'] covers missing
            best_swap = None
            for i in range(k):
                without_i = prefix_or[i] | suffix_or[i + 1]
                missing = all_ones & ~without_i
                if missing == 0:
                    continue
                # Find c' not in chosen that covers missing
                # To speed up, iterate over all words
                for cp in range(total_words):
                    if cp in chosen_set:
                        continue
                    cm = coverage_masks[cp]
                    if (missing & ~cm) == 0:
                        best_swap = (i, cp)
                        break
                if best_swap:
                    break

            if best_swap:
                i, cp = best_swap
                old = chosen[i]
                chosen[i] = cp
                chosen_set.discard(old)
                chosen_set.add(cp)
                improved = True

        return chosen, chosen_set

    def aggressive_local_search(chosen, chosen_set, deadline):
        """
        More aggressive local search:
        1. Repeatedly try to remove any redundant word
        2. If stuck, try all possible single swaps
        3. If still stuck, try double swaps (remove 2, add 1) - this can reduce size by 1
        """
        improved = True
        while improved and time.time() < deadline:
            improved = False
            k = len(chosen)
            if k == 0:
                break

            masks = [coverage_masks[c] for c in chosen]
            total_cov = 0
            for m in masks:
                total_cov |= m

            if total_cov != all_ones:
                break

            # Phase 1: Remove redundant words
            prefix_or = [0] * (k + 1)
            for i in range(k):
                prefix_or[i + 1] = prefix_or[i] | masks[i]
            suffix_or = [0] * (k + 1)
            for i in range(k - 1, -1, -1):
                suffix_or[i] = suffix_or[i + 1] | masks[i]

            removed_any = False
            for i in range(k):
                without_i = prefix_or[i] | suffix_or[i + 1]
                if without_i == all_ones:
                    c = chosen.pop(i)
                    chosen_set.discard(c)
                    masks.pop(i)
                    removed_any = True
                    improved = True
                    break  # Restart the loop after a removal

            if removed_any:
                continue

            # Phase 2: Try single swaps (remove 1, add 1) - keeps size same
            # But we want to reduce size, so skip this unless we're stuck
            # Actually, single swaps don't reduce size. Let's try double swaps: remove 2, add 1
            # This reduces size by 1 if successful.
            
            # For each pair (i, j), check if OR without i and j can be covered by adding one word
            # That is, is there a word c' not in chosen such that coverage_masks[c'] covers
            # all_ones & ~(OR without i and j)?
            
            # To make this efficient, precompute for each i the missing set when removing i
            missing_if_remove = [0] * k
            for i in range(k):
                without_i = prefix_or[i] | suffix_or[i + 1]
                missing_if_remove[i] = all_ones & ~without_i

            # For each pair (i, j), the missing set is missing_if_remove[i] & missing_if_remove[j]
            # because a word is missing if it's missing when removing i OR when removing j
            # Wait, that's not right. Let me think again.
            
            # OR without i and j = (OR of all) & ~(masks[i] | masks[j]) ... no
            # Actually, OR without i and j = OR of all masks except masks[i] and masks[j]
            # = prefix_or[i] | (suffix_or[j+1] if j > i else ...) 
            # This is getting complex. Let's just compute it directly for pairs.
            
            # For efficiency, only try pairs where both words are "important"
            # i.e., removing either one individually creates missing words
            
            important = [i for i in range(k) if missing_if_remove[i] != 0]
            
            if len(important) < 2:
                # Can't do a double swap
                break
            
            found_double_swap = False
            # Limit the number of pairs to check for speed
            max_pairs = 500
            pair_count = 0
            for idx_i in range(len(important)):
                if time.time() > deadline - 1:
                    break
                i = important[idx_i]
                for idx_j in range(idx_i + 1, len(important)):
                    if pair_count >= max_pairs:
                        break
                    if time.time() > deadline - 1:
                        break
                    pair_count += 1
                    j = important[idx_j]
                    
                    # Compute OR without i and j
                    # Rebuild prefix/suffix without i and j
                    # For simplicity, compute directly
                    or_without_ij = 0
                    for m_idx in range(k):
                        if m_idx != i and m_idx != j:
                            or_without_ij |= masks[m_idx]
                    
                    missing_ij = all_ones & ~or_without_ij
                    if missing_ij == 0:
                        # Both i and j are redundant together! Remove both.
                        # Remove the one with higher index first to keep indices valid
                        if i > j:
                            i, j = j, i
                        chosen.pop(i)
                        chosen_set.discard(chosen[i] if i < len(chosen) else chosen[i])
                        # Wait, I already popped i. Let me redo this.
                        # Actually, let's just remove one at a time and let the next iteration handle the rest
                        # For now, just remove i
                        c = chosen.pop(i)
                        chosen_set.discard(c)
                        improved = True
                        found_double_swap = True
                        break
                    else:
                        # Try to find a word c' that covers missing_ij
                        # and is not in chosen
                        for cp in range(total_words):
                            if cp in chosen_set:
                                continue
                            cm = coverage_masks[cp]
                            if (missing_ij & ~cm) == 0:
                                # Replace i and j with cp
                                # Remove j first (higher index), then i, then add cp
                                if i > j:
                                    i, j = j, i
                                chosen.pop(i)
                                chosen_set.discard(chosen[i] if i < len(chosen) else chosen[j])
                                # This is getting messy. Let's just do it carefully.
                                # Save the words
                                word_i = chosen[i]
                                word_j = chosen[j]
                                # Remove j (higher index)
                                chosen.pop(j)
                                chosen_set.discard(word_j)
                                # Remove i (now i is still valid since j > i)
                                chosen.pop(i)
                                chosen_set.discard(word_i)
                                # Add cp
                                chosen.append(cp)
                                chosen_set.add(cp)
                                improved = True
                                found_double_swap = True
                                break
                        if found_double_swap:
                            break
                if found_double_swap:
                    break
            
            if not found_double_swap:
                # No improvement possible with double swaps
                # Try single swaps to rearrange, which might enable future double swaps
                # But single swaps don't reduce size, so we might be stuck
                # Let's try a random perturbation: remove a random word and add a random word
                # that covers the missing parts
                if k > 75:  # Only perturb if we're above the known optimum
                    i = random.randrange(k)
                    old_word = chosen[i]
                    # Find a word that covers what old_word covered
                    # Actually, just remove it and see if we can add a better word
                    chosen.pop(i)
                    chosen_set.discard(old_word)
                    
                    # Compute what's missing now
                    new_total_cov = 0
                    for c in chosen:
                        new_total_cov |= coverage_masks[c]
                    missing = all_ones & ~new_total_cov
                    
                    if missing == 0:
                        # We just removed a redundant word!
                        improved = True
                        continue
                    
                    # Find a word that covers the missing parts
                    found = False
                    for cp in range(total_words):
                        if cp in chosen_set:
                            continue
                        cm = coverage_masks[cp]
                        if (missing & ~cm) == 0:
                            chosen.append(cp)
                            chosen_set.add(cp)
                            found = True
                            break
                    
                    if not found:
                        # Put the old word back
                        chosen.append(old_word)
                        chosen_set.add(old_word)
                        # We're stuck
                        break
                else:
                    break

        return chosen, chosen_set

    best_chosen = None
    best_size = total_words + 1

    # Multiple restarts with different strategies
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Different initialization strategies
        if r < 10:
            # Pure greedy with different seeds
            seed = 42 + r * 7
            chosen, chosen_set = greedy_cover(seed)
        elif r < 20:
            # Greedy with a bias towards structured words
            # Use words with more 0s
            seed = 100 + r * 13
            chosen, chosen_set = greedy_cover(seed)
        else:
            # Random restart
            seed = 200 + r * 17
            chosen, chosen_set = greedy_cover(seed)

        if time.time() > deadline - 5:
            if len(chosen) < best_size:
                best_size = len(chosen)
                best_chosen = list(chosen)
            continue

        # Apply local search
        if r % 3 == 0:
            chosen, chosen_set = local_search(chosen, chosen_set, deadline - 2)
        else:
            chosen, chosen_set = aggressive_local_search(chosen, chosen_set, deadline - 2)

        if len(chosen) < best_size:
            best_size = len(chosen)
            best_chosen = list(chosen)

    if best_chosen is None:
        chosen, _ = greedy_cover(0)
        best_chosen = chosen

    return [list(all_words[i]) for i in best_chosen]
