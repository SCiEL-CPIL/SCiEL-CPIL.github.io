# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def construct(q, n, radius):
    # Task: q=3, n=7, radius=1
    # Total words: 3^7 = 2187
    # Ball size: 1 + 7*2 = 15

    if q != 3 or n != 7 or radius != 1:
        # Fallback: greedy set cover for other parameters
        total = q ** n
        q_pows = [1]
        for i in range(n):
            q_pows.append(q_pows[-1] * q)
        
        ball_indices = [[] for _ in range(total)]
        for i in range(total):
            neighbors = [i]
            for pos in range(n):
                cur = (i // q_pows[pos]) % q
                for nd in range(q):
                    if nd == cur:
                        continue
                    neighbors.append(i + (nd - cur) * q_pows[pos])
            ball_indices[i] = neighbors
        
        deadline = time.time() + 115.0
        full_mask = (1 << total) - 1 if total <= 60 else None
        best_code = None
        best_size = float('inf')
        rng = random.Random(42)
        
        while time.time() < deadline:
            uncovered = set(range(total))
            code = []
            remaining = set(range(total))
            while uncovered and remaining:
                best_c = None
                best_cnt = -1
                for c in remaining:
                    cnt = len(set(ball_indices[c]) & uncovered)
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_c = c
                if best_c is None:
                    break
                code.append(best_c)
                remaining.discard(best_c)
                uncovered -= set(ball_indices[best_c])
            # minimize
            changed = True
            while changed:
                changed = False
                for i in range(len(code)-1, -1, -1):
                    c = code[i]
                    essential = False
                    for w in ball_indices[c]:
                        cnt = sum(1 for c2 in code if c2 != c and w in ball_indices[c2])
                        if cnt == 0:
                            essential = True
                            break
                    if not essential:
                        code.pop(i)
                        changed = True
                        break
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
        
        if best_code is None:
            best_code = list(range(total))
        
        def idx_to_word(idx):
            word = []
            temp = idx
            for _ in range(n):
                word.append(temp % q)
                temp //= q
            return word
        
        return [idx_to_word(idx) for idx in best_code]

    total_words = q ** n  # 2187
    deadline = time.time() + 115.0

    # Precompute powers of q
    q_pows = [1]
    for i in range(n):
        q_pows.append(q_pows[-1] * q)

    # Precompute balls: for each word index, list of indices within distance 1
    ball_indices = [[] for _ in range(total_words)]
    ball_masks = [0] * total_words
    for i in range(total_words):
        neighbors = [i]
        mask = 1 << i
        for pos in range(n):
            cur = (i // q_pows[pos]) % q
            for nd in range(q):
                if nd == cur:
                    continue
                new_idx = i + (nd - cur) * q_pows[pos]
                neighbors.append(new_idx)
                mask |= (1 << new_idx)
        ball_indices[i] = neighbors
        ball_masks[i] = mask

    full_mask = (1 << total_words) - 1

    # Precompute as numpy for fast bincount
    ball_idx_np = np.array(ball_indices, dtype=np.int32)  # (2187, 15)

    def idx_to_word(idx):
        word = []
        temp = idx
        for _ in range(n):
            word.append(temp % q)
            temp //= q
        return word

    def compute_counts(code):
        """Compute coverage counts for a list of codeword indices."""
        if not code:
            return np.zeros(total_words, dtype=np.int32)
        code_arr = np.array(code, dtype=np.int32)
        all_neigh = ball_idx_np[code_arr].ravel()
        return np.bincount(all_neigh, minlength=total_words).astype(np.int32)

    def minimize_code(code):
        """Remove redundant codewords iteratively."""
        counts = compute_counts(code)
        changed = True
        while changed:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                # Check if any word in c's ball has count == 1 (only covered by c)
                essential = False
                for w in ball_indices[c]:
                    if counts[w] == 1:
                        essential = True
                        break
                if not essential:
                    code.pop(i)
                    for w in ball_indices[c]:
                        counts[w] -= 1
                    changed = True
                    break
        return code

    def local_search_1swap(code, max_iters=300):
        """Try 1-swaps and deletions to reduce code size."""
        counts = compute_counts(code)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 5.0:
                break
            
            for i in range(len(code)):
                c = code[i]
                # Find words uniquely covered by c
                unique_words = [w for w in ball_indices[c] if counts[w] == 1]
                
                if not unique_words:
                    # Can delete c
                    code.pop(i)
                    for w in ball_indices[c]:
                        counts[w] -= 1
                    improved = True
                    break
                else:
                    # Try to find a replacement that covers all unique_words
                    # and is not already in code
                    # A candidate d can replace c if:
                    # 1. d covers all unique_words (i.e., all unique_words in ball_indices[d])
                    # 2. d is not already in code
                    # 3. After swap, no word becomes uncovered
                    #    i.e., for every w in ball_indices[c] \\ ball_indices[d], counts[w] >= 2
                    #    (since d's ball may not cover w, but other codewords must)
                    
                    # Efficient check: candidate d must have all unique_words in its ball
                    # Precompute: for each word w, which codeword indices cover it
                    # But that's expensive. Instead, iterate over candidates.
                    
                    # Only try candidates that cover the first unique word
                    uw0 = unique_words[0]
                    # Find all codeword indices that cover uw0
                    # We can check: d covers uw0 iff uw0 is in ball_indices[d]
                    # Equivalently, d is in ball_indices[uw0]
                    candidates_for_uw0 = ball_indices[uw0]
                    
                    code_set = set(code)
                    best_swap = None
                    best_gain = 0  # We want to swap (no size change) or find deletion
                    
                    for d in candidates_for_uw0:
                        if d in code_set:
                            continue
                        # Check d covers all unique_words
                        d_ball = ball_indices[d]
                        covers_all = all(w in d_ball for w in unique_words)
                        if not covers_all:
                            continue
                        # Check that removing c and adding d doesn't uncover anything
                        # Words in c's ball but not in d's ball must have count >= 2
                        ok = True
                        for w in ball_indices[c]:
                            if w not in d_ball:
                                if counts[w] < 2:
                                    ok = False
                                    break
                        if ok:
                            # This swap is valid (size stays same, but might enable future deletions)
                            best_swap = d
                            break
                    
                    if best_swap is not None:
                        code[i] = best_swap
                        code_set.discard(c)
                        code_set.add(best_swap)
                        for w in ball_indices[c]:
                            counts[w] -= 1
                        for w in ball_indices[best_swap]:
                            counts[w] += 1
                        improved = True
                        break
        
        return code

    def greedy_set_cover(rng, use_randomization=True):
        """Greedy set cover with optional randomization."""
        uncovered = full_mask
        code = []
        remaining = set(range(total_words))
        
        while uncovered and remaining:
            max_count = -1
            best_candidates = []
            
            # For speed, if many remaining, sample
            if len(remaining) > 300:
                sample_size = min(300, len(remaining))
                sample = rng.sample(list(remaining), sample_size)
            else:
                sample = list(remaining)
            
            for c in sample:
                count = (ball_masks[c] & uncovered).bit_count()
                if count > max_count:
                    max_count = count
                    best_candidates = [c]
                elif count == max_count:
                    best_candidates.append(c)
            
            if best_candidates:
                chosen = rng.choice(best_candidates)
            else:
                chosen = rng.choice(list(remaining))
            
            code.append(chosen)
            remaining.discard(chosen)
            uncovered &= ~ball_masks[chosen]
        
        return code

    def greedy_with_priority(rng):
        """Greedy that prioritizes covering rare words first."""
        uncovered = full_mask
        code = []
        remaining = set(range(total_words))
        
        while uncovered and remaining:
            # Find the uncovered word with fewest covering candidates
            # This is expensive, so approximate: pick the word in uncovered with min degree
            # For speed, just do standard greedy but with a twist:
            # Weight candidates by (newly covered / (1 + number of already-chosen that also cover those))
            
            max_score = -1
            best_candidates = []
            
            if len(remaining) > 300:
                sample = rng.sample(list(remaining), 300)
            else:
                sample = list(remaining)
            
            for c in sample:
                new_covered = ball_masks[c] & uncovered
                count = new_covered.bit_count()
                if count == 0:
                    continue
                # Simple score: count
                score = count
                if score > max_score:
                    max_score = score
                    best_candidates = [c]
                elif score == max_score:
                    best_candidates.append(c)
            
            if best_candidates:
                chosen = rng.choice(best_candidates)
            else:
                chosen = rng.choice(list(remaining))
            
            code.append(chosen)
            remaining.discard(chosen)
            uncovered &= ~ball_masks[chosen]
        
        return code

    best_code = None
    best_size = float('inf')

    # Strategy: multiple restarts with different greedy variants, then local search
    rng = random.Random(12345)
    iteration = 0

    # First, try a deterministic structured seed
    # Tetracode-based seed: 9*3^3 = 243 words
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_words = []
    for h in tetracode:
        for tail in __import__('itertools').product(range(3), repeat=3):
            seed_words.append(h + list(tail))
    seed_indices = []
    for w in seed_words:
        idx = 0
        for k in range(n):
            idx += w[k] * q_pows[k]
        seed_indices.append(idx)
    
    min_seed = minimize_code(seed_indices[:])
    ls_seed = local_search_1swap(min_seed[:], max_iters=200)
    best_size = len(ls_seed)
    best_code = ls_seed[:]

    if best_size <= 186:
        return [idx_to_word(idx) for idx in best_code]

    # Now run multiple randomized restarts
    while time.time() < deadline - 10.0:
        iteration += 1
        rng = random.Random(42 + iteration)
        
        if iteration % 3 == 1:
            code = greedy_set_cover(rng)
        elif iteration % 3 == 2:
            code = greedy_with_priority(rng)
        else:
            # Randomized greedy with different sampling
            uncovered = full_mask
            code = []
            remaining = set(range(total_words))
            while uncovered and remaining:
                max_count = -1
                best_candidates = []
                sample_size = min(200, len(remaining))
                sample = rng.sample(list(remaining), sample_size)
                for c in sample:
                    count = (ball_masks[c] & uncovered).bit_count()
                    if count > max_count:
                        max_count = count
                        best_candidates = [c]
                    elif count == max_count:
                        best_candidates.append(c)
                if best_candidates:
                    chosen = rng.choice(best_candidates)
                else:
                    chosen = rng.choice(list(remaining))
                code.append(chosen)
                remaining.discard(chosen)
                uncovered &= ~ball_masks[chosen]
        
        min_code = minimize_code(code[:])
        ls_code = local_search_1swap(min_code[:], max_iters=200)
        
        if len(ls_code) < best_size:
            best_size = len(ls_code)
            best_code = ls_code[:]
            if best_size <= 186:
                break

    # Final: try more aggressive local search on best code
    if best_code and time.time() < deadline - 5.0:
        # Multiple rounds of local search
        for round_num in range(5):
            if time.time() > deadline - 5.0:
                break
            improved = True
            while improved:
                improved = False
                counts = compute_counts(best_code)
                for i in range(len(best_code)):
                    c = best_code[i]
                    unique_words = [w for w in ball_indices[c] if counts[w] == 1]
                    if not unique_words:
                        best_code.pop(i)
                        for w in ball_indices[c]:
                            counts[w] -= 1
                        improved = True
                        break
                    else:
                        code_set = set(best_code)
                        uw0 = unique_words[0]
                        for d in ball_indices[uw0]:
                            if d in code_set:
                                continue
                            d_ball = ball_indices[d]
                            if not all(w in d_ball for w in unique_words):
                                continue
                            ok = True
                            for w in ball_indices[c]:
                                if w not in d_ball and counts[w] < 2:
                                    ok = False
                                    break
                            if ok:
                                best_code[i] = d
                                code_set.discard(c)
                                code_set.add(d)
                                for w in ball_indices[c]:
                                    counts[w] -= 1
                                for w in ball_indices[d]:
                                    counts[w] += 1
                                improved = True
                                break
                        if improved:
                            break

    result = [idx_to_word(idx) for idx in best_code]
    return result
# EVOLVE-BLOCK-END
