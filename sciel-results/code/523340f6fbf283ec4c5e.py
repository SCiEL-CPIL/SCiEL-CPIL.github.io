import numpy as np
import random
import time
import heapq

def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin for 120s limit

    # For the specific case of q=3, n=7, radius=1, we use specialized code
    if q == 3 and n == 7 and radius == 1:
        N = 3 ** n  # 2187
        ball_size = 1 + n * (q - 1)  # 15
        FULL_MASK = (1 << N) - 1

        def int_to_base3(num, length):
            res = [0] * length
            for i in range(length - 1, -1, -1):
                res[i] = num % 3
                num //= 3
            return res

        def base3_to_int(digits):
            res = 0
            for d in digits:
                res = res * 3 + d
            return res

        # Precompute cover_mask: bitset of words covered by each codeword
        cover_mask = [0] * N
        # Precompute cover_map: for each word, list of codewords covering it
        cover_map = [[] for _ in range(N)]

        for i in range(N):
            digits = int_to_base3(i, n)
            mask = 1 << i
            cover_map[i].append(i)
            
            for pos in range(n):
                original = digits[pos]
                for new_val in range(q):
                    if new_val != original:
                        new_digits = digits[:]
                        new_digits[pos] = new_val
                        w = base3_to_int(new_digits)
                        mask |= (1 << w)
                        cover_map[w].append(i)
            
            cover_mask[i] = mask

        def greedy_cover(seed, start_code=None):
            """Greedy set cover with random tie-breaking."""
            rng = random.Random(seed)
            uncovered = FULL_MASK
            code = []
            in_code = [False] * N
            
            if start_code:
                for c in start_code:
                    code.append(c)
                    in_code[c] = True
                    uncovered &= ~cover_mask[c]

            # We need to pick codewords that maximize new coverage.
            # Since N=2187, linear scan is fast enough.
            while uncovered:
                best_c = -1
                best_gain = 0
                # Randomize tie-breaking by iterating in random order?
                # Or just pick first max?
                # To introduce diversity, we can add small random noise to gain.
                # But gain is integer. Let's just pick one of the bests randomly.
                candidates = []
                for c in range(N):
                    if not in_code[c]:
                        gain = (cover_mask[c] & uncovered).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            candidates = [c]
                        elif gain == best_gain and gain > 0:
                            candidates.append(c)
                
                if not candidates:
                    break
                    
                c = rng.choice(candidates)
                code.append(c)
                in_code[c] = True
                uncovered &= ~cover_mask[c]

            return code

        def refine(code):
            """Remove redundant codewords."""
            if not code:
                return code

            cover_count = [0] * N
            for c in code:
                # Iterate bits in cover_mask[c]
                mask = cover_mask[c]
                while mask:
                    # Get index of lowest set bit
                    lsb = mask & -mask
                    idx = lsb.bit_length() - 1
                    cover_count[idx] += 1
                    mask ^= lsb

            code_list = list(code)
            changed = True
            while changed:
                changed = False
                for i in range(len(code_list)):
                    c = code_list[i]
                    safe = True
                    mask = cover_mask[c]
                    while mask:
                        lsb = mask & -mask
                        idx = lsb.bit_length() - 1
                        if cover_count[idx] == 1:
                            safe = False
                            break
                        mask ^= lsb
                    if safe:
                        # Remove c
                        mask = cover_mask[c]
                        while mask:
                            lsb = mask & -mask
                            idx = lsb.bit_length() - 1
                            cover_count[idx] -= 1
                            mask ^= lsb
                        code_list.pop(i)
                        changed = True
                        break
            return code_list

        def local_search(code, seed, time_limit):
            """Local search with deletion, 1-swap, and 2-swap moves using bitsets."""
            rng = random.Random(seed)
            code_list = list(code)
            if not code_list:
                return code_list

            # Initialize cover_count
            cover_count = [0] * N
            for c in code_list:
                mask = cover_mask[c]
                while mask:
                    lsb = mask & -mask
                    idx = lsb.bit_length() - 1
                    cover_count[idx] += 1
                    mask ^= lsb

            in_code = [False] * N
            for c in code_list:
                in_code[c] = True

            improved = True
            while improved and time.time() < time_limit:
                improved = False
                indices = list(range(len(code_list)))
                rng.shuffle(indices)
                
                # 1. Try Deletion and 1-Swap
                for i in indices:
                    if time.time() > time_limit:
                        break
                    c_remove = code_list[i]
                    mask_rm = cover_mask[c_remove]
                    
                    # Check if deletion is safe
                    newly_uncovered_mask = 0
                    # Words that will become uncovered are those with count 1
                    # We can't just AND masks. We need to know which words have count 1.
                    # Iterating bits of mask_rm is fast.
                    temp_mask = mask_rm
                    uncov_list = []
                    while temp_mask:
                        lsb = temp_mask & -temp_mask
                        idx = lsb.bit_length() - 1
                        if cover_count[idx] == 1:
                            uncov_list.append(idx)
                            newly_uncovered_mask |= lsb
                        temp_mask ^= lsb

                    if not uncov_list:
                        # Safe to delete
                        temp_mask = mask_rm
                        while temp_mask:
                            lsb = temp_mask & -temp_mask
                            idx = lsb.bit_length() - 1
                            cover_count[idx] -= 1
                            temp_mask ^= lsb
                        code_list.pop(i)
                        in_code[c_remove] = False
                        improved = True
                        break

                    # Try 1-Swap: Find c_add such that cover_mask[c_add] covers newly_uncovered_mask
                    found = -1
                    # Candidates must cover all words in uncov_list
                    # Intersect cover_map for each word in uncov_list
                    cand_pool = None
                    for w in uncov_list:
                        cands = [c for c in cover_map[w] if not in_code[c] and c != c_remove]
                        if cand_pool is None:
                            cand_pool = set(cands)
                        else:
                            cand_pool.intersection_update(cands)
                        if not cand_pool:
                            break
                    
                    if cand_pool:
                        # Pick one that minimizes "waste" or just random?
                        # Let's pick one that maximizes overlap with current code (redundancy)
                        # Or just pick the first one.
                        # To be fast, pick first.
                        found = next(iter(cand_pool))
                        
                        # Verify it covers all
                        # (It should, by construction of cand_pool)
                        
                        # Execute swap
                        # Remove c_remove
                        temp_mask = mask_rm
                        while temp_mask:
                            lsb = temp_mask & -temp_mask
                            idx = lsb.bit_length() - 1
                            cover_count[idx] -= 1
                            temp_mask ^= lsb
                        in_code[c_remove] = False
                        
                        # Add found
                        code_list[i] = found
                        in_code[found] = True
                        temp_mask = cover_mask[found]
                        while temp_mask:
                            lsb = temp_mask & -temp_mask
                            idx = lsb.bit_length() - 1
                            cover_count[idx] += 1
                            temp_mask ^= lsb
                            
                        improved = True
                        break
                    else:
                        # No swap found, do nothing (counts are unchanged)
                        pass

                # 2. Try 2-Swap: Remove c1, c2, add c3
                if time.time() < time_limit and len(code_list) >= 3:
                    n_codes = len(code_list)
                    # Randomly sample pairs to save time if n_codes is large?
                    # n_codes is ~186, pairs ~17k. This is fine.
                    for i2 in range(n_codes):
                        if time.time() > time_limit:
                            break
                        for j2 in range(i2 + 1, n_codes):
                            if time.time() > time_limit:
                                break
                            c_rem1 = code_list[i2]
                            c_rem2 = code_list[j2]
                            
                            # Calculate uncovered mask
                            uncov_list = []
                            # Temporarily decrement counts? 
                            # Or just check: word w is uncovered if count[w] == 1 and (w in mask1 or w in mask2) and not (w in mask1 AND w in mask2)?
                            # Actually, w is uncovered if (count[w] == 1 and w in mask1) OR (count[w] == 1 and w in mask2) ... wait.
                            # If count[w] == 2, and we remove 2 codewords covering it, it becomes 0.
                            # If count[w] == 1, and we remove the 1 codeword covering it, it becomes 0.
                            
                            # Efficient way:
                            # Iterate bits in mask1.
                            # If count[w] == 1, it's a candidate for uncovered.
                            # If count[w] == 2, check if w is in mask2. If yes, it becomes 0.
                            
                            uncov_mask = 0
                            temp_mask = cover_mask[c_rem1]
                            while temp_mask:
                                lsb = temp_mask & -temp_mask
                                idx = lsb.bit_length() - 1
                                if cover_count[idx] == 1:
                                    uncov_mask |= lsb
                                    uncov_list.append(idx)
                                elif cover_count[idx] == 2:
                                    # Check if c_rem2 also covers idx
                                    if cover_mask[c_rem2] & lsb:
                                        uncov_mask |= lsb
                                        uncov_list.append(idx)
                                temp_mask ^= lsb
                                
                            # Also check words covered ONLY by c_rem2
                            temp_mask = cover_mask[c_rem2] & ~cover_mask[c_rem1]
                            while temp_mask:
                                lsb = temp_mask & -temp_mask
                                idx = lsb.bit_length() - 1
                                if cover_count[idx] == 1:
                                    uncov_mask |= lsb
                                    uncov_list.append(idx)
                                elif cover_count[idx] == 2:
                                    # This case is handled? 
                                    # If count is 2, and w is in mask2 but not mask1.
                                    # If the other coverer is c_rem1, it would be in mask1.
                                    # So if w is in mask2, not mask1, and count is 2, the other coverer is NOT c_rem1.
                                    # So removing c_rem1 and c_rem2 leaves count 1. Not uncovered.
                                    # So only count==1 matters for "only in mask2".
                                    pass
                                temp_mask ^= lsb

                            if not uncov_list:
                                # Safe to remove both? No, we must add one.
                                # If uncov_list is empty, we can just remove one?
                                # If uncov_list is empty, it means removing c_rem1 and c_rem2 doesn't uncover anything.
                                # This implies we can remove one of them safely (deletion case).
                                # So we can just remove c_rem2 and keep c_rem1?
                                # Actually, if uncov_list is empty, we can remove c_rem1 AND c_rem2?
                                # No, we must add one to keep size constant? 
                                # The move is "2-for-1". If 0 uncovered, we can remove 2 and add 0? 
                                # That reduces size by 2. That's even better!
                                # But standard local search keeps size constant or reduces by 1.
                                # If we can reduce by 2, we should do it.
                                # Let's handle this: if uncov_list is empty, remove both.
                                # But wait, if uncov_list is empty, it means every word covered by c_rem1 or c_rem2 is covered by at least one other codeword.
                                # So we can remove both.
                                temp_mask = cover_mask[c_rem1]
                                while temp_mask:
                                    lsb = temp_mask & -temp_mask
                                    idx = lsb.bit_length() - 1
                                    cover_count[idx] -= 1
                                    temp_mask ^= lsb
                                temp_mask = cover_mask[c_rem2]
                                while temp_mask:
                                    lsb = temp_mask & -temp_mask
                                    idx = lsb.bit_length() - 1
                                    cover_count[idx] -= 1
                                    temp_mask ^= lsb
                                
                                code_list.pop(j2)
                                code_list.pop(i2)
                                in_code[c_rem1] = False
                                in_code[c_rem2] = False
                                improved = True
                                break
                            
                            # Find c_add covering uncov_list
                            cand_pool = None
                            for w in uncov_list:
                                cands = [c for c in cover_map[w] if not in_code[c] and c != c_rem1 and c != c_rem2]
                                if cand_pool is None:
                                    cand_pool = set(cands)
                                else:
                                    cand_pool.intersection_update(cands)
                                if not cand_pool:
                                    break
                            
                            found2 = -1
                            if cand_pool:
                                found2 = next(iter(cand_pool))
                                
                                # Execute 2-for-1
                                # Remove c_rem1, c_rem2
                                temp_mask = cover_mask[c_rem1]
                                while temp_mask:
                                    lsb = temp_mask & -temp_mask
                                    idx = lsb.bit_length() - 1
                                    cover_count[idx] -= 1
                                    temp_mask ^= lsb
                                temp_mask = cover_mask[c_rem2]
                                while temp_mask:
                                    lsb = temp_mask & -temp_mask
                                    idx = lsb.bit_length() - 1
                                    cover_count[idx] -= 1
                                    temp_mask ^= lsb
                                in_code[c_rem1] = False
                                in_code[c_rem2] = False
                                
                                # Add found2
                                code_list[j2] = found2
                                in_code[found2] = True
                                temp_mask = cover_mask[found2]
                                while temp_mask:
                                    lsb = temp_mask & -temp_mask
                                    idx = lsb.bit_length() - 1
                                    cover_count[idx] += 1
                                    temp_mask ^= lsb
                                
                                code_list.pop(i2)
                                improved = True
                                break
                            
                            if improved:
                                break
                        if improved:
                            break

            return code_list

        best_code = None
        best_len = float('inf')

        # Multiple restarts
        num_restarts = 100
        for i in range(num_restarts):
            if time.time() > deadline - 15:
                break
            code = greedy_cover(i)
            refined = refine(code)
            if time.time() > deadline - 10:
                if len(refined) < best_len:
                    best_len = len(refined)
                    best_code = refined
                continue
            ls_time = min(deadline - 8, time.time() + 4.0)
            ls_code = local_search(refined, i * 7 + 1, ls_time)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code

        # Kick / re-greedy restarts
        if best_code is not None:
            for kick_iter in range(12):
                if time.time() > deadline - 12:
                    break
                rng_k = random.Random(1000 + kick_iter)
                k = min(5, len(best_code))
                kick_set = set(rng_k.sample(best_code, k))
                remaining = [c for c in best_code if c not in kick_set]
                re_code = greedy_cover(2000 + kick_iter, start_code=remaining)
                re_refined = refine(re_code)
                if len(re_refined) < best_len:
                    best_len = len(re_refined)
                    best_code = re_refined
                if time.time() > deadline - 10:
                    break
                ls_code = local_search(re_refined, 3000 + kick_iter, min(deadline - 8, time.time() + 3.5))
                if len(ls_code) < best_len:
                    best_len = len(ls_code)
                    best_code = ls_code

        # Final polish
        if best_code is not None and time.time() < deadline - 5:
            ls_code = local_search(best_code, 42, deadline - 3)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code
            final_code = refine(best_code)
            if time.time() < deadline - 2 and len(final_code) > 150:
                ls_code2 = local_search(final_code, 99, deadline - 1)
                if len(ls_code2) < len(final_code):
                    final_code = refine(ls_code2)
        elif best_code is not None:
            final_code = refine(best_code)
        else:
            final_code = []

        result = []
        for c in final_code:
            result.append(int_to_base3(c, n))
        return result

    # Generic case
    N = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        return w

    nb_count = 1 + n * (q - 1)
    neighbors = np.empty((N, nb_count), dtype=np.int32)
    for idx in range(N):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        nb_list = [idx]
        for pos in range(n):
            for new_d in range(q):
                if new_d == w[pos]:
                    continue
                delta = (new_d - w[pos]) * q_powers[n - 1 - pos]
                nb_list.append(idx + delta)
        neighbors[idx] = nb_list

    uncovered = np.ones(N, dtype=bool)
    code = []
    while np.any(uncovered):
        if time.time() > deadline:
            break
        best_idx = -1
        max_gain = 0
        for i in range(N):
            gain = np.sum(uncovered[neighbors[i]])
            if gain > max_gain:
                max_gain = gain
                best_idx = i
        if max_gain == 0:
            break
        code.append(best_idx)
        uncovered[neighbors[best_idx]] = False

    return [idx_to_word(i) for i in code]
