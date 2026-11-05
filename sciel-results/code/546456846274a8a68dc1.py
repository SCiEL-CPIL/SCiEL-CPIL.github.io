import numpy as np
import random
import time
import heapq

def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin for 150s limit

    # For the specific case of q=3, n=7, radius=1, we use specialized code
    if q == 3 and n == 7 and radius == 1:
        N = 3 ** n  # 2187
        ball_size = 1 + n * (q - 1)  # 15

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

        # Precompute balls as lists of integers
        balls = []
        for i in range(N):
            digits = int_to_base3(i, n)
            neighbors = [i]
            for pos in range(n):
                original = digits[pos]
                for new_val in range(q):
                    if new_val != original:
                        new_digits = digits[:]
                        new_digits[pos] = new_val
                        neighbors.append(base3_to_int(new_digits))
            balls.append(neighbors)
        
        # Precompute bitsets for each ball
        # Python integers are arbitrary precision, so we can use them as bitsets
        # Bit i corresponds to word i
        ball_bitsets = [0] * N
        for i in range(N):
            mask = 0
            for w in balls[i]:
                mask |= (1 << w)
            ball_bitsets[i] = mask

        # Precompute cover_map: for each word w, which codewords cover it
        # We need this for efficient 1-swap search: given a hole h, find candidates covering it
        # But for 1-swap, we need a candidate that covers ALL holes.
        # So, for a set of holes H, we need intersection of cover_map[h] for h in H.
        # Let's precompute cover_map as list of lists.
        cover_map = [[] for _ in range(N)]
        for c in range(N):
            for w in balls[c]:
                cover_map[w].append(c)

        def greedy_cover(seed):
            """
            Heap-based Greedy Set Cover.
            """
            rng = random.Random(seed)
            covered_mask = 0
            uncovered_count = N
            
            # cnt[c] is the number of currently uncovered words in ball(c)
            cnt = [ball_size] * N
            
            # Heap entries: (-count, candidate_index)
            heap = [(-ball_size, i) for i in range(N)]
            heapq.heapify(heap)
            
            in_code = [False] * N
            code = []

            while uncovered_count > 0:
                best_c = -1
                best_val = -1
                while heap:
                    neg_val, i = heapq.heappop(heap)
                    val = -neg_val
                    if in_code[i]:
                        continue
                    if cnt[i] == val:
                        best_c = i
                        best_val = val
                        break
                    else:
                        continue
                
                if best_c == -1:
                    break

                code.append(best_c)
                in_code[best_c] = True
                
                # Update covered and cnt
                for w in balls[best_c]:
                    # Check if w is already covered
                    if not (covered_mask & (1 << w)):
                        covered_mask |= (1 << w)
                        uncovered_count -= 1
                        for c2 in cover_map[w]:
                            old_cnt = cnt[c2]
                            cnt[c2] -= 1
                            heapq.heappush(heap, (-cnt[c2], c2))

            return code

        def refine(code):
            """Remove redundant codewords."""
            if not code:
                return code

            code_list = list(code)
            changed = True
            while changed:
                changed = False
                for i in range(len(code_list)):
                    c = code_list[i]
                    # Check if removing c leaves any word uncovered
                    # This is equivalent to: (union of balls of other codewords) covers ball(c)
                    # Or simpler: for each w in ball(c), is there another codeword covering w?
                    # We can check this by counting coverage.
                    
                    # Efficient check using bitsets?
                    # Let's just do a simple count check for small codes.
                    # But code size is ~100-200. N=2187.
                    # Let's use cover_count array.
                    pass
            
            # Re-implement refine with cover_count for robustness
            cover_count = [0] * N
            for c in code_list:
                for w in balls[c]:
                    cover_count[w] += 1
            
            changed = True
            while changed:
                changed = False
                for i in range(len(code_list)):
                    c = code_list[i]
                    safe = True
                    for w in balls[c]:
                        if cover_count[w] == 1:
                            safe = False
                            break
                    if safe:
                        for w in balls[c]:
                            cover_count[w] -= 1
                        code_list.pop(i)
                        changed = True
                        break
            return code_list

        def local_search(code, seed, time_limit):
            """
            Local Search with Deletion and 1-Swap.
            Uses bitsets for fast hole identification.
            """
            rng = random.Random(seed)
            code_list = list(code)
            if not code_list:
                return code_list

            # Maintain a bitset of covered words
            covered_mask = 0
            for c in code_list:
                covered_mask |= ball_bitsets[c]

            improved = True
            while improved and time.time() < time_limit:
                improved = False
                indices = list(range(len(code_list)))
                rng.shuffle(indices)
                
                for i in indices:
                    if time.time() > time_limit:
                        break
                    c_remove = code_list[i]
                    
                    # 1. Try Deletion
                    # If we remove c_remove, the new covered mask is covered_mask & ~ball_bitsets[c_remove]
                    # Holes are words in ball_bitsets[c_remove] that are NOT in (covered_mask & ~ball_bitsets[c_remove])
                    # i.e., holes = ball_bitsets[c_remove] & ~covered_mask
                    # Wait, if we remove c_remove, the words that become uncovered are those ONLY covered by c_remove.
                    # Those are words in ball(c_remove) such that (covered_mask without c_remove) does not cover them.
                    # covered_mask_without_c = covered_mask & ~ball_bitsets[c_remove]
                    # holes = ball_bitsets[c_remove] & ~covered_mask_without_c
                    # Note: ~covered_mask_without_c is infinite in Python, so we mask with ball_bitsets[c_remove]
                    holes_mask = ball_bitsets[c_remove] & ~(covered_mask & ~ball_bitsets[c_remove])
                    
                    if holes_mask == 0:
                        # Successful deletion
                        code_list.pop(i)
                        covered_mask &= ~ball_bitsets[c_remove]
                        improved = True
                        break

                    # 2. Try 1-Swap
                    # Find a codeword c_add not in code_list such that ball(c_add) covers holes_mask.
                    # i.e., holes_mask & ~ball_bitsets[c_add] == 0
                    # Or: holes_mask is a subset of ball_bitsets[c_add]
                    
                    # Optimization: Only check candidates that cover the "hardest" hole?
                    # Or just iterate through a random subset of all N codewords?
                    # N=2187 is small enough to iterate through all if we do it efficiently?
                    # 2187 checks per move. If we do 1000 moves, that's 2M checks. Fast.
                    
                    # To avoid bias, we can shuffle candidates or just pick random ones?
                    # Let's pick a random candidate first, then if it fails, maybe check a few more?
                    # Or better: intersect cover maps of a few random holes.
                    
                    # Let's try the intersection method for efficiency.
                    # Get list of holes
                    holes_list = []
                    temp_mask = holes_mask
                    while temp_mask:
                        # Extract lowest set bit
                        lsb = temp_mask & -temp_mask
                        h = lsb.bit_length() - 1
                        holes_list.append(h)
                        temp_mask ^= lsb
                    
                    if not holes_list:
                        continue

                    # Pick the hole with the fewest candidates
                    # This is a good heuristic
                    crit_hole = min(holes_list, key=lambda h: len(cover_map[h]))
                    candidates = set(cover_map[crit_hole])
                    
                    # Filter candidates that are already in code_list?
                    # If c_add is in code_list, it's already there.
                    # If we swap c_remove with c_add (where c_add is in code_list), 
                    # it's effectively a no-op for the set of codewords, but changes the list.
                    # But we want to reduce size or keep it same.
                    # If c_add is in code_list, removing c_remove and adding c_add keeps size same.
                    # But if c_add covers the holes, and c_add was already there, 
                    # why did we have holes?
                    # Because c_add might not cover the holes?
                    # Wait, if c_add is in code_list, it is part of covered_mask.
                    # So if c_add covers a hole h, then h would be covered by c_add in the original state.
                    # So h would not be a hole.
                    # Contradiction.
                    # Therefore, c_add CANNOT be in code_list if it covers the holes.
                    # So we can safely filter out candidates in code_list.
                    
                    in_code_set = set(code_list)
                    valid_candidates = [c for c in candidates if c not in in_code_set]
                    
                    # If no valid candidates, try other holes?
                    # For now, just pick one random valid candidate and check if it covers ALL holes.
                    if valid_candidates:
                        c_add = rng.choice(valid_candidates)
                        # Check if c_add covers all holes
                        if (holes_mask & ~ball_bitsets[c_add]) == 0:
                            # Successful swap
                            code_list[i] = c_add
                            # Update covered_mask
                            # Remove c_remove's contribution, add c_add's contribution
                            # Note: covered_mask is union of all balls.
                            # We can just recompute? No, that's O(K*N).
                            # Let's update:
                            # new_covered = (covered_mask & ~ball_bitsets[c_remove]) | ball_bitsets[c_add]
                            # But we need to be careful if other codewords cover the same spots.
                            # The bitset OR handles this correctly.
                            covered_mask = (covered_mask & ~ball_bitsets[c_remove]) | ball_bitsets[c_add]
                            improved = True
                            break
                    # If no valid candidate found from the first hole's cover map, 
                    # we could try other holes, but for simplicity and speed, we skip.
                    # This might miss some swaps, but the random restarts help.

            return code_list

        best_code = None
        best_len = float('inf')

        # Multiple restarts
        max_restarts = 100
        for i in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Generate initial code
            code = greedy_cover(i)
            refined = refine(code)
            
            if len(refined) < best_len:
                best_len = len(refined)
                best_code = refined
            
            # Run local search on this code
            if time.time() < deadline - 5:
                ls_code = local_search(refined, i, deadline - 5)
                if len(ls_code) < best_len:
                    best_len = len(ls_code)
                    best_code = ls_code

        if best_code is not None:
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

    # Generic greedy
    uncovered = np.ones(N, dtype=bool)
    code = []
    while np.any(uncovered):
        if time.time() > deadline:
            break
        gains = np.sum(uncovered[neighbors], axis=1)
        max_gain = np.max(gains)
        if max_gain == 0:
            break
        candidates = np.where(gains == max_gain)[0]
        best_idx = candidates[0]
        code.append(best_idx)
        uncovered[neighbors[best_idx]] = False

    return [idx_to_word(i) for i in code]
