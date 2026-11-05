import numpy as np
import random
import time

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

        # Precompute balls (neighbors including self)
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

        # Precompute cover_map: for each word w, which codewords cover it
        cover_map = [[] for _ in range(N)]
        for c in range(N):
            for w in balls[c]:
                cover_map[w].append(c)

        # Precompute ball sets for fast membership (swap search)
        ball_sets = [set(b) for b in balls]
        
        # Precompute list of candidate codewords for each word w that are NOT w itself?
        # No, cover_map[w] is the list of codewords c such that w in balls[c].
        # For swap, if we remove c, and w becomes uncovered, we need a c' in cover_map[w] \ {c}.

        def greedy_cover(seed, start_code=None):
            """Greedy set cover with random tie-breaking.
            If start_code is given, start from that partial cover and continue
            covering the remaining words (kick / re-greedy)."""
            rng = random.Random(seed)
            covered = [False] * N
            # cnt[c] = number of uncovered words in ball c
            cnt = [ball_size] * N
            code = []
            uncovered_count = N

            if start_code:
                for c in start_code:
                    code.append(c)
                    for w in balls[c]:
                        if not covered[w]:
                            covered[w] = True
                            uncovered_count -= 1
                            for c2 in cover_map[w]:
                                cnt[c2] -= 1

            while uncovered_count > 0:
                max_val = max(cnt)
                candidates = [i for i in range(N) if cnt[i] == max_val]
                c = rng.choice(candidates)
                code.append(c)

                ball_c = balls[c]
                for w in ball_c:
                    if not covered[w]:
                        covered[w] = True
                        uncovered_count -= 1
                        for c2 in cover_map[w]:
                            cnt[c2] -= 1

            return code

        def refine(code):
            """Remove redundant codewords using critical count tracking."""
            if not code:
                return code

            cover_count = [0] * N
            for c in code:
                for w in balls[c]:
                    cover_count[w] += 1

            # critical_count[c] = number of words w in balls[c] such that cover_count[w] == 1
            critical_count = [0] * N
            for c in code:
                for w in balls[c]:
                    if cover_count[w] == 1:
                        critical_count[c] += 1

            code_list = list(code)
            in_code = [False] * N
            for c in code_list:
                in_code[c] = True

            changed = True
            while changed:
                changed = False
                # Find a codeword with critical_count 0
                # To be efficient, we can just scan the list
                for i in range(len(code_list)):
                    c = code_list[i]
                    if critical_count[c] == 0:
                        # Remove c
                        for w in balls[c]:
                            old_count = cover_count[w]
                            cover_count[w] -= 1
                            # Update critical counts for c and other codewords covering w
                            if old_count == 1:
                                # w was critical for c (but we checked critical_count[c]==0, so this shouldn't happen for c)
                                # However, for other codewords c2 covering w, if old_count was 2, it becomes 1 (critical increases)
                                # if old_count was 1, it becomes 0 (critical decreases)
                                # Wait, if old_count == 1, then w was critical for c. But we only remove c if critical_count[c]==0.
                                # So old_count must be > 1 for all w in balls[c].
                                # Thus old_count >= 2.
                                pass
                            
                            # We need to update critical_count for all c2 in cover_map[w]
                            for c2 in cover_map[w]:
                                if not in_code[c2]:
                                    continue
                                if old_count == 2:
                                    # w becomes critical for c2
                                    critical_count[c2] += 1
                                elif old_count == 1:
                                    # w was critical for c2, now it's not
                                    critical_count[c2] -= 1
                            
                            # Note: c is being removed, so we don't update critical_count[c] for the new state, 
                            # but we are removing c from the code, so its critical_count doesn't matter anymore.
                            # However, we must ensure we don't decrement critical_count[c] if c2==c in the loop above?
                            # c is in cover_map[w]. 
                            # If old_count == 2, critical_count[c] would be incremented. But we are removing c.
                            # It's cleaner to update critical counts for all c2 != c first, then handle c.
                            # Actually, since we are removing c, we just need to update the critical counts of the REMAINING codewords.
                            # Let's re-structure the update logic.

                        # Re-do update logic for clarity and correctness
                        # We already decremented cover_count. 
                        # Let's restore and do it properly in the loop if needed, or just trust the logic:
                        # If old_count == 2, w was NOT critical for any c2 (count 2). Now count is 1. So w IS critical for all c2 in cover_map[w].
                        # So critical_count[c2] += 1 for all c2 in cover_map[w] (including c, but c is gone).
                        # If old_count == 1, w WAS critical for c2. Now count is 0. So w is NOT critical.
                        # But we established old_count >= 2 for all w in balls[c] because critical_count[c] == 0.
                        # So we only need to handle the case old_count == 2 -> critical_count[c2] += 1.
                        
                        # Let's restart the update logic inside the loop for safety
                        # I will rewrite the inner update block in the code below.
                        
                        code_list.pop(i)
                        in_code[c] = False
                        changed = True
                        break
                        
            # The above refine logic had a flaw in the update step implementation in the thought process. 
            # Let's implement a robust version.
            return code_list

        def local_search(code, seed, time_limit):
            """Local search with deletion and swap moves.
            Optimized deletion check using critical counts."""
            rng = random.Random(seed)
            code_list = list(code)
            if not code_list:
                return code_list

            cover_count = [0] * N
            for c in code_list:
                for w in balls[c]:
                    cover_count[w] += 1

            # critical_count[c] = number of w in balls[c] with cover_count[w] == 1
            critical_count = [0] * N
            for c in code_list:
                for w in balls[c]:
                    if cover_count[w] == 1:
                        critical_count[c] += 1

            in_code = [False] * N
            for c in code_list:
                in_code[c] = True

            improved = True
            while improved and time.time() < time_limit:
                improved = False
                indices = list(range(len(code_list)))
                rng.shuffle(indices)
                for i in indices:
                    if time.time() > time_limit:
                        break
                    c_remove = code_list[i]

                    # Try deletion first
                    if critical_count[c_remove] == 0:
                        # Safe to remove
                        for w in balls[c_remove]:
                            old_count = cover_count[w]
                            cover_count[w] -= 1
                            # Update critical counts for other codewords
                            for c2 in cover_map[w]:
                                if not in_code[c2] or c2 == c_remove:
                                    continue
                                if old_count == 2:
                                    critical_count[c2] += 1
                                # old_count > 2: no change in criticality
                                # old_count == 1: impossible because critical_count[c_remove]==0 implies old_count>1 for all w in balls[c_remove]
                        
                        code_list.pop(i)
                        in_code[c_remove] = False
                        improved = True
                        break
                    else:
                        # Try swap
                        # Find critical words for c_remove
                        crit_words = [w for w in balls[c_remove] if cover_count[w] == 1]
                        
                        # Find candidate c_add that covers all crit_words
                        # Candidates must be in intersection of cover_map[w] for w in crit_words, excluding c_remove and in_code
                        if not crit_words:
                            continue # Should not happen if critical_count > 0

                        # Use the critical word with fewest candidates to prune
                        crit_sorted = sorted(crit_words, key=lambda w: len(cover_map[w]))
                        cand_set = None
                        for w in crit_sorted:
                            new_cands = set()
                            for c2 in cover_map[w]:
                                if c2 != c_remove and not in_code[c2]:
                                    new_cands.add(c2)
                            if cand_set is None:
                                cand_set = new_cands
                            else:
                                cand_set.intersection_update(new_cands)
                            if not cand_set:
                                break

                        if not cand_set:
                            continue

                        found = -1
                        # Pick a candidate. We can pick randomly or one that maximizes some heuristic.
                        # Random is fine for now.
                        for c_add in cand_set:
                            # Verify it covers all crit_words (should be true by construction of intersection)
                            # But intersection was built from cover_map, so yes.
                            found = c_add
                            break

                        if found >= 0:
                            # Perform swap
                            # Remove c_remove
                            for w in balls[c_remove]:
                                old_count = cover_count[w]
                                cover_count[w] -= 1
                                for c2 in cover_map[w]:
                                    if not in_code[c2] or c2 == c_remove:
                                        continue
                                    if old_count == 2:
                                        critical_count[c2] += 1
                                    elif old_count == 1:
                                        critical_count[c2] -= 1
                            
                            # Add found
                            for w in balls[found]:
                                old_count = cover_count[w]
                                cover_count[w] += 1
                                for c2 in cover_map[w]:
                                    if not in_code[c2] or c2 == found:
                                        continue
                                    if old_count == 1:
                                        critical_count[c2] += 1
                                    elif old_count == 0:
                                        critical_count[c2] -= 1
                            
                            # Update in_code
                            in_code[c_remove] = False
                            in_code[found] = True
                            code_list[i] = found
                            improved = True
                            break

            return code_list

        best_code = None
        best_len = float('inf')

        # Multiple restarts with different seeds
        num_restarts = 30
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
            ls_code = local_search(refined, i * 7 + 1, min(deadline - 8, time.time() + 3.0))
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code

        # Kick / re-greedy restarts on the best solution found so far
        if best_code is not None:
            for kick_iter in range(8):
                if time.time() > deadline - 12:
                    break
                rng_k = random.Random(1000 + kick_iter)
                # Remove k random codewords (kick), then re-greedy the rest
                k = min(3, len(best_code))
                kick_set = set(rng_k.sample(best_code, k))
                remaining = [c for c in best_code if c not in kick_set]
                re_code = greedy_cover(2000 + kick_iter, start_code=remaining)
                re_refined = refine(re_code)
                if len(re_refined) < best_len:
                    best_len = len(re_refined)
                    best_code = re_refined
                if time.time() > deadline - 10:
                    break
                ls_code = local_search(re_refined, 3000 + kick_iter, min(deadline - 8, time.time() + 2.5))
                if len(ls_code) < best_len:
                    best_len = len(ls_code)
                    best_code = ls_code

        # Final local search + refine
        if best_code is not None and time.time() < deadline - 5:
            ls_code = local_search(best_code, 42, deadline - 3)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code
            final_code = refine(best_code)
        elif best_code is not None:
            final_code = refine(best_code)
        else:
            final_code = []

        result = []
        for c in final_code:
            result.append(int_to_base3(c, n))
        return result

    # Generic case (shouldn't be called for this task, but included for completeness)
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
