import numpy as np
import random
import time
import heapq
from itertools import product

def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + 118.0  # Safety margin for 120s limit

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

        # Precompute ball sets for fast membership and intersection
        ball_sets = [set(b) for b in balls]

        def greedy_cover(seed, start_code=None, exclude_set=None):
            """
            Greedy set cover with heap-based candidate selection.
            If start_code is given, start from that partial cover.
            If exclude_set is given, those codewords cannot be added.
            """
            rng = random.Random(seed)
            covered = [False] * N
            cnt = [ball_size] * N
            code = []
            uncovered_count = N
            
            in_code = [False] * N
            if start_code:
                for c in start_code:
                    code.append(c)
                    in_code[c] = True
                    for w in balls[c]:
                        if not covered[w]:
                            covered[w] = True
                            uncovered_count -= 1
                            for c2 in cover_map[w]:
                                cnt[c2] -= 1

            # Initialize heap with all candidates not in code and not excluded
            heap = []
            if exclude_set is None:
                exclude_set = set()
                
            for i in range(N):
                if not in_code[i] and i not in exclude_set:
                    heap.append((-cnt[i], i))
            heapq.heapify(heap)

            while uncovered_count > 0:
                # Pop valid candidates from heap
                # A candidate is valid if its cnt matches the heap entry and it's not in code/excluded
                c = -1
                while heap:
                    neg_val, i = heapq.heappop(heap)
                    if cnt[i] == -neg_val and not in_code[i] and i not in exclude_set:
                        c = i
                        break
                
                if c == -1:
                    # No valid candidates left? This shouldn't happen if cover is possible
                    # but if exclude_set blocks all, we might fail. 
                    # In kick strategy, we might need to relax exclude_set or accept failure. 
                    break

                code.append(c)
                in_code[c] = True

                ball_c = balls[c]
                for w in ball_c:
                    if not covered[w]:
                        covered[w] = True
                        uncovered_count -= 1
                        for c2 in cover_map[w]:
                            cnt[c2] -= 1
                            if cnt[c2] > 0 and not in_code[c2] and c2 not in exclude_set:
                                heapq.heappush(heap, (-cnt[c2], c2))

            return code

        def refine(code):
            """Remove redundant codewords."""
            if not code:
                return code

            cover_count = [0] * N
            for c in code:
                for w in balls[c]:
                    cover_count[w] += 1

            code_list = list(code)
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

        def peeling(code):
            """
            Peeling algorithm based on Lovasz Local Lemma intuition.
            Iteratively remove codewords if they are not "critical" for any word in their ball.
            A codeword c is critical for word w if removing c makes w uncovered (cover_count[w] == 1).
            If c is not critical for any w in balls[c], it is safe to remove.
            We prioritize removing codewords that are "least critical" or just random if safe.
            This is effectively a more aggressive form of refine, but can be run repeatedly
            or combined with swaps. Here we just use it as a fast refinement step.
            """
            code_list = list(code)
            if not code_list:
                return code_list

            cover_count = [0] * N
            for c in code_list:
                for w in balls[c]:
                    cover_count[w] += 1

            in_code = [False] * N
            for c in code_list:
                in_code[c] = True

            changed = True
            while changed:
                changed = False
                # Find all safe codewords
                safe_indices = []
                for i, c in enumerate(code_list):
                    is_safe = True
                    for w in balls[c]:
                        if cover_count[w] == 1:
                            is_safe = False
                            break
                    if is_safe:
                        safe_indices.append(i)
                
                if safe_indices:
                    # Remove one safe codeword (randomly to break symmetry)
                    idx_to_remove = random.choice(safe_indices)
                    c = code_list[idx_to_remove]
                    for w in balls[c]:
                        cover_count[w] -= 1
                    code_list.pop(idx_to_remove)
                    in_code[c] = False
                    changed = True
            
            return code_list

        def local_search(code, seed, time_limit):
            """
            Local search with deletion and 1-swap moves.
            Uses critical word heuristic for swaps.
            """
            rng = random.Random(seed)
            code_list = list(code)
            if not code_list:
                return code_list

            cover_count = [0] * N
            for c in code_list:
                for w in balls[c]:
                    cover_count[w] += 1

            in_code = [False] * N
            for c in code_list:
                in_code[c] = True

            improved = True
            no_improve_count = 0
            while improved and time.time() < time_limit:
                improved = False
                indices = list(range(len(code_list)))
                rng.shuffle(indices)
                for i in indices:
                    if time.time() > time_limit:
                        break
                    c_remove = code_list[i]
                    
                    # Try deletion first
                    newly_uncovered = []
                    for w in balls[c_remove]:
                        cover_count[w] -= 1
                        if cover_count[w] == 0:
                            newly_uncovered.append(w)
                    
                    if not newly_uncovered:
                        # Safe to delete
                        code_list.pop(i)
                        in_code[c_remove] = False
                        improved = True
                        no_improve_count = 0
                        break

                    # Try 1-swap: find a replacement that covers all newly_uncovered
                    found = -1
                    if newly_uncovered:
                        # Heuristic: pick the critical word (fewest coverers)
                        # Sort uncovered words by number of available coverers
                        # We need coverers that are NOT in code and NOT c_remove
                        crit_w = None
                        min_cand_count = float('inf')
                        crit_cands = None
                        
                        for w in newly_uncovered:
                            # Count available coverers
                            cands = [c2 for c2 in cover_map[w] if not in_code[c2] and c2 != c_remove]
                            if len(cands) < min_cand_count:
                                min_cand_count = len(cands)
                                crit_w = w
                                crit_cands = cands
                                if min_cand_count == 0:
                                    break
                        
                        if crit_w is not None and crit_cands:
                            # Intersect candidates for all uncovered words
                            # Start with candidates for the critical word
                            candidate_pool = set(crit_cands)
                            
                            for w in newly_uncovered:
                                if w == crit_w:
                                    continue
                                cands_w = set(c for c in cover_map[w] if not in_code[c] and c != c_remove)
                                candidate_pool.intersection_update(cands_w)
                                if not candidate_pool:
                                    break
                            
                            if candidate_pool:
                                # Pick a candidate. 
                                # Heuristic: pick one that covers the most *other* uncovered words? 
                                # Or just random?
                                # Let's pick one that minimizes "waste" or just random from pool.
                                # Random is fine for diversity.
                                found = rng.choice(list(candidate_pool))
                    
                    if found >= 0:
                        # Perform swap
                        code_list[i] = found
                        in_code[c_remove] = False
                        in_code[found] = True
                        for w in balls[found]:
                            cover_count[w] += 1
                        improved = True
                        no_improve_count = 0
                        break
                    else:
                        # Restore cover_count
                        for w in balls[c_remove]:
                            cover_count[w] += 1

                if not improved:
                    no_improve_count += 1
                    # Kick strategy: if stuck for a while, remove a random codeword and re-greedy
                    if no_improve_count > 5 and len(code_list) > 0:
                        # Pick a random codeword to remove
                        kick_idx = rng.randrange(len(code_list))
                        kick_c = code_list[kick_idx]
                        
                        # Remove it
                        code_list.pop(kick_idx)
                        in_code[kick_c] = False
                        for w in balls[kick_c]:
                            cover_count[w] -= 1
                        
                        # Find uncovered words
                        uncovered_words = [w for w in range(N) if cover_count[w] == 0]
                        
                        if uncovered_words:
                            # Re-greedy to cover the remaining holes
                            # We pass the current code as start_code
                            # We exclude nothing, but we want to add new words
                            # The greedy function will add words to cover the uncovered parts
                            # Note: greedy_cover expects a list of codewords to start with.
                            # It will add to it.
                            # We need to be careful: greedy_cover returns the *entire* code.
                            # So we pass code_list.
                            new_code = greedy_cover(rng.randint(0, 10000), start_code=code_list)
                            
                            # Refine the new code
                            new_code = refine(new_code)
                            
                            # Update state
                            code_list = new_code
                            in_code = [False] * N
                            for c in code_list:
                                in_code[c] = True
                            
                            cover_count = [0] * N
                            for c in code_list:
                                for w in balls[c]:
                                    cover_count[w] += 1
                            
                            no_improve_count = 0
                            # Restart the while loop effectively by setting improved=True to continue
                            # But we need to break out of the for loop
                            break
                    else:
                        # If we've been stuck for a long time, give up on this local search
                        if no_improve_count > 20:
                            break

            return code_list

        best_code = None
        best_len = float('inf')

        # Phase 1: Multiple greedy restarts
        # We will run a few greedy restarts to find a good starting point
        num_greedy = 10
        for i in range(num_greedy):
            if time.time() > deadline - 60: # Reserve time for LS
                break
            
            code = greedy_cover(i)
            # Apply peeling for a more aggressive reduction
            peeled_code = peeling(code)
            refined = refine(peeled_code)
            if len(refined) < best_len:
                best_len = len(refined)
                best_code = refined

        # Phase 2: Local Search on the best code
        if best_code is not None and time.time() < deadline - 10:
            # Run local search for a good amount of time
            ls_time_limit = deadline - 5
            ls_code = local_search(best_code, 42, ls_time_limit)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code

        final_code = refine(best_code) if best_code else []

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
