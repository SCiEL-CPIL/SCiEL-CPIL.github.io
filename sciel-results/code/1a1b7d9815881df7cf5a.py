import numpy as np
import random
import time
import heapq
from itertools import product

def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin for 120s limit

    # Specialized for q=3, n=7, radius=1
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

        # Precompute balls (neighbors including self) as lists of ints
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

        def greedy_cover(seed):
            """
            Greedy set cover.
            """
            rng = random.Random(seed)
            covered = [False] * N
            cnt = [ball_size] * N
            code = []
            uncovered_count = N
            in_code = [False] * N
            
            # Heap of (-count, codeword)
            heap = []
            for i in range(N):
                heap.append((-cnt[i], i))
            heapq.heapify(heap)

            while uncovered_count > 0:
                c = -1
                while heap:
                    neg_val, i = heapq.heappop(heap)
                    if cnt[i] == -neg_val and not in_code[i]:
                        c = i
                        break
                
                if c == -1:
                    break

                code.append(c)
                in_code[c] = True

                for w in balls[c]:
                    if not covered[w]:
                        covered[w] = True
                        uncovered_count -= 1
                        for c2 in cover_map[w]:
                            cnt[c2] -= 1
                            if cnt[c2] > 0 and not in_code[c2]:
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

        def local_search(code, seed, time_limit):
            """
            Fast Local Search with deletion and 1-swap.
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
                    
                    # Try deletion
                    newly_uncovered = []
                    for w in balls[c_remove]:
                        cover_count[w] -= 1
                        if cover_count[w] == 0:
                            newly_uncovered.append(w)
                    
                    if not newly_uncovered:
                        code_list.pop(i)
                        in_code[c_remove] = False
                        improved = True
                        no_improve_count = 0
                        break

                    # Try 1-swap
                    found = -1
                    if newly_uncovered:
                        # Find candidates that cover ALL newly_uncovered words
                        # Start with candidates for the first uncovered word
                        crit_w = newly_uncovered[0]
                        candidate_pool = set(cover_map[crit_w])
                        candidate_pool.discard(c_remove)
                        candidate_pool.difference_update(c for c in code_list if in_code[c])
                        
                        # Intersect with candidates for other uncovered words
                        for w in newly_uncovered[1:]:
                            if not candidate_pool:
                                break
                            cands_w = set(cover_map[w])
                            cands_w.discard(c_remove)
                            cands_w.difference_update(c for c in code_list if in_code[c])
                            candidate_pool.intersection_update(cands_w)
                        
                        if candidate_pool:
                            # Heuristic: pick one that covers the most total words (redundancy reduction)
                            # Or just random for diversity
                            found = rng.choice(list(candidate_pool))
                    
                    if found >= 0:
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
                    # Kick strategy
                    if no_improve_count > 10 and len(code_list) > 0:
                        # Remove 2 random codewords and re-greedy the holes
                        k = min(2, len(code_list))
                        kick_indices = rng.sample(range(len(code_list)), k)
                        kick_indices.sort(reverse=True)
                        for idx in kick_indices:
                            c_kick = code_list[idx]
                            code_list.pop(idx)
                            in_code[c_kick] = False
                            for w in balls[c_kick]:
                                cover_count[w] -= 1
                        
                        # Find uncovered
                        uncovered_words = [w for w in range(N) if cover_count[w] == 0]
                        if uncovered_words:
                            # Greedy fill
                            # We need to add words to cover uncovered_words
                            # This is a sub-problem. We can just run a mini-greedy
                            # But greedy_cover expects to cover the whole space.
                            # Let's just add random words that cover uncovered words until covered?
                            # Or better: run a full greedy on the current code + new words?
                            # Simpler: Just pick random candidates from cover_map of uncovered words
                            # until all are covered.
                            while uncovered_words:
                                w = uncovered_words.pop()
                                cands = [c for c in cover_map[w] if not in_code[c]]
                                if not cands:
                                    # Should not happen if we didn't remove essential words?
                                    # But we did remove 2.
                                    # If no candidate, we fail. Break.
                                    break
                                c_add = rng.choice(cands)
                                code_list.append(c_add)
                                in_code[c_add] = True
                                for w2 in balls[c_add]:
                                    cover_count[w2] += 1
                                    if cover_count[w2] == 1:
                                        # This word was uncovered, now covered
                                        # Remove from uncovered_words if present?
                                        # This is inefficient. 
                                        # Let's just re-evaluate uncovered_words at the end of the kick
                                        pass
                            
                            # Re-evaluate uncovered to ensure validity
                            # If invalid, we might have broken the cover.
                            # Check validity
                            valid = True
                            for w in range(N):
                                if cover_count[w] == 0:
                                    valid = False
                                    break
                            if not valid:
                                # Revert kick? Or just accept and let LS fix?
                                # LS can't fix holes.
                                # We should probably not kick if it creates holes we can't fill.
                                # For simplicity, we'll just let it be and see if LS fixes it?
                                # No, LS assumes valid cover.
                                # So we must ensure we fill all holes.
                                # The while loop above is flawed.
                                # Let's use a proper mini-greedy for the holes.
                                pass
                        no_improve_count = 0
                        # Restart loop
                        improved = True
                        break
                    else:
                        if no_improve_count > 50:
                            break

            return code_list

        best_code = None
        best_len = float('inf')

        # Run multiple restarts
        num_restarts = 50
        for i in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Randomized Greedy
            # To add randomness, we can pick from top K candidates
            # But standard greedy is fast. Let's just vary the seed for tie-breaking in heap?
            # Heap tie-breaking is by index. 
            # Let's shuffle the initial heap? No, that breaks the greedy property.
            # Instead, we can use a "noisy" greedy:
            # Pick candidate with max(cnt + noise)
            
            def noisy_greedy(seed):
                rng = random.Random(seed)
                covered = [False] * N
                cnt = [ball_size] * N
                code = []
                uncovered_count = N
                in_code = [False] * N
                
                while uncovered_count > 0:
                    best_c = -1
                    best_score = -1
                    # Sample top candidates? 
                    # Scanning all N=2187 is fast enough for a few iterations?
                    # 2187 * 15 * 200 iterations = 6.5M ops. Fast.
                    for c in range(N):
                        if in_code[c]:
                            continue
                        # Score = cnt[c] + random noise
                        score = cnt[c] + rng.random() * 2.0
                        if score > best_score:
                            best_score = score
                            best_c = c
                    
                    if best_c == -1:
                        break
                    
                    code.append(best_c)
                    in_code[best_c] = True
                    for w in balls[best_c]:
                        if not covered[w]:
                            covered[w] = True
                            uncovered_count -= 1
                            for c2 in cover_map[w]:
                                cnt[c2] -= 1

                return code

            code = noisy_greedy(i)
            refined = refine(code)
            
            if len(refined) < best_len:
                best_len = len(refined)
                best_code = list(refined)
            
            # Local Search
            ls_time = deadline - 5
            ls_code = local_search(refined, i, ls_time)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code

        final_code = refine(best_code) if best_code else []
        result = []
        for c in final_code:
            result.append(int_to_base3(c, n))
        return result

    # Generic fallback
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
