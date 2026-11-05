import numpy as np
import itertools
import time
import random
import heapq

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

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
        # Precompute ball indices as numpy arrays for faster iteration in local search
        ball_indices = [np.array(b, dtype=np.int32) for b in balls]

        # Precompute gain matrix row-sums for greedy: gain[c] = number of
        # currently-uncovered words in ball c. We track it incrementally.
        # gain[c] = sum over w in ball c of (1 - covered[w])
        gain = [ball_size] * N
        # For quick update: when w becomes covered, decrement gain[c] for c in cover_map[w]

        def greedy_cover(seed, start_code=None):
            """Greedy set cover with heap-based candidate selection.
            If start_code is given, start from that partial cover and continue
            covering the remaining words (kick / re-greedy)."""
            rng = random.Random(seed)
            covered = [False] * N
            cnt = [ball_size] * N
            code = []
            uncovered_count = N

            # Build heap: (-cnt[i], i) for max-heap behavior
            heap = [(-ball_size, i) for i in range(N)]
            heapq.heapify(heap)

            if start_code:
                for c in start_code:
                    code.append(c)
                    for w in balls[c]:
                        if not covered[w]:
                            covered[w] = True
                            uncovered_count -= 1
                            for c2 in cover_map[w]:
                                cnt[c2] -= 1
                                if cnt[c2] > 0:
                                    heapq.heappush(heap, (-cnt[c2], c2))

            while uncovered_count > 0:
                # Pop valid candidates from heap
                while heap:
                    neg_val, i = heapq.heappop(heap)
                    if cnt[i] == -neg_val and not covered[i]:
                        break
                else:
                    break

                c = i
                code.append(c)

                ball_c = balls[c]
                for w in ball_c:
                    if not covered[w]:
                        covered[w] = True
                        uncovered_count -= 1
                        for c2 in cover_map[w]:
                            cnt[c2] -= 1
                            if cnt[c2] > 0:
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
            """Local search with deletion, 1-swap, and 2-swap moves.
            Uses precomputed ball_indices (numpy arrays) for fast iteration."""
            rng = random.Random(seed)
            code_list = list(code)
            if not code_list:
                return code_list

            cover_count = [0] * N
            for c in code_list:
                for w in ball_indices[c]:
                    cover_count[w] += 1

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

                    # Try deletion first: decrement and track uncovered
                    newly_uncovered = []
                    for w in ball_indices[c_remove]:
                        cover_count[w] -= 1
                        if cover_count[w] == 0:
                            newly_uncovered.append(w)

                    if not newly_uncovered:
                        code_list.pop(i)
                        in_code[c_remove] = False
                        improved = True
                        break

                    # Try 1-swap: find a single codeword covering all newly_uncovered
                    found = -1
                    if len(newly_uncovered) <= 6:
                        crit_sorted = sorted(newly_uncovered, key=lambda w: len(cover_map[w]))
                        cand_pool = None
                        for w in crit_sorted:
                            candidates_w = [c2 for c2 in cover_map[w] if not in_code[c2] and c2 != c_remove]
                            if cand_pool is None:
                                cand_pool = set(candidates_w)
                            else:
                                cand_pool.intersection_update(candidates_w)
                            if not cand_pool:
                                break
                        if cand_pool:
                            for c_add in cand_pool:
                                bs = ball_sets[c_add]
                                if all(w in bs for w in newly_uncovered):
                                    found = c_add
                                    break

                    if found >= 0:
                        code_list[i] = found
                        in_code[c_remove] = False
                        in_code[found] = True
                        for w in ball_indices[found]:
                            cover_count[w] += 1
                        improved = True
                        break

                    # Try 2-swap: remove c_remove and c2, add d1, d2
                    # Only if newly_uncovered is small
                    if len(newly_uncovered) <= 4:
                        # Find candidates for 2-swap
                        for j in range(len(code_list)):
                            if j == i or time.time() > time_limit:
                                continue
                            c2 = code_list[j]
                            # Tentatively remove both
                            temp_uncovered = list(newly_uncovered)
                            for w in ball_indices[c2]:
                                cover_count[w] -= 1
                                if cover_count[w] == 0 and w not in temp_uncovered:
                                    temp_uncovered.append(w)
                            
                            if len(temp_uncovered) > 4:
                                # Restore
                                for w in ball_indices[c2]:
                                    cover_count[w] += 1
                                continue

                            # Find pairs (d1, d2) that cover all temp_uncovered
                            # d1 covers some, d2 covers the rest
                            # Try to find d1 that covers at least 2 of temp_uncovered
                            found_pair = None
                            for d1 in cover_map[temp_uncovered[0]]:
                                if in_code[d1] and d1 != c_remove and d1 != c2:
                                    continue
                                if d1 == c_remove or d1 == c2:
                                    continue
                                if in_code[d1]:
                                    continue
                                # Check how many temp_uncovered d1 covers
                                d1_ball = ball_sets[d1]
                                covered_by_d1 = [w for w in temp_uncovered if w in d1_ball]
                                remaining = [w for w in temp_uncovered if w not in d1_ball]
                                
                                if not remaining:
                                    # d1 covers all, just a 1-swap case (should have been caught)
                                    continue
                                
                                if len(remaining) <= 2:
                                    # Find d2 that covers all remaining
                                    for d2 in cover_map[remaining[0]]:
                                        if in_code[d2] or d2 == c_remove or d2 == c2 or d2 == d1:
                                            continue
                                        d2_ball = ball_sets[d2]
                                        if all(w in d2_ball for w in remaining):
                                            found_pair = (d1, d2)
                                            break
                                    if found_pair:
                                        break
                            
                            if found_pair:
                                d1, d2 = found_pair
                                # Perform the 2-swap
                                code_list.pop(j)  # Remove c2 (adjust index if j > i)
                                if j < i:
                                    i -= 1
                                code_list.pop(i)  # Remove c_remove
                                in_code[c_remove] = False
                                in_code[c2] = False
                                
                                code_list.append(d1)
                                code_list.append(d2)
                                in_code[d1] = True
                                in_code[d2] = True
                                
                                for w in ball_indices[d1]:
                                    cover_count[w] += 1
                                for w in ball_indices[d2]:
                                    cover_count[w] += 1
                                improved = True
                                break
                            else:
                                # Restore
                                for w in ball_indices[c2]:
                                    cover_count[w] += 1
                        
                        if improved:
                            break

                    # Restore if no improvement
                    for w in ball_indices[c_remove]:
                        cover_count[w] += 1

            return code_list

        def run_greedy_local_search(seed):
            random.seed(seed)
            code = greedy_cover(seed)
            
            # Local Search
            code = refine(code)
            time_left = deadline - 5
            code = local_search(code, seed, time_left)
            
            # Final validity check
            cover_count = [0] * N
            for c in code:
                for w in balls[c]:
                    cover_count[w] += 1
            
            if min(cover_count) < 1:
                return None
                
            return code

        best_code = None
        best_size = float('inf')
        
        # Run multiple restarts
        num_restarts = 0
        while time.time() < deadline - 10:
            seed = num_restarts * 1000 + 42
            current_code = run_greedy_local_search(seed)
            num_restarts += 1
            
            if current_code is not None:
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
                
            # If we've done a few restarts, check time more strictly
            if num_restarts >= 5:
                if time.time() > deadline - 30:
                    break

        if best_code is None:
            # Fallback: Return a valid code from the first run if possible, or generate one
            best_code = run_greedy_local_search(0)
            if best_code is None:
                return []

        # Final Prune on the best code
        best_code = refine(best_code)
        
        # Return as list of lists
        words = [int_to_base3(i, n) for i in best_code]
        return words

    # Fallback for other cases
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    neighbors = []
    for i, w in enumerate(words):
        neigh = [i]
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                w_list[pos] = val
                idx = word_to_idx[tuple(w_list)]
                neigh.append(idx)
                w_list[pos] = orig_val
        neighbors.append(neigh)
    
    def get_counts(code_list):
        if not code_list:
            return np.zeros(N, dtype=np.int32)
        counts = np.zeros(N, dtype=np.int32)
        flat_neighbors = np.concatenate([neighbors[i] for i in code_list])
        np.add.at(counts, flat_neighbors, 1)
        return counts

    def prune_code(code_list):
        counts = get_counts(code_list)
        changed = True
        while changed:
            changed = False
            for i in range(len(code_list) - 1, -1, -1):
                c = code_list[i]
                if np.min(counts[neighbors[c]]) > 1:
                    code_list.pop(i)
                    for nb in neighbors[c]:
                        counts[nb] -= 1
                    changed = True
        return code_list

    def improve_by_swaps(code_list, max_iterations=100):
        counts = get_counts(code_list)
        code_set = set(code_list)
        improved = True
        iterations = 0
        
        neighbor_sets = [set(neigh) for neigh in neighbors]
        
        while improved and iterations < max_iterations and time.time() < deadline - 5:
            improved = False
            iterations += 1
            
            code_list = prune_code(code_list)
            code_set = set(code_list)
            counts = get_counts(code_list)
            
            random.shuffle(code_list)
            
            for c in code_list:
                if time.time() > deadline - 5:
                    break
                
                crit_points = []
                for nb in neighbors[c]:
                    if counts[nb] == 1:
                        crit_points.append(nb)
                        
                if not crit_points:
                    if np.min(counts[neighbors[c]]) > 1:
                        code_list.remove(c)
                        code_set.remove(c)
                        for nb in neighbors[c]:
                            counts[nb] -= 1
                        improved = True
                        break
                    continue
                
                p = crit_points[0]
                candidates = [x for x in neighbors[p] if x not in code_set]
                
                best_w = -1
                best_score = -1
                
                for w in candidates:
                    valid = True
                    for k in crit_points:
                        if k not in neighbor_sets[w]:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    score = 0
                    for nb in neighbors[w]:
                        if counts[nb] == 2:
                            score += 1
                            
                    if score > best_score:
                        best_score = score
                        best_w = w
                        
                if best_w != -1:
                    code_list.remove(c)
                    code_set.remove(c)
                    for nb in neighbors[c]:
                        counts[nb] -= 1
                        
                    code_list.append(best_w)
                    code_set.add(best_w)
                    for nb in neighbors[best_w]:
                        counts[nb] += 1
                        
                    improved = True
                    break
                    
        return code_list

    def run_greedy_local_search(seed):
        random.seed(seed)
        code = []
        uncovered = np.ones(N, dtype=bool)
        uncovered_count = N
        
        neighbor_array = np.zeros((N, 1 + n * (q - 1)), dtype=np.int32)
        for i, neigh in enumerate(neighbors):
            neighbor_array[i, :] = neigh
        
        while uncovered_count > 0 and time.time() < deadline - 20:
            gains = uncovered[neighbor_array].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            code.append(best_idx)
            
            new_uncovered = neighbors[best_idx]
            newly_covered = 0
            for nb in new_uncovered:
                if uncovered[nb]:
                    uncovered[nb] = False
                    newly_covered += 1
            uncovered_count -= newly_covered
            
        code = prune_code(code)
        code = improve_by_swaps(code)
        
        counts = get_counts(code)
        if np.min(counts) < 1:
            return None
            
        return code

    best_code = None
    best_size = float('inf')
    
    num_restarts = 0
    while time.time() < deadline - 10:
        seed = num_restarts * 1000 + 42
        current_code = run_greedy_local_search(seed)
        num_restarts += 1
        
        if current_code is not None:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
                
        if num_restarts >= 10:
             if time.time() > deadline - 30:
                 break

    if best_code is None:
        best_code = run_greedy_local_search(0)
        if best_code is None:
             return []

    best_code = prune_code(best_code)
    
    return [list(words[i]) for i in best_code]
