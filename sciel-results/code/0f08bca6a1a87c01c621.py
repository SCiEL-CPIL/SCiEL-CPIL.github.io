import itertools
import numpy as np
import time
import random

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if not (q == 3 and n == 8 and radius == 1):
        if q == 3 and n == 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(word) for word in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 115.0

    total_words = q ** n

    # Precompute digits for all words
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)

    # Precompute neighbor lists
    neighbors = [None] * total_words
    for i in range(total_words):
        d = digits_of[i]
        neigh = [i]
        for pos in range(n):
            orig_val = d[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_d = list(d)
                new_d[pos] = new_val
                idx = 0
                for j in range(n):
                    idx = idx * q + new_d[j]
                neigh.append(idx)
        neighbors[i] = np.array(neigh, dtype=np.int32)

    # --- Phase 1: Fast Greedy Initialization ---
    cov_count = np.zeros(total_words, dtype=np.int32)
    is_code = np.zeros(total_words, dtype=bool)
    code_list = []

    potential = np.full(total_words, len(neighbors[0]), dtype=np.int32)
    uncovered_remaining = total_words

    while uncovered_remaining > 0 and time.time() < deadline - 30.0:
        c = np.argmax(potential)
        if potential[c] == 0:
            break
        code_list.append(c)
        is_code[c] = True
        potential[c] = 0
        neigh_c = neighbors[c]
        for x in neigh_c:
            if cov_count[x] == 0:
                cov_count[x] = 1
                uncovered_remaining -= 1
                potential[neighbors[x]] -= 1
            else:
                cov_count[x] += 1

    current_code = code_list[:]
    current_size = len(current_code)
    best_code = current_code[:]
    best_size = current_size

    # --- Phase 2: Iterated Local Search ---
    current_cov_count = cov_count.copy()
    current_is_code = is_code.copy()
    current_code_set = set(current_code)

    random.seed(42)
    stuck_counter = 0
    swap_attempts = 0

    while time.time() < deadline - 5.0:
        # 1. Try to remove a word
        if current_size == 0:
            break
        idx = random.randint(0, current_size - 1)
        c = current_code[idx]

        if np.all(current_cov_count[neighbors[c]] > 1):
            current_cov_count[neighbors[c]] -= 1
            current_is_code[c] = False
            current_code_set.remove(c)
            current_code.pop(idx)
            current_size -= 1
            stuck_counter = 0
            swap_attempts = 0

            if current_size < best_size:
                best_size = current_size
                best_code = current_code[:]
        else:
            stuck_counter += 1

        # 2. If stuck, try multiple swap strategies
        if stuck_counter > 3 and time.time() < deadline - 10.0 and current_size > 0:
            # Strategy A: 1-opt swap - find c2 that covers all risky points of c1
            success = False
            for attempt in range(3):
                c1_idx = random.randint(0, current_size - 1)
                c1 = current_code[c1_idx]

                # Find risky points: x in neighbors[c1] where cov_count[x] == 1
                risky_points = [x for x in neighbors[c1] if current_cov_count[x] == 1]

                if not risky_points:
                    # Should have been caught in step 1, but just in case
                    current_cov_count[neighbors[c1]] -= 1
                    current_is_code[c1] = False
                    current_code_set.remove(c1)
                    current_code.pop(c1_idx)
                    current_size -= 1
                    stuck_counter = 0
                    swap_attempts = 0
                    if current_size < best_size:
                        best_size = current_size
                        best_code = current_code[:]
                    success = True
                    break
                else:
                    # Find c2 not in code such that c2 covers all risky points.
                    candidates = set(neighbors[risky_points[0]].tolist())
                    candidates.discard(c1)

                    for x in risky_points[1:]:
                        candidates &= set(neighbors[x].tolist())
                        if not candidates:
                            break

                    valid_candidates = [c2 for c2 in candidates if c2 not in current_code_set]

                    if valid_candidates:
                        c2 = random.choice(valid_candidates)

                        current_cov_count[neighbors[c1]] -= 1
                        current_is_code[c1] = False
                        current_code_set.remove(c1)
                        current_code[c1_idx] = c2

                        current_cov_count[neighbors[c2]] += 1
                        current_is_code[c2] = True
                        current_code_set.add(c2)

                        stuck_counter = 0
                        swap_attempts = 0
                        success = True
                        break

            if success:
                continue

            # Strategy B: 2-opt - remove 2, add 1
            if not success and current_size >= 2 and stuck_counter > 8:
                for attempt in range(3):
                    c1_idx = random.randint(0, current_size - 1)
                    c2_idx = random.randint(0, current_size - 1)
                    if c2_idx == c1_idx:
                        c2_idx = (c2_idx + 1) % current_size
                    c1 = current_code[c1_idx]
                    c2 = current_code[c2_idx]
                    
                    # Collect risky points for c1 and c2
                    risky_c1 = [x for x in neighbors[c1] if current_cov_count[x] == 1]
                    risky_c2 = [x for x in neighbors[c2] if current_cov_count[x] == 1]
                    all_risky = list(set(risky_c1 + risky_c2))
                    
                    if all_risky:
                        # Find c3 that covers all all_risky
                        candidates_2opt = set(neighbors[all_risky[0]].tolist())
                        candidates_2opt.discard(c1)
                        candidates_2opt.discard(c2)
                        
                        for x in all_risky[1:]:
                            candidates_2opt &= set(neighbors[x].tolist())
                            if not candidates_2opt:
                                break
                        
                        valid_candidates_2opt = [c3 for c3 in candidates_2opt if c3 not in current_code_set]
                        
                        if valid_candidates_2opt:
                            c3 = random.choice(valid_candidates_2opt)
                            
                            # Update counts
                            current_cov_count[neighbors[c1]] -= 1
                            current_cov_count[neighbors[c2]] -= 1
                            current_cov_count[neighbors[c3]] += 1
                            
                            # Update sets
                            current_is_code[c1] = False
                            current_is_code[c2] = False
                            current_is_code[c3] = True
                            current_code_set.remove(c1)
                            current_code_set.remove(c2)
                            current_code_set.add(c3)
                            
                            # Update list
                            indices_to_remove = sorted([c1_idx, c2_idx], reverse=True)
                            for idx_rm in indices_to_remove:
                                current_code.pop(idx_rm)
                            current_code.append(c3)
                            current_size -= 1
                            stuck_counter = 0
                            swap_attempts = 0
                            
                            if current_size < best_size:
                                best_size = current_size
                                best_code = current_code[:]
                            success = True
                            break

            # Strategy C: Add then remove - temporarily add a word, then try to remove two
            if not success and current_size > 0 and stuck_counter > 15:
                # Pick a random uncovered-heavy area and add a word there
                # Find the word with highest potential (most uncovered neighbors)
                temp_potential = np.zeros(total_words, dtype=np.int32)
                for i in range(current_size):
                    ci = current_code[i]
                    temp_potential[neighbors[ci]] += 1
                
                # Words with low coverage are good candidates to add
                low_cov = np.where(temp_potential <= 1)[0]
                if len(low_cov) > 0:
                    c_add = low_cov[random.randint(0, len(low_cov) - 1)]
                    if c_add not in current_code_set:
                        # Add it
                        current_cov_count[neighbors[c_add]] += 1
                        current_is_code[c_add] = True
                        current_code_set.add(c_add)
                        current_code.append(c_add)
                        current_size += 1
                        
                        # Now try to remove two words
                        removed_count = 0
                        for _ in range(2):
                            if current_size <= 1:
                                break
                            rm_idx = random.randint(0, current_size - 1)
                            rm_c = current_code[rm_idx]
                            if rm_c == c_add:
                                continue
                            if np.all(current_cov_count[neighbors[rm_c]] > 1):
                                current_cov_count[neighbors[rm_c]] -= 1
                                current_is_code[rm_c] = False
                                current_code_set.remove(rm_c)
                                current_code.pop(rm_idx)
                                current_size -= 1
                                removed_count += 1
                        
                        if removed_count >= 1:
                            stuck_counter = 0
                            swap_attempts = 0
                            if current_size < best_size:
                                best_size = current_size
                                best_code = current_code[:]
                        else:
                            # Revert: remove the added word
                            current_cov_count[neighbors[c_add]] -= 1
                            current_is_code[c_add] = False
                            current_code_set.remove(c_add)
                            current_code.pop()
                            current_size -= 1

            swap_attempts += 1
            if swap_attempts > 50:
                swap_attempts = 0

    # --- Phase 3: Polish with multiple passes ---
    final_code = best_code[:]
    final_cov_count = np.zeros(total_words, dtype=np.int32)
    for idx in final_code:
        final_cov_count[neighbors[idx]] += 1

    # Multiple polish passes
    for pass_num in range(5):
        if time.time() >= deadline - 2.0:
            break
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Randomize order for better exploration
            order = list(range(len(final_code)))
            random.shuffle(order)
            for i in order:
                c = final_code[i]
                if np.all(final_cov_count[neighbors[c]] > 1):
                    final_cov_count[neighbors[c]] -= 1
                    final_code.pop(i)
                    improved = True

    # Final verification and repair if needed
    if not np.all(final_cov_count > 0):
        # Find uncovered words and add minimal repairs
        uncovered = np.where(final_cov_count == 0)[0]
        if len(uncovered) > 0:
            # Greedily add words to cover uncovered
            temp_cov = final_cov_count.copy()
            temp_code = final_code[:]
            while len(uncovered) > 0 and time.time() < deadline - 1.0:
                # Find word that covers most uncovered
                best_c = -1
                best_count = 0
                for c in range(total_words):
                    if c in set(temp_code):
                        continue
                    count = np.sum(temp_cov[neighbors[c]] == 0)
                    if count > best_count:
                        best_count = count
                        best_c = c
                if best_c == -1 or best_count == 0:
                    break
                temp_cov[neighbors[best_c]] += 1
                temp_code.append(best_c)
                uncovered = np.where(temp_cov == 0)[0]
            final_code = temp_code
            final_cov_count = temp_cov

    return [list(digits_of[i]) for i in final_code]
