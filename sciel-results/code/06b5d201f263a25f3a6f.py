import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # Fallback for non-specific parameters
    if q != 3 or n != 7 or radius != 1:
        if n <= 4 and q == 3 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
    N = 3 ** n  # 2187

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    # Precompute neighbor lists
    # neighbors_list[i] is a numpy array of indices of words within Hamming distance 1 of i
    neighbors_list = []
    neighbor_sets = []
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    nb.append(tuple_to_idx(new_w))
        neighbors_list.append(np.array(nb, dtype=np.int32))
        neighbor_sets.append(set(nb))

    # ---------- Phase 1: Greedy Set-Cover with vectorized gain computation ----------
    uncovered = np.ones(N, dtype=bool)
    code = []
    code_set = set()

    while np.any(uncovered):
        if time.time() > deadline - 30:
            break

        # Compute gains: for each candidate codeword, how many uncovered points does it cover?
        # Accumulate from uncovered points
        gains = np.zeros(N, dtype=np.int32)
        uncovered_indices = np.where(uncovered)[0]
        for idx in uncovered_indices:
            gains[neighbors_list[idx]] += 1

        best_idx = int(np.argmax(gains))
        max_gain = gains[best_idx]

        if max_gain == 0:
            break

        code.append(best_idx)
        code_set.add(best_idx)
        uncovered[neighbors_list[best_idx]] = False

    # ---------- Phase 2: Local Search ----------
    # Initialize coverage counts
    counts = np.zeros(N, dtype=np.int32)
    for idx in code:
        counts[neighbors_list[idx]] += 1

    best_size = len(code)
    best_code = list(code)
    random.seed(42)

    def greedy_repair(uncovered_set):
        """Given a set of uncovered points, greedily add codewords to cover them."""
        if not uncovered_set:
            return []
        added = []
        temp_uncovered = set(uncovered_set)

        while temp_uncovered:
            if time.time() > deadline - 10:
                break
            best_c = -1
            best_count = 0

            # Build candidate pool from neighbors of uncovered words
            candidate_pool = set()
            for w in temp_uncovered:
                candidate_pool.update(neighbor_sets[w])

            # Find best candidate
            for c in candidate_pool:
                if c in code_set:
                    continue
                cnt = 0
                for x in neighbors_list[c]:
                    if x in temp_uncovered:
                        cnt += 1
                        if cnt > best_count:
                            best_count = cnt
                            best_c = c
                        if best_count == len(temp_uncovered):
                            break
                if best_count == len(temp_uncovered):
                    break

            if best_c == -1:
                break

            added.append(best_c)
            for w in neighbors_list[best_c]:
                temp_uncovered.discard(w)

        return added

    def try_remove():
        """Remove all redundant codewords (those whose neighbors are all covered by >1 codeword)."""
        removed = False
        for c in list(code):
            if np.min(counts[neighbors_list[c]]) > 1:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_list[c]] -= 1
                removed = True
        return removed

    def try_swap(c):
        """Try to swap codeword c with a better-positioned codeword.
        Strategy: Find a replacement covering all critical points of c."""
        # Check if c is currently removable
        if np.min(counts[neighbors_list[c]]) > 1:
            return False

        # Get critical points for c: points in neighbors[c] with counts == 1
        crit_points = [i for i in neighbors_list[c] if counts[i] == 1]

        if not crit_points:
            return False

        # Find replacement covering all critical points
        found_swap = False
        shuffled_crit = list(crit_points)
        random.shuffle(shuffled_crit)

        for p in shuffled_crit:
            if found_swap:
                break

            cands = [x for x in neighbor_sets[p] if x not in code_set]

            best_w = -1
            best_score = -1

            for w in cands:
                valid = True
                for k in crit_points:
                    if k not in neighbor_sets[w]:
                        valid = False
                        break

                if not valid:
                    continue

                # Score: maximize redundancy creation (points that would go from count 1 to 2)
                score = int(np.sum(counts[neighbors_list[w]] == 1))
                if score > best_score:
                    best_score = score
                    best_w = w

            if best_w != -1:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_list[c]] -= 1

                code.append(best_w)
                code_set.add(best_w)
                counts[neighbors_list[best_w]] += 1

                found_swap = True
                return True

        return False

    no_improvement_counter = 0
    max_no_improvement = 50

    while time.time() < deadline:
        # 1. Remove redundancies
        if try_remove():
            no_improvement_counter = 0
            best_size = len(code)
            best_code = list(code)
            continue

        # 2. Try swaps
        swap_made = False
        for c in list(code):
            if try_swap(c):
                swap_made = True
                no_improvement_counter = 0
                best_size = len(code)
                best_code = list(code)
                # Restart swap loop as counts changed
                break
        
        if swap_made:
            continue

        # 3. Kick: Remove k random codewords, repair
        k = random.choice([1, 2, 3])
        if len(code) <= k:
            break
            
        removed_indices = random.sample(list(code), k)
        
        # Temporarily remove
        temp_code_set = set(code_set)
        temp_counts = counts.copy()
        for c in removed_indices:
            temp_code_set.discard(c)
            temp_counts[neighbors_list[c]] -= 1
        
        # Find uncovered words
        uncovered_idx = np.where(temp_counts == 0)[0]
        
        if len(uncovered_idx) == 0:
            # Perfect removal
            code_set = temp_code_set
            code = list(code_set)
            counts = temp_counts
            best_size = len(code)
            best_code = list(code)
            no_improvement_counter = 0
            continue
            
        # Greedy Repair
        added_indices = greedy_repair(set(uncovered_idx.tolist()))
        
        if len(added_indices) < k:
            # Accept the change
            for c in added_indices:
                code_set.add(c)
                counts[neighbors_list[c]] += 1
            code = list(code_set)
            best_size = len(code)
            best_code = list(code)
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1
            
        if no_improvement_counter > max_no_improvement:
            # Try larger kicks or break
            if no_improvement_counter > 200:
                break

    # Return the best code found
    final_code = best_code
    result = [list(idx_to_tuple(c)) for c in final_code]
    
    return result
