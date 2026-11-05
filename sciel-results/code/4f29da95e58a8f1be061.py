import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before the 150s kill

    # Fast path for the canonical instance
    if q == 3 and n == 8 and radius == 1:
        return _solve_3_8_1(deadline)

    # Fallback for small instances or non-canonical parameters
    total_words = q ** n
    if total_words <= 81:
        # Brute force exhaustive search via greedy + removal
        digits_of = []
        for i in range(total_words):
            d = []
            x = i
            for _ in range(n):
                d.append(x % q)
                x //= q
            d.reverse()
            digits_of.append(tuple(d))
        
        neighbors = []
        for i in range(total_words):
            d = list(digits_of[i])
            neigh = [i]
            for pos in range(n):
                orig = d[pos]
                for nv in range(q):
                    if nv == orig:
                        continue
                    nd = d[:]
                    nd[pos] = nv
                    idx = 0
                    for j in range(n):
                        idx = idx * q + nd[j]
                    neigh.append(idx)
            neighbors.append(neigh)

        cov_count = np.zeros(total_words, dtype=np.int32)
        code = []
        
        # Greedy init
        potential = np.full(total_words, len(neighbors[0]), dtype=np.int32)
        uncovered = total_words
        while uncovered > 0:
            c = int(np.argmax(potential))
            if potential[c] == 0:
                break
            code.append(c)
            potential[c] = 0
            for x in neighbors[c]:
                if cov_count[x] == 0:
                    cov_count[x] = 1
                    uncovered -= 1
                    potential[neighbors[x]] -= 1
                else:
                    cov_count[x] += 1
        
        # Removal polish
        changed = True
        while changed:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                if np.all(cov_count[neighbors[c]] > 1):
                    cov_count[neighbors[c]] -= 1
                    code.pop(i)
                    changed = True
        
        return [list(digits_of[i]) for i in code]

    # General fallback: return full space (valid but large)
    return [list(word) for word in itertools.product(range(q), repeat=n)]


def _solve_3_8_1(deadline):
    q, n, radius = 3, 8, 1
    total_words = q ** n
    k = 1 + n * (q - 1)  # 17

    # Precompute digits
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)

    # Precompute neighbors as numpy arrays
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

    # Precompute neighbor sets for fast intersection/membership in local search
    neighbor_sets = [set(nbrs.tolist()) for nbrs in neighbors]

    # --- Phase 1: Greedy Initialization ---
    cov_count = np.zeros(total_words, dtype=np.int32)
    is_code = np.zeros(total_words, dtype=bool)
    code_list = []

    potential = np.full(total_words, k, dtype=np.int32)
    uncovered_remaining = total_words

    while uncovered_remaining > 0 and time.time() < deadline - 20.0:
        c = int(np.argmax(potential))
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

    # --- Phase 2: Iterated Local Search with Smart Swaps ---
    # This phase combines the removal-first logic of the current program
    # with the "risky point intersection" swap strategy from the inspiration.
    
    current_cov_count = cov_count.copy()
    current_is_code = is_code.copy()
    current_code_set = set(current_code)
    
    random.seed(42)
    stuck_counter = 0
    no_improve_count = 0
    max_stuck_before_restart = 200

    while time.time() < deadline - 5.0:
        # 1. Try to remove a word
        if current_size == 0:
            break
        
        # Randomly pick a codeword to try removing
        idx = random.randint(0, current_size - 1)
        c = current_code[idx]

        # Check if removable
        if np.all(current_cov_count[neighbors[c]] > 1):
            current_cov_count[neighbors[c]] -= 1
            current_is_code[c] = False
            current_code_set.remove(c)
            current_code.pop(idx)
            current_size -= 1
            stuck_counter = 0
            no_improve_count = 0

            if current_size < best_size:
                best_size = current_size
                best_code = current_code[:]
        else:
            stuck_counter += 1
            no_improve_count += 1

        # 2. If stuck, try a smart swap
        if stuck_counter > 3 and time.time() < deadline - 10.0 and current_size > 0:
            c1_idx = random.randint(0, current_size - 1)
            c1 = current_code[c1_idx]

            # Find risky points: x in neighbors[c1] where cov_count[x] == 1
            # These are the points that would become uncovered if c1 is removed.
            risky_points = [x for x in neighbors[c1] if current_cov_count[x] == 1]

            if not risky_points:
                # Should have been removed, but just in case
                current_cov_count[neighbors[c1]] -= 1
                current_is_code[c1] = False
                current_code_set.remove(c1)
                current_code.pop(c1_idx)
                current_size -= 1
                stuck_counter = 0
                no_improve_count = 0
                if current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
            else:
                # Find c2 not in code such that c2 covers ALL risky points.
                # This ensures that swapping c1 -> c2 keeps all previously covered points covered.
                # c2 must be in the intersection of neighbors[risky_points].
                
                # Start with neighbors of the first risky point
                candidates = set(neighbor_sets[risky_points[0]])
                candidates.discard(c1)
                
                # Intersect with neighbors of other risky points
                for x in risky_points[1:]:
                    candidates &= neighbor_sets[x]
                    if not candidates:
                        break

                # Filter out codewords already in the code
                valid_candidates = [c2 for c2 in candidates if c2 not in current_code_set]

                if valid_candidates:
                    # Pick a candidate. We can pick randomly or based on other criteria.
                    # Picking one that covers the most *other* points might be good, but random is fine.
                    c2 = random.choice(valid_candidates)

                    # Perform swap
                    current_cov_count[neighbors[c1]] -= 1
                    current_is_code[c1] = False
                    current_code_set.remove(c1)
                    current_code[c1_idx] = c2

                    current_cov_count[neighbors[c2]] += 1
                    current_is_code[c2] = True
                    current_code_set.add(c2)

                    stuck_counter = 0
                    # Note: Size doesn't change, so no_improve_count is not reset here
                    # unless we consider a successful swap as "progress" towards escaping a local min.
                    # Let's reset stuck_counter to avoid immediate re-swap, but keep no_improve_count.
                else:
                    # No valid swap found. This is a dead end for this c1.
                    # We could try a different c1, but for now we just continue.
                    pass

        # 3. Restart logic
        # If we've been stuck for a long time without improvement, restart from best
        if no_improve_count > max_stuck_before_restart:
            # Restore best code
            current_code = best_code[:]
            current_size = len(current_code)
            current_code_set = set(current_code)
            current_cov_count = np.zeros(total_words, dtype=np.int32)
            for idx in current_code:
                current_cov_count[neighbors[idx]] += 1
            current_is_code = np.zeros(total_words, dtype=bool)
            for idx in current_code:
                current_is_code[idx] = True
            
            no_improve_count = 0
            stuck_counter = 0
            max_stuck_before_restart = int(max_stuck_before_restart * 0.8)
            if max_stuck_before_restart < 50:
                max_stuck_before_restart = 50

    # --- Phase 3: Final Polish ---
    final_code = best_code[:]
    final_cov_count = np.zeros(total_words, dtype=np.int32)
    for idx in final_code:
        final_cov_count[neighbors[idx]] += 1

    # Deterministic removal pass
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i in range(len(final_code) - 1, -1, -1):
            c = final_code[i]
            if np.all(final_cov_count[neighbors[c]] > 1):
                final_cov_count[neighbors[c]] -= 1
                final_code.pop(i)
                improved = True

    # Verify
    if not np.all(final_cov_count > 0):
        return [list(digits_of[i]) for i in best_code]

    return [list(digits_of[i]) for i in final_code]
