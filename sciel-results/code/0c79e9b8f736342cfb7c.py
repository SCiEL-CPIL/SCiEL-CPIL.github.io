import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit

    # Fallback for non-standard parameters
    if q != 3 or n != 7 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            all_words = list(itertools.product(range(q), repeat=n))
            return all_words

    total_words = q ** n  # 2187

    # Precompute all words
    all_words = []
    for i in range(total_words):
        w = [0] * n
        idx = i
        for pos in range(n - 1, -1, -1):
            w[pos] = idx % q
            idx //= q
        all_words.append(w)

    # Precompute spheres as lists of indices
    spheres = []
    for i in range(total_words):
        w = all_words[i]
        s = [i]
        for pos in range(n):
            for delta in [-1, 1]:
                new_val = (w[pos] + delta) % q
                if new_val != w[pos]:
                    nw = w.copy()
                    nw[pos] = new_val
                    j = 0
                    for d in nw:
                        j = j * q + d
                    s.append(j)
        spheres.append(s)

    # Convert spheres to numpy arrays for fast operations
    sphere_arrays = [np.array(s, dtype=np.int32) for s in spheres]

    # Precompute sphere bitmasks as Python integers for fast intersection
    sphere_bits = [0] * total_words
    for i in range(total_words):
        mask = 0
        for j in spheres[i]:
            mask |= (1 << j)
        sphere_bits[i] = mask

    best_code = None
    best_size = float('inf')

    # Helper: greedy set cover with randomization
    def greedy_set_cover(randomize_order=True, rng=None):
        if rng is None:
            rng = random
        uncovered = (1 << total_words) - 1
        code_indices = []

        while uncovered:
            best_gain = -1
            best_c = -1

            if randomize_order:
                candidates = list(range(total_words))
                rng.shuffle(candidates)
            else:
                candidates = range(total_words)

            for c in candidates:
                gain = bin(sphere_bits[c] & uncovered).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                if best_gain == 15:
                    break

            if best_c == -1:
                break

            code_indices.append(best_c)
            uncovered &= ~sphere_bits[best_c]

        return code_indices

    # Helper: prune redundant codewords using numpy
    def prune_code(code_indices, rng=None):
        if rng is None:
            rng = random
        count = np.zeros(total_words, dtype=np.int32)
        for c in code_indices:
            count[sphere_arrays[c]] += 1

        current_code = set(code_indices)
        to_check = list(code_indices)
        rng.shuffle(to_check)

        for c in to_check:
            if c not in current_code:
                continue
            # Check if removing c is safe: all words in sphere[c] must have count > 1
            if np.min(count[sphere_arrays[c]]) > 1:
                current_code.remove(c)
                count[sphere_arrays[c]] -= 1

        return list(current_code)

    # Helper: local search via removal, swap, and 2-for-1 moves
    def local_search(code_indices, max_iters=300, rng=None):
        if rng is None:
            rng = random
        code_set = set(code_indices)
        count = np.zeros(total_words, dtype=np.int32)
        for c in code_indices:
            count[sphere_arrays[c]] += 1

        current_size = len(code_indices)
        improved = True
        iters = 0

        while improved and iters < max_iters:
            improved = False
            iters += 1

            # Try removing a codeword
            for c in list(code_set):
                if np.min(count[sphere_arrays[c]]) > 1:
                    code_set.remove(c)
                    count[sphere_arrays[c]] -= 1
                    current_size -= 1
                    improved = True
                    break

            if improved:
                continue

            # Try 2-for-1 swap: remove one codeword, add another that covers all newly uncovered
            critical_words = np.where(count == 1)[0]
            if len(critical_words) == 0:
                break

            candidates_remove = list(code_set)
            rng.shuffle(candidates_remove)

            for c_remove in candidates_remove[:30]:
                # Simulate removal
                temp_count = count.copy()
                temp_count[sphere_arrays[c_remove]] -= 1

                # Find words that would become uncovered
                uncovered_after_remove = np.where(temp_count == 0)[0]
                if len(uncovered_after_remove) == 0:
                    # Safe to remove
                    code_set.remove(c_remove)
                    count = temp_count
                    current_size -= 1
                    improved = True
                    break

                # Try to find a single codeword that covers all uncovered_after_remove
                # and is not already in the code
                for c_add in range(total_words):
                    if c_add in code_set:
                        continue
                    # Check if c_add covers all uncovered_after_remove
                    add_sphere = set(spheres[c_add])
                    covers_all = all(j in add_sphere for j in uncovered_after_remove)
                    if not covers_all:
                        continue

                    # Check if after swap, no word has count < 1
                    # New count = temp_count + sphere[c_add]
                    new_count = temp_count.copy()
                    new_count[sphere_arrays[c_add]] += 1
                    if np.min(new_count) >= 0:
                        # Valid swap
                        code_set.remove(c_remove)
                        code_set.add(c_add)
                        count = new_count
                        # Size stays the same, but we might enable further removals
                        improved = True
                        break

                if improved:
                    break

            if improved:
                continue

            # Try 1-for-2: add one codeword, then try to remove two
            # This is more expensive; only try if we have time
            if iters < max_iters // 2:
                # Find a codeword to add that has high overlap with current code
                # but allows removing two codewords
                # Simplified: try adding a random codeword and see if we can remove two
                for _ in range(5):
                    c_add = rng.randint(0, total_words - 1)
                    if c_add in code_set:
                        continue

                    temp_count = count.copy()
                    temp_count[sphere_arrays[c_add]] += 1

                    # Try to find two codewords that can be removed
                    removed = []
                    for c1 in list(code_set):
                        if c1 == c_add:
                            continue
                        if np.min(temp_count[sphere_arrays[c1]]) > 1:
                            temp_count[sphere_arrays[c1]] -= 1
                            removed.append(c1)
                            if len(removed) == 1:
                                break

                    if len(removed) == 1:
                        c1 = removed[0]
                        for c2 in list(code_set):
                            if c2 == c_add or c2 == c1:
                                continue
                            if np.min(temp_count[sphere_arrays[c2]]) > 1:
                                temp_count[sphere_arrays[c2]] -= 1
                                removed.append(c2)
                                break

                    if len(removed) == 2:
                        # Valid 1-for-2 move
                        code_set.add(c_add)
                        for c in removed:
                            code_set.remove(c)
                        count = temp_count
                        current_size -= 1
                        improved = True
                        break

            if not improved:
                break

        return list(code_set), current_size

    # Simulated annealing approach with improved moves
    def simulated_annealing(initial_code, max_iters=3000, t_start=1.0, t_end=0.005, rng=None):
        if rng is None:
            rng = random
        code_set = set(initial_code)
        count = np.zeros(total_words, dtype=np.int32)
        for c in initial_code:
            count[sphere_arrays[c]] += 1

        uncovered_count = int(np.sum(count == 0))
        current_size = len(code_set)
        current_energy = current_size * 1000 + uncovered_count * 10000

        best_code = set(code_set)
        best_energy = current_energy
        best_size = current_size

        t = t_start
        cooling = (t_end / t_start) ** (1.0 / max_iters)

        for it in range(max_iters):
            if time.time() > deadline:
                break

            # Generate a move: add, remove, or swap
            move_type = rng.random()
            new_code_set = set(code_set)
            new_count = count.copy()

            if move_type < 0.3:
                # Remove a random codeword
                if len(code_set) > 0:
                    c_remove = rng.choice(list(code_set))
                    new_code_set.remove(c_remove)
                    new_count[sphere_arrays[c_remove]] -= 1
            elif move_type < 0.6:
                # Add a codeword that covers an uncovered word if possible
                if uncovered_count > 0:
                    # Pick a random uncovered word
                    uncovered_indices = np.where(count == 0)[0]
                    if len(uncovered_indices) > 0:
                        target_word = rng.choice(uncovered_indices.tolist())
                        # Find codewords that cover this word
                        # We can use the precomputed sphere info: c covers target_word if target_word in spheres[c]
                        # For efficiency, pick a random codeword and check
                        c_add = rng.randint(0, total_words - 1)
                        if c_add not in code_set:
                            new_code_set.add(c_add)
                            new_count[sphere_arrays[c_add]] += 1
                else:
                    c_add = rng.randint(0, total_words - 1)
                    if c_add not in code_set:
                        new_code_set.add(c_add)
                        new_count[sphere_arrays[c_add]] += 1
            else:
                # Swap: remove one, add another
                if len(code_set) > 0:
                    c_remove = rng.choice(list(code_set))
                    c_add = rng.randint(0, total_words - 1)
                    if c_add != c_remove:
                        new_code_set.remove(c_remove)
                        new_code_set.add(c_add)
                        new_count[sphere_arrays[c_remove]] -= 1
                        new_count[sphere_arrays[c_add]] += 1

            new_uncovered = int(np.sum(new_count == 0))
            new_size = len(new_code_set)
            new_energy = new_size * 1000 + new_uncovered * 10000

            delta_e = new_energy - current_energy

            if delta_e < 0 or rng.random() < np.exp(-delta_e / max(t, 1e-6)):
                code_set = new_code_set
                count = new_count
                current_size = new_size
                current_energy = new_energy
                uncovered_count = new_uncovered

                if new_uncovered == 0 and new_size < best_size:
                    best_code = set(new_code_set)
                    best_energy = new_energy
                    best_size = new_size

            t *= cooling

        return list(best_code), best_size

    # Run multiple greedy runs with different randomizations
    num_greedy_runs = 40

    for run in range(num_greedy_runs):
        if time.time() > deadline - 15:
            break

        # Create a new RNG for this run
        rng = random.Random(run * 12345 + 67890)

        # Greedy set cover
        code_indices = greedy_set_cover(randomize_order=True, rng=rng)

        # Prune
        code_indices = prune_code(code_indices, rng=rng)

        # Local search
        code_indices, size_after_ls = local_search(code_indices, max_iters=200, rng=rng)

        if size_after_ls < best_size:
            best_size = size_after_ls
            best_code = code_indices

            if best_size <= 186:
                break

        # Simulated annealing on the best found so far
        if run % 4 == 0 and best_code is not None:
            if time.time() > deadline - 25:
                break
            sa_code, sa_size = simulated_annealing(best_code, max_iters=1500, rng=rng)
            if sa_size < best_size:
                best_size = sa_size
                best_code = sa_code
                if best_size <= 186:
                    break

    # Final local search on the best code
    if best_code is not None and time.time() < deadline - 5:
        rng_final = random.Random(99999)
        best_code, best_size = local_search(best_code, max_iters=500, rng=rng_final)

    # Final SA pass if we have time
    if best_code is not None and time.time() < deadline - 10:
        rng_final2 = random.Random(88888)
        sa_code, sa_size = simulated_annealing(best_code, max_iters=2000, rng=rng_final2)
        if sa_size < best_size:
            best_size = sa_size
            best_code = sa_code

    if best_code is None:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Convert indices back to word lists
    result = [all_words[i] for i in best_code]
    return result
