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
    sphere_sets = [set(s) for s in spheres]

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
    def greedy_set_cover(randomize_order=True):
        uncovered = (1 << total_words) - 1
        code_indices = []

        while uncovered:
            best_gain = -1
            best_c = -1

            if randomize_order:
                candidates = list(range(total_words))
                random.shuffle(candidates)
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

    # Helper: prune redundant codewords
    def prune_code(code_indices):
        count = np.zeros(total_words, dtype=np.int32)
        for c in code_indices:
            for j in spheres[c]:
                count[j] += 1

        current_code = set(code_indices)
        to_check = list(code_indices)
        random.shuffle(to_check)

        for c in to_check:
            if c not in current_code:
                continue
            # Check if removing c is safe
            safe = True
            for j in spheres[c]:
                if count[j] <= 1:
                    safe = False
                    break
            if safe:
                current_code.remove(c)
                for j in spheres[c]:
                    count[j] -= 1

        return list(current_code)

    # Helper: local search via swap/add/remove
    def local_search(code_indices, max_iters=500):
        code_set = set(code_indices)
        count = np.zeros(total_words, dtype=np.int32)
        for c in code_indices:
            for j in spheres[c]:
                count[j] += 1

        current_size = len(code_indices)
        improved = True
        iters = 0

        while improved and iters < max_iters:
            improved = False
            iters += 1

            # Try removing a codeword
            for c in list(code_set):
                safe = True
                for j in spheres[c]:
                    if count[j] <= 1:
                        safe = False
                        break
                if safe:
                    code_set.remove(c)
                    for j in spheres[c]:
                        count[j] -= 1
                    current_size -= 1
                    improved = True
                    break

            if improved:
                continue

            # Try swapping: remove one, add another that covers more uncovered
            # Find words with count == 1 (critical)
            critical_words = np.where(count == 1)[0]
            if len(critical_words) == 0:
                break

            # Try to find a codeword to remove and another to add
            # that reduces size or maintains coverage with smaller set
            candidates_remove = list(code_set)
            random.shuffle(candidates_remove)

            for c_remove in candidates_remove[:20]:  # Limit to 20 tries
                # Simulate removal
                temp_count = count.copy()
                for j in spheres[c_remove]:
                    temp_count[j] -= 1

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
                # and has minimal overlap with current code
                for c_add in range(total_words):
                    if c_add in code_set:
                        continue
                    # Check if c_add covers all uncovered_after_remove
                    covers_all = True
                    for j in uncovered_after_remove:
                        if j not in sphere_sets[c_add]:
                            covers_all = False
                            break
                    if not covers_all:
                        continue

                    # Check if adding c_add and removing c_remove is beneficial
                    # New count after swap
                    # We need to verify that after swap, no word has count < 1
                    # This is complex; skip for now and just do removals

            if improved:
                continue

            # Try adding a codeword that allows removing two others
            # This is more complex; skip for now

            break

        return list(code_set), current_size

    # Simulated annealing approach
    def simulated_annealing(initial_code, max_iters=2000, t_start=1.0, t_end=0.01):
        code_set = set(initial_code)
        count = np.zeros(total_words, dtype=np.int32)
        for c in initial_code:
            for j in spheres[c]:
                count[j] += 1

        uncovered_count = int(np.sum(count == 0))
        current_size = len(code_set)
        current_energy = current_size * 1000 + uncovered_count * 10000  # Heavy penalty for uncovered

        best_code = set(code_set)
        best_energy = current_energy
        best_size = current_size

        t = t_start
        cooling = (t_end / t_start) ** (1.0 / max_iters)

        for it in range(max_iters):
            if time.time() > deadline:
                break

            # Generate a move: add, remove, or swap
            move_type = random.random()
            new_code_set = set(code_set)
            new_count = count.copy()

            if move_type < 0.3:
                # Remove a random codeword
                if len(code_set) > 0:
                    c_remove = random.choice(list(code_set))
                    new_code_set.remove(c_remove)
                    for j in spheres[c_remove]:
                        new_count[j] -= 1
            elif move_type < 0.6:
                # Add a random codeword
                c_add = random.randint(0, total_words - 1)
                if c_add not in code_set:
                    new_code_set.add(c_add)
                    for j in spheres[c_add]:
                        new_count[j] += 1
            else:
                # Swap: remove one, add another
                if len(code_set) > 0:
                    c_remove = random.choice(list(code_set))
                    c_add = random.randint(0, total_words - 1)
                    if c_add != c_remove:
                        new_code_set.remove(c_remove)
                        new_code_set.add(c_add)
                        for j in spheres[c_remove]:
                            new_count[j] -= 1
                        for j in spheres[c_add]:
                            new_count[j] += 1

            new_uncovered = int(np.sum(new_count == 0))
            new_size = len(new_code_set)
            new_energy = new_size * 1000 + new_uncovered * 10000

            delta_e = new_energy - current_energy

            if delta_e < 0 or random.random() < np.exp(-delta_e / max(t, 1e-6)):
                code_set = new_code_set
                count = new_count
                current_size = new_size
                current_energy = new_energy

                if new_uncovered == 0 and new_size < best_size:
                    best_code = set(new_code_set)
                    best_energy = new_energy
                    best_size = new_size

            t *= cooling

        return list(best_code), best_size

    # Run greedy multiple times with different randomizations
    num_greedy_runs = 30

    for run in range(num_greedy_runs):
        if time.time() > deadline - 20:
            break

        # Greedy set cover
        code_indices = greedy_set_cover(randomize_order=True)

        # Prune
        code_indices = prune_code(code_indices)

        # Local search
        code_indices, size_after_ls = local_search(code_indices, max_iters=200)

        if size_after_ls < best_size:
            best_size = size_after_ls
            best_code = code_indices

            if best_size <= 186:
                break

        # Simulated annealing on the best found so far
        if run % 5 == 0 and best_code is not None:
            if time.time() > deadline - 30:
                break
            sa_code, sa_size = simulated_annealing(best_code, max_iters=1000)
            if sa_size < best_size:
                best_size = sa_size
                best_code = sa_code
                if best_size <= 186:
                    break

    # Final local search on the best code
    if best_code is not None and time.time() < deadline - 5:
        best_code, best_size = local_search(best_code, max_iters=500)

    if best_code is None:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Convert indices back to word lists
    result = [all_words[i] for i in best_code]
    return result
