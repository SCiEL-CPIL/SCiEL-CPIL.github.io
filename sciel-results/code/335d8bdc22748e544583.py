import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 142.0

    total_space = q ** n

    # Precompute positional powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i - 1] * q

    # Precompute neighbor masks for all words (word w covers itself + all words at dist <= radius)
    # For radius=1, q=3, n=7: each word covers 1 + n*(q-1) = 1 + 7*2 = 15 words
    masks = [0] * total_space
    for i in range(total_space):
        m = 1 << i
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                m |= (1 << (i + diff * powers[pos]))
        masks[i] = m

    full_mask = (1 << total_space) - 1

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    def polish(selected, time_limit):
        """Remove redundant codewords iteratively."""
        selected = selected[:]
        if not selected:
            return selected

        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            mm = masks[w]
            while mm:
                lowbit = mm & -mm
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                mm ^= lowbit

        changed = True
        while changed and time.time() < time_limit:
            changed = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                m = masks[w]

                removable = True
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        removable = False
                        break
                    mm ^= lowbit

                if removable:
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    selected.pop(idx)
                    changed = True

        return selected

    def run_greedy():
        """Standard greedy set cover to get an initial solution."""
        covered = 0
        selected = []
        remaining = full_mask

        while remaining and time.time() < deadline - 10:
            best_gain = -1
            best_word = -1
            for w in range(total_space):
                gain = (masks[w] & remaining).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_word = w

            if best_word < 0:
                break

            selected.append(best_word)
            covered |= masks[best_word]
            remaining = full_mask & ~covered

        return selected

    def simulated_annealing(initial, time_limit):
        """Simulated annealing to minimize uncovered count, then size."""
        if not initial:
            return []

        # Build coverage counts
        counts = np.zeros(total_space, dtype=np.int32)
        for w in initial:
            mm = masks[w]
            while mm:
                lowbit = mm & -mm
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                mm ^= lowbit

        # Uncovered = words with count 0
        uncovered_count = np.sum(counts == 0)
        current_energy = uncovered_count + 0.01 * len(initial)

        best_energy = current_energy
        best_set = initial[:]

        temp = 5.0
        cooling = 0.9995

        while time.time() < time_limit:
            # Pick a random move
            move_type = random.random()
            new_set = None

            if move_type < 0.4:
                # 1-for-1 swap
                if len(initial) < 2:
                    continue
                i = random.randrange(len(initial))
                old_w = initial[i]
                # Try a random replacement
                new_w = random.randrange(total_space)
                if new_w == old_w:
                    continue
                # Check if new_w is already in the set
                if new_w in initial:
                    continue

                # Compute energy delta
                # Remove old_w, add new_w
                m_old = masks[old_w]
                m_new = masks[new_w]

                # Words that become uncovered (covered only by old_w, not by new_w)
                newly_uncovered = 0
                mm = m_old & ~m_new
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        newly_uncovered += 1
                    mm ^= lowbit

                # Words that become newly covered (uncovered, covered by new_w)
                newly_covered = 0
                mm = m_new & ~m_old
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 0:
                        newly_covered += 1
                    mm ^= lowbit

                delta_uncovered = newly_uncovered - newly_covered
                delta_energy = delta_uncovered  # size unchanged

                if delta_energy <= 0 or random.random() < np.exp(-delta_energy / max(temp, 1e-10)):
                    # Apply move
                    mm = m_old
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    mm = m_new
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] += 1
                        mm ^= lowbit
                    initial[i] = new_w
                    current_energy += delta_energy
                    new_set = initial[:]

            elif move_type < 0.7:
                # Deletion
                if len(initial) < 1:
                    continue
                i = random.randrange(len(initial))
                w = initial[i]
                m = masks[w]

                # Count how many words become uncovered
                newly_uncovered = 0
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        newly_uncovered += 1
                    mm ^= lowbit

                delta_energy = newly_uncovered - 0.01
                if delta_energy <= 0 or random.random() < np.exp(-delta_energy / max(temp, 1e-10)):
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    initial.pop(i)
                    current_energy += delta_energy
                    new_set = initial[:]

            else:
                # Insertion
                new_w = random.randrange(total_space)
                if new_w in initial:
                    continue
                m = masks[new_w]

                newly_covered = 0
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 0:
                        newly_covered += 1
                    mm ^= lowbit

                delta_energy = -newly_covered + 0.01
                if delta_energy <= 0 or random.random() < np.exp(-delta_energy / max(temp, 1e-10)):
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] += 1
                        mm ^= lowbit
                    initial.append(new_w)
                    current_energy += delta_energy
                    new_set = initial[:]

            if new_set is None:
                continue

            if current_energy < best_energy:
                best_energy = current_energy
                best_set = initial[:]

            temp *= cooling

        return best_set

    # Get initial solution via greedy
    if time.time() > deadline - 15:
        return [int_to_tuple(v) for v in range(total_space)]

    sol = run_greedy()
    best_size = len(sol)
    best_selected = sol[:]

    # Polish the greedy solution
    if time.time() < deadline - 10:
        polished = polish(best_selected[:], deadline - 10)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]

    # Run simulated annealing with multiple restarts
    sa_time_budget = deadline - 20
    if time.time() < sa_time_budget:
        # Use the polished solution as starting point
        sa_result = simulated_annealing(best_selected[:], sa_time_budget)
        if sa_result:
            polished_sa = polish(sa_result[:], deadline - 5)
            if len(polished_sa) < best_size:
                best_size = len(polished_sa)
                best_selected = polished_sa[:]

    # Final polish
    if time.time() < deadline - 2:
        polished = polish(best_selected[:], deadline - 2)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]

    best_words = [int_to_tuple(v) for v in best_selected]
    return best_words
