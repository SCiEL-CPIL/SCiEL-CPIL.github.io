import time
import random
import math
import sys


def _row_diffs_list(row):
    """Return list of all positive differences within a row (i<j)."""
    n = len(row)
    diffs = []
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.append(row[j] - ri)
    return diffs


def _row_mask(row):
    """Bitmask of differences in a row."""
    m = 0
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            m |= (1 << (row[j] - ri))
    return m


def _scope(rows):
    return max(r[-1] for r in rows)


def _is_valid(rows):
    """Check all positive differences are globally distinct."""
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs_list(row))
    return len(all_diffs) == len(set(all_diffs))


def _greedy_construct(n_rows, num_elements, rng):
    """Greedy construction: build rows one by one, picking the smallest valid value."""
    rows = []
    used_mask = 0

    for row_idx in range(n_rows):
        row = [0]
        for pos in range(1, num_elements):
            found = False
            high = 250

            candidates = list(range(row[-1] + 1, high + 1))

            if len(candidates) > 30:
                sample_size = min(25, len(candidates))
                weights = [1.0 / (1.0 + 0.5 * i) for i in range(len(candidates))]
                selected = []
                pool = list(range(len(candidates)))
                for _ in range(sample_size):
                    total_w = sum(weights[i] for i in pool)
                    r = rng.random() * total_w
                    cum = 0
                    for idx, pi in enumerate(pool):
                        cum += weights[pi]
                        if r <= cum:
                            selected.append(candidates[pi])
                            pool.remove(pi)
                            break
                candidates = sorted(selected)

            for val in candidates:
                new_diffs = {val - prev for prev in row}
                if len(new_diffs) != len(set(new_diffs)):
                    continue
                new_mask = 0
                for d in new_diffs:
                    new_mask |= (1 << d)
                if new_mask & used_mask:
                    continue
                existing_diffs = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        existing_diffs.add(row[j] - row[i])
                if new_diffs & existing_diffs:
                    continue

                row.append(val)
                used_mask |= new_mask
                found = True
                break

            if not found:
                row.append(row[-1] + 10 + rng.randint(0, 20))
                new_diffs = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        new_diffs.add(row[j] - row[i])
                for d in new_diffs:
                    used_mask |= (1 << d)

        rows.append(row)

    return rows


def _compute_all_diffs(rows):
    """Return (global_mask, row_masks) for all rows."""
    row_masks = []
    global_mask = 0
    for row in rows:
        m = _row_mask(row)
        row_masks.append(m)
        global_mask |= m
    return global_mask, row_masks


def _try_move_element(rows, row_idx, pos, new_val, global_mask, row_masks):
    """Try to move element at rows[row_idx][pos] to new_val."""
    row = rows[row_idx]
    n = len(row)

    if pos > 0 and new_val <= row[pos - 1]:
        return False, None, None
    if pos < n - 1 and new_val >= row[pos + 1]:
        return False, None, None

    old_val = row[pos]
    if new_val == old_val:
        return False, None, None

    row[pos] = new_val

    diffs_in_row = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            if d in diffs_in_row:
                row[pos] = old_val
                return False, None, None
            diffs_in_row.add(d)

    new_row_mask = 0
    for d in diffs_in_row:
        new_row_mask |= (1 << d)

    other_mask = global_mask & ~row_masks[row_idx]
    if new_row_mask & other_mask:
        row[pos] = old_val
        return False, None, None

    new_global_mask = (global_mask & ~row_masks[row_idx]) | new_row_mask
    row[pos] = old_val

    return True, new_global_mask, new_row_mask


def _shift_row_down(rows, row_idx, global_mask, row_masks, max_shift=20):
    """
    Try to shift an entire row down by some amount (rigid frame move).
    Returns (success, new_global_mask, new_row_mask, shift_amount).
    """
    row = rows[row_idx]
    n = len(row)
    old_row_mask = row_masks[row_idx]
    other_mask = global_mask & ~old_row_mask

    # Try shifts from 1 to max_shift
    for shift in range(1, max_shift + 1):
        # Check if shifting down by 'shift' is valid
        # New row: [row[0] - shift, row[1] - shift, ..., row[n-1] - shift]
        # But row[0] must stay >= 0, and row must still be strictly increasing
        # Since all elements shift by same amount, strict increase is preserved
        # row[0] - shift >= 0 => shift <= row[0]
        if shift > row[0]:
            break

        # New differences are the same (rigid frame), so new_row_mask = old_row_mask
        # But we need to check if any difference now conflicts with other rows
        # Actually, the differences don't change when shifting! Only the absolute values change.
        # Wait - the differences a_j - a_i are invariant under translation.
        # So shifting a row doesn't change its difference set at all!
        # This means shifting can't help with difference conflicts.
        # The only way to improve scope is to change the internal structure.
        # So rigid frame shifting is useless for this problem.
        break

    return False, None, None, 0


def _ga_crossover(parent1, parent2, n_rows, rng):
    """
    Crossover: for each row position, pick the row from the parent with smaller scope contribution.
    This is a uniform crossover at the row level.
    """
    child = []
    for i in range(n_rows):
        if rng.random() < 0.5:
            child.append(parent1[i][:])
        else:
            child.append(parent2[i][:])
    return child


def _ga_mutation(rows, num_elements, rng, mutation_rate=0.3):
    """
    Mutation: for each row, with some probability, randomly perturb one element.
    """
    mutated = [r[:] for r in rows]
    for row_idx in range(len(mutated)):
        if rng.random() < mutation_rate:
            row = mutated[row_idx]
            pos = rng.randint(1, num_elements - 1)
            old_val = row[pos]
            low = row[pos - 1] + 1
            high = row[pos + 1] - 1 if pos < num_elements - 1 else old_val + 30
            if low < high:
                new_val = rng.randint(low, high)
                row[pos] = new_val
    return mutated


def _ga_repair(rows, num_elements, rng, time_limit):
    """
    Greedy repair: iteratively find the element causing the most conflicts
    and try to move it to a valid position. This repairs invalid solutions.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope

    n_rows = len(current_rows)
    n_elem = num_elements

    max_iter = 50000
    stagnation = 0
    max_stagnation = 5000

    for it in range(max_iter):
        if time.time() > time_limit:
            break
        if stagnation > max_stagnation:
            break

        # Check validity
        if not _is_valid(current_rows):
            # Find a conflict and try to fix it
            all_diffs = {}
            for row_idx, row in enumerate(current_rows):
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        d = row[j] - row[i]
                        if d in all_diffs:
                            # Conflict found
                            # Try to fix by moving an element in one of the conflicting rows
                            target_row = all_diffs[d]
                            target_pos = rng.randint(1, n_elem - 1)
                            target_row_data = current_rows[target_row]
                            old_val = target_row_data[target_pos]

                            low = target_row_data[target_pos - 1] + 1
                            high = target_row_data[target_pos + 1] - 1 if target_pos < n_elem - 1 else current_scope + 10

                            if low < high:
                                new_val = rng.randint(low, high)
                                global_mask, row_masks = _compute_all_diffs(current_rows)
                                success, new_global_mask, new_row_mask = _try_move_element(
                                    current_rows, target_row, target_pos, new_val, global_mask, row_masks
                                )
                                if success:
                                    current_rows[target_row][target_pos] = new_val
                                    continue
                            break
                        all_diffs[d] = row_idx
            continue

        # Valid - try to improve scope
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, n_elem - 1)
        row = current_rows[row_idx]
        old_val = row[pos]

        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < n_elem - 1 else current_scope

        if low >= old_val:
            stagnation += 1
            continue

        new_val = rng.randint(low, old_val - 1)
        global_mask, row_masks = _compute_all_diffs(current_rows)
        success, new_global_mask, new_row_mask = _try_move_element(
            current_rows, row_idx, pos, new_val, global_mask, row_masks
        )

        if success:
            current_rows[row_idx][pos] = new_val
            new_scope = _scope(current_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
                stagnation = 0
            else:
                stagnation += 1
        else:
            stagnation += 1

    return best_rows, best_scope


def _genetic_algorithm(n_rows, num_elements, rng, time_limit, pop_size=12):
    """
    Genetic algorithm with row-level crossover and element-level mutation.
    """
    start = time.time()

    # Initialize population
    population = []
    for i in range(pop_size):
        rng_local = random.Random(rng.randint(0, 1000000))
        rows = _greedy_construct(n_rows, num_elements, rng_local)
        if _is_valid(rows):
            population.append(rows)

    if not population:
        # Fallback: simple construction
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
        population = [rows]

    best_rows = min(population, key=_scope)
    best_scope = _scope(best_rows)

    gen = 0
    max_generations = 200

    while time.time() < time_limit and gen < max_generations:
        gen += 1

        # Evaluate and sort population by scope
        population.sort(key=_scope)

        # Elitism: keep top 2
        new_population = [population[0][:], population[1][:]] if len(population) > 1 else [population[0][:]]

        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > time_limit:
                break

            # Tournament selection
            def tournament_select(pop):
                k = min(3, len(pop))
                candidates = rng.sample(pop, k)
                return min(candidates, key=_scope)

            parent1 = tournament_select(population)
            parent2 = tournament_select(population)

            # Crossover
            child = _ga_crossover(parent1, parent2, n_rows, rng)

            # Mutation
            child = _ga_mutation(child, num_elements, rng, mutation_rate=0.4)

            # Repair if invalid
            if not _is_valid(child):
                repaired, scope = _ga_repair(child, num_elements, rng, time_limit=min(time.time() + 2, time_limit))
                if _is_valid(repaired):
                    child = repaired
                else:
                    # Skip invalid offspring
                    continue

            new_population.append(child)

        population = new_population

        # Update best
        current_best = min(population, key=_scope)
        current_scope = _scope(current_best)
        if current_scope < best_scope:
            best_scope = current_scope
            best_rows = current_best

    return best_rows, best_scope


def _local_search_refine(rows, num_elements, rng, time_limit):
    """Simple local search: try to reduce scope by moving elements down."""
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope

    global_mask, row_masks = _compute_all_diffs(current_rows)
    n_rows = len(current_rows)
    n_elem = num_elements

    max_iter = 100000
    stagnation = 0
    max_stagnation = 5000

    for it in range(max_iter):
        if time.time() > time_limit:
            break
        if stagnation > max_stagnation:
            break

        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, n_elem - 1)
        row = current_rows[row_idx]
        old_val = row[pos]

        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < n_elem - 1 else current_scope

        if low >= old_val:
            continue

        new_val = rng.randint(low, old_val - 1)

        success, new_global_mask, new_row_mask = _try_move_element(
            current_rows, row_idx, pos, new_val, global_mask, row_masks
        )

        if success:
            current_rows[row_idx][pos] = new_val
            old_row_mask = row_masks[row_idx]
            row_masks[row_idx] = new_row_mask
            global_mask = new_global_mask

            new_scope = _scope(current_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
                stagnation = 0
            else:
                stagnation += 1
        else:
            stagnation += 1

    return best_rows, best_scope


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = start_time + 140.0

    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))

    rng = random.Random(42)

    # Phase 1: Generate initial solution
    best_rows = None
    best_scope = float('inf')

    for seed in range(10):
        if time.time() > DEADLINE - 100:
            break
        rng_local = random.Random(seed * 7 + 1)
        rows = _greedy_construct(n_rows, num_elements, rng_local)
        if _is_valid(rows):
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = rows

    if best_rows is None:
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
        best_rows = rows
        best_scope = _scope(best_rows)

    # Phase 2: Genetic Algorithm (main new strategy)
    if time.time() < DEADLINE - 40:
        ga_rows, ga_scope = _genetic_algorithm(
            n_rows, num_elements, rng,
            time_limit=DEADLINE - 40,
            pop_size=12
        )
        if ga_scope < best_scope:
            best_scope = ga_scope
            best_rows = ga_rows

    # Phase 3: Local search refinement
    if time.time() < DEADLINE - 15:
        improved_rows, improved_scope = _local_search_refine(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 15
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows

    # Final validation
    if _is_valid(best_rows):
        return best_rows
    else:
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
        return rows
