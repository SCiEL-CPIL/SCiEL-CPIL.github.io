# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large square-difference-free subset of Z/205Z.
    Uses bitmask-based conflict checking, CRT structure exploitation,
    randomized greedy with multiple restarts, and local search with 2-opt.
    """
    m = 205
    deadline = time.monotonic() + 610.0  # safety margin before 630s kill

    # --- Precomputation ---
    forbidden = set()
    for x in range(1, m):
        f = (x * x) % m
        if f != 0:
            forbidden.add(f)
    forbidden.discard(0)
    forbidden_set = forbidden

    # Bitmask for conflicts: conflict_mask[i] is a 205-bit integer
    # where bit j is set if i and j conflict
    conflict_mask = np.zeros(m, dtype=np.uint64)  # need 205 bits, use 4 uint64s
    # Actually, use a list of 4 uint64 per element, or use Python integers
    # For speed, use Python's arbitrary precision integers
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            d = (j - i) % m
            if d in forbidden_set:
                mask |= (1 << j)
        conflict_mask[i] = mask

    # Also precompute for each element, the set of non-conflicting elements
    # that are NOT in the current set - we'll check dynamically

    # Precompute conflict as a boolean matrix for quick lookup
    conflict_bool = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(m):
            if i != j:
                d = (j - i) % m
                if d in forbidden_set:
                    conflict_bool[i, j] = True
                    conflict_bool[j, i] = True

    def can_add(subset_mask, cand):
        """Check if cand can be added to subset (represented as bitmask)."""
        # subset_mask has bits set for elements in subset
        # Check if any element in subset conflicts with cand
        # conflict_mask[cand] has bits set for elements that conflict with cand
        # If (subset_mask & conflict_mask[cand]) == 0, then cand can be added
        return (subset_mask & conflict_mask[cand]) == 0

    def is_valid_subset(subset_mask):
        """Check if subset has no forbidden pairs."""
        elems = [i for i in range(m) if (subset_mask >> i) & 1]
        for idx, a in enumerate(elems):
            for b in elems[idx+1:]:
                if conflict_bool[a, b]:
                    return False
        return True

    def mask_to_set(subset_mask):
        return set(i for i in range(m) if (subset_mask >> i) & 1)

    def set_to_mask(s):
        mask = 0
        for x in s:
            mask |= (1 << x)
        return mask

    # --- Strategy 1: Randomized greedy with bitmask ---
    def randomized_greedy(rng):
        order = list(range(m))
        rng.shuffle(order)
        subset_mask = 0
        for cand in order:
            if can_add(subset_mask, cand):
                subset_mask |= (1 << cand)
        return subset_mask

    # --- Strategy 2: Local search with bitmask ---
    def local_search(initial_mask, rng, max_iters=30000):
        """Improve by trying adds, swaps, and 2-opt moves."""
        current = initial_mask
        cur_size = bin(current).count('1')
        best = current
        best_size = cur_size
        no_improve = 0
        all_elements = list(range(m))

        for it in range(max_iters):
            if time.monotonic() > deadline:
                break
            if no_improve > 8000:
                break

            # Try to add a random element
            added = False
            for _ in range(15):
                cand = rng.randint(0, m - 1)
                if not ((current >> cand) & 1):
                    if can_add(current, cand):
                        current |= (1 << cand)
                        added = True
                        new_size = bin(current).count('1')
                        if new_size > best_size:
                            best = current
                            best_size = new_size
                            no_improve = 0
                        break
            if added:
                continue

            # Try a 1-swap: remove one, add one
            swap_done = False
            cur_elems = [i for i in range(m) if (current >> i) & 1]
            if cur_elems:
                to_remove = rng.choice(cur_elems)
                trial = current & ~(1 << to_remove)
                # Try to add a better element
                found_add = False
                for _ in range(25):
                    cand = rng.randint(0, m - 1)
                    if not ((trial >> cand) & 1):
                        if can_add(trial, cand):
                            trial |= (1 << cand)
                            found_add = True
                            new_size = bin(trial).count('1')
                            if new_size > best_size:
                                current = trial
                                best = trial
                                best_size = new_size
                                no_improve = 0
                            elif new_size > cur_size:
                                current = trial
                                cur_size = new_size
                            break
                if not found_add:
                    # Undo
                    current |= (1 << to_remove)

            # Try a 2-swap: remove two, add two
            if len(cur_elems) >= 2:
                r1, r2 = rng.sample(cur_elems, 2)
                trial = current & ~(1 << r1) & ~(1 << r2)
                found_adds = 0
                for _ in range(20):
                    cand = rng.randint(0, m - 1)
                    if not ((trial >> cand) & 1) and cand not in (r1, r2):
                        if can_add(trial, cand):
                            trial |= (1 << cand)
                            found_adds += 1
                            if found_adds == 2:
                                break
                if found_adds == 2:
                    new_size = bin(trial).count('1')
                    if new_size > best_size:
                        current = trial
                        best = trial
                        best_size = new_size
                        no_improve = 0
                    elif new_size > cur_size:
                        current = trial
                        cur_size = new_size

            # Track no improvement
            cur_size = bin(current).count('1')
            if cur_size <= best_size:
                no_improve += 1

        return best

    # --- Strategy 3: Simulated annealing with bitmask ---
    def simulated_annealing(initial_mask, rng, steps=80000):
        current = initial_mask
        cur_size = bin(current).count('1')
        best = current
        best_size = cur_size
        T = 1.0
        cooling = 0.99995
        all_elements = list(range(m))

        for step in range(steps):
            if time.monotonic() > deadline:
                break

            move_type = rng.random()
            if move_type < 0.35:
                # Add
                cand = rng.randint(0, m - 1)
                if not ((current >> cand) & 1):
                    if can_add(current, cand):
                        current |= (1 << cand)
                        new_size = bin(current).count('1')
                        if new_size > best_size:
                            best = current
                            best_size = new_size
            elif move_type < 0.65:
                # Remove
                cur_elems = [i for i in range(m) if (current >> i) & 1]
                if cur_elems:
                    to_remove = rng.choice(cur_elems)
                    current &= ~(1 << to_remove)
            else:
                # Swap
                cur_elems = [i for i in range(m) if (current >> i) & 1]
                if cur_elems:
                    to_remove = rng.choice(cur_elems)
                    trial = current & ~(1 << to_remove)
                    cand = rng.randint(0, m - 1)
                    if not ((trial >> cand) & 1):
                        if can_add(trial, cand):
                            trial |= (1 << cand)
                            new_size = bin(trial).count('1')
                            if new_size >= cur_size or rng.random() < np.exp((new_size - cur_size) / T):
                                current = trial
                                cur_size = new_size
                                if new_size > best_size:
                                    best = current
                                    best_size = new_size
                        else:
                            # Keep original
                            pass
                    else:
                        pass

            T *= cooling

        return best

    # --- Strategy 4: CRT-based structured search ---
    def crt_structured_search(rng):
        """
        Use the CRT structure: Z/205Z ≅ Z/5Z × Z/41Z
        Differences with mod-5 component in {0, 2, 3} are always allowed.
        This means elements with the same mod-5 residue are never in conflict
        with each other (since their difference has mod-5 component 0).
        """
        # For each mod-5 residue class, pick elements
        # Elements in the same mod-5 class never conflict with each other
        # But elements in different mod-5 classes may conflict

        # Group elements by their mod-5 residue
        classes = {r: [] for r in range(5)}
        for i in range(m):
            r = i % 5
            classes[r].append(i)

        # For each class, we can potentially take many elements
        # The conflicts are between different classes
        # Let's try to maximize by taking elements from all classes

        # Randomized approach: for each class, randomly select elements
        # and check cross-class conflicts
        best_mask = 0
        best_size = 0

        for restart in range(50):
            if time.monotonic() > deadline:
                break

            subset_mask = 0
            # Process classes in random order
            class_order = list(range(5))
            rng.shuffle(class_order)

            for r in class_order:
                # Try to add elements from this class
                elems_in_class = classes[r][:]
                rng.shuffle(elems_in_class)
                for cand in elems_in_class:
                    if can_add(subset_mask, cand):
                        subset_mask |= (1 << cand)

            new_size = bin(subset_mask).count('1')
            if new_size > best_size:
                best_mask = subset_mask
                best_size = new_size

        return best_mask

    # --- Main search loop ---
    best_solution_mask = 0
    best_size = 0

    rng = random.Random(42)

    # Phase 0: CRT-based structured search
    crt_mask = crt_structured_search(rng)
    crt_size = bin(crt_mask).count('1')
    if crt_size > best_size:
        best_solution_mask = crt_mask
        best_size = crt_size

    # Phase 1: Many randomized greedy restarts
    greedy_count = 0
    while time.monotonic() < deadline - 60:  # reserve 60s for refinement
        sol = randomized_greedy(rng)
        sol_size = bin(sol).count('1')
        greedy_count += 1
        if sol_size > best_size:
            best_solution_mask = sol
            best_size = sol_size
        # Periodically try local search on the best
        if greedy_count % 3 == 0:
            improved = local_search(best_solution_mask, rng, max_iters=15000)
            improved_size = bin(improved).count('1')
            if improved_size > best_size:
                best_solution_mask = improved
                best_size = improved_size

    # Phase 2: More local search on best
    if time.monotonic() < deadline - 20:
        improved = local_search(best_solution_mask, rng, max_iters=50000)
        improved_size = bin(improved).count('1')
        if improved_size > best_size:
            best_solution_mask = improved
            best_size = improved_size

    # Phase 3: Simulated annealing from best
    if time.monotonic() < deadline - 10:
        improved = simulated_annealing(best_solution_mask, rng, steps=100000)
        improved_size = bin(improved).count('1')
        if improved_size > best_size:
            best_solution_mask = improved
            best_size = improved_size

    # Phase 4: Final local search polish
    if time.monotonic() < deadline - 5:
        improved = local_search(best_solution_mask, rng, max_iters=30000)
        improved_size = bin(improved).count('1')
        if improved_size > best_size:
            best_solution_mask = improved
            best_size = improved_size

    # Phase 5: Another round of greedy + local search with different seed
    rng2 = random.Random(12345)
    greedy_count2 = 0
    while time.monotonic() < deadline - 2:
        sol = randomized_greedy(rng2)
        sol_size = bin(sol).count('1')
        greedy_count2 += 1
        if sol_size > best_size:
            best_solution_mask = sol
            best_size = sol_size
        if greedy_count2 % 2 == 0:
            improved = local_search(best_solution_mask, rng2, max_iters=10000)
            improved_size = bin(improved).count('1')
            if improved_size > best_size:
                best_solution_mask = improved
                best_size = improved_size

    # Ensure we have at least one element
    if best_solution_mask == 0:
        best_solution_mask = 1  # element 0

    # Convert to array
    result = np.array([i for i in range(m) if (best_solution_mask >> i) & 1], dtype=np.int64)
    return result


# EVOLVE-BLOCK-END
