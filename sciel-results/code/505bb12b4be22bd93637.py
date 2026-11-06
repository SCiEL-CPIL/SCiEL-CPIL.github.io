import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin from 90s limit

    n = 128
    target_low = 128
    target_high = 256
    targets = list(range(target_low, target_high + 1))

    def compute_reps(A_set):
        """Compute representation counts for all targets efficiently."""
        reps = {}
        for t in targets:
            count = 0
            for a in A_set:
                b = t - a
                if b in A_set and a <= b:
                    count += 1
            reps[t] = count
        return reps

    def is_valid(A_set):
        """Check if every target is covered by at least one pair."""
        for t in targets:
            found = False
            for a in A_set:
                b = t - a
                if b in A_set and a <= b:
                    found = True
                    break
            if not found:
                return False
        return True

    def get_peak(A_set):
        """Get the maximum representation multiplicity."""
        reps = compute_reps(A_set)
        return max(reps[t] for t in targets)

    def get_score(A_set):
        """Composite score: (peak, sum_of_squares) - lower is better."""
        reps = compute_reps(A_set)
        peak = max(reps[t] for t in targets)
        sq_sum = sum(v * v for v in reps.values())
        return (peak, sq_sum)

    # Phase 1: Greedy construction from empty set
    # This often finds a smaller set than starting from [64,128]
    A_greedy = set()
    uncovered = set(targets)

    for _ in range(200):
        if time.time() > deadline - 30:
            break
        if not uncovered:
            break

        best_elem = None
        best_gain = 0

        for x in range(0, 129):
            if x in A_greedy:
                continue
            gain = 0
            for y in A_greedy:
                t = x + y
                if t in uncovered:
                    gain += 1
            t2 = 2 * x
            if t2 in uncovered:
                gain += 1
            if gain > best_gain:
                best_gain = gain
                best_elem = x

        if best_elem is None or best_gain == 0:
            break

        A_greedy.add(best_elem)
        # Update uncovered
        for t in list(uncovered):
            found = False
            for a in A_greedy:
                b = t - a
                if b in A_greedy and a <= b:
                    found = True
                    break
            if found:
                uncovered.discard(t)

    # Choose the better starting point
    if is_valid(A_greedy):
        best_A = set(A_greedy)
    else:
        best_A = set(range(64, 129))

    best_peak = get_peak(best_A)
    best_score = get_score(best_A)

    # Phase 2: Greedy removal - remove elements while maintaining coverage
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False
        for elem in list(best_A):
            if time.time() > deadline - 20:
                break
            A_try = best_A - {elem}
            if is_valid(A_try):
                score_try = get_score(A_try)
                if score_try < best_score:
                    best_score = score_try
                    best_peak = score_try[0]
                    best_A = A_try
                    improved = True
                    break

    # Phase 3: Local search with swaps
    current_score = get_score(best_A)
    random.seed(42)

    while time.time() < deadline - 5:
        improved = False
        A_list = list(best_A)
        random.shuffle(A_list)

        for a_out in A_list:
            if time.time() > deadline - 5:
                break

            best_swap = None
            best_swap_score = current_score

            # Try candidate elements to add
            for a_in in range(0, 129):
                if a_in in best_A:
                    continue

                A_temp = best_A - {a_out}
                A_temp.add(a_in)

                if not is_valid(A_temp):
                    continue

                score = get_score(A_temp)
                if score < best_swap_score:
                    best_swap_score = score
                    best_swap = a_in

            if best_swap is not None:
                best_A.remove(a_out)
                best_A.add(best_swap)
                current_score = best_swap_score
                improved = True
                break

        if not improved:
            # Try random perturbations to escape local minima
            if time.time() > deadline - 10:
                break

            A_perturb = set(best_A)
            # Remove a random element
            if A_perturb:
                elem_to_remove = random.choice(list(A_perturb))
                A_perturb.remove(elem_to_remove)
                # Add a random element
                candidates = [x for x in range(0, 129) if x not in A_perturb]
                if candidates:
                    A_perturb.add(random.choice(candidates))

            if is_valid(A_perturb):
                score_perturb = get_score(A_perturb)
                # Accept if better or slightly worse (simulated annealing style)
                if score_perturb < current_score:
                    best_A = A_perturb
                    current_score = score_perturb
                elif random.random() < 0.1:
                    # Occasionally accept worse solutions to escape local minima
                    best_A = A_perturb
                    current_score = score_perturb

    return list(best_A)
