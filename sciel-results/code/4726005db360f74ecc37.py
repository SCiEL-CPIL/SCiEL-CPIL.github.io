"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import time
    import random
    import math

    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s kill

    # Denominators are 2,3,...,48 -> indices 0..46
    n = 47
    denominators = list(range(2, 49))
    weights = [Fraction(1, d) for d in denominators]
    # Float versions for speed in comparisons
    weight_floats = [1.0 / d for d in denominators]

    def is_valid(indices_set):
        """Check if sum is strictly below 1"""
        s = sum(weight_floats[i] for i in indices_set)
        return s < 1.0

    def get_deficit(indices_set):
        """Return 1 - sum (the deficit we want to minimize)"""
        s = sum(weight_floats[i] for i in indices_set)
        return 1.0 - s

    # Greedy construction with different orderings
    def greedy_construct(order):
        """Build a solution by adding items in given order if they fit"""
        current_set = set()
        current_sum = 0.0
        for i in order:
            if current_sum + weight_floats[i] < 1.0:
                current_set.add(i)
                current_sum += weight_floats[i]
        return current_set, current_sum

    # Generate diverse initial solutions using different orderings
    initial_solutions = []

    # 1. Ascending denominator (largest weights first)
    order_asc = list(range(n))
    initial_solutions.append(greedy_construct(order_asc))

    # 2. Descending denominator (smallest weights first)
    order_desc = list(range(n - 1, -1, -1))
    initial_solutions.append(greedy_construct(order_desc))

    # 3. Random orderings
    random.seed(42)
    for _ in range(20):
        order = list(range(n))
        random.shuffle(order)
        initial_solutions.append(greedy_construct(order))

    # 4. Weighted random (bias towards larger weights)
    for _ in range(10):
        order = list(range(n))
        # Sort by weight with some randomness
        random.shuffle(order)
        initial_solutions.append(greedy_construct(order))

    # Find best initial solution
    best_set = None
    best_deficit = float('inf')
    for sol_set, sol_sum in initial_solutions:
        if sol_set:
            deficit = get_deficit(sol_set)
            if deficit < best_deficit:
                best_deficit = deficit
                best_set = set(sol_set)

    if best_set is None:
        # Fallback: just pick index 0 (1/2)
        best_set = {0}
        best_deficit = 0.5

    # Simulated Annealing refinement
    def simulated_annealing(initial_set, time_budget):
        """Refine a solution using simulated annealing"""
        current_set = set(initial_set)
        current_deficit = get_deficit(current_set)
        best_set_sa = set(current_set)
        best_deficit_sa = current_deficit

        # Temperature schedule
        t_start = 0.1
        t_end = 0.001
        iterations = 0

        while time.time() - start_time < time_budget:
            iterations += 1

            # Generate neighbor: swap one selected with one unselected,
            # or add one unselected, or remove one selected
            move_type = random.random()

            if move_type < 0.6:
                # Swap
                if not current_set:
                    continue
                selected_list = list(current_set)
                unselected_list = [i for i in range(n) if i not in current_set]
                if not unselected_list:
                    continue
                i = random.choice(selected_list)
                j = random.choice(unselected_list)
                new_set = set(current_set)
                new_set.remove(i)
                new_set.add(j)
            elif move_type < 0.85:
                # Add
                unselected_list = [i for i in range(n) if i not in current_set]
                if not unselected_list:
                    continue
                j = random.choice(unselected_list)
                new_set = set(current_set)
                new_set.add(j)
            else:
                # Remove
                if not current_set:
                    continue
                i = random.choice(list(current_set))
                new_set = set(current_set)
                new_set.remove(i)

            if not new_set:
                continue

            # Check validity
            new_sum = sum(weight_floats[i] for i in new_set)
            if new_sum >= 1.0:
                continue

            new_deficit = 1.0 - new_sum
            delta = new_deficit - current_deficit

            # Temperature decay
            progress = min(iterations / 50000.0, 1.0)
            temperature = t_start * (t_end / t_start) ** progress

            # Acceptance criterion
            if delta < 0 or random.random() < math.exp(-delta / max(temperature, 1e-10)):
                current_set = new_set
                current_deficit = new_deficit

                if current_deficit < best_deficit_sa:
                    best_deficit_sa = current_deficit
                    best_set_sa = set(current_set)

        return best_set_sa, best_deficit_sa

    # Run simulated annealing from the best initial solution
    sa_time_budget = deadline - 5.0
    sa_set, sa_deficit = simulated_annealing(best_set, sa_time_budget)

    if sa_deficit < best_deficit:
        best_set = sa_set
        best_deficit = sa_deficit

    # Also try SA from a few other good initial solutions if time permits
    if time.time() - start_time < deadline - 10.0:
        # Sort initial solutions by quality
        sorted_sols = sorted(initial_solutions, key=lambda x: get_deficit(x[0]) if x[0] else float('inf'))
        for sol_set, _ in sorted_sols[:3]:
            if time.time() - start_time > deadline - 5.0:
                break
            if not sol_set:
                continue
            sa_set2, sa_deficit2 = simulated_annealing(sol_set, deadline - 5.0)
            if sa_deficit2 < best_deficit:
                best_set = sa_set2
                best_deficit = sa_deficit2

    # Final local search: try all single swaps to see if we can improve
    if time.time() - start_time < deadline - 2.0:
        improved = True
        while improved and time.time() - start_time < deadline - 2.0:
            improved = False
            current_deficit = get_deficit(best_set)
            selected_list = list(best_set)
            unselected_list = [i for i in range(n) if i not in best_set]

            for i in selected_list:
                if time.time() - start_time > deadline - 2.0:
                    break
                for j in unselected_list:
                    if weight_floats[j] <= weight_floats[i]:
                        continue
                    new_set = set(best_set)
                    new_set.remove(i)
                    new_set.add(j)
                    new_sum = sum(weight_floats[k] for k in new_set)
                    if new_sum < 1.0:
                        new_deficit = 1.0 - new_sum
                        if new_deficit < current_deficit:
                            best_set = new_set
                            current_deficit = new_deficit
                            improved = True
                            break
                if improved:
                    break

            # Also try adding items
            if not improved and time.time() - start_time < deadline - 2.0:
                current_sum = sum(weight_floats[i] for i in best_set)
                for j in unselected_list:
                    if current_sum + weight_floats[j] < 1.0:
                        best_set.add(j)
                        improved = True
                        break

    # Final verification
    assert len(best_set) > 0, "Must select at least one item"
    total_check = sum(Fraction(1, denominators[i]) for i in best_set)
    assert total_check < 1, f"Sum {total_check} not strictly below 1"

    return list(best_set)
# EVOLVE-BLOCK-END
