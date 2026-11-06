"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free_subset(elements, S):
        """Check if elements (a set) is sum-free with respect to S."""
        elist = list(elements)
        n = len(elist)
        for i in range(n):
            for j in range(i, n):  # a = b allowed
                s = elist[i] + elist[j]
                if s in elements:
                    return False
        return True

    def count_conflicts(element, current_set, S_set):
        """Count how many conflicts element would create with current_set."""
        conflicts = 0
        for a in current_set:
            # element + a = c in current_set?
            if element + a in current_set:
                conflicts += 1
            # a + element = c in current_set? (same as above since + is commutative)
            # element + element = c in current_set?
            if element + element in current_set and element + element != element:
                conflicts += 1
        return conflicts

    def greedy_construct(S, order):
        """Greedy: add elements in given order if they don't create conflicts."""
        selected = set()
        for x in order:
            # Check if adding x creates any conflict
            ok = True
            for a in selected:
                if x + a in selected:
                    ok = False
                    break
                if a + x in selected:
                    ok = False
                    break
            if ok:
                # Check x + x
                if x + x in selected and x + x != x:
                    ok = False
            if ok:
                selected.add(x)
        return selected

    def local_search(S, initial_set, max_iterations=5000):
        """Local search: try to add, remove, or swap elements to improve size."""
        selected = set(initial_set)
        S_set = set(S)
        best = len(selected)
        no_improve = 0
        iteration = 0

        while no_improve < 200 and iteration < max_iterations:
            if time.time() > DEADLINE - 5:
                break
            improved = False
            iteration += 1

            # Try adding any element from S not in selected
            for x in S:
                if x in selected:
                    continue
                # Check if x can be added
                ok = True
                for a in selected:
                    if x + a in selected:
                        ok = False
                        break
                if ok:
                    if x + x in selected:
                        ok = False
                if ok:
                    selected.add(x)
                    improved = True
                    break

            if improved:
                no_improve = 0
                best = max(best, len(selected))
                continue

            # Try removing one element to allow adding two others
            removed_improved = False
            for a in list(selected):
                if time.time() > DEADLINE - 5:
                    break
                selected.remove(a)
                # Try to add elements
                added_any = False
                for x in S:
                    if x in selected or x == a:
                        continue
                    ok = True
                    for b in selected:
                        if x + b in selected:
                            ok = False
                            break
                    if ok and x + x not in selected:
                        selected.add(x)
                        added_any = True
                        break
                if added_any and len(selected) > best:
                    removed_improved = True
                    best = len(selected)
                    no_improve = 0
                    break
                else:
                    selected.add(a)  # restore

            if removed_improved:
                continue

            # Try swapping: remove a, add b
            swap_improved = False
            for a in list(selected):
                if time.time() > DEADLINE - 5:
                    break
                for b in S:
                    if b in selected or b == a:
                        continue
                    selected.remove(a)
                    ok = True
                    for c in selected:
                        if b + c in selected:
                            ok = False
                            break
                    if ok and b + b not in selected:
                        selected.add(b)
                        if len(selected) > best:
                            best = len(selected)
                            no_improve = 0
                            swap_improved = True
                            break
                    else:
                        selected.add(a)  # restore
                    if swap_improved:
                        break
                if swap_improved:
                    break

            if not improved and not removed_improved and not swap_improved:
                no_improve += 1

        return selected

    def solve_instance(S):
        """Solve a single instance using multiple strategies."""
        S_set = set(S)
        n = len(S)
        best_solution = set()
        best_size = 0

        # Strategy 1: Greedy with different orderings
        orderings = [
            sorted(S),                          # ascending
            sorted(S, reverse=True),            # descending
            sorted(S, key=lambda x: abs(x)),    # by absolute value
            sorted(S, key=lambda x: abs(x), reverse=True),
        ]

        # Add some random orderings
        import random
        rng = random.Random(42)
        for _ in range(10):
            shuffled = S[:]
            rng.shuffle(shuffled)
            orderings.append(shuffled)

        for order in orderings:
            if time.time() > DEADLINE - 10:
                break
            sol = greedy_construct(S, order)
            if len(sol) > best_size:
                best_size = len(sol)
                best_solution = set(sol)

        # Strategy 2: Local search from best greedy solution
        if best_solution:
            if time.time() > DEADLINE - 15:
                return best_solution
            improved = local_search(S, best_solution, max_iterations=2000)
            if len(improved) > best_size:
                best_size = len(improved)
                best_solution = improved

        # Strategy 3: Multiple random restarts with local search
        for restart in range(20):
            if time.time() > DEADLINE - 10:
                break
            rng = random.Random(1000 + restart)
            shuffled = S[:]
            rng.shuffle(shuffled)
            sol = greedy_construct(S, shuffled)
            if len(sol) > best_size:
                best_size = len(sol)
                best_solution = set(sol)
            improved = local_search(S, sol, max_iterations=1000)
            if len(improved) > best_size:
                best_size = len(improved)
                best_solution = improved

        return best_solution

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > DEADLINE - 5:
            # Fallback: use odd numbers
            S = instance(size, seed)
            results.append([x for x in S if x % 2])
            continue
        S = instance(size, seed)
        sol = solve_instance(S)
        results.append(sorted(sol))

    return results
# EVOLVE-BLOCK-END
