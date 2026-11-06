"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)
    unit_fracs = [Fraction(1, d) for d in denominators]

    # Greedy baseline: pick largest unit fractions that keep sum < 1
    greedy_selected = []
    greedy_total = Fraction(0)
    # Sort by reciprocal descending (i.e., denominator ascending)
    for i in range(n):
        if greedy_total + unit_fracs[i] < 1:
            greedy_selected.append(i)
            greedy_total += unit_fracs[i]

    best_indices = greedy_selected
    best_deficit = 1 - greedy_total

    if best_deficit <= 0:
        return best_indices

    # Try local search: swap one element in/out to reduce deficit
    # Also try adding elements that fit
    improved = True
    while improved:
        improved = False
        current_set = set(best_indices)
        current_total = sum(unit_fracs[i] for i in best_indices)
        current_deficit = 1 - current_total

        # Try adding any element not in set
        for i in range(n):
            if i not in current_set:
                new_total = current_total + unit_fracs[i]
                if new_total < 1:
                    new_deficit = 1 - new_total
                    if new_deficit < current_deficit:
                        best_indices.append(i)
                        current_set.add(i)
                        current_total = new_total
                        current_deficit = new_deficit
                        improved = True
                        break

        if improved:
            continue

        # Try swapping: remove one, add another
        for i_out in list(best_indices):
            total_without = current_total - unit_fracs[i_out]
            for i_in in range(n):
                if i_in in current_set:
                    continue
                new_total = total_without + unit_fracs[i_in]
                if new_total < 1:
                    new_deficit = 1 - new_total
                    if new_deficit < current_deficit:
                        best_indices.remove(i_out)
                        best_indices.append(i_in)
                        current_set.remove(i_out)
                        current_set.add(i_in)
                        current_total = new_total
                        current_deficit = new_deficit
                        improved = True
                        break
            if improved or time.time() > deadline:
                break

        if time.time() > deadline:
            break

    # Try 2-swap moves
    if time.time() < deadline:
        current_set = set(best_indices)
        current_total = sum(unit_fracs[i] for i in best_indices)
        current_deficit = 1 - current_total
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i_out1, i_out2 in combinations(sorted(best_indices), 2):
                total_without = current_total - unit_fracs[i_out1] - unit_fracs[i_out2]
                for i_in1 in range(n):
                    if i_in1 in current_set:
                        continue
                    for i_in2 in range(i_in1 + 1, n):
                        if i_in2 in current_set:
                            continue
                        new_total = total_without + unit_fracs[i_in1] + unit_fracs[i_in2]
                        if new_total < 1:
                            new_deficit = 1 - new_total
                            if new_deficit < current_deficit:
                                best_indices.remove(i_out1)
                                best_indices.remove(i_out2)
                                best_indices.append(i_in1)
                                best_indices.append(i_in2)
                                current_set.discard(i_out1)
                                current_set.discard(i_out2)
                                current_set.add(i_in1)
                                current_set.add(i_in2)
                                current_total = new_total
                                current_deficit = new_deficit
                                improved = True
                                break
                    if improved or time.time() > deadline:
                        break
                if improved or time.time() > deadline:
                    break
            if time.time() > deadline:
                break

    # Try random restarts with local search if time permits
    if time.time() < deadline - 5.0:
        import random
        random.seed(42)
        for _ in range(5000):
            if time.time() > deadline:
                break
            # Random subset
            subset_size = random.randint(3, 12)
            candidate = set(random.sample(range(n), subset_size))
            total = sum(unit_fracs[i] for i in candidate)
            if total >= 1:
                continue
            deficit = 1 - total
            if deficit >= current_deficit:
                continue
            # Local improvement
            changed = True
            while changed and time.time() < deadline:
                changed = False
                # Add
                for i in range(n):
                    if i in candidate:
                        continue
                    new_total = total + unit_fracs[i]
                    if new_total < 1:
                        new_deficit = 1 - new_total
                        if new_deficit < deficit:
                            candidate.add(i)
                            total = new_total
                            deficit = new_deficit
                            changed = True
                            break
                if changed:
                    continue
                # Swap
                for i_out in list(candidate):
                    total_without = total - unit_fracs[i_out]
                    for i_in in range(n):
                        if i_in in candidate:
                            continue
                        new_total = total_without + unit_fracs[i_in]
                        if new_total < 1:
                            new_deficit = 1 - new_total
                            if new_deficit < deficit:
                                candidate.discard(i_out)
                                candidate.add(i_in)
                                total = new_total
                                deficit = new_deficit
                                changed = True
                                break
                    if changed:
                        break
            if deficit < current_deficit:
                best_indices = list(candidate)
                current_deficit = deficit

    # Final validation
    final_total = sum(unit_fracs[i] for i in best_indices)
    if final_total >= 1 or not best_indices:
        # Fallback to greedy
        best_indices = greedy_selected
        final_total = sum(unit_fracs[i] for i in best_indices)

    return best_indices
# EVOLVE-BLOCK-END
