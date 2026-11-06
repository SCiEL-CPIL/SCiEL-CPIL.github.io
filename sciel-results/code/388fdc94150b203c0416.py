"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    N = 256

    # Greedy construction: add numbers in order of largest reciprocal (i.e., smallest numbers first)
    # but check for 3-AP constraint
    def is_valid_addition(selected_set, x):
        """Check if x can be added to selected_set without creating a 3-AP."""
        for a in selected_set:
            # Check if a, x, c form a 3-AP: a, x, c => 2x = a + c => c = 2x - a
            c = 2 * x - a
            if c in selected_set and c != a:
                return False
            # Check if b, a, x form a 3-AP: b, a, x => 2a = b + x => b = 2a - x
            b = 2 * a - x
            if b in selected_set and b != a:
                return False
        return True

    # Start with greedy: try to add all numbers from 1 to 256, prioritizing small numbers (large reciprocals)
    selected_set = set()
    for x in range(1, N + 1):
        if is_valid_addition(selected_set, x):
            selected_set.add(x)

    def get_score(selected):
        """Compute exact sum of reciprocals."""
        total = Fraction(0, 1)
        for x in selected:
            total += Fraction(1, x)
        return float(total)

    def get_exact_score(selected):
        """Compute exact Fraction sum of reciprocals."""
        total = Fraction(0, 1)
        for x in selected:
            total += Fraction(1, x)
        return total

    best_set = set(selected_set)
    best_score = get_score(best_set)

    # Local search: try swaps to improve the reciprocal sum
    # A swap removes one element and adds another
    def try_swap(selected_set, remove, add):
        """Try removing 'remove' and adding 'add'. Returns new set if valid, else None."""
        if remove == add:
            return None
        if remove not in selected_set:
            return None
        if add in selected_set:
            return None
        # Check if removing 'remove' and adding 'add' creates no 3-AP
        new_set = selected_set - {remove}
        # Check if 'add' can be added to new_set
        for a in new_set:
            c = 2 * add - a
            if c in new_set and c != a:
                return None
            b = 2 * a - add
            if b in new_set and b != a:
                return None
        return new_set | {add}

    def local_search(selected_set, time_limit):
        """Perform local search with swaps."""
        current_set = set(selected_set)
        current_score = get_score(current_set)
        improved = True
        while improved:
            improved = False
            if time.time() > time_limit:
                break
            # Try all possible swaps
            for remove in list(current_set):
                if time.time() > time_limit:
                    break
                for add in range(1, N + 1):
                    if add in current_set:
                        continue
                    new_set = try_swap(current_set, remove, add)
                    if new_set is not None:
                        new_score = get_score(new_set)
                        if new_score > current_score + 1e-15:
                            current_set = new_set
                            current_score = new_score
                            improved = True
                            break
                if improved:
                    break
        return current_set, current_score

    # Run local search with a time budget
    search_deadline = deadline - 5.0  # leave 5 seconds for final verification
    best_set, best_score = local_search(best_set, search_deadline)

    # Final verification
    if time.time() < deadline:
        # Verify the result is valid
        valid = True
        for x in best_set:
            for a in best_set:
                if a == x:
                    continue
                c = 2 * x - a
                if c in best_set and c != a and c != x:
                    valid = False
                    break
                b = 2 * a - x
                if b in best_set and b != a and b != x:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            # Fallback to greedy result
            best_set = set()
            for x in range(1, N + 1):
                if is_valid_addition(best_set, x):
                    best_set.add(x)

    return sorted(best_set)
# EVOLVE-BLOCK-END
