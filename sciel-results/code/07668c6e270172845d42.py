# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    Uses randomized greedy construction with local search (swap-based) and multiple restarts.
    """
    m = 205
    deadline = time.time() + 580  # safety margin before 600s limit

    # Precompute forbidden differences: nonzero squares mod 205
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)

    # Precompute conflict matrix: conflict[i][j] = True if |i-j| mod 205 is in forbidden
    # Since the relation is symmetric, we can use a 2D boolean array
    conflict = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            d = (i - j) % m
            if d in forbidden:
                conflict[i][j] = True

    def is_valid_addition(selected_set, candidate):
        """Check if candidate can be added to selected_set without conflict."""
        for s in selected_set:
            if conflict[candidate][s]:
                return False
        return True

    def greedy_init(order):
        """Build a set using greedy selection in the given order."""
        selected = set()
        for candidate in order:
            if is_valid_addition(selected, candidate):
                selected.add(candidate)
        return selected

    def local_search(selected_set):
        """
        Try to improve the set by swapping elements.
        For each pair (in_set, out_set), try removing in_set and adding out_set.
        Also try pure additions.
        Returns improved set.
        """
        selected = set(selected_set)
        improved = True
        while improved:
            improved = False
            # Try additions first
            for candidate in range(m):
                if candidate not in selected:
                    if is_valid_addition(selected, candidate):
                        selected.add(candidate)
                        improved = True
            # Try swaps
            in_list = list(selected)
            for in_elem in in_list:
                for out_elem in range(m):
                    if out_elem in selected:
                        continue
                    # Try removing in_elem and adding out_elem
                    new_selected = selected - {in_elem}
                    if is_valid_addition(new_selected, out_elem):
                        selected = new_selected
                        selected.add(out_elem)
                        improved = True
                        break
                if improved:
                    break
        return selected

    def local_search_v2(selected_set):
        """
        More aggressive local search: try all possible single swaps and additions.
        Use numpy for faster conflict checking.
        """
        selected = list(selected_set)
        selected_set = set(selected)
        improved = True
        while improved:
            improved = False
            # Try additions
            for candidate in range(m):
                if candidate not in selected_set:
                    ok = True
                    for s in selected_set:
                        if conflict[candidate][s]:
                            ok = False
                            break
                    if ok:
                        selected_set.add(candidate)
                        improved = True
            # Try swaps: remove one, add one
            in_list = list(selected_set)
            for in_elem in in_list:
                for out_elem in range(m):
                    if out_elem in selected_set:
                        continue
                    new_set = selected_set - {in_elem}
                    ok = True
                    for s in new_set:
                        if conflict[out_elem][s]:
                            ok = False
                            break
                    if ok:
                        selected_set = new_set
                        selected_set.add(out_elem)
                        improved = True
                        break
                if improved:
                    break
        return selected_set

    best_set = set()
    best_size = 0
    rng = random.Random(42)

    # Multiple restarts with different random orderings
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        # Create a random ordering
        order = list(range(m))
        rng.shuffle(order)
        
        # Greedy initialization
        current_set = greedy_init(order)
        current_size = len(current_set)
        
        # Local search
        current_set = local_search_v2(current_set)
        current_size = len(current_set)
        
        if current_size > best_size:
            best_size = current_size
            best_set = set(current_set)
        
        # If we've found a very large set, we can be more patient
        # Otherwise, keep trying
        if best_size >= 14:
            # Do a few more targeted searches
            for _ in range(5):
                if time.time() >= deadline:
                    break
                order = list(range(m))
                rng.shuffle(order)
                current_set = greedy_init(order)
                current_set = local_search_v2(current_set)
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
            break

    # Final polish: try to add any remaining elements
    final_set = set(best_set)
    for candidate in range(m):
        if candidate not in final_set:
            if is_valid_addition(final_set, candidate):
                final_set.add(candidate)

    return np.array(sorted(final_set), dtype=np.int64)


# EVOLVE-BLOCK-END
