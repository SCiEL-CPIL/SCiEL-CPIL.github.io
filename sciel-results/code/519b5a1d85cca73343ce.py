"""Combined program for the suite task."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    """
    Find a large subset of [1, 128] with no three distinct elements whose
    three pairwise gcds are equal.
    """
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    N = 128
    
    # Precompute GCD matrix
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    def has_bad_triple_with_new(selected, new_elem):
        """
        Check if adding new_elem creates a bad triple with any pair in selected.
        """
        # Optimization: Precompute gcd of new_elem with all in selected
        g_new = {}
        for a in selected:
            g_new[a] = gcd_mat[new_elem][a]
            
        # Check all pairs in selected
        n = len(selected)
        for i in range(n):
            a = selected[i]
            g_an = g_new[a]
            for j in range(i + 1, n):
                b = selected[j]
                g_bn = g_new[b]
                if g_an == g_bn and g_an == gcd_mat[a][b]:
                    return True
        return False

    def is_valid_set(S):
        """
        Verify the entire set is valid. Used for final check or debugging.
        """
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                g_ab = gcd_mat[a][b]
                for k in range(j + 1, n):
                    c = S_list[k]
                    if gcd_mat[a][c] == g_ab and gcd_mat[b][c] == g_ab:
                        return False
        return True

    def greedy_build(order):
        """Build set greedily following given order."""
        selected = []
        selected_set = set()
        for x in order:
            if x not in selected_set:
                if not has_bad_triple_with_new(selected, x):
                    selected.append(x)
                    selected_set.add(x)
        return selected

    def local_search(set_list, max_iters=2000):
        """
        Improve the set via additions and swaps.
        """
        set_list = list(set_list)
        set_list_set = set(set_list)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # 1. Try adding elements
            # Shuffle candidates to avoid bias
            candidates = [x for x in range(1, N + 1) if x not in set_list_set]
            random.shuffle(candidates)
            for x in candidates:
                if not has_bad_triple_with_new(set_list, x):
                    set_list.append(x)
                    set_list_set.add(x)
                    improved = True
                    break
            
            # 2. Try swaps: remove one, add another
            if not improved and time.time() < deadline:
                # Try a limited number of random swaps to save time
                swaps_tried = 0
                while swaps_tried < 20 and time.time() < deadline:
                    swaps_tried += 1
                    if not set_list: continue
                    i = random.randint(0, len(set_list) - 1)
                    removed = set_list[i]
                    candidate_set = set_list[:i] + set_list[i+1:]
                    
                    # Find a valid x to add
                    found = False
                    for x in range(1, N + 1):
                        if x in set_list_set:
                            continue
                        if not has_bad_triple_with_new(candidate_set, x):
                            set_list = candidate_set + [x]
                            set_list_set.remove(removed)
                            set_list_set.add(x)
                            improved = True
                            found = True
                            break
                    if found:
                        break

        return set_list

    def perturb(set_list, k):
        """
        Randomly remove k elements and greedily re-add them or others.
        """
        set_list = list(set_list)
        set_list_set = set(set_list)
        
        # Remove k random elements
        to_remove = random.sample(set_list, min(k, len(set_list)))
        for r in to_remove:
            set_list.remove(r)
            set_list_set.remove(r)
        
        # Try to add back removed elements or new ones greedily
        candidates = list(to_remove) + [x for x in range(1, N + 1) if x not in set_list_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if x not in set_list_set and not has_bad_triple_with_new(set_list, x):
                set_list.append(x)
                set_list_set.add(x)
                
        return set_list

    # Main search loop
    best_set = []
    
    # Initial greedy builds with different orderings
    base_order = list(range(1, N + 1))
    
    # Try a few different initial orders
    orders = [
        base_order,
        list(reversed(base_order)),
        list(range(1, N + 1, 2)),  # odds
        list(range(2, N + 1, 2)),  # evens
    ]
    
    random.shuffle(orders)
    
    for order in orders:
        if time.time() > deadline:
            break
        current = greedy_build(order)
        current = local_search(current)
        if len(current) > len(best_set):
            best_set = current
    
    # If we still have time, do more random restarts
    restarts = 0
    while time.time() < deadline - 5:
        random.shuffle(base_order)
        current = greedy_build(base_order)
        current = local_search(current)
        if len(current) > len(best_set):
            best_set = current
        restarts += 1
        
        # Occasional perturbation of the best set
        if restarts % 3 == 0 and best_set:
            perturbed = perturb(best_set, k=random.randint(2, 5))
            perturbed = local_search(perturbed)
            if len(perturbed) > len(best_set):
                best_set = perturbed

    # Final verification
    if not is_valid_set(best_set):
        # Fallback to a simple valid set if something went wrong
        best_set = greedy_build(base_order)
        if not is_valid_set(best_set):
            # Extremely unlikely fallback
            best_set = [1]

    return best_set
# EVOLVE-BLOCK-END
