"""Rewritten program for B2[2] set construction in {1..1000}."""
import time
import random

# EVOLVE-BLOCK-START

def _greedy_build(order, deadline):
    """Greedy construction following a given order of candidate integers.
    
    Returns (chosen, counts) where chosen is the list of selected integers
    and counts is a dict mapping sum -> count of representations.
    """
    chosen = []
    counts = {}
    n = len(order)
    
    for i, x in enumerate(order):
        if time.time() > deadline:
            break
        # Compute new sums
        new_sums = [x + y for y in chosen] + [2 * x]
        
        # Check if adding x keeps all counts <= 2
        valid = True
        for s in new_sums:
            c = counts.get(s, 0) + 1
            if c > 2:
                valid = False
                break
        
        if valid:
            chosen.append(x)
            for s in new_sums:
                counts[s] = counts.get(s, 0) + 1
    
    return chosen, counts


def _local_search_swap(chosen, counts, deadline, max_swaps=5000):
    """Improve the set by trying swap operations: remove one element, add another.
    
    Returns improved (chosen, counts).
    """
    chosen_set = set(chosen)
    improved = True
    swap_count = 0
    
    while improved and swap_count < max_swaps and time.time() < deadline:
        improved = False
        # Try removing each element and adding a better one
        for i, x in enumerate(chosen):
            if time.time() > deadline:
                break
            # Remove x
            new_counts = dict(counts)
            for y in chosen:
                if y == x:
                    continue
                s = x + y
                if s in new_counts:
                    new_counts[s] -= 1
                    if new_counts[s] == 0:
                        del new_counts[s]
            s = 2 * x
            if s in new_counts:
                new_counts[s] -= 1
                if new_counts[s] == 0:
                    del new_counts[s]
            
            # Try to add a candidate not in chosen
            best_add = None
            best_gain = 0
            for cand in range(1, 1001):
                if cand in chosen_set:
                    continue
                # Check if cand can be added
                valid = True
                for y in chosen:
                    if y == x:
                        continue
                    s = cand + y
                    if new_counts.get(s, 0) >= 2:
                        valid = False
                        break
                if valid:
                    s = 2 * cand
                    if new_counts.get(s, 0) >= 2:
                        valid = False
                if valid:
                    # We found a valid replacement; just take the first one
                    best_add = cand
                    break
            
            if best_add is not None:
                # Perform the swap
                chosen[i] = best_add
                chosen_set.discard(x)
                chosen_set.add(best_add)
                # Update counts
                for y in chosen:
                    if y == best_add:
                        continue
                    s = best_add + y
                    counts[s] = counts.get(s, 0) + 1
                s = 2 * best_add
                counts[s] = counts.get(s, 0) + 1
                improved = True
                swap_count += 1
                break  # Restart the loop after a swap
    
    return chosen, counts


def _try_orders(deadline):
    """Try multiple greedy orders and local search improvements.
    
    Returns the best (chosen, counts) found.
    """
    best_chosen = []
    best_counts = {}
    best_size = 0
    
    # Define several order strategies
    strategies = []
    
    # 1. Ascending
    strategies.append(list(range(1, 1001)))
    
    # 2. Descending
    strategies.append(list(range(1000, 0, -1)))
    
    # 3. Sparse (every 2nd)
    strategies.append(list(range(1, 1001, 2)))
    
    # 4. Random (seeded)
    rng = random.Random(42)
    rand_order = list(range(1, 1001))
    rng.shuffle(rand_order)
    strategies.append(rand_order)
    
    # 5. Another random
    rng2 = random.Random(123)
    rand_order2 = list(range(1, 1001))
    rng2.shuffle(rand_order2)
    strategies.append(rand_order2)
    
    # 6. Sparse descending
    strategies.append(list(range(1000, 0, -2)))
    
    # 7. Mix: evens then odds
    strategies.append(list(range(2, 1001, 2)) + list(range(1, 1001, 2)))
    
    # 8. Odds then evens
    strategies.append(list(range(1, 1001, 2)) + list(range(2, 1001, 2)))
    
    for order in strategies:
        if time.time() > deadline:
            break
        chosen, counts = _greedy_build(order, deadline)
        size = len(chosen)
        
        if size > best_size:
            best_size = size
            best_chosen = list(chosen)
            best_counts = dict(counts)
    
    # Local search on the best result
    if best_chosen and time.time() < deadline:
        best_chosen, best_counts = _local_search_swap(
            best_chosen, best_counts, deadline, max_swaps=3000
        )
        best_size = len(best_chosen)
    
    return best_chosen, best_counts


def construct():
    start_time = time.time()
    # Leave 15s safety margin before the 150s kill
    deadline = start_time + 135.0
    
    chosen, counts = _try_orders(deadline)
    
    return chosen

# EVOLVE-BLOCK-END
