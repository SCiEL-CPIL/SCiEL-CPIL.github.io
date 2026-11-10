import time
import random
import numpy as np

def construct():
    N = 1000
    max_sum = 2 * N
    best_set = []
    best_len = 0
    
    start_time = time.time()
    timeout = 110.0 # Safety margin for 120s limit

    def get_counts(chosen):
        counts = [0] * (max_sum + 1)
        n = len(chosen)
        for i in range(n):
            x = chosen[i]
            counts[2 * x] += 1
            for j in range(i + 1, n):
                y = chosen[j]
                counts[x + y] += 1
        return counts

    def can_add(chosen, counts, x):
        # Check 2x
        if counts[2 * x] >= 2:
            return False
        # Check x + y for y in chosen
        for y in chosen:
            if counts[x + y] >= 2:
                return False
        return True

    def add_element(chosen, counts, x):
        for y in chosen:
            counts[x + y] += 1
        counts[2 * x] += 1
        chosen.append(x)

    def remove_element(chosen, counts, x):
        # Remove x from chosen
        chosen.remove(x)
        # Update counts
        counts[2 * x] -= 1
        for y in chosen:
            counts[x + y] -= 1

    def local_search(chosen, counts, time_limit, start_time):
        """
        Iterative local search trying to add elements, and if stuck, trying swaps.
        """
        improved = True
        while improved:
            if time.time() - start_time > time_limit:
                break
            improved = False
            
            # 1. Try to add any element
            # Shuffle candidates to avoid bias
            candidates = [x for x in range(1, N + 1) if x not in set(chosen)]
            random.shuffle(candidates)
            
            added = False
            for x in candidates:
                if can_add(chosen, counts, x):
                    add_element(chosen, counts, x)
                    improved = True
                    added = True
                    break
            
            if added:
                continue
            
            # 2. If no add, try swaps
            # To be efficient, we only try a few swaps per round
            # Or we try to find a swap that increases the number of available candidates
            
            # Simple swap: remove random x, try to add any y
            # This might not improve size, but changes state.
            # Better: remove x, check if we can add y. If yes, swap.
            
            # We will try up to 10 random swaps
            swap_done = False
            for _ in range(10):
                if not chosen:
                    break
                x_to_remove = random.choice(chosen)
                remove_element(chosen, counts, x_to_remove)
                
                # Try to add any element (excluding x_to_remove? No, we can add it back, but that's a no-op swap)
                # We want to add a *different* element or multiple?
                # If we add x_to_remove back, we are back to start.
                # So we should try to add y != x_to_remove.
                
                # Find if any y can be added
                added_y = None
                # Check all y not in chosen
                # Optimization: just check a random subset or all?
                # All is safer.
                for y in range(1, N + 1):
                    if y == x_to_remove:
                        continue
                    if y in chosen:
                        continue
                    if can_add(chosen, counts, y):
                        add_element(chosen, counts, y)
                        added_y = y
                        break
                        
                if added_y is None:
                    # Put x_to_remove back
                    add_element(chosen, counts, x_to_remove)
                else:
                    # We swapped x_to_remove with added_y. Size is same.
                    # But the set changed. We might be able to add more in the next round.
                    improved = True
                    swap_done = True
                    break # Break out of swap loop, go to next while iteration
            
            if not improved:
                # Stuck in local optimum.
                # Perturbation: Remove a few random elements
                if len(chosen) > 5:
                    k = min(3, len(chosen))
                    for _ in range(k):
                        if chosen:
                            x_to_remove = random.choice(chosen)
                            remove_element(chosen, counts, x_to_remove)
                    # The while loop will restart and try to add.
                    # We don't set improved=True here because we didn't improve size,
                    # but we changed state. We need to force the loop to continue?
                    # The current logic: if not improved, loop ends.
                    # So we should set improved=True to continue searching from the perturbed state.
                    improved = True
                    # But we must ensure we don't loop infinitely on the same perturbation.
                    # Since the state is random, it should be fine.

        return chosen, counts

    # Strategy: Multiple restarts
    # 1. Greedy with squares
    # 2. Random restarts with local search

    # Helper for greedy
    def greedy_init(order='random', seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
            
        if order == 'squares':
            squares = [i*i for i in range(1, int(np.sqrt(N)) + 1) if i*i <= N]
            others = [x for x in range(1, N + 1) if x not in squares]
            rng.shuffle(others)
            candidates = squares + others
        else:
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
            
        chosen = []
        counts = [0] * (max_sum + 1)
        
        for x in candidates:
            if can_add(chosen, counts, x):
                add_element(chosen, counts, x)
                
        return chosen, counts

    # Run 1: Squares
    current_set, current_counts = greedy_init('squares', 42)
    current_set, current_counts = local_search(current_set, current_counts, timeout - (time.time() - start_time), start_time)
    if len(current_set) > best_len:
        best_len = len(current_set)
        best_set = current_set[:]
        
    # Run 2: Random restarts
    while time.time() - start_time < timeout - 2:
        current_set, current_counts = greedy_init('random', None)
        current_set, current_counts = local_search(current_set, current_counts, timeout - (time.time() - start_time), start_time)
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
            
    return best_set
