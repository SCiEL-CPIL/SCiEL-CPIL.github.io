import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    denominators = list(range(2, 49))
    n = len(denominators)
    inv = [Fraction(1, d) for d in denominators]
    one = Fraction(1, 1)
    
    # Precompute all pairs of indices for swaps? 
    # Or just iterate.
    
    best_deficit = None
    best_subset = None
    best_sum = None

    def is_valid(subset, total):
        if not subset:
            return False
        return total < one

    def update_best(subset, total):
        nonlocal best_deficit, best_subset, best_sum
        if is_valid(subset, total):
            deficit = one - total
            if best_deficit is None or deficit < best_deficit:
                best_deficit = deficit
                best_subset = set(subset)
                best_sum = total
                return True
        return False

    def get_deficit(total):
        return one - total

    def local_search(subset, total, max_time=None):
        """
        Performs local search with 1-swap, 2-swap, and 3-swap.
        Returns (best_subset, best_total, best_deficit)
        """
        current_subset = set(subset)
        current_total = total
        current_deficit = one - current_total
        
        if not is_valid(current_subset, current_total):
            return None, None, None
            
        improved = True
        last_improvement_time = time.time()
        
        while improved:
            if max_time and time.time() > max_time:
                break
            improved = False
            
            # 1. Single Insertion (Add)
            # Try adding an element if it keeps sum < 1
            for i in range(n):
                if i not in current_subset:
                    new_total = current_total + inv[i]
                    if new_total < one:
                        new_deficit = one - new_total
                        if new_deficit < current_deficit:
                            current_subset.add(i)
                            current_total = new_total
                            current_deficit = new_deficit
                            improved = True
                            last_improvement_time = time.time()
            
            # 2. Single Removal (Remove) - only useful if we somehow got invalid, 
            # but we maintain validity. However, sometimes removing a large one 
            # allows adding two smaller ones. This is handled by swaps.
            
            # 3. Single Swap (Remove i, Add j)
            for i in list(current_subset):
                for j in range(n):
                    if j not in current_subset:
                        new_total = current_total - inv[i] + inv[j]
                        if new_total < one:
                            new_deficit = one - new_total
                            if new_deficit < current_deficit:
                                current_subset.remove(i)
                                current_subset.add(j)
                                current_total = new_total
                                current_deficit = new_deficit
                                improved = True
                                last_improvement_time = time.time()
                                # Note: We don't break, as another swap might be better, 
                                # but typically we restart the loop to re-evaluate all moves.
                                # For efficiency, we can continue, but the set changed.
                                # It's safer to restart the while loop structure for correctness 
                                # of "improved" flag logic in complex cases, but simple hill climbing 
                                # usually accepts first improvement or best improvement.
                                # Let's accept first improvement to keep it fast, then restart loop.
                                break
                if improved:
                    break
            
            # 4. Double Swap (Remove i1, i2; Add j1, j2)
            if not improved:
                sel_list = list(current_subset)
                unselected = [j for j in range(n) if j not in current_subset]
                
                # To keep it fast, we limit the number of pairs we check if the set is large
                # But n=47 is small enough.
                for a_idx in range(len(sel_list)):
                    if max_time and time.time() > max_time:
                        break
                    i1 = sel_list[a_idx]
                    for b_idx in range(a_idx + 1, len(sel_list)):
                        if max_time and time.time() > max_time:
                            break
                        i2 = sel_list[b_idx]
                        
                        # Pre-calculate removal sum
                        rem_sum = inv[i1] + inv[i2]
                        
                        for j1_idx in range(len(unselected)):
                            if max_time and time.time() > max_time:
                                break
                            j1 = unselected[j1_idx]
                            for j2_idx in range(j1_idx + 1, len(unselected)):
                                if max_time and time.time() > max_time:
                                    break
                                j2 = unselected[j2_idx]
                                
                                add_sum = inv[j1] + inv[j2]
                                new_total = current_total - rem_sum + add_sum
                                
                                if new_total < one:
                                    new_deficit = one - new_total
                                    if new_deficit < current_deficit:
                                        current_subset.remove(i1)
                                        current_subset.remove(i2)
                                        current_subset.add(j1)
                                        current_subset.add(j2)
                                        current_total = new_total
                                        current_deficit = new_deficit
                                        improved = True
                                        last_improvement_time = time.time()
                                        break
                            if improved:
                                break
                    if improved:
                        break
                        
            # 5. Triple Swap (Remove 3, Add 3) - Only if time allows
            if not improved and (max_time is None or time.time() < max_time - 5):
                sel_list = list(current_subset)
                unselected = [j for j in range(n) if j not in current_subset]
                
                # This is O(N^6) if done naively. We need to be careful.
                # We will sample a subset of triplets to keep it fast.
                # Or just skip if sets are large.
                if len(sel_list) >= 3 and len(unselected) >= 3:
                    # Randomly sample some triplets to avoid O(N^6) explosion
                    samples = 100 
                    for _ in range(samples):
                        if max_time and time.time() > max_time:
                            break
                        i1, i2, i3 = random.sample(sel_list, 3)
                        j1, j2, j3 = random.sample(unselected, 3)
                        
                        rem_sum = inv[i1] + inv[i2] + inv[i3]
                        add_sum = inv[j1] + inv[j2] + inv[j3]
                        new_total = current_total - rem_sum + add_sum
                        
                        if new_total < one:
                            new_deficit = one - new_total
                            if new_deficit < current_deficit:
                                current_subset.remove(i1)
                                current_subset.remove(i2)
                                current_subset.remove(i3)
                                current_subset.add(j1)
                                current_subset.add(j2)
                                current_subset.add(j3)
                                current_total = new_total
                                current_deficit = new_deficit
                                improved = True
                                last_improvement_time = time.time()
                                break

        return current_subset, current_total, current_deficit

    def greedy_seed():
        subset = set()
        total = Fraction(0, 1)
        # Sort by value descending (1/2, 1/3, ...) which is already the order
        for i in range(n):
            if total + inv[i] < one:
                subset.add(i)
                total += inv[i]
        return subset, total

    def random_seed():
        subset = set()
        total = Fraction(0, 1)
        # Randomly add elements
        indices = list(range(n))
        random.shuffle(indices)
        for i in indices:
            if total + inv[i] < one:
                subset.add(i)
                total += inv[i]
        if not subset:
            return None, None
        return subset, total

    def simulated_annealing(initial_subset, initial_total, max_time):
        current_subset = set(initial_subset)
        current_total = initial_total
        current_deficit = one - current_total
        
        if not is_valid(current_subset, current_total):
            return None, None, None
            
        best_subset = set(current_subset)
        best_total = current_total
        best_deficit = current_deficit
        
        # Annealing parameters
        T = 0.1  # Initial temperature (deficit scale)
        cooling = 0.99
        min_T = 1e-10
        
        start_sa = time.time()
        
        while T > min_T:
            if time.time() > max_time:
                break
                
            # Generate a neighbor
            move_type = random.random()
            neighbor_subset = set(current_subset)
            neighbor_total = current_total
            
            if move_type < 0.3:
                # 1-swap
                if current_subset:
                    i = random.choice(list(current_subset))
                    unselected = [j for j in range(n) if j not in current_subset]
                    if unselected:
                        j = random.choice(unselected)
                        neighbor_subset.remove(i)
                        neighbor_subset.add(j)
                        neighbor_total = current_total - inv[i] + inv[j]
                    else:
                        continue
            elif move_type < 0.6:
                # 2-swap
                if len(current_subset) >= 1:
                    unselected = [j for j in range(n) if j not in current_subset]
                    if len(unselected) >= 1:
                        # Remove 1, Add 1 is 1-swap. Let's do Remove 2, Add 2 or Remove 1 Add 2?
                        # Standard 2-swap: Remove 2, Add 2.
                        if len(current_subset) >= 2 and len(unselected) >= 2:
                            i1, i2 = random.sample(list(current_subset), 2)
                            j1, j2 = random.sample(unselected, 2)
                            neighbor_subset.remove(i1)
                            neighbor_subset.remove(i2)
                            neighbor_subset.add(j1)
                            neighbor_subset.add(j2)
                            neighbor_total = current_total - inv[i1] - inv[i2] + inv[j1] + inv[j2]
                        else:
                            continue
                    else:
                        continue
            elif move_type < 0.8:
                # Insert
                unselected = [j for j in range(n) if j not in current_subset]
                if unselected:
                    j = random.choice(unselected)
                    neighbor_subset.add(j)
                    neighbor_total = current_total + inv[j]
                else:
                    continue
            else:
                # Remove
                if current_subset:
                    i = random.choice(list(current_subset))
                    neighbor_subset.remove(i)
                    neighbor_total = current_total - inv[i]
                else:
                    continue

            # Check validity
            if not is_valid(neighbor_subset, neighbor_total):
                # Invalid neighbor (sum >= 1 or empty)
                # We can still accept if it's "close" but we must return valid at the end.
                # Better to just reject invalid moves in SA to stay in feasible space?
                # The problem requires STRICTLY BELOW 1.
                # If we allow invalid states in SA, we might get stuck.
                # Let's reject invalid states.
                continue
            
            neighbor_deficit = one - neighbor_total
            delta_deficit = neighbor_deficit - current_deficit
            
            # Acceptance probability
            if delta_deficit < 0:
                # Improvement
                accept = True
            else:
                # Worse, accept with probability exp(-delta/T)
                # delta is positive (deficit increased)
                if T > 0:
                    prob = random.expovariate(1.0) # No, use exp directly
                    import math
                    p = math.exp(-delta_deficit / T)
                    accept = random.random() < p
                else:
                    accept = False
            
            if accept:
                current_subset = neighbor_subset
                current_total = neighbor_total
                current_deficit = neighbor_deficit
                
                if current_deficit < best_deficit:
                    best_subset = set(current_subset)
                    best_total = current_total
                    best_deficit = current_deficit
            
            T *= cooling
        
        return best_subset, best_total, best_deficit

    # 1. Greedy Start
    g_subset, g_total = greedy_seed()
    if g_subset:
        update_best(g_subset, g_total)
        # Local search on greedy
        ls_subset, ls_total, ls_deficit = local_search(g_subset, g_total, max_time=start_time + 5)
        if ls_subset:
            update_best(ls_subset, ls_total)

    # 2. Random Restarts with SA
    # Allocate remaining time to SA runs
    # Each SA run + local search should take ~1-2 seconds.
    # We have ~50 seconds left. Let's do 20-30 runs.
    
    random.seed(42) # Fixed seed for reproducibility in eval? Or random?
    # The prompt doesn't specify determinism, but fixed seed is safer for debugging.
    # However, different seeds might find different optima. 
    # Let's use a fixed seed for the first few, then random.
    
    num_restarts = 25
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for k in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Alternate between random seed and perturbed greedy
        if k % 2 == 0:
            r_subset, r_total = random_seed()
        else:
            # Perturb greedy: remove a random few, then re-add greedily?
            # Or just start from greedy and do a big jump
            r_subset, r_total = greedy_seed()
            if r_subset:
                # Remove 2 random elements to create a "hole"
                if len(r_subset) > 2:
                    to_remove = random.sample(list(r_subset), 2)
                    for i in to_remove:
                        r_subset.remove(i)
                        r_total -= inv[i]
        
        if not r_subset:
            continue
            
        # Run SA
        sa_max_time = time.time() + time_per_restart * 0.8
        sa_subset, sa_total, sa_deficit = simulated_annealing(r_subset, r_total, sa_max_time)
        
        if sa_subset:
            update_best(sa_subset, sa_total)
            
            # Final local search on the SA result
            ls_max_time = time.time() + time_per_restart * 0.2
            ls_subset, ls_total, ls_deficit = local_search(sa_subset, sa_total, max_time=ls_max_time)
            if ls_subset:
                update_best(ls_subset, ls_total)

    # Final cleanup: ensure we return the best found
    if best_subset is None:
        # Fallback to greedy if something went wrong
        g_subset, g_total = greedy_seed()
        return list(g_subset)
        
    return list(best_subset)

# EVOLVE-BLOCK-END
