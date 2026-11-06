import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    N = 256
    
    # Precompute all 3-APs and 4-APs
    # 3-AP: (a, a+d, a+2d)
    # 4-AP: (a, a+d, a+2d, a+3d)
    
    all_3aps = []
    all_4aps = []
    
    for a in range(1, N + 1):
        max_d_3 = (N - a) // 2
        for d in range(1, max_d_3 + 1):
            b = a + d
            c = a + 2 * d
            all_3aps.append((a, b, c))
            
            # Check for 4-AP
            if a + 3 * d <= N:
                e = a + 3 * d
                all_4aps.append((a, b, c, e))
    
    num_3aps = len(all_3aps)
    num_4aps = len(all_4aps)
    
    # Build adjacency lists
    # adj3[x] = list of indices of 3-APs containing x
    # adj4[x] = list of indices of 4-APs containing x
    adj3 = [[] for _ in range(N + 1)]
    adj4 = [[] for _ in range(N + 1)]
    
    for i, (a, b, c) in enumerate(all_3aps):
        adj3[a].append(i)
        adj3[b].append(i)
        adj3[c].append(i)
        
    for i, (a, b, c, e) in enumerate(all_4aps):
        adj4[a].append(i)
        adj4[b].append(i)
        adj4[c].append(i)
        adj4[e].append(i)
        
    # State representation: boolean array
    in_set = np.zeros(N + 1, dtype=bool)
    
    def count_3aps():
        """Count total 3-APs in current set."""
        count = 0
        for a, b, c in all_3aps:
            if in_set[a] and in_set[b] and in_set[c]:
                count += 1
        return count
        
    def is_valid_addition(x):
        """Check if adding x creates a 4-AP."""
        if in_set[x]:
            return False
        for idx in adj4[x]:
            a, b, c, e = all_4aps[idx]
            others = [a, b, c, e]
            others.remove(x)
            if all(in_set[o] for o in others):
                return False
        return True
        
    def gain_3aps_add(x):
        """Calculate gain in 3-APs if x is added."""
        if in_set[x]:
            return 0
        gain = 0
        for idx in adj3[x]:
            a, b, c = all_3aps[idx]
            others = [a, b, c]
            others.remove(x)
            if all(in_set[o] for o in others):
                gain += 1
        return gain
        
    def loss_3aps_remove(x):
        """Calculate loss in 3-APs if x is removed."""
        if not in_set[x]:
            return 0
        loss = 0
        for idx in adj3[x]:
            a, b, c = all_3aps[idx]
            others = [a, b, c]
            others.remove(x)
            if all(in_set[o] for o in others):
                loss += 1
        return loss
        
    def greedy_construct():
        """Greedy construction: add elements that maximize 3-AP gain without creating 4-APs."""
        current_gain = 0
        # To speed up, we can just iterate through all possible elements and add the best one
        # But N=256 is small, so we can do a simple loop.
        # Actually, a better greedy is: for each x in 1..N, if valid and gain>0, add it.
        # However, order matters. Let's do a steepest ascent greedy.
        
        # Initialize empty
        in_set[:] = False
        current_3ap_count = 0
        
        remaining = list(range(1, N + 1))
        while remaining:
            if time.time() > deadline:
                break
            best_x = -1
            best_gain = -1
            for x in remaining:
                if not is_valid_addition(x):
                    continue
                g = gain_3aps_add(x)
                if g > best_gain:
                    best_gain = g
                    best_x = x
                    
            if best_x == -1:
                break
                
            in_set[best_x] = True
            current_3ap_count += best_gain
            remaining.remove(best_x)
            
        return current_3ap_count
        
    def local_search(initial_set):
        """Steepest ascent local search with swaps."""
        # Copy initial state
        in_set[:] = np.array(initial_set, dtype=bool)
        current_3ap_count = count_3aps()
        best_3ap_count = current_3ap_count
        best_set = set(i for i in range(1, N+1) if in_set[i])
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
                
            # 1. Try adding elements
            best_add_x = -1
            best_add_gain = -1
            for x in range(1, N + 1):
                if in_set[x]:
                    continue
                if not is_valid_addition(x):
                    continue
                g = gain_3aps_add(x)
                if g > best_add_gain:
                    best_add_gain = g
                    best_add_x = x
                    
            if best_add_x != -1 and best_add_gain > 0:
                in_set[best_add_x] = True
                current_3ap_count += best_add_gain
                if current_3ap_count > best_3ap_count:
                    best_3ap_count = current_3ap_count
                    best_set = set(i for i in range(1, N+1) if in_set[i])
                improved = True
                continue # Restart loop as state changed significantly
                
            # 2. Try swaps (remove one, add one)
            # To optimize, we can precompute the best add for each removal?
            # Or just iterate.
            best_swap_remove = -1
            best_swap_add = -1
            best_swap_net = -1
            
            # List of elements currently in set
            current_elements = [i for i in range(1, N + 1) if in_set[i]]
            
            for x_rem in current_elements:
                if time.time() > deadline:
                    break
                # Temporarily remove
                in_set[x_rem] = False
                loss = loss_3aps_remove(x_rem)
                current_3ap_count -= loss
                
                # Find best add
                best_add_x_for_rem = -1
                best_add_gain_for_rem = -1
                for x_add in range(1, N + 1):
                    if in_set[x_add]:
                        continue
                    if not is_valid_addition(x_add):
                        continue
                    g = gain_3aps_add(x_add)
                    if g > best_add_gain_for_rem:
                        best_add_gain_for_rem = g
                        best_add_x_for_rem = x_add
                        
                if best_add_x_for_rem != -1:
                    net = best_add_gain_for_rem - loss
                    if net > best_swap_net:
                        best_swap_net = net
                        best_swap_remove = x_rem
                        best_swap_add = best_add_x_for_rem
                        best_swap_add_gain = best_add_gain_for_rem
                        
                # Revert removal
                in_set[x_rem] = True
                current_3ap_count += loss
                
            if best_swap_remove != -1 and best_swap_net > 0:
                in_set[best_swap_remove] = False
                in_set[best_swap_add] = True
                current_3ap_count += best_swap_add_gain - (loss_3aps_remove(best_swap_remove)) # Wait, loss was calculated before revert
                # Let's recalculate to be safe or track carefully
                # Actually, loss_3aps_remove(best_swap_remove) is the loss.
                # So current_3ap_count = current_3ap_count - loss + gain
                # We already reverted, so current_3ap_count is back to original.
                # Let's just apply the change directly.
                in_set[best_swap_remove] = False
                in_set[best_swap_add] = True
                # Recalculate count is expensive, but we know the delta.
                # Delta = gain_add - loss_rem
                # We stored best_swap_net which is gain - loss.
                current_3ap_count += best_swap_net
                
                if current_3ap_count > best_3ap_count:
                    best_3ap_count = current_3ap_count
                    best_set = set(i for i in range(1, N+1) if in_set[i])
                improved = True
                
        return best_set, best_3ap_count

    # 1. Initial Greedy Construction
    # We run greedy a few times with different random orders to get diverse starting points
    best_overall_set = set()
    best_overall_score = 0
    
    # Run greedy once to get a baseline
    greedy_construct()
    base_set, base_score = local_search(set(i for i in range(1, N+1) if in_set[i]))
    best_overall_set = base_set
    best_overall_score = base_score
    
    # 2. Iterated Local Search
    # Perturb the best solution and run local search again
    
    while time.time() < deadline:
        # Perturb: randomly remove k elements from best_overall_set
        k = random.randint(2, 5)
        perturbed_set = set(best_overall_set)
        
        # Ensure we don't remove too many
        if len(perturbed_set) > k:
            elements_to_remove = random.sample(list(perturbed_set), k)
            for e in elements_to_remove:
                perturbed_set.remove(e)
        else:
            # If set is small, just start from scratch or random
            perturbed_set = set(random.sample(range(1, N+1), 10))
            
        # Run local search on perturbed set
        # Note: local_search modifies the global in_set, so we need to be careful
        # Our local_search function takes initial_set and copies it to in_set
        p_set, p_score = local_search(perturbed_set)
        
        if p_score > best_overall_score:
            best_overall_score = p_score
            best_overall_set = p_set
            
    # Final validation and return
    # Convert set to sorted list
    return sorted(best_overall_set)

