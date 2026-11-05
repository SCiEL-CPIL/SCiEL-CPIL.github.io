import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    N = 1000
    max_sum = 2 * N
    
    def greedy_init(seed=None):
        if seed is not None:
            random.seed(seed)
        A = []
        in_A = set()
        sum_counts = [0] * (max_sum + 1)
        
        def try_add(x):
            if x in in_A:
                return False
            if sum_counts[2*x] >= 2:
                return False
            for y in A:
                if sum_counts[x+y] >= 2:
                    return False
            in_A.add(x)
            A.append(x)
            sum_counts[2*x] += 1
            for y in A[:-1]:
                sum_counts[x+y] += 1
            return True
        
        for x in range(1, N + 1):
            try_add(x)
        return A, in_A, sum_counts

    def local_search(A, in_A, sum_counts, time_limit):
        best_set = A[:]
        best_size = len(A)
        
        # Convert to numpy for faster operations
        A_np = np.array(A, dtype=np.int32)
        
        def get_candidates():
            all_nums = np.arange(1, N+1, dtype=np.int32)
            A_set = np.array(sorted(in_A), dtype=np.int32)
            return np.setdiff1d(all_nums, A_set)
        
        # Simulated annealing parameters
        temperature = 50.0
        cooling_rate = 0.9995
        min_temperature = 0.1
        
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() < time_limit:
            # Try to add elements
            added = False
            candidates = get_candidates()
            if len(candidates) > 0:
                # Try random candidates
                for _ in range(min(20, len(candidates))):
                    x = int(random.choice(candidates))
                    if x not in in_A:
                        valid = True
                        if sum_counts[2*x] >= 2:
                            valid = False
                        else:
                            for y in A:
                                if sum_counts[x+y] >= 2:
                                    valid = False
                                    break
                        if valid:
                            in_A.add(x)
                            A.append(x)
                            sum_counts[2*x] += 1
                            for y in A[:-1]:
                                sum_counts[x+y] += 1
                            added = True
                            if len(A) > best_size:
                                best_size = len(A)
                                best_set = A[:]
                                no_improve_count = 0
                            break
            
            if added:
                continue
            
            # Try swaps with simulated annealing
            swapped = False
            if A and len(candidates) > 0:
                for _ in range(30):
                    y = random.choice(A)
                    x = int(random.choice(candidates))
                    if x in in_A or y not in in_A:
                        continue
                    
                    # Check if swap is valid
                    valid = True
                    
                    # Check 2*x
                    if sum_counts[2*x] >= 2:
                        # Check if removing y would help
                        contrib = 0
                        if (2*x - y) in in_A and (2*x - y) != y:
                            contrib += 1
                        elif (2*x - y) == y:
                            contrib += 1
                        if sum_counts[2*x] - contrib + 1 > 2:
                            valid = False
                    
                    if valid:
                        # Check x+z for all z in A, z != y
                        for z in A:
                            if z == y:
                                continue
                            s = x + z
                            contrib = 0
                            if (s - y) in in_A and (s - y) != y:
                                contrib += 1
                            elif (s - y) == y:
                                contrib += 1
                            if sum_counts[s] - contrib + 1 > 2:
                                valid = False
                                break
                    
                    if valid:
                        # Perform swap
                        # Remove y
                        in_A.remove(y)
                        A.remove(y)
                        sum_counts[2*y] -= 1
                        for z in A:
                            sum_counts[y+z] -= 1
                        # Add x
                        in_A.add(x)
                        A.append(x)
                        sum_counts[2*x] += 1
                        for z in A[:-1]:
                            sum_counts[x+z] += 1
                        swapped = True
                        no_improve_count += 1
                        break
                
                # If no valid swap, consider accepting a worse swap with probability
                if not swapped and temperature > min_temperature:
                    # Try a random swap and accept with probability based on "cost"
                    y = random.choice(A)
                    x = int(random.choice(candidates))
                    if x not in in_A and y in in_A:
                        # Calculate how "bad" this swap is
                        violations = 0
                        if sum_counts[2*x] >= 2:
                            violations += 1
                        for z in A:
                            if z != y and sum_counts[x+z] >= 2:
                                violations += 1
                        
                        if violations <= 1:  # Accept small violations
                            prob = np.exp(-violations * 2.0 / temperature)
                            if random.random() < prob:
                                # Perform swap
                                in_A.remove(y)
                                A.remove(y)
                                sum_counts[2*y] -= 1
                                for z in A:
                                    sum_counts[y+z] -= 1
                                in_A.add(x)
                                A.append(x)
                                sum_counts[2*x] += 1
                                for z in A[:-1]:
                                    sum_counts[x+z] += 1
                                swapped = True
                                no_improve_count += 1
                        else:
                            no_improve_count += 1
            
            if not swapped:
                # Try removing an element to break deadlock
                if A:
                    # Find element that blocks the most additions
                    removal_scores = {}
                    for y in A:
                        score = 0
                        # Count how many candidate additions are blocked by y
                        for x in candidates[:50]:  # Sample candidates
                            if sum_counts[2*x] >= 2 and (2*x - y) in in_A:
                                score += 1
                            for z in A:
                                if z != y and sum_counts[x+z] >= 2 and (x+z-y) in in_A:
                                    score += 1
                                    break
                        removal_scores[y] = score
                    
                    if removal_scores:
                        max_score = max(removal_scores.values())
                        if max_score > 0:
                            to_remove = [y for y in A if removal_scores[y] == max_score]
                        else:
                            to_remove = [random.choice(A)]
                        y = random.choice(to_remove)
                        in_A.remove(y)
                        A.remove(y)
                        sum_counts[2*y] -= 1
                        for z in A:
                            sum_counts[y+z] -= 1
                        no_improve_count += 1
            
            # Cool down
            temperature *= cooling_rate
            
            if no_improve_count > max_no_improve:
                # Restart local search with current set
                no_improve_count = 0
                temperature = max(min_temperature, temperature * 0.5)
        
        return best_set, best_size

    # Run multiple restarts
    best_set = []
    best_size = 0
    restart_count = 0
    
    while time.time() < deadline:
        # Greedy init with random seed
        seed = restart_count * 7919 + 13
        A, in_A, sum_counts = greedy_init(seed)
        
        # Local search with remaining time budget
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        
        # Use 80% of remaining time for local search
        ls_time = time.time() + remaining * 0.8
        best_ls_set, ls_size = local_search(A, in_A, sum_counts, ls_time)
        
        if ls_size > best_size:
            best_size = ls_size
            best_set = best_ls_set
        
        restart_count += 1
        
        # Check if we should continue
        if time.time() > deadline - 10.0:
            break
        
        # If we've done several restarts, reduce time per restart
        if restart_count >= 3:
            break
    
    return best_set
