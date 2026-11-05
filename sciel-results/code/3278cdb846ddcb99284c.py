import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer
    
    N = 1000
    max_sum = 2 * N
    
    # Data structures
    A = []
    in_A = set()
    sum_counts = [0] * (2 * N + 1)
    
    best_set = []
    best_size = 0
    
    def try_add(A, in_A, sum_counts, x):
        if x in in_A:
            return False
        if sum_counts[2*x] >= 2:
            return False
        for y in A:
            if sum_counts[x+y] >= 2:
                return False
        # Valid
        in_A.add(x)
        A.append(x)
        sum_counts[2*x] += 1
        for y in A[:-1]:
            sum_counts[x+y] += 1
        return True

    def remove_element(A, in_A, sum_counts, x):
        if x not in in_A:
            return False
        in_A.remove(x)
        A.remove(x)
        sum_counts[2*x] -= 1
        for y in A:
            sum_counts[x+y] -= 1
        return True

    def try_swap(A, in_A, sum_counts, y, x):
        if x in in_A or y not in in_A:
            return False
        
        # Check 2x
        s = 2 * x
        contrib_y = 1 if (s - y) in in_A and (s - y) != y else 0
        if sum_counts[s] - contrib_y + 1 > 2:
            return False
            
        # Check x+z for z in A, z != y
        for z in A:
            if z == y:
                continue
            s = x + z
            contrib_y = 1 if (s - y) in in_A and (s - y) != y else 0
            if sum_counts[s] - contrib_y + 1 > 2:
                return False
                
        # Perform swap
        remove_element(A, in_A, sum_counts, y)
        in_A.add(x)
        A.append(x)
        sum_counts[2*x] += 1
        for z in A[:-1]:
            sum_counts[x+z] += 1
        return True

    # Initialize with greedy
    for x in range(1, N + 1):
        if try_add(A, in_A, sum_counts, x):
            continue
    
    best_set = A[:]
    best_size = len(A)
    
    # Track which sums are saturated (count == 2) for quick rejection
    def get_saturated_sums():
        return [s for s in range(1, 2*N+1) if sum_counts[s] == 2]
    
    # Multi-phase local search
    phase = 0
    T = 15.0
    cooling_rate = 0.998
    min_T = 0.5
    
    # Phase 0: Aggressive adding (first 20% of time)
    # Phase 1: Focused swaps (next 40% of time)  
    # Phase 2: Fine-tuning with occasional removals (last 40% of time)
    
    total_time_budget = deadline - start_time
    phase0_end = start_time + total_time_budget * 0.2
    phase1_end = start_time + total_time_budget * 0.6
    
    while time.time() < deadline:
        current_time = time.time()
        
        # Determine current phase
        if current_time < phase0_end:
            phase = 0
        elif current_time < phase1_end:
            phase = 1
        else:
            phase = 2
        
        # 1. Try to add any element (always accept improvements)
        added = False
        candidates = [x for x in range(1, N + 1) if x not in in_A]
        
        if phase == 0:
            # Aggressive: try many candidates
            random.shuffle(candidates)
            for x in candidates[:100]:  # Limit to first 100 for speed
                if try_add(A, in_A, sum_counts, x):
                    added = True
                    if len(A) > best_size:
                        best_size = len(A)
                        best_set = A[:]
                    break
        else:
            # Smarter: prioritize candidates that hit less-constrained sums
            scored_candidates = []
            for x in candidates:
                # Count how many sums would be at capacity 1 (can still add)
                flexible_sums = 0
                for y in A:
                    s = x + y
                    if sum_counts[s] == 1:
                        flexible_sums += 1
                    elif sum_counts[s] >= 2:
                        break  # Can't add if any sum is saturated
                if flexible_sums > 0 and sum_counts[2*x] < 2:
                    scored_candidates.append((flexible_sums, x))
            
            if scored_candidates:
                # Sort by flexibility (more flexible first)
                scored_candidates.sort(reverse=True)
                # Try top candidates
                for _, x in scored_candidates[:20]:
                    if try_add(A, in_A, sum_counts, x):
                        added = True
                        if len(A) > best_size:
                            best_size = len(A)
                            best_set = A[:]
                        break
        
        if added:
            T = max(T * 0.9, min_T)
            continue
        
        # 2. Try swaps with simulated annealing
        swapped = False
        
        if phase == 0:
            swap_attempts = 30
        elif phase == 1:
            swap_attempts = 80
        else:
            swap_attempts = 50
        
        for _ in range(swap_attempts):
            if not A or not candidates:
                break
            y = random.choice(A)
            x = random.choice(candidates)
            if try_swap(A, in_A, sum_counts, y, x):
                swapped = True
                if len(A) > best_size:
                    best_size = len(A)
                    best_set = A[:]
                break
        
        if swapped:
            T *= cooling_rate
            continue
            
        # 3. If no swap, try a random move (accept with probability based on temperature)
        if A and candidates:
            y = random.choice(A)
            x = random.choice(candidates)
            # Calculate "cost" of this swap (how many constraints it would violate)
            # For simplicity, just accept with probability based on temperature
            accept_prob = min(1.0, T / 15.0)
            if random.random() < accept_prob:
                if try_swap(A, in_A, sum_counts, y, x):
                    T *= cooling_rate
                    continue
        
        # 4. If stuck, remove a random element to break deadlock
        # In phase 2, be more aggressive with removals
        removal_prob = 0.2 if phase < 2 else 0.4
        if A and random.random() < removal_prob:
            # Prefer removing elements that block many additions
            if phase == 2:
                # Find element whose removal frees the most sums
                best_y = None
                best_freed = -1
                for y in A:
                    freed = 0
                    for s in range(y+1, min(2*N+1, y + len(A) + 1)):
                        if sum_counts[s] == 2:
                            # Check if removing y would help
                            other = s - y
                            if other in in_A and other != y:
                                freed += 1
                    if freed > best_freed:
                        best_freed = freed
                        best_y = y
                if best_y is not None:
                    y = best_y
                else:
                    y = random.choice(A)
            else:
                y = random.choice(A)
            
            remove_element(A, in_A, sum_counts, y)
            T = max(T * 1.2, 2.0)  # Reheat significantly
        
        T *= cooling_rate
        T = max(T, min_T)
    
    return best_set
