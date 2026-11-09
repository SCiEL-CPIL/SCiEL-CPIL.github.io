import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit

    N = 256

    # Precompute 4-APs as bitmasks for fast conflict checking
    # four_ap_by_elem[x] contains masks of all 4-APs containing x
    four_ap_by_elem = [[] for _ in range(N)]
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            a0 = a - 1
            b0 = a + d - 1
            c0 = a + 2 * d - 1
            d0 = a + 3 * d - 1
            mask = (1 << a0) | (1 << b0) | (1 << c0) | (1 << d0)
            four_ap_by_elem[a0].append(mask)
            four_ap_by_elem[b0].append(mask)
            four_ap_by_elem[c0].append(mask)
            four_ap_by_elem[d0].append(mask)

    # Precompute 3-AP neighbors for score calculation
    # three_ap_by_elem[x] contains (u, v) such that {x, u, v} is a 3-AP
    three_ap_by_elem = [[] for _ in range(N)]
    for d in range(1, (N - 1) // 2 + 1):
        max_a = N - 2 * d
        for a in range(1, max_a + 1):
            a0 = a - 1
            b0 = a + d - 1
            c0 = a + 2 * d - 1
            three_ap_by_elem[a0].append((b0, c0))
            three_ap_by_elem[b0].append((a0, c0))
            three_ap_by_elem[c0].append((a0, b0))

    def is_valid(state):
        """Check if state contains no 4-APs."""
        for x in range(N):
            if (state >> x) & 1:
                for mask in four_ap_by_elem[x]:
                    if (state & mask) == mask:
                        return False
        return True

    def count_3aps(state):
        """Count distinct 3-APs in state."""
        count = 0
        for x in range(N):
            if (state >> x) & 1:
                for u, v in three_ap_by_elem[x]:
                    if ((state >> u) & 1) and ((state >> v) & 1):
                        count += 1
        return count // 3

    def repair(state):
        """Remove elements to make state valid. Prioritize removing elements with high 3-AP contribution."""
        # Calculate contribution of each element to 3-APs
        # An element x is in a 3-AP if there exists u,v in state such that {x,u,v} is 3-AP.
        # We want to remove elements that are in many 4-APs but few 3-APs?
        # Actually, simpler: while invalid, find a violating 4-AP and remove the element
        # in that 4-AP that has the lowest "value" (number of 3-APs it participates in).
        
        valid = True
        while True:
            # Find first violation
            violation_mask = 0
            violation_elements = []
            for x in range(N):
                if (state >> x) & 1:
                    for mask in four_ap_by_elem[x]:
                        if (state & mask) == mask:
                            violation_mask = mask
                            # Get elements in mask
                            for i in range(N):
                                if (mask >> i) & 1:
                                    violation_elements.append(i)
                            break
                    if violation_mask:
                        break
            
            if not violation_mask:
                break
                
            # Among violation_elements, find the one with lowest 3-AP contribution
            # to minimize score loss
            min_score = float('inf')
            min_elem = -1
            for e in violation_elements:
                # Count 3-APs involving e in current state
                c = 0
                for u, v in three_ap_by_elem[e]:
                    if ((state >> u) & 1) and ((state >> v) & 1):
                        c += 1
                if c < min_score:
                    min_score = c
                    min_elem = e
            
            if min_elem != -1:
                state &= ~(1 << min_elem)
            else:
                # Should not happen
                break
                
        return state

    def greedy_construct():
        """Fast greedy construction to initialize population."""
        state = 0
        available = list(range(N))
        random.shuffle(available)
        
        for x in available:
            # Check if adding x creates a 4-AP
            if is_valid(state | (1 << x)):
                state |= (1 << x)
        
        return state

    # Initialize Population
    pop_size = 20
    population = []
    
    # Add some greedy starts
    for _ in range(pop_size // 2):
        state = greedy_construct()
        population.append(state)
        
    # Add some random starts (with repair)
    for _ in range(pop_size // 2):
        state = 0
        # Random subset
        for i in range(N):
            if random.random() < 0.4:
                state |= (1 << i)
        state = repair(state)
        population.append(state)

    # Evaluate initial population
    scores = [count_3aps(s) for s in population]
    best_idx = max(range(pop_size), key=lambda i: scores[i])
    best_state = population[best_idx]
    best_score = scores[best_idx]
    
    # Genetic Algorithm Loop
    # Parameters
    offspring_size = pop_size
    mutation_rate = 0.1
    crossover_rate = 0.8
    
    while time.time() < deadline - 5.0:
        # Selection: Tournament Selection
        def select():
            i1 = random.randint(0, pop_size - 1)
            i2 = random.randint(0, pop_size - 1)
            if scores[i1] > scores[i2]:
                return i1
            else:
                return i2
        
        new_population = []
        
        # Keep the best individual (Elitism)
        best_idx = max(range(pop_size), key=lambda i: scores[i])
        new_population.append(population[best_idx])
        
        for _ in range(offspring_size - 1):
            if time.time() > deadline - 5.0:
                break
                
            p1_idx = select()
            p2_idx = select()
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            
            if random.random() < crossover_rate:
                # Uniform Crossover
                child = 0
                for i in range(N):
                    if (p1 >> i) & 1:
                        if random.random() < 0.5:
                            child |= (1 << i)
                    elif (p2 >> i) & 1:
                        if random.random() < 0.5:
                            child |= (1 << i)
            else:
                # Clone
                child = p1
            
            # Mutation: Random bit flips
            # We need to be careful: flipping bits might create 4-APs.
            # We will mutate and then repair.
            for i in range(N):
                if random.random() < mutation_rate:
                    child ^= (1 << i)
            
            # Repair child to ensure validity
            child = repair(child)
            
            new_population.append(child)
        
        # Update population
        population = new_population
        
        # Evaluate new population
        for i in range(len(population)):
            scores[i] = count_3aps(population[i])
        
        # Update best
        best_idx = max(range(len(population)), key=lambda i: scores[i])
        if scores[best_idx] > best_score:
            best_score = scores[best_idx]
            best_state = population[best_idx]
            
        # Optional: Local Search on Best Individual occasionally
        # To save time, do a simple hill climb on the best individual every few generations
        if len(population) % 5 == 0 and time.time() < deadline - 10.0:
            current_state = best_state
            current_score = best_score
            improved = True
            max_local_iters = 100
            iter_count = 0
            
            while improved and iter_count < max_local_iters and time.time() < deadline - 5.0:
                improved = False
                iter_count += 1
                
                # Try Add
                for x in range(N):
                    if (current_state >> x) & 1:
                        continue
                    if is_valid(current_state | (1 << x)):
                        # Check gain
                        gain = 0
                        for u, v in three_ap_by_elem[x]:
                            if ((current_state >> u) & 1) and ((current_state >> v) & 1):
                                gain += 1
                        if gain > 0:
                            current_state |= (1 << x)
                            current_score += gain
                            improved = True
                            break
                            
                if improved:
                    continue
                
                # Try Swap
                state_indices = [i for i in range(N) if (current_state >> i) & 1]
                non_state_indices = [i for i in range(N) if not (current_state >> i) & 1]
                if not state_indices or not non_state_indices:
                    break
                random.shuffle(state_indices)
                random.shuffle(non_state_indices)
                
                for x in state_indices:
                    if time.time() > deadline - 5.0:
                        break
                    for y in non_state_indices:
                        if time.time() > deadline - 5.0:
                            break
                        new_state = (current_state & ~(1 << x)) | (1 << y)
                        if not is_valid(new_state):
                            continue
                        new_score = count_3aps(new_state)
                        if new_score > current_score:
                            current_state = new_state
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break
            
            if current_score > best_score:
                best_score = current_score
                best_state = current_state
                # Inject improved best into population
                population[random.randint(0, len(population)-1)] = best_state
                scores[random.randint(0, len(population)-1)] = best_score

    # Final Result
    result = []
    for i in range(N):
        if (best_state >> i) & 1:
            result.append(i + 1)

    return sorted(result)
