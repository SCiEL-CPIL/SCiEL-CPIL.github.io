import time
import random
import numpy as np

def _compute_counts(chosen):
    """Compute sum counts for a chosen set using numpy convolution."""
    if not chosen:
        return {}
    arr = np.zeros(1002, dtype=np.int32)
    for x in chosen:
        arr[x] = 1
    # Convolution gives counts of sums
    conv = np.convolve(arr, arr, mode='full')
    counts = {}
    for s in range(2, len(conv)):
        c = conv[s]
        if c > 0:
            counts[s] = int(c)
    return counts

def _energy(chosen, target_size=None):
    """
    Energy function: penalizes sum-count violations (>2) and rewards set size.
    Lower energy is better.
    """
    if not chosen:
        return 1000.0  # High penalty for empty set
    
    counts = _compute_counts(chosen)
    
    # Penalty for violations (sums with count > 2)
    violation_penalty = 0
    for s, c in counts.items():
        if c > 2:
            violation_penalty += (c - 2) * 100  # Heavy penalty
    
    # Reward for set size (we want to maximize size)
    size_reward = -len(chosen) * 1.0
    
    # If target_size is given, penalize being below it
    if target_size is not None:
        if len(chosen) < target_size:
            violation_penalty += (target_size - len(chosen)) * 10
    
    return violation_penalty + size_reward

def _try_greedy(order):
    chosen = []
    counts = {}
    
    for x in order:
        new_sums = []
        for y in chosen:
            new_sums.append(x + y)
        new_sums.append(2 * x)
        
        valid = True
        for s in new_sums:
            if counts.get(s, 0) == 2:
                valid = False
                break
        
        if valid:
            chosen.append(x)
            for s in new_sums:
                counts[s] = counts.get(s, 0) + 1

    return chosen

def _simulated_annealing(initial_chosen, time_limit, rng):
    """
    Simulated annealing on the bitset representation.
    Moves: flip a bit (add or remove an element).
    Energy: penalizes violations, rewards size.
    """
    chosen = list(initial_chosen)
    chosen_set = set(chosen)
    
    # Initial energy
    current_energy = _energy(chosen)
    best_chosen = list(chosen)
    best_energy = current_energy
    
    # Temperature schedule
    initial_temp = 50.0
    final_temp = 0.1
    cooling_rate = 0.995
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        temp = initial_temp * (cooling_rate ** (int((time.time() - start_time) * 100)))
        if temp < final_temp:
            temp = final_temp
        
        # Pick a random move: flip a bit
        x = rng.randint(1, 1000)
        
        if x in chosen_set:
            # Remove x
            new_chosen = [y for y in chosen if y != x]
            new_chosen_set = set(new_chosen)
        else:
            # Add x
            new_chosen = chosen + [x]
            new_chosen_set = chosen_set | {x}
        
        new_energy = _energy(new_chosen)
        delta_e = new_energy - current_energy
        
        # Acceptance criterion
        if delta_e < 0:
            # Better state, always accept
            chosen = new_chosen
            chosen_set = new_chosen_set
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chosen = list(chosen)
        else:
            # Worse state, accept with probability exp(-delta_e / temp)
            if rng.random() < np.exp(-delta_e / temp):
                chosen = new_chosen
                chosen_set = new_chosen_set
                current_energy = new_energy
        
        # Check if we should stop early
        if time.time() - start_time > time_limit * 0.95:
            break
    
    return best_chosen

def _local_search(chosen, counts, time_limit, rng):
    """
    Local search: try to improve the set by removing elements and adding others.
    Returns improved (chosen, counts) if found within time limit.
    """
    chosen_set = set(chosen)
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        candidates_to_remove = list(chosen_set)
        rng.shuffle(candidates_to_remove)
        
        for remove_x in candidates_to_remove:
            if time.time() >= time_limit:
                break
            
            new_chosen = [y for y in chosen if y != remove_x]
            new_chosen_set = set(new_chosen)
            
            new_counts = {}
            for i, a in enumerate(new_chosen):
                for b in new_chosen[i:]:
                    s = a + b
                    new_counts[s] = new_counts.get(s, 0) + 1
            
            available = [x for x in range(1, 1001) if x not in new_chosen_set]
            rng.shuffle(available)
            
            added_count = 0
            for add_x in available:
                new_sums = []
                for y in new_chosen:
                    new_sums.append(add_x + y)
                new_sums.append(2 * add_x)
                
                valid = True
                for s in new_sums:
                    if new_counts.get(s, 0) == 2:
                        valid = False
                        break
                
                if valid:
                    new_chosen.append(add_x)
                    new_chosen_set.add(add_x)
                    for s in new_sums:
                        new_counts[s] = new_counts.get(s, 0) + 1
                    added_count += 1
                
                if len(new_chosen) >= len(chosen):
                    continue
                else:
                    break
            
            if len(new_chosen) > len(chosen):
                chosen = new_chosen
                chosen_set = new_chosen_set
                counts = new_counts
                improved = True
                break
    
    return chosen, counts

def construct():
    start_time = time.time()
    limit_time = 115.0  # Safety margin
    
    best_set = []
    best_counts = {}
    
    # 1. Deterministic Greedy (Ascending)
    s1 = _try_greedy(range(1, 1001))
    if len(s1) > len(best_set):
        best_set = s1
        best_counts = _compute_counts(s1)
    
    # 2. Deterministic Greedy (Descending)
    s2 = _try_greedy(range(1000, 0, -1))
    if len(s2) > len(best_set):
        best_set = s2
        best_counts = _compute_counts(s2)
    
    # 3. Randomized Greedy with multiple seeds
    rng = random.Random(42)
    candidates = list(range(1, 1001))
    
    # Run randomized restarts until time limit
    while time.time() - start_time < limit_time - 40:  # Leave 40s for SA + local search
        rng.shuffle(candidates)
        s = _try_greedy(candidates)
        if len(s) > len(best_set):
            best_set = s
            best_counts = _compute_counts(s)
    
    # 4. Simulated Annealing
    if time.time() - start_time < limit_time - 10:
        sa_time = min(30.0, limit_time - 10 - (time.time() - start_time))
        if sa_time > 5:
            sa_result = _simulated_annealing(best_set, sa_time, rng)
            if len(sa_result) > len(best_set):
                best_set = sa_result
                best_counts = _compute_counts(best_set)
    
    # 5. Local search to improve the best solution
    if time.time() - start_time < limit_time - 5:
        best_set, best_counts = _local_search(best_set, best_counts, limit_time - 5, rng)
    
    return best_set
