"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def _compute_threshold_bitmask(a_map, b_map):
    """Compute shortest reset word length using bitmask BFS."""
    full_mask = (1 << N) - 1
    
    # Precompute transitions for each state and letter
    # trans[state][letter] = destination
    trans = [[a_map[s], b_map[s]] for s in range(N)]
    
    # BFS over subsets (represented as bitmasks)
    visited = set()
    visited.add(full_mask)
    queue = deque()
    queue.append((full_mask, 0))
    
    while queue:
        mask, dist = queue.popleft()
        
        # Check if this is a singleton (reset achieved)
        if mask & (mask - 1) == 0:  # power of 2
            return dist
        
        # Generate next states for letter 'a'
        new_mask_a = 0
        for s in range(N):
            if mask & (1 << s):
                new_mask_a |= (1 << trans[s][0])
        
        if new_mask_a not in visited:
            visited.add(new_mask_a)
            queue.append((new_mask_a, dist + 1))
        
        # Generate next states for letter 'b'
        new_mask_b = 0
        for s in range(N):
            if mask & (1 << s):
                new_mask_b |= (1 << trans[s][1])
        
        if new_mask_b not in visited:
            visited.add(new_mask_b)
            queue.append((new_mask_b, dist + 1))
    
    return -1  # Not synchronizing

def _pawlik_automaton():
    """Generate the Pawlik automaton structure."""
    a_map = [(i + 1) % N for i in range(N)]
    b_map = [0 if i == 0 else i - 1 for i in range(N)]
    return [a_map, b_map]

def _random_synchronizing():
    """Generate a random automaton and check if it synchronizes."""
    a_map = [random.randint(0, N - 1) for _ in range(N)]
    b_map = [random.randint(0, N - 1) for _ in range(N)]
    return [a_map, b_map]

def _local_search(initial_automaton, max_steps=5000, time_limit=50.0):
    """Perform local search to improve the reset threshold."""
    start_time = time.time()
    best_a = initial_automaton[0][:]
    best_b = initial_automaton[1][:]
    best_score = _compute_threshold_bitmask(best_a, best_b)
    
    if best_score < 0:
        return None
    
    current_a = best_a[:]
    current_b = best_b[:]
    current_score = best_score
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Random perturbation: change one state's transition for one letter
        letter = random.choice([0, 1])
        state = random.randint(0, N - 1)
        old_val = current_a[state] if letter == 0 else current_b[state]
        new_val = random.randint(0, N - 1)
        
        if letter == 0:
            current_a[state] = new_val
        else:
            current_b[state] = new_val
        
        new_score = _compute_threshold_bitmask(current_a, current_b)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_a = current_a[:]
                best_b = current_b[:]
        else:
            # Revert
            if letter == 0:
                current_a[state] = old_val
            else:
                current_b[state] = old_val
    
    return [best_a, best_b] if best_score > 0 else None

def _simulated_annealing(time_budget):
    """Simulated annealing search for high-threshold automata."""
    start_time = time.time()
    
    # Start from Pawlik
    current = _pawlik_automaton()
    current_score = _compute_threshold_bitmask(current[0], current[1])
    
    best = [current[0][:], current[1][:]]
    best_score = current_score
    
    temperature = 5.0
    cooling = 0.999
    
    while time.time() - start_time < time_budget:
        # Perturb
        letter = random.choice([0, 1])
        state = random.randint(0, N - 1)
        old_val = current[letter][state]
        new_val = random.randint(0, N - 1)
        current[letter][state] = new_val
        
        new_score = _compute_threshold_bitmask(current[0], current[1])
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best[0] = current[0][:]
                best[1] = current[1][:]
        elif new_score >= 0:
            # Accept worse with probability
            delta = new_score - current_score
            if random.random() < min(1.0, temperature / max(1, abs(delta))):
                current_score = new_score
        else:
            # Not synchronizing - revert
            current[letter][state] = old_val
        
        temperature *= cooling
        if temperature < 0.01:
            break
    
    return [best[0], best[1]] if best_score > 0 else None

def construct():
    """
    Constructs a synchronizing automaton with high reset threshold.
    Uses simulated annealing and local search to find good structures.
    """
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    # Start with Pawlik as baseline
    pawlik = _pawlik_automaton()
    pawlik_score = _compute_threshold_bitmask(pawlik[0], pawlik[1])
    
    best = [pawlik[0][:], pawlik[1][:]]
    best_score = pawlik_score
    
    # Try simulated annealing with time budget
    sa_budget = 40.0
    if time.time() + sa_budget < deadline:
        sa_result = _simulated_annealing(sa_budget)
        if sa_result:
            sa_score = _compute_threshold_bitmask(sa_result[0], sa_result[1])
            if sa_score > best_score:
                best = sa_result
                best_score = sa_score
    
    # Try local search from best found
    ls_budget = 30.0
    if time.time() + ls_budget < deadline:
        ls_result = _local_search(best, max_steps=10000, time_limit=ls_budget)
        if ls_result:
            ls_score = _compute_threshold_bitmask(ls_result[0], ls_result[1])
            if ls_score > best_score:
                best = ls_result
                best_score = ls_score
    
    # Final check
    final_score = _compute_threshold_bitmask(best[0], best[1])
    if final_score <= 0:
        # Fallback to Pawlik if something went wrong
        return pawlik
    
    return best
# EVOLVE-BLOCK-END
