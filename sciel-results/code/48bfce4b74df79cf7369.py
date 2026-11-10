"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

_DEADLINE_SECONDS = 85  # safety margin before 90s kill

def _precompute_gcd_table(n):
    """Precompute gcd for all pairs in [1..n]."""
    table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            g = gcd(i, j)
            table[i][j] = g
            table[j][i] = g
    return table

def _can_add(x, selected, table):
    """Check if adding x to selected keeps the set valid."""
    n = len(selected)
    for i in range(n):
        a = selected[i]
        g_ax = table[a][x]
        for j in range(i + 1, n):
            b = selected[j]
            g_ab = table[a][b]
            if g_ab == g_ax:
                if table[b][x] == g_ax:
                    return False
    return True

def _greedy_seed(table, rng):
    """Generate a random greedy seed."""
    all_nums = list(range(1, 129))
    rng.shuffle(all_nums)
    selected = []
    for x in all_nums:
        if _can_add(x, selected, table):
            selected.append(x)
    return selected

def _simulated_annealing(initial_set, table, rng, time_limit):
    """
    Perform Simulated Annealing starting from initial_set.
    Returns the best set found.
    """
    current = list(initial_set)
    best = list(current)
    best_len = len(current)
    
    # Temperature parameters
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.995
    
    T = T_start
    steps = 0
    max_steps = 500000 # Hard cap on steps to prevent infinite loop if time check fails
    
    in_set = set(current)
    excluded = [i for i in range(1, 129) if i not in in_set]
    
    while T > T_end and steps < max_steps:
        if time.time() > time_limit:
            break
        
        # Choose a move
        # 1. Add a random excluded element
        # 2. Remove a random included element
        # 3. Swap a random included with a random excluded
        
        move_type = rng.random()
        
        new_current = current[:] # Copy for trial
        
        if move_type < 0.4 and excluded:
            # Add
            x = rng.choice(excluded)
            if _can_add(x, current, table):
                new_current.append(x)
                delta = 1
            else:
                continue # Invalid move, skip
        elif move_type < 0.8 and current:
            # Remove
            idx = rng.randint(0, len(current) - 1)
            x = current[idx]
            new_current.pop(idx)
            delta = -1
        else:
            # Swap
            if not current or not excluded:
                continue
            idx_in = rng.randint(0, len(current) - 1)
            x_in = current[idx_in]
            x_out = rng.choice(excluded)
            
            # Try swap: remove x_in, add x_out
            temp_set = current[:]
            temp_set.pop(idx_in)
            if _can_add(x_out, temp_set, table):
                new_current = temp_set
                new_current.append(x_out)
                delta = 0 # Size unchanged, but structure changed
            else:
                continue # Invalid swap
        
        # Evaluate
        if delta > 0:
            # Always accept improvements
            current = new_current
            in_set = set(current)
            excluded = [i for i in range(1, 129) if i not in in_set]
            if len(current) > best_len:
                best = list(current)
                best_len = len(current)
        else:
            # Accept with probability exp(delta / T)
            # delta is -1 or 0
            # If delta is 0, we are doing a random walk. 
            # To avoid getting stuck in 0-delta loops, we might want to bias towards -1 moves less?
            # Actually, for maximizing size, 0-delta moves are useful to escape local minima.
            prob = 1.0
            if delta < 0:
                prob = min(1.0, exp(delta / T))
            
            if rng.random() < prob:
                current = new_current
                in_set = set(current)
                excluded = [i for i in range(1, 129) if i not in in_set]
        
        T *= cooling_rate
        steps += 1
        
    return best

import math
def exp(x):
    return math.exp(x)

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE_SECONDS
    
    table = _precompute_gcd_table(128)
    
    best_set = []
    best_len = 0
    rng = random.Random(42)
    
    # Multi-start SA
    # Run several independent SA chains
    num_restarts = 5
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Generate a seed
        seed_set = _greedy_seed(table, rng)
        
        # Run SA with a time slice
        # Give each restart a portion of the time
        time_slice = deadline - (r * (deadline - start_time) / num_restarts)
        # Simpler: just run until deadline, but break if no improvement for a while?
        # Let's just run SA for a fixed number of steps or time.
        
        # We'll run SA for the remaining time, but limit steps per restart to ensure diversity
        # Actually, let's just run one long SA from a good seed.
        # Better: Run 3 restarts.
        
        # Let's refine: Run 4 restarts.
        # Each restart: Greedy Seed -> SA
        
        if r == 0:
             # Use a specific good heuristic seed?
             # Just random is fine.
             pass

        current_best = _simulated_annealing(seed_set, table, rng, deadline)
        
        if len(current_best) > best_len:
            best_set = current_best
            best_len = len(current_best)
            
    # Final polish on the best set found
    if time.time() < deadline - 5: # Leave 5s margin
        # Run one more SA on the best set
        final_best = _simulated_annealing(best_set, table, rng, deadline)
        if len(final_best) > best_len:
            best_set = final_best
            best_len = len(final_best)

    if not best_set:
        best_set = [1]
        
    return best_set

# EVOLVE-BLOCK-END
