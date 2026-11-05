import random
import time
from math import gcd
from itertools import combinations

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55  # 55 seconds to be safe
    
    # 1. Precomputation
    N = 128
    # gcd_table[i][j] for 1 <= i,j <= N
    # We only need i < j for pairs, but let's make it symmetric or just use function
    # Precomputing all pairs is fast
    gcd_vals = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            g = gcd(i, j)
            gcd_vals[i][j] = g
            gcd_vals[j][i] = g
            
    # Find all bad triples
    # A triple (a,b,c) is bad if gcd(a,b) == gcd(a,c) == gcd(b,c)
    bad_triples = []
    # element_to_triples[x] = list of indices in bad_triples
    element_to_triples = [[] for _ in range(N+1)]
    
    # Iterate through all triples
    # To speed up, we can filter
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            gab = gcd_vals[a][b]
            for c in range(b+1, N+1):
                if gcd_vals[a][c] == gab and gcd_vals[b][c] == gab:
                    idx = len(bad_triples)
                    bad_triples.append((a, b, c))
                    element_to_triples[a].append(idx)
                    element_to_triples[b].append(idx)
                    element_to_triples[c].append(idx)
                    
    # If no bad triples, return all
    if not bad_triples:
        return list(range(1, N+1))

    # 2. Simulated Annealing Setup
    # State: selected (boolean list), active_count
    # We start with all selected
    selected = [True] * (N+1)
    
    # Calculate initial active count
    active_count = 0
    # We need to know which triples are active initially
    # Since all are selected, all bad_triples are active
    active_count = len(bad_triples)
    
    best_selected = [False] * (N+1)
    best_size = 0
    
    # Objective function
    # We want to maximize size, minimize violations
    # J = size - penalty * active_count
    penalty = 1000
    
    def get_size():
        return sum(1 for i in range(1, N+1) if selected[i])

    # Helper to update active_count when toggling x
    # Returns delta_active
    def toggle_active(x):
        nonlocal active_count
        # If x is being added, check triples containing x
        # If the other two are selected, active_count increases
        # If x is being removed, check triples containing x
        # If the other two are selected, active_count decreases (it was active)
        
        # We need to know if x is currently selected to determine direction
        # This function assumes we are about to flip selected[x]
        # Let's pass the new state? Or just calculate based on current state and flip?
        # Better: Calculate change before flipping.
        
        current_state = selected[x]
        
        delta = 0
        for idx in element_to_triples[x]:
            a, b, c = bad_triples[idx]
            # Find the other two
            others = [y for y in (a, b, c) if y != x]
            y1, y2 = others
            
            if selected[y1] and selected[y2]:
                if current_state:
                    # x is selected, others selected -> triple is active.
                    # We are removing x -> triple becomes inactive.
                    delta -= 1
                else:
                    # x is not selected, others selected -> triple is inactive.
                    # We are adding x -> triple becomes active.
                    delta += 1
        return delta

    # Simulated Annealing Loop
    T = 1.0
    cooling = 0.99
    min_T = 0.001
    
    current_size = get_size()
    current_J = current_size - penalty * active_count
    
    # Update best if valid
    if active_count == 0 and current_size > best_size:
        best_size = current_size
        best_selected = selected.copy()
        
    random.seed(42)
    
    while T > min_T:
        if time.time() > deadline:
            break
            
        # Pick random element
        x = random.randint(1, N)
        
        # Calculate delta J
        delta_active = toggle_active(x)
        # Size changes by +1 if adding, -1 if removing
        if selected[x]:
            delta_size = -1
        else:
            delta_size = 1
            
        delta_J = delta_size - penalty * delta_active
        
        # Acceptance
        if delta_J > 0:
            accept = True
        else:
            # Probability of accepting worse solution
            # delta_J is negative
            exp_val = delta_J / T
            if exp_val < -700: # Underflow protection
                accept = False
            else:
                accept = random.random() < pow(2.718281828, exp_val)
                
        if accept:
            # Apply move
            selected[x] = not selected[x]
            active_count += delta_active
            current_size += delta_size
            current_J += delta_J
            
            # Update best
            if active_count == 0 and current_size > best_size:
                best_size = current_size
                best_selected = selected.copy()
                
        T *= cooling
        
    # Final Cleanup: Ensure best_selected is valid
    # If best_selected is empty, we should try a greedy fallback
    if best_size == 0:
        # Greedy Removal
        selected = [True] * (N+1)
        active_count = len(bad_triples)
        
        # Re-calculate active_count properly for greedy
        # Actually, if we start with all True, active_count is len(bad_triples)
        
        # Iteratively remove element with most active triples
        # We need a way to count active triples per element
        # This is slower, but only if SA fails
        pass # SA should find something
        
    # Return the best found set
    result = [i for i in range(1, N+1) if best_selected[i]]
    
    # If result is empty, return a known valid small set
    if not result:
        # 1 is coprime to all. 
        # Let's just return [1] or similar
        # Actually, any single element is valid.
        return [1]
        
    return result

# EVOLVE-BLOCK-END
