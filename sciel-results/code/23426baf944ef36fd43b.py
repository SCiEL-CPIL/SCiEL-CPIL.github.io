"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import lcm
from fractions import Fraction

DENOMS = list(range(2, 49))
N = len(DENOMS)

# Precompute weights as integers for exact comparison
# We want sum(1/d_i) < 1.
# Multiply by L = lcm(2..48).
# Let W_i = L / d_i.
# We want to maximize sum(W_i) subject to sum(W_i) < L.
# Deficit in original space = 1 - sum(1/d_i) = (L - sum(W_i)) / L.
# Minimizing deficit is equivalent to maximizing sum(W_i) subject to sum(W_i) < L.

L = 1
for d in DENOMS:
    L = lcm(L, d)

WEIGHTS = [L // d for d in DENOMS]

# Greedy initialization: take largest weights that fit
# Sort indices by weight descending
sorted_indices = sorted(range(N), key=lambda i: -WEIGHTS[i])

def _greedy_init():
    selected = [False] * N
    current_sum = 0
    for i in sorted_indices:
        if current_sum + WEIGHTS[i] < L:
            selected[i] = True
            current_sum += WEIGHTS[i]
    return selected, current_sum

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # 1. Initialize with greedy solution
    selected, current_sum = _greedy_init()
    best_selected = selected[:]
    best_sum = current_sum
    
    # Simulated Annealing
    # State: selected (list of bools), current_sum
    # Move: flip one bit
    # If new_sum < L:
    #   If new_sum > current_sum: accept (improvement)
    #   Else: accept with probability exp((new_sum - current_sum) / T)
    # Else (new_sum >= L): reject (invalid)
    
    T = 1000.0  # Initial temperature (in weight units)
    cooling_rate = 0.999
    
    # To speed up, we can precompute a random order or just pick random indices
    # We need to track current_sum accurately.
    
    # Random seed for reproducibility in evaluation if needed, but usually random is fine
    rng = random.Random(42)
    
    while time.time() < deadline:
        if T < 1e-9:
            break
            
        # Pick a random index to flip
        i = rng.randint(0, N - 1)
        
        if selected[i]:
            # Try to remove i
            new_sum = current_sum - WEIGHTS[i]
            # Removing always keeps sum < L (since current_sum < L)
            # But we want to MAXIMIZE sum. So removing is only good if we can add something else later?
            # In SA, we just evaluate the move.
            # new_sum < current_sum, so it's a "worse" move in terms of objective value.
            # We might accept it to escape local minima, but since constraint is sum < L,
            # removing doesn't violate constraint.
            
            # Acceptance probability
            delta = new_sum - current_sum # negative
            if delta < 0:
                # Compute probability
                # To avoid underflow/overflow issues with exp, we can compare log
                # But delta is integer, T is float.
                # If T is large, exp is close to 1.
                # If T is small, exp is close to 0.
                # Optimization: if delta / T > -50, exp is ~0, reject.
                ratio = delta / T
                if ratio > -50:
                    if rng.random() < (2.718281828 ** ratio):
                        # Accept
                        selected[i] = False
                        current_sum = new_sum
                        # Update best if improved? No, it decreased.
                # Else reject
            else:
                # Should not happen for removal if current_sum > 0 and weight > 0
                # But if weight is 0? No, L/d >= 1.
                selected[i] = False
                current_sum = new_sum
        else:
            # Try to add i
            new_sum = current_sum + WEIGHTS[i]
            if new_sum < L:
                # Valid move
                delta = new_sum - current_sum # positive
                # Always accept improvements
                selected[i] = True
                current_sum = new_sum
                
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_selected = selected[:]
            else:
                # Invalid move (new_sum >= L)
                # Reject
                pass
        
        # Cool down
        T *= cooling_rate
    
    # Convert best_selected to list of indices
    best_indices = [i for i in range(N) if best_selected[i]]
    
    # Fallback if empty (should not happen)
    if not best_indices:
        best_indices = [0]
        
    return best_indices
# EVOLVE-BLOCK-END
