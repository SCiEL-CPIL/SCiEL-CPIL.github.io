"""Rewritten program: optimized randomized local search using NumPy for B2[2] sets."""

import numpy as np
import time
import math
import random

def construct():
    """
    Returns a list of distinct integers in 1..1000 such that every integer
    has at most two representations a + b with a <= b from the set.
    Maximizes the size using simulated annealing over binary masks.
    """
    N = 1000
    max_time = 115.0  # seconds
    
    # Initial greedy construction
    def greedy_construct():
        chosen = []
        counts = np.zeros(2 * N + 1, dtype=int)
        for x in range(1, N + 1):
            # Check if adding x violates constraint
            # Sums: x + y for y in chosen, y < x, and 2*x
            # Since we iterate x in increasing order, all y in chosen are < x
            if not chosen:
                new_sums = [2 * x]
            else:
                y_vals = np.array(chosen)
                new_sums = np.append(x + y_vals, 2 * x)
            
            # Check counts
            # We need to see if any count[s] >= 2
            # If yes, adding x would make it >= 3, which is a violation?
            # Wait, the constraint is "at most two representations".
            # So count[s] must be <= 2.
            # If current count[s] is 2, adding x makes it 3 -> Violation.
            # If current count[s] is 1, adding x makes it 2 -> OK.
            # If current count[s] is 0, adding x makes it 1 -> OK.
            
            # So we check if any count[s] >= 2 for s in new_sums
            if np.any(counts[new_sums] >= 2):
                continue
            
            # Accept x
            chosen.append(x)
            counts[new_sums] += 1
            
        return chosen

    greedy_set = greedy_construct()
    
    # Initialize mask and counts
    mask = np.zeros(N + 1, dtype=bool)
    for x in greedy_set:
        mask[x] = True
        
    counts = np.zeros(2 * N + 1, dtype=int)
    # Rebuild counts for the greedy set
    for i in range(1, N + 1):
        if mask[i]:
            # Pairs (j, i) for j < i in set, and (i, i)
            j_idx = np.where(mask[:i])[0]
            sums = np.append(i + j_idx, 2 * i)
            counts[sums] += 1

    current_pen = 0
    for s in range(1, 2 * N + 1):
        if counts[s] > 2:
            current_pen += counts[s] - 2
            
    current_size = np.sum(mask)
    
    best_mask = mask.copy()
    best_pen = current_pen
    best_size = current_size
    
    random.seed(42)
    np.random.seed(42)
    
    start_time = time.time()
    T_start = 10.0
    T_end = 0.01
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_time:
            break
        
        progress = min(elapsed / max_time, 1.0)
        T = T_start * (T_end / T_start) ** progress
        
        # Choose move
        # Bias: if penalty is 0, try to add. Else, random.
        if current_pen == 0 and random.random() < 0.7:
            # Add a random element not in set
            candidates = np.where(~mask)[0]
            candidates = candidates[candidates > 0] # exclude 0
            if len(candidates) > 0:
                i = candidates[random.randint(0, len(candidates) - 1)]
            else:
                i = random.randint(1, N)
        else:
            i = random.randint(1, N)
            
        if mask[i]:
            # Remove i
            # Sums to remove: i + j for j in mask, j < i, and 2*i
            j_idx = np.where(mask[:i])[0]
            sums = np.append(i + j_idx, 2 * i)
            
            # Delta penalty
            # Removing a representation from sum s:
            # old count c -> new count c-1
            # penalty change: max(0, c-1-2) - max(0, c-2)
            # if c >= 3: (c-3) - (c-2) = -1
            # if c < 3: 0
            delta_pen = -np.sum(counts[sums] >= 3)
            
            new_pen = current_pen + delta_pen
            new_size = current_size - 1
            
            # Acceptance
            if new_pen < current_pen:
                accept = True
            elif new_pen == current_pen:
                accept = random.random() < 0.5
            else:
                accept_prob = math.exp(-delta_pen / max(T, 1e-10))
                accept = random.random() < accept_prob
            
            if accept:
                mask[i] = False
                counts[sums] -= 1
                current_pen = new_pen
                current_size = new_size
                
                if current_pen < best_pen or (current_pen == best_pen and current_size > best_size):
                    best_pen = current_pen
                    best_size = current_size
                    best_mask = mask.copy()
        else:
            # Add i
            # Sums to add: i + j for j in mask, j < i, and 2*i
            j_idx = np.where(mask[:i])[0]
            sums = np.append(i + j_idx, 2 * i)
            
            # Delta penalty
            # Adding a representation to sum s:
            # old count c -> new count c+1
            # penalty change: max(0, c+1-2) - max(0, c-2)
            # if c >= 2: (c-1) - (c-2) = 1
            # if c < 2: 0
            delta_pen = np.sum(counts[sums] >= 2)
            
            new_pen = current_pen + delta_pen
            new_size = current_size + 1
            
            # Acceptance
            if new_pen < current_pen:
                accept = True
            elif new_pen == current_pen:
                accept = True # Always accept neutral moves to explore
            else:
                accept_prob = math.exp(-delta_pen / max(T, 1e-10))
                accept = random.random() < accept_prob
            
            if accept:
                mask[i] = True
                counts[sums] += 1
                current_pen = new_pen
                current_size = new_size
                
                if current_pen < best_pen or (current_pen == best_pen and current_size > best_size):
                    best_pen = current_pen
                    best_size = current_size
                    best_mask = mask.copy()

    result = [int(i) for i in np.where(best_mask)[0] if i > 0]
    return result

if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    # Quick verification
    counts = {}
    for i in range(len(result)):
        for j in range(i, len(result)):
            s = result[i] + result[j]
            counts[s] = counts.get(s, 0) + 1
    violations = sum(1 for c in counts.values() if c > 2)
    print(f"Violations: {violations}")
    print(f"Max count: {max(counts.values()) if counts else 0}")
