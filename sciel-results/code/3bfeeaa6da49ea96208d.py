import time
import numpy as np
import random

N = 18

def construct():
    # Generate Odd-Even Transposition Network
    def odd_even_transposition(n):
        comps = []
        for phase in range(n):
            for i in range(phase % 2, n - 1, 2):
                comps.append((i, i + 1))
        return comps

    # Pre-compute base inputs for verification
    # Shape: (2^18, 18)
    # Using bit shifting
    base_inputs = (np.arange(2**N)[:, None] >> np.arange(N-1, -1, -1)) & 1

    # Fast Verifier using NumPy
    def is_valid(comparators):
        inputs = base_inputs.copy()
        
        for i, j in comparators:
            min_val = np.minimum(inputs[:, i], inputs[:, j])
            max_val = np.maximum(inputs[:, i], inputs[:, j])
            inputs[:, i] = min_val
            inputs[:, j] = max_val
            
        # Check if sorted
        # If any row has a 1 followed by a 0, it's invalid
        if np.any(np.diff(inputs, axis=1) < 0):
            return False
        return True

    current_net = odd_even_transposition(N)
    best_net = current_net
    best_len = len(current_net)
    
    deadline = time.time() + 85.0 # Safety margin
    
    # Greedy Removal and Swap Loop
    stuck_count = 0
    while time.time() < deadline:
        improved = False
        
        # 1. Try to remove a comparator
        for i in range(len(current_net)):
            if time.time() > deadline:
                break
            candidate = current_net[:i] + current_net[i+1:]
            if is_valid(candidate):
                current_net = candidate
                best_net = candidate
                best_len = len(candidate)
                improved = True
                stuck_count = 0
                break # Restart scan from 0
        
        # 2. If no removal possible, try swaps
        if not improved:
            for i in range(len(current_net) - 1):
                if time.time() > deadline:
                    break
                c1 = current_net[i]
                c2 = current_net[i+1]
                # Check if disjoint (wires don't overlap)
                if len(set(c1) & set(c2)) == 0:
                    candidate = current_net[:i] + [c2, c1] + current_net[i+2:]
                    if is_valid(candidate):
                        current_net = candidate
                        if len(current_net) < best_len:
                            best_net = candidate
                            best_len = len(current_net)
                        improved = True
                        stuck_count = 0
                        break # Restart swap scan
            
        # 3. If stuck, perturb
        if not improved:
            stuck_count += 1
            if best_len > 75 and stuck_count < 500: # Only perturb if we have room to improve and haven't tried too much
                # Random swap of two comparators
                if len(current_net) > 1:
                    i, j = random.sample(range(len(current_net)), 2)
                    current_net[i], current_net[j] = current_net[j], current_net[i]
            else:
                # If we are close to optimal and stuck, break
                break

    return best_net
