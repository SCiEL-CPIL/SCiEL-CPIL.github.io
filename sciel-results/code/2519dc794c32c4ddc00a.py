import time
import math
import random
import numpy as np

def compute_score(U):
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0
    return 1 + log_ratio / log_denom

def construct():
    start = time.time()
    deadline = start + 140  # Leave 10s margin for safety
    
    best_score = 0
    best_U = [0, 1, 3]
    
    # 1. Initial Exploration: Exhaustive search for small sets to find good local optima
    # This provides a strong starting point for the annealing.
    
    # Exhaustive search for 3-element sets {0, a, b}
    for a in range(1, 30):
        if time.time() > deadline - 60:
            break
        for b in range(a + 1, 30):
            U = [0, a, b]
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_U = U[:]

    # Exhaustive search for 4-element sets {0, a, b, c}
    for a in range(1, 20):
        if time.time() > deadline - 60:
            break
        for b in range(a + 1, 20):
            for c in range(b + 1, 20):
                U = [0, a, b, c]
                s = compute_score(U)
                if s > best_score:
                    best_score = s
                    best_U = U[:]

    # 2. Structured Families
    families = []
    # Consecutive
    for n in range(2, 20):
        families.append(list(range(n)))
    # Powers of 2
    for n in range(2, 20):
        families.append([0] + [2**i for i in range(1, n)])
    # Mersenne
    for n in range(2, 20):
        families.append([0] + [2**i - 1 for i in range(1, n)])
    # Arithmetic progressions
    for n in range(2, 20):
        for step in range(1, 5):
            families.append([i * step for i in range(n)])
    # Mixed base representations
    for base in range(2, 10):
        for n in range(3, 15):
            U = [0]
            val = 1
            for i in range(n - 1):
                U.append(val)
                val *= base
            families.append(U)
            
    # Evaluate families
    for U in families:
        if time.time() > deadline - 40:
            break
        U = sorted(set(U))
        if len(U) < 2:
            continue
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]

    # 3. Simulated Annealing
    # State: list of integers
    # Move: Add, Remove, or Modify an element
    # Objective: Maximize score
    
    current_U = list(best_U)
    current_score = best_score
    
    T_start = 0.1
    T_end = 1e-6
    cooling_rate = 0.995
    
    # We will run until deadline - 10s
    steps = 0
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 10:
        steps += 1
        
        # Generate a neighbor
        U_new = current_U[:]
        op = random.choice(['add', 'remove', 'modify', 'modify'])
        max_val = max(U_new) if U_new else 1
        min_val = min(U_new) if U_new else 0
        
        if op == 'add':
            # Add a random element, possibly large
            if len(U_new) < 50: # Limit size to avoid O(N^2) explosion in compute_score
                new_val = random.randint(1, max(2, int(max_val * 1.5)))
                if new_val not in U_new:
                    U_new.append(new_val)
                    U_new.sort()
                else:
                    U_new = current_U[:]
        elif op == 'remove':
            if len(U_new) > 2:
                # Don't remove 0
                idx = random.randint(1, len(U_new) - 1)
                U_new.pop(idx)
            else:
                U_new = current_U[:]
        else: # modify
            if len(U_new) > 1:
                idx = random.randint(1, len(U_new) - 1)
                # Perturb the value
                delta = random.randint(-max(1, max_val//10), max(1, max_val//10))
                new_val = U_new[idx] + delta
                if new_val >= 0:
                    U_new[idx] = new_val
                    U_new.sort()
                    # Handle duplicates
                    U_new = sorted(set(U_new))
                else:
                    U_new = current_U[:]
        
        if len(U_new) < 2:
            U_new = current_U[:]
        
        new_score = compute_score(U_new)
        
        # Acceptance probability
        delta_score = new_score - current_score
        
        if delta_score >= 0:
            # Accept improvement
            current_U = U_new
            current_score = new_score
            no_improve_count = 0
            
            if new_score > best_score:
                best_score = new_score
                best_U = U_new[:]
        else:
            # Accept worse move with probability exp(delta_score / T)
            # Note: delta_score is negative
            T = T_start * (T_end / T_start) ** (steps / 100000.0) # Adaptive cooling
            if T < T_end:
                T = T_end
            
            # Avoid log of 0
            if T > 0:
                prob = math.exp(delta_score / T)
            else:
                prob = 0
                
            if random.random() < prob:
                current_U = U_new
                current_score = new_score
                no_improve_count += 1
            else:
                no_improve_count += 1
        
        # Restart logic: If stuck for too long, randomize or reset T
        if no_improve_count > max_no_improve:
            # Try a bigger jump
            if random.random() < 0.5:
                # Reset to best
                current_U = list(best_U)
                current_score = best_score
            else:
                # Random perturbation of current
                for i in range(random.randint(1, 3)):
                    if len(current_U) > 2:
                        idx = random.randint(1, len(current_U)-1)
                        current_U[idx] += random.randint(-10, 10)
                current_U = sorted(set(x for x in current_U if x >= 0))
                if 0 not in current_U:
                    current_U.insert(0, 0)
                current_score = compute_score(current_U)
            
            no_improve_count = 0
            # Increase temperature slightly to escape
            # We don't explicitly track T in the loop above as a variable, 
            # but we can force a "hot" phase by resetting steps counter or similar.
            # For simplicity, the randomization above helps.
            
        # Check deadline
        if time.time() > deadline - 5:
            break

    return best_U
