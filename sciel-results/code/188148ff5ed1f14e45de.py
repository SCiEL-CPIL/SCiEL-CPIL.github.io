import time
import numpy as np
import random

BASE = 51975

def get_divisors(n):
    divs = []
    for i in range(2, n + 1):
        if n % i == 0:
            divs.append(i)
    return divs

DIVISORS = get_divisors(BASE)
DIVISORS.sort()

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for safety
    
    # count[i] stores how many classes cover residue i
    count = np.zeros(BASE, dtype=np.int8)
    classes = []
    used_moduli = set()
    
    # 1. Greedy Initialization
    # At each step, pick (m, r) that covers the most currently uncovered residues.
    # Gain(m, r) = sum(1 for i in r::m if count[i] == 0)
    
    while time.time() < deadline - 10:
        best_m = None
        best_r = None
        best_gain = 0
        
        for m in DIVISORS:
            if m in used_moduli:
                continue
            
            # Calculate gain for all residues r for this m
            # We can vectorize this:
            # Reshape count to (-1, m)
            # gain[r] = sum(count[r::m] == 0)
            # count is flat. count.reshape(-1, m) works if BASE % m == 0.
            try:
                c_reshaped = count.reshape(-1, m)
                # We want sum of (c_reshaped == 0) along axis 0
                # This gives an array of length m, where index r is the gain
                gains = np.sum(c_reshaped == 0, axis=0)
                max_gain = np.max(gains)
                
                if max_gain > best_gain:
                    best_gain = max_gain
                    best_m = m
                    best_r = np.argmax(gains)
            except:
                continue
        
        if best_gain == 0:
            break
            
        # Apply best move
        m = best_m
        r = int(best_r)
        count[r::m] += 1
        classes.append([m, r])
        used_moduli.add(m)
        
        if len(used_moduli) == len(DIVISORS):
            break

    current_score = np.sum(count == 0)
    
    if current_score == 0:
        return classes

    # 2. Local Search with Simulated Annealing
    # Moves:
    # A. Change residue r for an existing modulus m.
    # B. Swap an existing modulus m1 with an unused modulus m2.
    
    # We'll use a simple hill climbing with random restarts or SA.
    # Given the time, a few seconds of SA is good.
    
    T = 5.0
    cooling = 0.99
    min_T = 0.1
    
    # Precompute indices for speed? No, slices are fast.
    
    while time.time() < deadline - 2 and T > min_T:
        improved = False
        
        # Try a number of moves
        # Number of moves per temperature step?
        # Let's try 100 moves per T step, or until time runs out.
        
        for _ in range(100):
            if time.time() > deadline - 2:
                break
                
            if random.random() < 0.7:
                # Move A: Change residue
                if not classes:
                    continue
                idx = random.randint(0, len(classes) - 1)
                m, r_old = classes[idx]
                
                # Find best new r
                # We want to minimize new score.
                # Delta U = Z_created - Z_removed
                # Z_created = sum(count[r_old::m] == 1)
                # Z_removed = sum(count[r_new::m] == 0) -- wait, count is updated after remove.
                
                # Let's calculate delta for all possible r_new
                # To do this efficiently, we can compute:
                # 1. Temporarily remove old class: count[r_old::m] -= 1
                # 2. Compute gains for all r_new: sum(count[r_new::m] == 0)
                # 3. The best r_new maximizes gain.
                # 4. The net change in score is:
                #    Old Score = S
                #    After Remove: S + Z_created (where Z_created is zeros created by removal)
                #    Z_created = sum(count[r_old::m] == 1) BEFORE removal?
                #    Let's check:
                #    If count[i] was 1, after decrement it is 0. So it becomes uncovered.
                #    So Z_created = np.sum(count[r_old::m] == 1).
                #    After removal, count is updated.
                #    Then we add best r_new.
                #    Z_removed = np.sum(count[r_new::m] == 0) AFTER removal.
                #    So Delta U = Z_created - Z_removed.
                
                z_created = np.sum(count[r_old::m] == 1)
                
                # Perform removal
                count[r_old::m] -= 1
                
                # Find best r_new
                try:
                    c_reshaped = count.reshape(-1, m)
                    gains = np.sum(c_reshaped == 0, axis=0)
                    best_r_new = np.argmax(gains)
                    z_removed = gains[best_r_new]
                except:
                    best_r_new = r_old
                    z_removed = 0
                    
                delta_u = z_created - z_removed
                
                # Revert removal for now to check if we accept
                count[r_old::m] += 1
                
                # Acceptance criterion
                if delta_u < 0 or (random.random() < np.exp(-delta_u / T)):
                    # Accept move
                    # Remove old
                    count[r_old::m] -= 1
                    # Add new
                    r_new = int(best_r_new)
                    count[r_new::m] += 1
                    classes[idx] = [m, r_new]
                    current_score += delta_u
                    improved = True
                    if current_score == 0:
                        return classes
            else:
                # Move B: Swap modulus
                if not classes:
                    continue
                
                # Pick random existing class to remove
                idx = random.randint(0, len(classes) - 1)
                m_old, r_old = classes[idx]
                
                # Pick random unused modulus
                unused = [m for m in DIVISORS if m not in used_moduli]
                if not unused:
                    continue
                
                m_new = random.choice(unused)
                
                # Calculate delta for swapping (m_old, r_old) with (m_new, r_best_new)
                
                # 1. Remove old class
                z_created_old = np.sum(count[r_old::m_old] == 1)
                count[r_old::m_old] -= 1
                
                # 2. Find best r for m_new
                try:
                    c_reshaped = count.reshape(-1, m_new)
                    gains = np.sum(c_reshaped == 0, axis=0)
                    best_r_new = np.argmax(gains)
                    z_removed_new = gains[best_r_new]
                except:
                    best_r_new = 0
                    z_removed_new = 0
                    
                delta_u = z_created_old - z_removed_new
                
                # Revert removal
                count[r_old::m_old] += 1
                
                if delta_u < 0 or (random.random() < np.exp(-delta_u / T)):
                    # Accept swap
                    # Remove old
                    count[r_old::m_old] -= 1
                    used_moduli.remove(m_old)
                    classes.pop(idx)
                    
                    # Add new
                    r_new = int(best_r_new)
                    count[r_new::m_new] += 1
                    classes.append([m_new, r_new])
                    used_moduli.add(m_new)
                    
                    current_score += delta_u
                    improved = True
                    if current_score == 0:
                        return classes
        
        T *= cooling
        
        # If no improvement for a while, we might be stuck.
        # But SA handles this with T.
        
    return classes

# Ensure random seed is set for reproducibility if needed, but random is fine.
# random.seed(42)
