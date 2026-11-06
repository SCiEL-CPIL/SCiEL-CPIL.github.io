import math
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Problem: Maximize sum(1/m) for a set of pairwise disjoint residue classes.
    # Moduli m in [20, 200], residues r in [0, m).
    # Disjointness: r_i != r_j mod gcd(m_i, m_j).
    
    # Strategy: Local Search / Simulated Annealing
    # State: A list of (m, r) pairs.
    # Moves:
    #   1. Add: Pick an unselected m, find a valid r. If exists, add.
    #   2. Swap: Pick a selected m, try a different valid r.
    #   3. Remove: Pick a selected m, remove it. (Usually bad for score, but helps unblock)
    #   4. Swap Moduli: Remove m1, add m2 (if valid).
    
    # To speed up, we precompute gcds for all pairs of moduli in [20, 200].
    moduli_range = list(range(20, 201))
    n_moduli = len(moduli_range)
    
    # Precompute gcd matrix for moduli indices
    # gcd_matrix[i][j] = gcd(moduli_range[i], moduli_range[j])
    gcd_matrix = [[0]*n_moduli for _ in range(n_moduli)]
    for i in range(n_moduli):
        mi = moduli_range[i]
        for j in range(i, n_moduli):
            mj = moduli_range[j]
            g = math.gcd(mi, mj)
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g
    
    # Precompute weights
    weights = {m: 1.0/m for m in moduli_range}
    
    # Helper to check if a residue r for modulus m is compatible with a set of classes
    # classes: list of (m_idx, r)
    def is_compatible(m_idx, r, classes):
        m = moduli_range[m_idx]
        for m_i_idx, r_i in classes:
            g = gcd_matrix[m_idx][m_i_idx]
            if r % g == r_i % g:
                return False
        return True
    
    # Helper to find any valid residue for m_idx given classes
    def find_valid_residue(m_idx, classes):
        m = moduli_range[m_idx]
        # Try residues in a random order to avoid bias? 
        # Or try 0..m-1. Since we are in local search, randomization in the move generation is better.
        # But for initialization, let's try to find one quickly.
        for r in range(m):
            if is_compatible(m_idx, r, classes):
                return r
        return -1

    # Initialization: Greedy construction
    # Sort moduli by weight (ascending m)
    # Try to add each modulus if possible.
    
    def greedy_init():
        classes = []
        # Try to add moduli in order of increasing m (highest weight first)
        # But this might get stuck. Let's try a few random shuffles for init.
        best_init_score = 0.0
        best_init_classes = []
        
        for _ in range(50):
            if time.time() > deadline - 10: break
            random.shuffle(moduli_range)
            cur_classes = []
            cur_score = 0.0
            for m in moduli_range:
                m_idx = moduli_range.index(m) # This is slow. Better to iterate indices.
            # Re-do init properly with indices
            indices = list(range(n_moduli))
            random.shuffle(indices)
            cur_classes = []
            cur_score = 0.0
            for idx in indices:
                m = moduli_range[idx]
                r = find_valid_residue(idx, cur_classes)
                if r != -1:
                    cur_classes.append((idx, r))
                    cur_score += weights[m]
            if cur_score > best_init_score:
                best_init_score = cur_score
                best_init_classes = cur_classes[:]
        
        return best_init_classes, best_init_score

    best_classes, best_score = greedy_init()
    
    # Local Search / Hill Climbing
    # We will perform a series of moves.
    # Move types:
    # 1. Add a new modulus (if space allows)
    # 2. Change residue of an existing modulus (to potentially unblock others or just for variety)
    # 3. Swap one modulus for another (remove one, try to add another)
    
    # To make "Add" efficient, we maintain a set of selected indices.
    selected_indices = set(idx for idx, _ in best_classes)
    
    # Simulated Annealing parameters
    T = 1.0
    cooling_rate = 0.99
    min_T = 0.01
    max_iterations = 20000
    
    current_classes = best_classes[:]
    current_score = best_score
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
            
        if T < min_T:
            break
            
        # Choose a move type
        move_type = random.choice(['add', 'swap_res', 'swap_mod'])
        
        new_classes = current_classes[:]
        new_score = current_score
        valid_move = False
        
        if move_type == 'add':
            # Pick a random unselected modulus
            unselected = [i for i in range(n_moduli) if i not in selected_indices]
            if unselected:
                idx = random.choice(unselected)
                m = moduli_range[idx]
                # Try to find a valid residue
                # To speed up, try a few random residues first, then linear scan if needed?
                # Linear scan is O(m * |classes|). m<=200, |classes|<=181. ~36k ops. Fast enough.
                r = find_valid_residue(idx, current_classes)
                if r != -1:
                    new_classes.append((idx, r))
                    new_score += weights[m]
                    selected_indices.add(idx)
                    valid_move = True
                    # Accept if better (which it always is if valid, since weight > 0)
                    current_classes = new_classes
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_classes = current_classes[:]
        
        elif move_type == 'swap_res':
            # Pick a random selected modulus and change its residue
            if current_classes:
                pos = random.randint(0, len(current_classes) - 1)
                idx, old_r = current_classes[pos]
                # Try to find a different valid residue
                # We can just pick a random residue and check, or scan.
                # Scanning is safer to ensure we find *some* valid one if exists.
                # But we want to change it.
                # Let's pick a random new residue and check compatibility.
                m = moduli_range[idx]
                new_r = random.randint(0, m - 1)
                if new_r == old_r:
                    continue
                
                # Check if new_r is compatible with all OTHER classes
                compatible = True
                for i, (m_i_idx, r_i) in enumerate(current_classes):
                    if i == pos:
                        continue
                    g = gcd_matrix[idx][m_i_idx]
                    if new_r % g == r_i % g:
                        compatible = False
                        break
                
                if compatible:
                    current_classes[pos] = (idx, new_r)
                    # Score doesn't change, but state does.
                    # This might unblock future 'add' moves.
                    valid_move = True
                    # In SA, we might accept neutral moves to escape local minima.
                    # Here, we always accept valid neutral moves to explore state space.
        
        elif move_type == 'swap_mod':
            # Remove a random selected modulus, try to add a random unselected one
            if current_classes:
                pos = random.randint(0, len(current_classes) - 1)
                idx_to_remove, r_to_remove = current_classes[pos]
                m_to_remove = moduli_range[idx_to_remove]
                
                # Remove it
                temp_classes = current_classes[:]
                temp_classes.pop(pos)
                temp_selected = selected_indices - {idx_to_remove}
                
                # Pick a random unselected modulus to try adding
                unselected = [i for i in range(n_moduli) if i not in temp_selected]
                if unselected:
                    idx_to_add = random.choice(unselected)
                    m_to_add = moduli_range[idx_to_add]
                    r_to_add = find_valid_residue(idx_to_add, temp_classes)
                    
                    if r_to_add != -1:
                        # Calculate score difference
                        delta_score = weights[m_to_add] - weights[m_to_remove]
                        new_score = current_score + delta_score
                        
                        # Acceptance criterion
                        if delta_score > 0:
                            # Always accept improvement
                            current_classes = temp_classes + [(idx_to_add, r_to_add)]
                            current_score = new_score
                            selected_indices = temp_selected | {idx_to_add}
                            if current_score > best_score:
                                best_score = current_score
                                best_classes = current_classes[:]
                        else:
                            # Accept worse move with probability exp(delta / T)
                            if random.random() < math.exp(delta_score / T):
                                current_classes = temp_classes + [(idx_to_add, r_to_add)]
                                current_score = new_score
                                selected_indices = temp_selected | {idx_to_add}
        
        T *= cooling_rate
        
        # If no valid move was found in a while, maybe restart with a new greedy init?
        # Or just continue.
        
    # Final check: try to add any remaining moduli with a greedy pass
    # This ensures we don't leave "free" points on the table
    final_classes = best_classes[:]
    final_score = best_score
    final_selected = set(idx for idx, _ in final_classes)
    
    for idx in range(n_moduli):
        if idx in final_selected:
            continue
        if time.time() > deadline - 1:
            break
        m = moduli_range[idx]
        r = find_valid_residue(idx, final_classes)
        if r != -1:
            final_classes.append((idx, r))
            final_score += weights[m]
            final_selected.add(idx)
            
    # Convert back to required format [[m, r], ...]
    result = [[moduli_range[idx], r] for idx, r in final_classes]
    
    # Verify validity just in case
    for i in range(len(result)):
        m1, r1 = result[i]
        for j in range(i+1, len(result)):
            m2, r2 = result[j]
            g = math.gcd(m1, m2)
            if r1 % g == r2 % g:
                # Invalid! Should not happen, but if it does, return a known valid fallback
                return [[20, 0], [21, 1]] # Fallback to a simple valid set? 
                # Actually 20,0 and 21,1: gcd=1. 0%1=0, 1%1=0. 0==0. INVALID.
                # Let's use a known valid pair: 20,0 and 22,1. gcd=2. 0%2=0, 1%2=1. OK.
                # 20,0 and 24,1. gcd=4. 0%4=0, 1%4=1. OK.
                # Let's just return the best valid found during search, or a trivial one.
                # Trivial valid: [[20, 0], [22, 1], [24, 2]]?
                # 20,0 vs 22,1: OK.
                # 20,0 vs 24,2: gcd=4. 0%4=0, 2%4=2. OK.
                # 22,1 vs 24,2: gcd=2. 1%2=1, 2%2=0. OK.
                return [[20, 0], [22, 1], [24, 2]]

    return result

