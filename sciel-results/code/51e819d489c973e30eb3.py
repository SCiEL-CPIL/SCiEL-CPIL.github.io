import math
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 110  # Safety margin before 120s kill
    
    # Moduli in [20, 200]
    moduli = list(range(20, 201))
    n_mods = len(moduli)
    mod_index = {m: i for i, m in enumerate(moduli)}
    
    # Precompute gcd for all pairs
    gcd_matrix = [[0] * n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(i, n_mods):
            g = math.gcd(moduli[i], moduli[j])
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g
            
    # Precompute 1/m for each modulus
    reciprocals = [1.0 / moduli[i] for i in range(n_mods)]
    
    def check_validity(indices, residues):
        """Check if the set of (modulus, residue) pairs is pairwise disjoint."""
        n = len(indices)
        for i in range(n):
            mi = indices[i]
            ri = residues[i]
            for j in range(i + 1, n):
                mj = indices[j]
                rj = residues[j]
                g = gcd_matrix[mi][mj]
                if ri % g == rj % g:
                    return False
        return True
        
    def get_score(indices):
        return sum(reciprocals[i] for i in indices)
        
    def greedy_init():
        """
        Greedy construction: add moduli in order of increasing m (decreasing 1/m)
        if a valid residue exists.
        """
        selected_indices = []
        selected_residues = []
        
        # Sort by 1/m descending (i.e., m ascending)
        for m_idx in range(n_mods):
            m = moduli[m_idx]
            found = False
            # Try residues 0 to m-1
            for r in range(m):
                # Check against all selected
                valid = True
                for k, sel_idx in enumerate(selected_indices):
                    g = gcd_matrix[m_idx][sel_idx]
                    if r % g == selected_residues[k] % g:
                        valid = False
                        break
                if valid:
                    selected_indices.append(m_idx)
                    selected_residues.append(r)
                    found = True
                    break
            # If not found, skip
        return selected_indices, selected_residues

    def local_search(selected_indices, selected_residues, time_limit):
        """
        Simulated Annealing / Local Search to improve the set.
        """
        current_score = get_score(selected_indices)
        best_indices = list(selected_indices)
        best_residues = list(selected_residues)
        best_score = current_score
        
        start_time = time.time()
        temp = 0.5
        cooling = 0.9999
        
        while time.time() < time_limit:
            # Generate a move
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to Add a new modulus
                unselected = [i for i in range(n_mods) if i not in selected_indices]
                if not unselected:
                    continue
                m_idx = random.choice(unselected)
                m = moduli[m_idx]
                # Find a valid residue
                valid_rs = []
                for r in range(m):
                    valid = True
                    for k, sel_idx in enumerate(selected_indices):
                        g = gcd_matrix[m_idx][sel_idx]
                        if r % g == selected_residues[k] % g:
                            valid = False
                            break
                    if valid:
                        valid_rs.append(r)
                if valid_rs:
                    r = random.choice(valid_rs)
                    # Apply move
                    new_indices = selected_indices + [m_idx]
                    new_residues = selected_residues + [r]
                    new_score = get_score(new_indices)
                    delta = new_score - current_score
                    if delta > 0 or random.random() < math.exp(delta / temp):
                        selected_indices = new_indices
                        selected_residues = new_residues
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_indices = list(selected_indices)
                            best_residues = list(selected_residues)
                            
            elif move_type < 0.7:
                # Try to Remove a modulus
                if not selected_indices:
                    continue
                rem_idx = random.choice(selected_indices)
                pos = selected_indices.index(rem_idx)
                new_indices = selected_indices[:pos] + selected_indices[pos+1:]
                new_residues = selected_residues[:pos] + selected_residues[pos+1:]
                new_score = get_score(new_indices)
                delta = new_score - current_score
                if random.random() < math.exp(delta / temp):
                    selected_indices = new_indices
                    selected_residues = new_residues
                    current_score = new_score
                    
            else:
                # Try to Change Residue of an existing modulus
                if not selected_indices:
                    continue
                chg_idx = random.choice(selected_indices)
                pos = selected_indices.index(chg_idx)
                m = moduli[chg_idx]
                old_r = selected_residues[pos]
                
                found_new_r = False
                for _ in range(10):
                    new_r = random.randint(0, m - 1)
                    if new_r == old_r:
                        continue
                    valid = True
                    for k, sel_idx in enumerate(selected_indices):
                        if k == pos:
                            continue
                        g = gcd_matrix[chg_idx][sel_idx]
                        if new_r % g == selected_residues[k] % g:
                            valid = False
                            break
                    if valid:
                        new_residues = selected_residues[:]
                        new_residues[pos] = new_r
                        selected_residues = new_residues
                        found_new_r = True
                        break
                        
            temp *= cooling
            
        return best_indices, best_residues, best_score

    # Run multiple restarts
    best_global_indices, best_global_residues = greedy_init()
    best_global_score = get_score(best_global_indices)
    
    start_time = time.time()
    num_restarts = 0
    while time.time() < deadline - 5:
        init_indices, init_residues = greedy_init()
        
        sa_time = min(10.0, deadline - 5 - time.time())
        if sa_time <= 0:
            break
            
        res_indices, res_residues, res_score = local_search(init_indices, init_residues, sa_time)
        
        if res_score > best_global_score:
            best_global_score = res_score
            best_global_indices = res_indices
            best_global_residues = res_residues
            
        num_restarts += 1
        
    return [[moduli[i], r] for i, r in zip(best_global_indices, best_global_residues)]
