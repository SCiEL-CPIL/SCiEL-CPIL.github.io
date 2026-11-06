import math
import time
import random

def construct():
    # Deadline guard: 120s limit, use 110s to be safe
    deadline = time.time() + 110
    
    moduli = list(range(20, 201))
    num_moduli = len(moduli)
    
    # Precompute GCD matrix for speed
    # gcd_mat[i][j] = gcd(moduli[i], moduli[j])
    gcd_mat = [[0] * num_moduli for _ in range(num_moduli)]
    for i in range(num_moduli):
        for j in range(i, num_moduli):
            g = math.gcd(moduli[i], moduli[j])
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g
            
    best_score = 0.0
    best_classes = []
    
    # Multi-start loop
    while time.time() < deadline:
        current_classes = []
        # List of indices of selected moduli for faster GCD lookup
        selected_indices = []
        
        for idx_m, m in enumerate(moduli):
            # Find valid residues for m
            valid_rs = []
            for r in range(m):
                is_valid = True
                # Check against all selected
                for idx_sel in selected_indices:
                    g = gcd_mat[idx_m][idx_sel]
                    r_sel = current_classes[idx_sel][1]
                    if r % g == r_sel % g:
                        is_valid = False
                        break
                if is_valid:
                    valid_rs.append(r)
            
            if valid_rs:
                # Pick random valid residue
                r = random.choice(valid_rs)
                current_classes.append((m, r))
                selected_indices.append(idx_m)
                
        # Calculate score
        score = sum(1.0 / m for m, _ in current_classes)
        
        if score > best_score:
            best_score = score
            best_classes = current_classes
            
    # Convert to required format [[m, r], ...]
    return [[m, r] for m, r in best_classes]
