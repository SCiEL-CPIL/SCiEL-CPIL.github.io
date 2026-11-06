"""Rewritten program: Randomized greedy with restarts for pairwise disjoint residue classes."""
# EVOLVE-BLOCK-START
import math
import random


def construct():
    moduli = list(range(20, 201))
    n_moduli = len(moduli)
    
    # Precompute gcds for all pairs of moduli
    # Index 0..179 corresponds to m=20..200
    # gcd_matrix[i][j] = gcd(moduli[i], moduli[j])
    gcd_matrix = [[0] * n_moduli for _ in range(n_moduli)]
    for i in range(n_moduli):
        for j in range(i, n_moduli):
            g = math.gcd(moduli[i], moduli[j])
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g
            
    best_classes = []
    best_score = 0.0
    
    # Number of restarts
    # Estimate: ~180 moduli, ~100 chosen, ~10 checks per residue try, ~5 tries per modulus.
    # Ops per run ~ 180 * 5 * 100 * 5 = 450,000 gcd checks.
    # 1000 runs ~ 450,000,000 checks. Might be tight.
    # Let's start with 500 restarts.
    num_restarts = 500
    
    for _ in range(num_restarts):
        chosen = []  # list of (m_idx, r)
        score = 0.0
        
        for i in range(n_moduli):
            m = moduli[i]
            valid_rs = []
            
            # Check all residues for this modulus
            for r in range(m):
                is_ok = True
                # Check against all previously chosen
                for (j, r2) in chosen:
                    g = gcd_matrix[i][j]
                    if r % g == r2 % g:
                        is_ok = False
                        break
                if is_ok:
                    valid_rs.append(r)
            
            if not valid_rs:
                continue
                
            # Randomly select a valid residue
            r = random.choice(valid_rs)
            chosen.append((i, r))
            score += 1.0 / m
            
        if score > best_score:
            best_score = score
            # Convert back to [m, r] format
            best_classes = [[moduli[idx], r] for (idx, r) in chosen]
            
    return best_classes
# EVOLVE-BLOCK-END
