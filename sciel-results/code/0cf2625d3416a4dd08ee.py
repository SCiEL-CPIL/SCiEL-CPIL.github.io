"""Improved program: Simulated Annealing for pairwise disjoint residue classes."""
# EVOLVE-BLOCK-START
import math
import random


def construct():
    moduli = list(range(20, 201))
    n_moduli = len(moduli)
    
    # Precompute gcds for all pairs of moduli
    gcd_matrix = [[0] * n_moduli for _ in range(n_moduli)]
    for i in range(n_moduli):
        for j in range(i, n_moduli):
            g = math.gcd(moduli[i], moduli[j])
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g
            
    def is_valid_addition(idx_m, r, chosen_indices, chosen_rs):
        for k in range(len(chosen_indices)):
            idx_c = chosen_indices[k]
            r_c = chosen_rs[k]
            g = gcd_matrix[idx_m][idx_c]
            if r % g == r_c % g:
                return False
        return True
        
    def get_score(chosen_indices):
        return sum(1.0 / moduli[idx] for idx in chosen_indices)
        
    # Greedy initialization
    def greedy_init():
        chosen_indices = []
        chosen_rs = []
        indices = list(range(n_moduli))
        # Sort by 1/m descending (m ascending)
        indices.sort(key=lambda i: moduli[i])
        
        for i in indices:
            m = moduli[i]
            # Try random residues
            rs = list(range(m))
            random.shuffle(rs)
            for r in rs:
                if is_valid_addition(i, r, chosen_indices, chosen_rs):
                    chosen_indices.append(i)
                    chosen_rs.append(r)
                    break
        return chosen_indices, chosen_rs
        
    best_indices, best_rs = greedy_init()
    best_score = get_score(best_indices)
    
    current_indices, current_rs = best_indices[:], best_rs[:]
    current_score = best_score
    
    # Simulated Annealing
    T = 0.1
    cooling = 0.9995
    num_iterations = 20000
    
    for _ in range(num_iterations):
        # Random move
        move = random.random()
        
        if move < 0.4: # Add
            current_set = set(current_indices)
            available = [i for i in range(n_moduli) if i not in current_set]
            if available:
                i = random.choice(available)
                m = moduli[i]
                # Try to find valid r
                for _ in range(5):
                    r = random.randint(0, m-1)
                    if is_valid_addition(i, r, current_indices, current_rs):
                        new_indices = current_indices + [i]
                        new_rs = current_rs + [r]
                        new_score = current_score + 1.0/m
                        delta = new_score - current_score
                        if delta > 0 or random.random() < math.exp(delta/T):
                            current_indices = new_indices
                            current_rs = new_rs
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_indices = current_indices[:]
                                best_rs = current_rs[:]
                        break
                    
        elif move < 0.7: # Remove
            if current_indices:
                k = random.randint(0, len(current_indices)-1)
                i = current_indices[k]
                new_indices = current_indices[:k] + current_indices[k+1:]
                new_rs = current_rs[:k] + current_rs[k+1:]
                new_score = current_score - 1.0/moduli[i]
                delta = new_score - current_score
                if random.random() < math.exp(delta/T):
                    current_indices = new_indices
                    current_rs = new_rs
                    current_score = new_score
                    
        else: # Change residue
            if current_indices:
                k = random.randint(0, len(current_indices)-1)
                i = current_indices[k]
                m = moduli[i]
                for _ in range(5):
                    r_new = random.randint(0, m-1)
                    if r_new == current_rs[k]: continue
                    # Check validity
                    valid = True
                    for j in range(len(current_indices)):
                        if j == k: continue
                        idx_c = current_indices[j]
                        r_c = current_rs[j]
                        g = gcd_matrix[i][idx_c]
                        if r_new % g == r_c % g:
                            valid = False
                            break
                    if valid:
                        new_rs = current_rs[:]
                        new_rs[k] = r_new
                        # Score unchanged, but might help future moves
                        # Accept with probability 1 (or similar) to explore
                        current_rs = new_rs
                        break
        
        T *= cooling
        
    return [[moduli[idx], r] for idx, r in zip(best_indices, best_rs)]
# EVOLVE-BLOCK-END
