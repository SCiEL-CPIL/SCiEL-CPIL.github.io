"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit
    
    # Moduli in [20, 200]
    moduli = list(range(20, 201))
    
    best_score = 0.0
    best_classes = []
    
    def is_compatible(m_new, r_new, selected_list):
        """Check if (m_new, r_new) is disjoint from all in selected_list."""
        for m_old, r_old in selected_list:
            g = math.gcd(m_new, m_old)
            if (r_new - r_old) % g == 0:
                return False
        return True
    
    def get_valid_residues(m_new, selected_list):
        """Get all valid residues for m_new given selected_list."""
        valid = []
        for r in range(m_new):
            if is_compatible(m_new, r, selected_list):
                valid.append(r)
        return valid
    
    def run_greedy_randomized(moduli_list, seed):
        """Run greedy randomized construction."""
        rng = random.Random(seed)
        shuffled = moduli_list[:]
        rng.shuffle(shuffled)
        
        selected = []
        for m in shuffled:
            valid_r = get_valid_residues(m, selected)
            if valid_r:
                r = rng.choice(valid_r)
                selected.append([m, r])
        
        # Calculate score
        score = sum(1.0 / m for m, r in selected)
        return selected, score
    
    def local_search(classes, seed, max_iter=500):
        """Try to improve by reassigning residues or swapping moduli."""
        rng = random.Random(seed)
        classes = [c[:] for c in classes]
        current_score = sum(1.0 / m for m, r in classes)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try reassigning a residue for an existing modulus
            idx = rng.randrange(len(classes))
            m, r = classes[idx]
            others = [classes[i] for i in range(len(classes)) if i != idx]
            
            valid_r = get_valid_residues(m, others)
            if len(valid_r) > 1:
                new_r = rng.choice([x for x in valid_r if x != r])
                if new_r != r:
                    classes[idx][1] = new_r
                    # This doesn't change score, but might unblock others
                    improved = True
            
            # Try adding a new modulus
            if not improved:
                used_moduli = set(m for m, r in classes)
                candidates = [m for m in moduli if m not in used_moduli]
                if candidates:
                    m_new = rng.choice(candidates)
                    valid_r = get_valid_residues(m_new, classes)
                    if valid_r:
                        r_new = rng.choice(valid_r)
                        new_classes = classes + [[m_new, r_new]]
                        new_score = current_score + 1.0 / m_new
                        if new_score > current_score:
                            classes = new_classes
                            current_score = new_score
                            improved = True
            
            # Try removing a large modulus and adding a smaller one
            if not improved and len(classes) > 1:
                # Find the largest modulus
                max_idx = max(range(len(classes)), key=lambda i: classes[i][0])
                m_remove = classes[max_idx][0]
                others = [classes[i] for i in range(len(classes)) if i != max_idx]
                
                used_moduli = set(m for m, r in others)
                candidates = [m for m in moduli if m not in used_moduli and m < m_remove]
                if candidates:
                    m_new = rng.choice(candidates)
                    valid_r = get_valid_residues(m_new, others)
                    if valid_r:
                        r_new = rng.choice(valid_r)
                        new_classes = others + [[m_new, r_new]]
                        new_score = sum(1.0 / m for m, r in new_classes)
                        if new_score > current_score:
                            classes = new_classes
                            current_score = new_score
                            improved = True
        
        return classes, current_score
    
    # Run multiple restarts with different strategies
    for seed in range(10000):
        if time.time() > deadline:
            break
        
        # Strategy 1: Greedy randomized
        classes, score = run_greedy_randomized(moduli, seed)
        if score > best_score:
            best_score = score
            best_classes = classes
        
        # Strategy 2: Local search from greedy result
        if time.time() < deadline - 5:
            classes2, score2 = local_search(classes, seed, max_iter=200)
            if score2 > best_score:
                best_score = score2
                best_classes = classes2
    
    # Final local search on best result
    if time.time() < deadline - 5 and best_classes:
        classes_final, score_final = local_search(best_classes, 42, max_iter=1000)
        if score_final > best_score:
            best_score = score_final
            best_classes = classes_final
    
    # If we didn't find anything, return a basic valid construction
    if not best_classes:
        best_classes = [[20 * j, j - 1] for j in range(1, 11)]
        best_score = sum(1.0 / m for m, r in best_classes)
    
    return best_classes
# EVOLVE-BLOCK-END
