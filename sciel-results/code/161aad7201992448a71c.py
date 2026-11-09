import time
from math import gcd
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    
    # Precompute gcd table
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def check_pair_disjoint(m1_idx, r1, m2_idx, r2):
        """Check if two classes are disjoint."""
        g = gcd_table[m1_idx][m2_idx]
        if g == 1:
            return True
        return (r1 % g) != (r2 % g)
    
    def is_valid(classes):
        """Check if all pairs are disjoint."""
        n = len(classes)
        for i in range(n):
            for j in range(i + 1, n):
                m1, r1 = classes[i]
                m2, r2 = classes[j]
                g = gcd(m1, m2)
                if g > 1 and (r1 % g) == (r2 % g):
                    return False
        return True
    
    def score(classes):
        """Compute the score."""
        total = Fraction(0, 1)
        for m, r in classes:
            total += Fraction(1, m)
        return float(total)
    
    def try_add(classes, m, r):
        """Check if we can add (m, r) to classes."""
        for m2, r2 in classes:
            g = gcd(m, m2)
            if g > 1 and (r % g) == (r2 % g):
                return False
        return True
    
    def greedy_construct(seed):
        """Greedy construction with randomization."""
        rng = random.Random(seed)
        # Sort moduli by m (ascending) for greedy, but add some randomization
        order = list(range(n_mod))
        # Primarily by m ascending, with small random perturbation
        order.sort(key=lambda i: (moduli[i] + rng.random() * 2, rng.random()))
        
        classes = []
        for idx in order:
            m = moduli[idx]
            # Try all residues for this modulus
            best_r = None
            for r in range(m):
                if try_add(classes, m, r):
                    best_r = r
                    break
            if best_r is not None:
                classes.append([m, best_r])
        return classes
    
    def local_search(classes, seed):
        """Local search to improve the solution."""
        rng = random.Random(seed)
        classes = [list(c) for c in classes]
        current_score = score(classes)
        best_classes = [list(c) for c in classes]
        best_score = current_score
        
        # Try to add unselected moduli
        selected_mods = set(m for m, r in classes)
        
        max_iters = 5000
        for it in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            
            # Try to add a random unselected modulus
            unselected = [m for m in moduli if m not in selected_mods]
            if unselected:
                m = rng.choice(unselected)
                # Try all residues
                for r in range(m):
                    if try_add(classes, m, r):
                        new_classes = classes + [[m, r]]
                        new_score = score(new_classes)
                        if new_score > best_score:
                            best_score = new_score
                            best_classes = [list(c) for c in new_classes]
                            classes = new_classes
                            selected_mods.add(m)
                        break
            
            # Try swapping a residue to enable adding another modulus
            if classes:
                i = rng.randint(0, len(classes) - 1)
                m_i, r_i = classes[i]
                new_r = rng.randint(0, m_i - 1)
                old_r = classes[i][1]
                classes[i][1] = new_r
                
                # Check validity
                valid = True
                for j in range(len(classes)):
                    if j == i:
                        continue
                    m_j, r_j = classes[j]
                    g = gcd(m_i, m_j)
                    if g > 1 and (new_r % g) == (r_j % g):
                        valid = False
                        break
                
                if valid:
                    new_score = score(classes)
                    if new_score > best_score:
                        best_score = new_score
                        best_classes = [list(c) for c in classes]
                else:
                    classes[i][1] = old_r
        
        return best_classes
    
    # Multi-start
    best_classes = None
    best_score = 0.0
    
    # Start with greedy from different seeds
    for seed in range(20):
        if time.time() > deadline - 5.0:
            break
        classes = greedy_construct(seed)
        s = score(classes)
        if s > best_score:
            best_score = s
            best_classes = [list(c) for c in classes]
    
    # Local search from best solution
    if best_classes is not None:
        if time.time() < deadline - 10.0:
            improved = local_search(best_classes, 42)
            s = score(improved)
            if s > best_score:
                best_score = s
                best_classes = improved
    
    # More local search iterations with different seeds
    for seed in range(42, 62):
        if time.time() > deadline - 5.0:
            break
        if best_classes is None:
            break
        improved = local_search(best_classes, seed)
        s = score(improved)
        if s > best_score:
            best_score = s
            best_classes = improved
    
    if best_classes is None:
        # Fallback
        return [[20, 0], [21, 0], [22, 0], [23, 0], [24, 0]]
    
    return best_classes

