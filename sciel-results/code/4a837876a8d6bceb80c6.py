"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Precompute gcd table for moduli 20..200
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    mod_idx = {m: i for i, m in enumerate(moduli)}
    
    # gcd_table[i][j] = gcd(moduli[i], moduli[j])
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def can_add(classes, m, r):
        """Check if class (m, r) is disjoint from all classes in the set."""
        mi = mod_idx[m]
        for (m2, r2) in classes:
            mj = mod_idx[m2]
            g = gcd_table[mi][mj]
            if (r - r2) % g == 0:
                return False
        return True
    
    def score(classes):
        """Compute sum of 1/m for the set of classes."""
        total = Fraction(0, 1)
        for (m, r) in classes:
            total += Fraction(1, m)
        return total
    
    def greedy_construct(order=None):
        """Greedy construction: add classes in the given order if possible."""
        if order is None:
            order = sorted(moduli, key=lambda m: 1/m)  # smallest moduli first
        classes = []
        for m in order:
            # Try all residues for this modulus
            best_r = None
            # We just need one valid residue; try 0 first, then others
            # Actually, for greedy, we can try residues in a specific order
            # Let's try residue 0 first, then 1, etc.
            for r in range(m):
                if can_add(classes, m, r):
                    best_r = r
                    break
            if best_r is not None:
                classes.append([m, best_r])
        return classes
    
    def local_search(classes, max_iter=500):
        """Try to improve by swapping residues or removing/adding classes."""
        best_score = score(classes)
        best_classes = [list(c) for c in classes]
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try changing residue of each class
            for i in range(len(best_classes)):
                m, r = best_classes[i]
                # Try other residues
                for new_r in range(m):
                    if new_r == r:
                        continue
                    # Check if new residue is valid with all other classes
                    valid = True
                    for j in range(len(best_classes)):
                        if i == j:
                            continue
                        m2, r2 = best_classes[j]
                        g = gcd_table[mod_idx[m]][mod_idx[m2]]
                        if (new_r - r2) % g == 0:
                            valid = False
                            break
                    if valid:
                        # Check if this improves score (it doesn't change score directly,
                        # but might allow adding more classes later)
                        # For now, just accept if valid (residue doesn't affect score)
                        pass
            
            # Try adding a new class
            added = False
            for m in moduli:
                if any(m == cm for (cm, cr) in best_classes):
                    continue
                for r in range(m):
                    if can_add(best_classes, m, r):
                        new_classes = best_classes + [[m, r]]
                        new_score = score(new_classes)
                        if new_score > best_score:
                            best_classes = new_classes
                            best_score = new_score
                            added = True
                            improved = True
                            break
                if added:
                    break
            
            # Try removing a class to see if we can add two better ones
            # This is complex, skip for now
            
            if not improved:
                break
        
        return best_classes
    
    # Multi-start with different orderings
    best_classes = None
    best_score = Fraction(0, 1)
    
    # Start with greedy (smallest moduli first)
    classes = greedy_construct()
    classes = local_search(classes)
    s = score(classes)
    if s > best_score:
        best_score = s
        best_classes = [list(c) for c in classes]
    
    # Try randomized greedy multiple times
    random.seed(42)
    num_starts = 50
    for start in range(num_starts):
        if time.time() > deadline:
            break
        
        # Create a random ordering with bias toward small moduli
        weights = [1.0 / m for m in moduli]
        order = []
        remaining = list(moduli)
        while remaining:
            # Weighted random selection
            total_w = sum(weights[i] for i in range(len(remaining)))
            r = random.random() * total_w
            cum = 0
            chosen_idx = 0
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    chosen_idx = i
                    break
            order.append(remaining[chosen_idx])
            remaining.pop(chosen_idx)
        
        classes = greedy_construct(order)
        classes = local_search(classes, max_iter=200)
        s = score(classes)
        if s > best_score:
            best_score = s
            best_classes = [list(c) for c in classes]
    
    # Final local search on best solution
    if best_classes is not None:
        best_classes = local_search(best_classes, max_iter=1000)
        best_score = score(best_classes)
    
    return best_classes if best_classes is not None else []
# EVOLVE-BLOCK-END
