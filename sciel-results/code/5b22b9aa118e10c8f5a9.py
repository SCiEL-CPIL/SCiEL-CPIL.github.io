"""Model-and-optimize approach for pairwise disjoint residue classes."""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
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
    
    # For each pair of moduli, precompute which residues are compatible
    # compatible[i][j] is a set of residues r_i in [0, moduli[i]) that are compatible
    # with each residue r_j in [0, moduli[j])
    # Actually, let's think differently.
    
    # The constraint: for classes (m1, r1) and (m2, r2), we need r1 != r2 mod gcd(m1, m2)
    # So for a given modulus m and residue r, it conflicts with (m2, r2) if r ≡ r2 (mod gcd(m, m2))
    
    # Strategy: This is essentially a constraint satisfaction / optimization problem.
    # We want to select a subset of moduli and assign residues to maximize sum of 1/m.
    
    # Key insight: If two moduli are coprime (gcd=1), then ANY residue is valid for both
    # because r1 != r2 mod 1 is always true (since there's only 1 residue class mod 1).
    # So coprime moduli are always compatible regardless of residues.
    
    # For moduli with gcd > 1, we need to avoid r1 ≡ r2 (mod g).
    
    # Model: We can think of this as a graph problem where nodes are (m, r) pairs,
    # but that's too large. Instead, let's use a different approach.
    
    # Key observation: For a given set of moduli, we can check if there exist valid residues
    # using a backtracking or constraint propagation approach.
    
    # Better approach: Since we want to maximize sum of 1/m, we should prefer smaller moduli.
    # The optimal solution likely includes many small moduli.
    
    # Let's use a greedy + local search approach with a smarter residue selection.
    
    def can_add(classes, m, r):
        """Check if class (m, r) is disjoint from all classes in the set."""
        mi = mod_idx[m]
        for (m2, r2) in classes:
            mj = mod_idx[m2]
            g = gcd_table[mi][mj]
            if (r - r2) % g == 0:
                return False
        return True
    
    def find_best_residue(classes, m):
        """Find the best residue for modulus m given existing classes.
        Returns (residue, num_conflicts_avoided) or (None, 0) if no valid residue."""
        mi = mod_idx[m]
        valid_residues = []
        for r in range(m):
            if can_add(classes, m, r):
                valid_residues.append(r)
        
        if not valid_residues:
            return None, 0
        
        # Score each valid residue by how much "slack" it provides
        # A residue r has slack if for each existing class (m2, r2),
        # the constraint (r - r2) % g != 0 is "far from being violated"
        # We can measure this as the minimum over all (m2,r2) of 
        # min(|r - r2 mod g|, g - |r - r2 mod g|)
        # Larger minimum distance means more robust.
        
        best_r = valid_residues[0]
        best_slack = -1
        
        for r in valid_residues:
            min_dist = m
            for (m2, r2) in classes:
                mj = mod_idx[m2]
                g = gcd_table[mi][mj]
                if g > 1:
                    diff = (r - r2) % g
                    dist = min(diff, g - diff)
                    min_dist = min(min_dist, dist)
            if min_dist > best_slack:
                best_slack = min_dist
                best_r = r
        
        return best_r, best_slack
    
    def score(classes):
        """Compute sum of 1/m for the set of classes."""
        total = 0.0
        for (m, r) in classes:
            total += 1.0 / m
        return total
    
    def score_exact(classes):
        """Compute exact score using Fraction."""
        total = Fraction(0, 1)
        for (m, r) in classes:
            total += Fraction(1, m)
        return total
    
    def greedy_construct(order=None):
        """Greedy construction: add classes in the given order if possible."""
        if order is None:
            order = sorted(moduli, key=lambda m: 1.0/m)  # smallest moduli first (highest 1/m)
        classes = []
        used_moduli = set()
        for m in order:
            if m in used_moduli:
                continue
            r, slack = find_best_residue(classes, m)
            if r is not None:
                classes.append([m, r])
                used_moduli.add(m)
        return classes
    
    def try_add_all(classes, max_time=5.0):
        """Try to add all remaining moduli, starting with smallest."""
        used_moduli = set(m for m, r in classes)
        remaining = [m for m in sorted(moduli) if m not in used_moduli]
        
        for m in remaining:
            if time.time() - start_time > deadline - 2:
                break
            r, slack = find_best_residue(classes, m)
            if r is not None:
                classes.append([m, r])
                used_moduli.add(m)
        
        return classes
    
    def local_search(classes, max_iter=200):
        """Improve by trying to add missing moduli or swap residues."""
        best_classes = [list(c) for c in classes]
        best_score = score(best_classes)
        used_moduli = set(m for m, r in best_classes)
        n_classes = len(best_classes)
        
        # Precompute gcd for pairs involving current moduli for faster checks
        for iteration in range(max_iter):
            if time.time() - start_time > deadline - 3:
                break
            
            improved = False
            n_classes = len(best_classes)
            
            # Try adding a new modulus (smallest first for best 1/m)
            for m in sorted(moduli):
                if m in used_moduli:
                    continue
                r, slack = find_best_residue(best_classes, m)
                if r is not None:
                    new_classes = best_classes + [[m, r]]
                    new_score = best_score + 1.0/m
                    if new_score > best_score + 1e-12:
                        best_classes = new_classes
                        best_score = new_score
                        used_moduli.add(m)
                        improved = True
                        break
            
            # Try swapping a residue for a better one (only if it enables adding more)
            if not improved:
                n_classes = len(best_classes)
                for i in range(n_classes):
                    m, r = best_classes[i]
                    mi = mod_idx[m]
                    # Try all other residues for this modulus
                    best_r = r
                    best_slack = -1
                    for r2 in range(m):
                        if r2 == r:
                            continue
                        # Check if r2 is valid
                        valid = True
                        for j in range(n_classes):
                            if i == j:
                                continue
                            m2, r2_other = best_classes[j]
                            mj = mod_idx[m2]
                            g = gcd_table[mi][mj]
                            if (r2 - r2_other) % g == 0:
                                valid = False
                                break
                        if not valid:
                            continue
                        # Calculate slack
                        min_dist = m
                        for j in range(n_classes):
                            if i == j:
                                continue
                            m2, r2_other = best_classes[j]
                            mj = mod_idx[m2]
                            g = gcd_table[mi][mj]
                            if g > 1:
                                diff = (r2 - r2_other) % g
                                dist = min(diff, g - diff)
                                min_dist = min(min_dist, dist)
                        if min_dist > best_slack:
                            best_slack = min_dist
                            best_r = r2
                    
                    if best_r != r:
                        # Swapping might allow adding more moduli
                        temp_classes = [list(c) for c in best_classes]
                        temp_classes[i][1] = best_r
                        # Try to add more
                        added = try_add_all(temp_classes, max_time=0.5)
                        if score(added) > best_score + 1e-12:
                            best_classes = added
                            best_score = score(best_classes)
                            used_moduli = set(m for m, r in best_classes)
                            improved = True
                            break
            
            if not improved:
                break
        
        return best_classes
    
    # Multi-start approach
    best_classes = None
    best_score = 0.0
    
    # Start 1: Greedy with smallest moduli first
    classes = greedy_construct()
    classes = try_add_all(classes)
    classes = local_search(classes, max_iter=500)
    s = score(classes)
    if s > best_score:
        best_score = s
        best_classes = [list(c) for c in classes]
    
    # Start 2: Greedy with random ordering biased toward small moduli
    random.seed(123)
    num_starts = 80
    
    for start_idx in range(num_starts):
        if time.time() - start_time > deadline - 15:
            break
        
        # Create a random ordering
        remaining = list(moduli)
        order = []
        while remaining:
            # Weight by 1/m^alpha to bias toward small moduli
            alpha = 1.0 + start_idx * 0.05
            weights = [1.0 / (m ** alpha) for m in remaining]
            total_w = sum(weights)
            if total_w <= 0:
                order.append(remaining[0])
                remaining.pop(0)
            else:
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
        classes = try_add_all(classes)
        
        if time.time() - start_time > deadline - 8:
            s = score(classes)
            if s > best_score:
                best_score = s
                best_classes = [list(c) for c in classes]
            continue
        
        classes = local_search(classes, max_iter=150)
        s = score(classes)
        if s > best_score:
            best_score = s
            best_classes = [list(c) for c in classes]
    
    # Final polish: intensive local search on best solution
    if best_classes is not None:
        # Run multiple rounds of local search with increasing iteration counts
        for round_num in range(3):
            if time.time() - start_time > deadline - 2:
                break
            max_iter = 300 + round_num * 200
            improved_classes = local_search(best_classes, max_iter=max_iter)
            improved_score = score(improved_classes)
            if improved_score > best_score + 1e-12:
                best_classes = improved_classes
                best_score = improved_score
            else:
                break
    
    return best_classes if best_classes is not None else []
# EVOLVE-BLOCK-END
