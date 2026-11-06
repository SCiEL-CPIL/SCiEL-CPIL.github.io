"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    """
    Maximize a family of 4-element subsets of [1,16] with no three distinct sets
    having all three pairwise intersections equal (sunflower condition).
    
    Strategy: Greedy construction with randomization and local search.
    1. Precompute all C(16,4) = 1820 four-element subsets.
    2. Use a greedy algorithm that adds sets in a random order, checking the sunflower constraint.
    3. Try multiple random orderings and keep the best result.
    4. Apply local search: try to add more sets or swap sets to improve the family size.
    
    Time budget: ~85 seconds (with safety margin before 90s kill).
    """
    start_time = time.time()
    DEADLINE = 85.0  # safety margin
    
    # Precompute all 4-subsets of [1,16]
    all_sets = list(combinations(range(1, 17), 4))
    n_sets = len(all_sets)  # 1820
    
    # Convert to frozensets for fast intersection
    all_fsets = [frozenset(s) for s in all_sets]
    
    # For each pair of sets, precompute whether they form a "conflict" with any third set
    # Actually, the sunflower condition: no three distinct sets A, B, C where
    # |A∩B| = |A∩C| = |B∩C| (all three pairwise intersections have the same size)
    
    # Wait, let me re-read: "No three distinct sets may have all three pairwise intersections equal"
    # This means |A∩B| = |A∩C| = |B∩C| for some three distinct sets A, B, C.
    # The "including an empty common core" suggests that even if the intersection is empty,
    # if all three pairwise intersections are empty (size 0), that's also forbidden.
    
    # So the constraint is: for any three distinct sets A, B, C in the family,
    # it must NOT be the case that |A∩B| = |A∩C| = |B∩C|.
    
    # Precompute pairwise intersection sizes
    # inter[i][j] = |all_fsets[i] ∩ all_fsets[j]|
    inter = [[0]*n_sets for _ in range(n_sets)]
    for i in range(n_sets):
        si = all_fsets[i]
        for j in range(i+1, n_sets):
            ij = len(si & all_fsets[j])
            inter[i][j] = ij
            inter[j][i] = ij
    
    def check_sunflower(family_indices):
        """Check if the family (list of indices) has no sunflower triple."""
        fam = family_indices
        n = len(fam)
        for i in range(n):
            a = fam[i]
            for j in range(i+1, n):
                b = fam[j]
                ab = inter[a][b]
                for k in range(j+1, n):
                    c = fam[k]
                    if inter[a][c] == ab and inter[b][c] == ab:
                        return False
        return True
    
    def can_add(family_indices, new_idx):
        """Check if adding new_idx to family_indices violates the sunflower constraint."""
        fam = family_indices
        n = len(fam)
        for i in range(n):
            a = fam[i]
            for j in range(i+1, n):
                b = fam[j]
                ab = inter[a][b]
                if inter[a][new_idx] == ab and inter[b][new_idx] == ab:
                    return False
        return True
    
    # Greedy with random ordering
    best_family = []
    best_size = 0
    
    random.seed(42)
    
    max_iterations = 200  # number of random restarts to try
    iter_count = 0
    
    while iter_count < max_iterations:
        if time.time() - start_time > DEADLINE - 5.0:
            break
        
        # Random permutation of all sets
        order = list(range(n_sets))
        random.shuffle(order)
        
        family = []
        for idx in order:
            if can_add(family, idx):
                family.append(idx)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
        
        iter_count += 1
    
    # Local search: try to improve the best family
    if time.time() - start_time < DEADLINE - 2.0:
        family = best_family[:]
        improved = True
        while improved and time.time() - start_time < DEADLINE - 2.0:
            improved = False
            # Try adding any set not in family
            in_family = set(family)
            for idx in range(n_sets):
                if idx not in in_family:
                    if can_add(family, idx):
                        family.append(idx)
                        in_family.add(idx)
                        improved = True
                        if len(family) > best_size:
                            best_size = len(family)
                            best_family = family[:]
                        break  # restart scanning from beginning after an addition
    
    # Try more random restarts if time permits
    if time.time() - start_time < DEADLINE - 1.0:
        while time.time() - start_time < DEADLINE - 1.0:
            order = list(range(n_sets))
            random.shuffle(order)
            family = []
            for idx in order:
                if can_add(family, idx):
                    family.append(idx)
            if len(family) > best_size:
                best_size = len(family)
                best_family = family[:]
    
    # Return the best family found
    return [sorted(all_sets[i]) for i in best_family]
# EVOLVE-BLOCK-END
