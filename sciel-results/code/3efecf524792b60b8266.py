import time
import math
import random

def _is_disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    MAX_MOD = 128
    
    # Precompute gcds
    gcds = [[0] * (MAX_MOD + 1) for _ in range(MAX_MOD + 1)]
    for i in range(2, MAX_MOD + 1):
        for j in range(2, MAX_MOD + 1):
            gcds[i][j] = math.gcd(i, j)

    # Moduli to consider
    moduli = list(range(2, MAX_MOD + 1))
    
    best_count = 0
    best_result = []
    
    # Backtracking search
    # We will try to add moduli in a specific order.
    # To find the maximum, we can use a greedy approach with randomization and restarts,
    # but with a better local search or smarter ordering.
    
    # Heuristic: Moduli with fewer coprime partners are "safer" to add?
    # Or moduli with larger gcds with others are more constrained.
    # Let's try sorting moduli by the number of other moduli they share a gcd > 1 with.
    
    # Count constraints for each modulus
    constraint_count = [0] * (MAX_MOD + 1)
    for m in moduli:
        for m2 in moduli:
            if m != m2 and gcds[m][m2] > 1:
                constraint_count[m] += 1
    
    # Sort by constraint count (descending) to handle constrained ones first
    # Or ascending to handle easy ones first?
    # Usually, most constrained first is better for CSP.
    
    def solve(moduli_order, selected, forbidden_sets):
        """
        selected: list of (r, m)
        forbidden_sets: dict m -> set of forbidden residues mod g for each g? 
        Actually, for a new m, we just check against selected.
        To speed up, we can maintain for each potential new modulus m, 
        a bitmask or set of forbidden residues in [0, m).
        """
        pass

    # Simpler approach: Randomized Greedy with Local Search
    # 1. Generate a random permutation of moduli.
    # 2. Greedily add if possible.
    # 3. If we get stuck, try to swap residues of existing moduli to free up space for others.
    
    def find_valid_residue(m, selected):
        # Returns a valid residue for m given selected list, or None
        # For speed, we can check all residues
        for r in range(m):
            valid = True
            for r_s, m_s in selected:
                if not _is_disjoint(r, m, r_s, m_s):
                    valid = False
                    break
            if valid:
                return r
        return None

    # Run multiple passes with different orderings
    # Ordering 1: Random
    # Ordering 2: Sorted by constraint count
    # Ordering 3: Sorted by value
    
    orderings = [
        moduli[:],
        sorted(moduli, key=lambda x: constraint_count[x], reverse=True),
        sorted(moduli, key=lambda x: constraint_count[x]),
        sorted(moduli, reverse=True),
        sorted(moduli)
    ]
    
    for _ in range(200): # Number of restarts
        if time.time() > deadline:
            break
            
        # Pick an ordering
        order = random.choice(orderings)
        if random.random() < 0.5:
            random.shuffle(order)
        else:
            # Perturb the sorted order
            pass
            
        current_selected = []
        
        for m in order:
            # Try to find a residue
            # Shuffle residues to break symmetry?
            residues = list(range(m))
            random.shuffle(residues)
            
            found = False
            for r in residues:
                valid = True
                for r_s, m_s in current_selected:
                    if not _is_disjoint(r, m, r_s, m_s):
                        valid = False
                        break
                if valid:
                    current_selected.append((r, m))
                    found = True
                    break
        
        if len(current_selected) > best_count:
            best_count = len(current_selected)
            best_result = [[r, m] for r, m in current_selected]
            
            # If we found a very large set, we can try to polish it
            if best_count > 20:
                # Try to add more? The greedy already tried all in the order.
                # Try local search: swap residues to allow adding another modulus
                # This is complex. Let's rely on more restarts.
                pass

    # Fallback
    if not best_result or best_count < 7:
        best_result = [[0, 2], [1, 4], [3, 8], [5, 16], [7, 32], [15, 64], [23, 128]]
        
    return best_result
