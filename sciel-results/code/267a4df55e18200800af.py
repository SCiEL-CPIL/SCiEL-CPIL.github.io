import time
from math import gcd
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Precompute gcd table for moduli 20..200
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    mod_idx = {m: i for i, m in enumerate(moduli)}
    
    # gcd[m1_idx][m2_idx]
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def can_add(classes, m_idx, r):
        """Check if adding (m_idx, r) to classes keeps them disjoint."""
        for mi, ri in classes:
            g = gcd_table[mi][m_idx]
            if g == 1:
                return False # Coprime moduli cannot be disjoint
            if ri % g == r % g:
                return False
        return True

    def score(classes):
        """Sum of 1/m for the classes."""
        total = Fraction(0, 1)
        for m_idx, r in classes:
            total += Fraction(1, moduli[m_idx])
        return total

    def is_disjoint_set(classes):
        """Check if all pairs are disjoint. classes: list of (m_idx, r)"""
        n = len(classes)
        for i in range(n):
            mi, ri = classes[i]
            for j in range(i + 1, n):
                mj, rj = classes[j]
                g = gcd_table[mi][mj]
                if g == 1:
                    return False
                if ri % g == rj % g:
                    return False
        return True

    # Strategy: Use branch-and-bound on a conflict graph
    # Nodes: (m_idx, r) pairs
    # Edges: conflicting pairs (same residue mod gcd)
    
    # First, build the conflict graph structure
    # For each modulus, we have m residues, so total nodes = sum(m for m in 20..200)
    # That's about 181*110 = 19910 nodes, which is manageable
    
    # Precompute conflicts between moduli
    # Two moduli m1, m2 conflict if gcd(m1,m2) = 1 (no valid residue pair)
    # or if they have the same residue mod gcd(m1,m2)
    
    # For branch-and-bound, we need to:
    # 1. Order nodes by weight (1/m) descending
    # 2. Use bitsets for neighbor tracking
    # 3. Bound by sum of remaining weights
    
    # Build node list
    nodes = []
    node_to_mod = []
    for m_idx in range(n_mod):
        m = moduli[m_idx]
        for r in range(m):
            nodes.append((m_idx, r))
            node_to_mod.append(m_idx)
    
    n_nodes = len(nodes)
    
    # Weight for each node: 1/m
    weights = [Fraction(1, moduli[node_to_mod[i]]) for i in range(n_nodes)]
    
    # Sort nodes by weight descending (prefer small moduli)
    node_order = sorted(range(n_nodes), key=lambda i: -weights[i])
    
    # Reorder nodes and weights
    nodes = [nodes[i] for i in node_order]
    weights = [weights[i] for i in node_order]
    node_to_mod = [node_to_mod[i] for i in node_order]
    
    # Build conflict bitsets
    # conflict[i] is a bitset of nodes that conflict with node i
    # Use Python integers as bitsets (arbitrary precision)
    
    # First, precompute for each pair of moduli whether they can coexist
    # and if so, which residues conflict
    
    # For efficiency, precompute for each pair of moduli (i,j):
    # - if gcd(i,j) == 1: all residues conflict (can't use both moduli)
    # - else: residues r1, r2 conflict if r1 % g == r2 % g
    
    # Build adjacency list first, then convert to bitsets
    # This is O(n_nodes^2) which is ~400M operations, might be too slow
    
    # Alternative: Build conflict on the fly during branch-and-bound
    # Or use a smarter approach: group by modulus
    
    # Let's use a different approach: 
    # Since moduli are distinct, we can think of this as selecting residues for each modulus
    # such that for any two selected moduli m1, m2 with gcd g, r1 % g != r2 % g
    
    # This is a constraint satisfaction problem. Let's use backtracking with pruning.
    
    # Order moduli by size (smaller first, higher weight)
    mod_order = sorted(range(n_mod), key=lambda i: moduli[i])
    
    # Precompute for each modulus, the residues that are compatible with already chosen ones
    # Use a constraint propagation approach
    
    best_classes = []
    best_score = Fraction(0, 1)
    
    # Backtracking search
    # State: current classes, next modulus to consider
    # Pruning: upper bound on achievable score
    
    # For upper bound, we can use: sum of 1/m for all remaining moduli that could potentially be added
    # But this is loose. Better: for each remaining modulus, check if it can be added at all
    
    def backtrack(classes, m_idx, remaining_mods):
        """
        classes: list of (m_idx, r) chosen so far
        m_idx: current modulus index to consider
        remaining_mods: set of modulus indices not yet considered
        """
        nonlocal best_classes, best_score
        
        if time.time() > deadline - 5.0:
            return
        
        # Update best if current is better
        if classes:
            cur_score = score(classes)
            if cur_score > best_score:
                best_score = cur_score
                best_classes = list(classes)
        
        if not remaining_mods:
            return
        
        # Upper bound: sum of 1/m for all remaining moduli
        # Plus current score
        upper_bound = score(classes)
        for mi in remaining_mods:
            upper_bound += Fraction(1, moduli[mi])
        
        if upper_bound <= best_score:
            return
        
        # Try to add each remaining modulus
        # Order by weight descending
        sorted_remaining = sorted(remaining_mods, key=lambda mi: -Fraction(1, moduli[mi]))
        
        for mi in sorted_remaining:
            if time.time() > deadline - 5.0:
                return
            
            m = moduli[mi]
            
            # Try all residues for this modulus
            # Order residues: try ones that are most flexible first
            # A residue r is flexible if it doesn't conflict with many future moduli
            
            # Precompute: for each residue r, count how many future moduli it conflicts with
            # This is expensive, so use a heuristic: try residues in order of increasing r
            # Or random order
            
            # Try a few residues in random order, then fall back to systematic
            residues = list(range(m))
            random.shuffle(residues)
            
            # Try first 10 random residues, then systematic
            tried = set()
            
            for r in residues[:10]:
                if r in tried:
                    continue
                tried.add(r)
                
                if can_add(classes, mi, r):
                    new_classes = classes + [(mi, r)]
                    new_remaining = remaining_mods - {mi}
                    backtrack(new_classes, mi, new_remaining)
            
            # If time allows, try more residues systematically
            if time.time() < deadline - 10.0:
                for r in range(m):
                    if r in tried:
                        continue
                    tried.add(r)
                    
                    if can_add(classes, mi, r):
                        new_classes = classes + [(mi, r)]
                        new_remaining = remaining_mods - {mi}
                        backtrack(new_classes, mi, new_remaining)
        
        # Also try skipping this modulus
        for mi in sorted_remaining:
            if time.time() > deadline - 5.0:
                return
            
            new_remaining = remaining_mods - {mi}
            backtrack(classes, mi, new_remaining)
    
    # Start backtracking
    all_mods = set(range(n_mod))
    backtrack([], 0, all_mods)
    
    # Local search to improve
    def local_search(classes, max_iter=2000):
        classes = list(classes)
        current_score = score(classes)
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iter and time.time() < deadline - 5.0:
            improved = False
            iter_count += 1
            
            # Try to add a missing modulus
            for m_idx in range(n_mod):
                if any(c[0] == m_idx for c in classes):
                    continue
                
                m = moduli[m_idx]
                # Try random residues
                for _ in range(10):
                    r = random.randint(0, m - 1)
                    if can_add(classes, m_idx, r):
                        new_classes = classes + [(m_idx, r)]
                        new_score = score(new_classes)
                        if new_score > current_score:
                            classes = new_classes
                            current_score = new_score
                            improved = True
                            break
                if improved:
                    break
                if time.time() > deadline - 5.0:
                    break

            # Try swap move
            for i in range(len(classes)):
                mi, ri = classes[i]
                m_old = moduli[mi]
                
                for m_idx in range(n_mod):
                    if m_idx == mi:
                        continue
                    if any(c[0] == m_idx for c in classes if c[0] != mi):
                        continue
                    m_new = moduli[m_idx]
                    if m_new >= m_old:
                        continue
                    
                    # Try to find a residue for m_new that fits
                    found_swap = False
                    for _ in range(10):
                        r_new = random.randint(0, m_new - 1)
                        valid = True
                        for j, (mj, rj) in enumerate(classes):
                            if j == i:
                                continue
                            g = gcd_table[mj][m_idx]
                            if g == 1:
                                valid = False
                                break
                            if rj % g == r_new % g:
                                valid = False
                                break
                        
                        if valid:
                            test_classes = classes[:i] + [(m_idx, r_new)] + classes[i+1:]
                            new_score = score(test_classes)
                            if new_score > current_score:
                                classes = test_classes
                                current_score = new_score
                                improved = True
                                found_swap = True
                                break
                    
                    if found_swap:
                        break
                if improved:
                    break
                    
        return classes, current_score
    
    if best_classes:
        improved_classes, improved_score = local_search(best_classes)
        if improved_score > best_score:
            best_score = improved_score
            best_classes = improved_classes
    
    # Fallback if backtracking didn't find anything
    if not best_classes:
        classes = []
        for m_idx in range(n_mod):
            m = moduli[m_idx]
            for r in range(m):
                if can_add(classes, m_idx, r):
                    classes.append((m_idx, r))
                    break
        best_classes = classes
        best_score = score(classes)

    result = [[moduli[m_idx], r] for m_idx, r in best_classes]
    
    # Verify validity
    assert is_disjoint_set(best_classes), "Result is not pairwise disjoint!"
    assert len(set(moduli[m_idx] for m_idx, r in best_classes)) == len(best_classes), "Moduli not distinct!"
    
    return result

# Test
if __name__ == "__main__":
    result = construct()
    print(f"Number of classes: {len(result)}")
    print(f"Score: {sum(1/m for m, r in result)}")
    print(f"Classes: {result[:10]}...")
