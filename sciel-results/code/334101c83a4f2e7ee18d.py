"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Moduli in [20, 200]
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    mod_idx = {m: i for i, m in enumerate(moduli)}
    
    # Precompute gcd table
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Build conflict graph: nodes are (m_idx, r) pairs
    # Two nodes (i, r1) and (j, r2) conflict if r1 ≡ r2 (mod gcd(m_i, m_j))
    # Also, same modulus always conflicts (r1 != r2 needed, but since we pick at most one per mod, we handle that)
    
    # For each modulus i, list of residues 0..m_i-1
    # Node id: i * 200 + r (since max modulus is 200)
    
    # Precompute conflicts using bitsets
    # For each node, which other nodes does it conflict with?
    
    # Total nodes: sum of m for m in 20..200 = sum(20..200) = (20+200)*181/2 = 19380
    # That's a lot. Let's be smarter.
    
    # Actually, let's use a different approach: for each modulus, we can only pick one residue.
    # So we have a "modulus selection" problem with residue constraints.
    
    # Better: build a graph where nodes are (m, r) but use bitset operations efficiently.
    
    # Let's assign each (m, r) a unique id
    node_to_mr = []
    mr_to_node = {}
    for i, m in enumerate(moduli):
        for r in range(m):
            node_id = len(node_to_mr)
            node_to_mr.append((m, r, i))  # (modulus, residue, mod_idx)
            mr_to_node[(m, r)] = node_id
    
    n_nodes = len(node_to_mr)
    
    # Build conflict bitsets
    # conflict[i] is a bitset of nodes that conflict with node i
    # Using Python integers as bitsets (arbitrary precision)
    
    # For efficiency, process by modulus groups
    # Nodes of same modulus always conflict with each other
    # Nodes of different moduli i,j conflict if r1 ≡ r2 (mod gcd(m_i, m_j))
    
    # This is O(n_nodes^2) which is ~376M operations for 19380 nodes - too slow.
    
    # Better approach: for each pair of moduli (i, j), determine which residue pairs conflict.
    # Then build the bitset incrementally.
    
    # Let's use a different strategy: branch and bound with pruning based on upper bound.
    # Upper bound: sum of 1/m for all remaining moduli that could potentially be added.
    
    # First, let's get a good initial solution using greedy
    def greedy_solution():
        """Get a good initial solution using greedy with small moduli first."""
        classes = []
        used_moduli = set()
        
        # Sort by 1/m descending (m ascending)
        for m in sorted(moduli):
            if m in used_moduli:
                continue
            # Check if m is compatible with all existing (gcd > 1 for all)
            compatible = True
            for (m2, r2) in classes:
                if gcd(m, m2) == 1:
                    compatible = False
                    break
            if not compatible:
                continue
            
            # Find a valid residue
            # Try residues that minimize conflicts with future moduli
            best_r = None
            best_conflict_count = float('inf')
            
            for r in range(m):
                # Check if r is valid with existing classes
                valid = True
                for (m2, r2) in classes:
                    g = gcd(m, m2)
                    if (r - r2) % g == 0:
                        valid = False
                        break
                if valid:
                    # Count potential conflicts with future moduli
                    conflict_count = 0
                    for m2 in moduli:
                        if m2 in used_moduli or m2 == m:
                            continue
                        g = gcd(m, m2)
                        if g == 1:
                            continue
                        # How many residues of m2 conflict with r mod g?
                        # Residues r2 such that r2 ≡ r (mod g)
                        # That's ceil(m2/g) or floor(m2/g) + 1 depending
                        conflict_count += (m2 + g - 1) // g  # approximate
                    if conflict_count < best_conflict_count:
                        best_conflict_count = conflict_count
                        best_r = r
                    if best_conflict_count == 0:
                        break
            
            if best_r is not None:
                classes.append([m, best_r])
                used_moduli.add(m)
        
        return classes
    
    # Get initial solution
    best_classes = greedy_solution()
    
    def compute_score(classes):
        total = Fraction(0, 1)
        for (m, r) in classes:
            total += Fraction(1, m)
        return total
    
    best_score = compute_score(best_classes)
    
    # Now try to improve using local search with residue reassignment
    def local_search_improve(classes, max_time=30.0):
        """Improve solution by trying to add moduli and reassigning residues."""
        classes = [list(c) for c in classes]
        current_score = compute_score(classes)
        used_moduli = set(m for m, r in classes)
        start = time.time()
        
        improved = True
        while improved and (time.time() - start) < max_time:
            improved = False
            
            # Try to add each unused modulus
            for m in sorted([m for m in moduli if m not in used_moduli]):
                # Check gcd compatibility
                compatible = True
                for (m2, r2) in classes:
                    if gcd(m, m2) == 1:
                        compatible = False
                        break
                if not compatible:
                    continue
                
                # Try all residues
                for r in range(m):
                    valid = True
                    for (m2, r2) in classes:
                        g = gcd(m, m2)
                        if (r - r2) % g == 0:
                            valid = False
                            break
                    if valid:
                        new_classes = classes + [[m, r]]
                        new_score = compute_score(new_classes)
                        if new_score > current_score:
                            classes = new_classes
                            current_score = new_score
                            used_moduli.add(m)
                            improved = True
                        break  # Only need first valid r for this m
        
            # Try to swap: remove one modulus, try to add two better ones
            # This is more complex, skip for now
        
        return classes, current_score
    
    best_classes, best_score = local_search_improve(best_classes, max_time=25.0)
    
    # Try multiple random restarts
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Random greedy with different ordering
        order = moduli.copy()
        random.shuffle(order)
        # Bias towards small moduli
        order.sort(key=lambda m: random.random() * 50 + m / 100)
        
        classes = []
        used_moduli = set()
        for m in order:
            if m in used_moduli:
                continue
            compatible = True
            for (m2, r2) in classes:
                if gcd(m, m2) == 1:
                    compatible = False
                    break
            if not compatible:
                continue
            
            # Random valid residue
            valid_residues = []
            for r in range(m):
                valid = True
                for (m2, r2) in classes:
                    g = gcd(m, m2)
                    if (r - r2) % g == 0:
                        valid = False
                        break
                if valid:
                    valid_residues.append(r)
            
            if valid_residues:
                r = random.choice(valid_residues)
                classes.append([m, r])
                used_moduli.add(m)
        
        # Local search
        classes, sc = local_search_improve(classes, max_time=10.0)
        if sc > best_score:
            best_score = sc
            best_classes = [list(c) for c in classes]
    
    return best_classes
# EVOLVE-BLOCK-END
