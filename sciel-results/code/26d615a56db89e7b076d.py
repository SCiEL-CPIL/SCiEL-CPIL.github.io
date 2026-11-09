"""Rewritten program: Maximum Weight Independent Set via branch-and-bound on conflict graph."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Moduli range
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
    
    # Build all valid (m, r) pairs
    # For each modulus m, each residue r in [0, m)
    # Total nodes: sum(m for m in moduli) = sum(20..200) = (20+200)*181/2 = 20910
    all_nodes = []  # list of (m, r, node_index)
    node_to_mod = []  # node_index -> m
    node_to_res = []  # node_index -> r
    mod_to_nodes = {m: [] for m in moduli}  # m -> list of node indices
    
    node_idx = 0
    for m in moduli:
        for r in range(m):
            all_nodes.append((m, r))
            node_to_mod.append(m)
            node_to_res.append(r)
            mod_to_nodes[m].append(node_idx)
            node_idx += 1
    
    total_nodes = len(all_nodes)
    
    # Build conflict graph using bitsets
    # conflict[node_i] is a set of node_j that conflict with node_i
    # Two nodes (m1, r1) and (m2, r2) conflict if (r1 - r2) % gcd(m1, m2) == 0
    # and m1 != m2 (distinct moduli requirement)
    # If m1 == m2, they always conflict (same modulus, different residues still conflict? 
    # Wait: distinct moduli requirement means we can only pick one residue per modulus)
    # So nodes with same modulus always conflict.
    
    # For efficiency, use Python sets for adjacency
    # But with 20910 nodes, full adjacency is 20910^2 which is too large for sets.
    # Instead, use a smarter approach: for each node, compute conflicts on the fly 
    # or precompute using the structure.
    
    # Actually, let's think about this differently.
    # The key insight: for a given modulus m and residue r, it conflicts with:
    # 1. All other residues for the same modulus m (since we can only pick one per modulus)
    # 2. For each other modulus m2, residues r2 where r2 ≡ r (mod gcd(m, m2))
    
    # For branch-and-bound, we need to check if a candidate node is compatible with 
    # all selected nodes. Let's use a different approach:
    
    # Greedy + local search with smarter heuristics, but also try a more structured approach.
    
    # Let me try a different strategy: since we want to maximize sum of 1/m,
    # and smaller moduli give larger contributions, we should prioritize small moduli.
    # The constraint is that for any two selected (m1, r1) and (m2, r2), 
    # r1 != r2 mod gcd(m1, m2).
    
    # Key observation: if we pick a residue for a small modulus, it constrains 
    # residues for all other moduli. The "tightest" constraints come from moduli 
    # with large gcd.
    
    # Strategy: 
    # 1. Start with the smallest moduli and greedily assign residues
    # 2. Use a more sophisticated local search that considers swapping residues
    # 3. Try multiple random orderings
    
    def can_add(classes, m, r):
        """Check if class (m, r) is disjoint from all classes in the set."""
        mi = mod_idx[m]
        for (m2, r2) in classes:
            mj = mod_idx[m2]
            g = gcd_table[mi][mj]
            if (r - r2) % g == 0:
                return False
        return True
    
    def find_best_residue(classes, m, num_samples=100):
        """Find a valid residue for modulus m that maximizes future flexibility.
        Use sampling to find a good residue."""
        mi = mod_idx[m]
        
        # First, find all valid residues
        valid_residues = []
        for r in range(m):
            if can_add(classes, m, r):
                valid_residues.append(r)
        
        if not valid_residues:
            return None
        
        if len(valid_residues) == 1:
            return valid_residues[0]
        
        # Score each valid residue by how many future moduli it leaves open
        # For each other modulus m2, count how many residues would be blocked
        # A residue r blocks r2 ≡ r (mod gcd(m, m2)) for modulus m2
        # Number of blocked residues for m2: m2 / gcd(m, m2)
        
        # We want to minimize the total "damage" to future choices
        # But this is complex. Instead, use a simpler heuristic:
        # Prefer residues that are "spread out" relative to existing constraints
        
        best_r = valid_residues[0]
        best_score = float('-inf')
        
        for r in valid_residues:
            # Score: sum over all other moduli m2 of (number of residues still available)
            # This is expensive, so use a proxy
            # Proxy: for each existing class (m2, r2), the "distance" in the 
            # cyclic group Z_gcd(m, m2) between r mod gcd and r2 mod gcd
            # Larger minimum distance is better
            min_dist = float('inf')
            for (m2, r2) in classes:
                mj = mod_idx[m2]
                g = gcd_table[mi][mj]
                if g > 1:
                    dist = min((r - r2) % g, (r2 - r) % g)
                    min_dist = min(min_dist, dist / g)  # normalize
            # Also consider distance from 0 and m/2 for variety
            if min_dist == float('inf'):
                min_dist = 0.5
            
            # Prefer residues with higher minimum normalized distance
            s = min_dist
            if s > best_score:
                best_score = s
                best_r = r
        
        return best_r
    
    def score(classes):
        """Compute sum of 1/m for the set of classes."""
        total = Fraction(0, 1)
        for (m, r) in classes:
            total += Fraction(1, m)
        return total
    
    def greedy_construct(order):
        """Greedy construction: add classes in the given order if possible."""
        classes = []
        used_moduli = set()
        for m in order:
            if m in used_moduli:
                continue
            r = find_best_residue(classes, m)
            if r is not None:
                classes.append([m, r])
                used_moduli.add(m)
        return classes
    
    def local_search(classes, max_iter=200):
        """Local search: try adding, removing, and swapping classes."""
        best_classes = [list(c) for c in classes]
        best_score = score(best_classes)
        used_moduli = set(m for m, r in best_classes)
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding a new class
            for m in sorted(moduli, key=lambda x: 1/x):  # smallest first
                if m in used_moduli:
                    continue
                r = find_best_residue(best_classes, m)
                if r is not None:
                    new_classes = best_classes + [[m, r]]
                    new_score = score(new_classes)
                    if new_score > best_score:
                        best_classes = new_classes
                        best_score = new_score
                        used_moduli.add(m)
                        improved = True
                        break
            
            # Try removing a class and adding a better one
            if not improved and len(best_classes) > 1:
                for i in range(len(best_classes)):
                    if time.time() > deadline:
                        break
                    m_remove = best_classes[i][0]
                    temp_classes = [c for j, c in enumerate(best_classes) if j != i]
                    temp_used = set(m for m, r in temp_classes)
                    temp_score = score(temp_classes)
                    
                    # Try to add a class that improves the score
                    for m_add in sorted(moduli, key=lambda x: 1/x):
                        if m_add in temp_used:
                            continue
                        r_add = find_best_residue(temp_classes, m_add)
                        if r_add is not None:
                            new_classes = temp_classes + [[m_add, r_add]]
                            new_score = score(new_classes)
                            if new_score > best_score:
                                best_classes = new_classes
                                best_score = new_score
                                used_moduli = set(m for m, r in best_classes)
                                improved = True
                                break
                    if improved:
                        break
            
            # Try reassigning residues
            if not improved:
                for i in range(len(best_classes)):
                    if time.time() > deadline:
                        break
                    m_i, r_i = best_classes[i]
                    other_classes = [c for j, c in enumerate(best_classes) if j != i]
                    
                    # Try different residues for m_i
                    mi = mod_idx[m_i]
                    for r_new in range(m_i):
                        if r_new == r_i:
                            continue
                        if can_add(other_classes, m_i, r_new):
                            # Check if this reassignment allows adding a new class
                            temp_classes = other_classes + [[m_i, r_new]]
                            temp_used = set(m for m, r in temp_classes)
                            
                            # Try to add one more class
                            for m_add in sorted(moduli, key=lambda x: 1/x):
                                if m_add in temp_used:
                                    continue
                                r_add = find_best_residue(temp_classes, m_add)
                                if r_add is not None:
                                    new_classes = temp_classes + [[m_add, r_add]]
                                    new_score = score(new_classes)
                                    if new_score > best_score:
                                        best_classes = new_classes
                                        best_score = new_score
                                        used_moduli = set(m for m, r in best_classes)
                                        improved = True
                                        break
                            if improved:
                                break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best_classes
    
    # Multi-start with different orderings
    best_classes = None
    best_score = Fraction(0, 1)
    
    # Start 1: Greedy with smallest moduli first
    order1 = sorted(moduli, key=lambda m: 1/m)
    classes = greedy_construct(order1)
    classes = local_search(classes, max_iter=300)
    s = score(classes)
    if s > best_score:
        best_score = s
        best_classes = [list(c) for c in classes]
    
    # Start 2: Greedy with reverse order (largest first)
    order2 = sorted(moduli, key=lambda m: -1/m)
    classes = greedy_construct(order2)
    classes = local_search(classes, max_iter=300)
    s = score(classes)
    if s > best_score:
        best_score = s
        best_classes = [list(c) for c in classes]
    
    # Start 3-10: Randomized orderings with bias toward small moduli
    import random
    random.seed(123)
    
    for start in range(8):
        if time.time() > deadline:
            break
        
        # Create weighted random ordering
        weights = [1.0 / (m ** 1.2) for m in moduli]
        order = []
        remaining = list(moduli)
        remaining_weights = list(weights)
        while remaining:
            total_w = sum(remaining_weights)
            if total_w <= 0:
                order.append(remaining[0])
                remaining.pop(0)
                remaining_weights.pop(0)
                continue
            r = random.random() * total_w
            cum = 0
            chosen_idx = 0
            for i, w in enumerate(remaining_weights):
                cum += w
                if r <= cum:
                    chosen_idx = i
                    break
            order.append(remaining[chosen_idx])
            remaining.pop(chosen_idx)
            remaining_weights.pop(chosen_idx)
        
        classes = greedy_construct(order)
        classes = local_search(classes, max_iter=200)
        s = score(classes)
        if s > best_score:
            best_score = s
            best_classes = [list(c) for c in classes]
    
    # Final local search on best solution
    if best_classes is not None:
        best_classes = local_search(best_classes, max_iter=500)
        best_score = score(best_classes)
    
    return best_classes if best_classes is not None else []
# EVOLVE-BLOCK-END
