"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # Safety margin before 150s kill

    # Moduli in [20, 200]
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    mod_idx = {m: i for i, m in enumerate(moduli)}

    # Precompute gcd for all pairs of moduli
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Precompute conflict structure: for each modulus i, which moduli j share gcd > 1
    # and the gcd value
    non_coprime = [set() for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(n_mod):
            if i != j and gcd_table[i][j] > 1:
                non_coprime[i].add(j)

    # For each modulus i, precompute for each residue r the set of (j, r_j) pairs that conflict
    # A residue r for modulus m_i conflicts with residue r_j for modulus m_j iff (r - r_j) % gcd(m_i, m_j) == 0
    # For a given (i, r), the conflicting residues for modulus j are: r_j ≡ r (mod gcd(m_i, m_j))
    # So for modulus j, the valid residues are those NOT congruent to r mod gcd(m_i, m_j)

    def can_add_residue(classes, m, r):
        """Check if residue r for modulus m is disjoint from all classes in classes."""
        for (m2, r2) in classes:
            g = gcd(m, m2)
            if (r - r2) % g == 0:
                return False
        return True

    def score(classes):
        total = Fraction(0, 1)
        for (m, r) in classes:
            total += Fraction(1, m)
        return total

    def solve_greedy_flexible(order):
        """Greedy construction with flexibility-based residue selection."""
        classes = []
        used = set()
        for m in order:
            if m in used:
                continue
            # Check pairwise non-coprime with all existing
            compatible = True
            for (m2, r2) in classes:
                if gcd(m, m2) == 1:
                    compatible = False
                    break
            if not compatible:
                continue

            # Pick residue that maximizes flexibility
            best_r = None
            best_flex = -1
            for r in range(m):
                if not can_add_residue(classes, m, r):
                    continue
                flex = 0
                for m2 in moduli:
                    if m2 in used or m2 == m:
                        continue
                    g = gcd(m, m2)
                    if g == 1:
                        continue
                    r0 = r % g
                    if r0 < m2:
                        conflict_count = (m2 - 1 - r0) // g + 1
                    else:
                        conflict_count = 0
                    flex += m2 - conflict_count
                if flex > best_flex:
                    best_flex = flex
                    best_r = r
            if best_r is not None:
                classes.append([m, best_r])
                used.add(m)
        return classes

    def local_search(classes, max_time=20.0):
        """Improve solution: add moduli, remove-and-readd, re-optimize residues."""
        classes = [list(c) for c in classes]
        current_score = score(classes)
        used_moduli = set(m for m, r in classes)
        start = time.time()

        improved = True
        while improved and (time.time() - start) < max_time:
            improved = False

            # Phase 1: Try to add each unused modulus (small first for higher reciprocal)
            for m in sorted([m for m in moduli if m not in used_moduli]):
                compatible = True
                for (m2, r2) in classes:
                    if gcd(m, m2) == 1:
                        compatible = False
                        break
                if not compatible:
                    continue

                best_r = None
                best_flex = -1
                for r in range(m):
                    if not can_add_residue(classes, m, r):
                        continue
                    flex = 0
                    for m2 in moduli:
                        if m2 in used_moduli or m2 == m:
                            continue
                        g = gcd(m, m2)
                        if g == 1:
                            continue
                        r0 = r % g
                        if r0 < m2:
                            conflict_count = (m2 - 1 - r0) // g + 1
                        else:
                            conflict_count = 0
                        flex += m2 - conflict_count
                    if flex > best_flex:
                        best_flex = flex
                        best_r = r
                if best_r is not None:
                    new_classes = classes + [[m, best_r]]
                    new_score = score(new_classes)
                    if new_score > current_score:
                        classes = new_classes
                        current_score = new_score
                        used_moduli.add(m)
                        improved = True
                        break

            # Phase 2: Try removing the largest modulus and re-adding better ones
            if not improved and classes:
                idx_to_remove = max(range(len(classes)), key=lambda i: classes[i][0])
                temp_classes = [c for i, c in enumerate(classes) if i != idx_to_remove]
                temp_used = set(m for m, r in temp_classes)

                for m_add in sorted([m for m in moduli if m not in temp_used]):
                    compatible = True
                    for (m2, r2) in temp_classes:
                        if gcd(m_add, m2) == 1:
                            compatible = False
                            break
                    if not compatible:
                        continue

                    best_r = None
                    best_flex = -1
                    for r in range(m_add):
                        if not can_add_residue(temp_classes, m_add, r):
                            continue
                        flex = 0
                        for m2 in moduli:
                            if m2 in temp_used or m2 == m_add:
                                continue
                            g = gcd(m_add, m2)
                            if g == 1:
                                continue
                            r0 = r % g
                            if r0 < m2:
                                conflict_count = (m2 - 1 - r0) // g + 1
                            else:
                                conflict_count = 0
                            flex += m2 - conflict_count
                        if flex > best_flex:
                            best_flex = flex
                            best_r = r
                    if best_r is not None:
                        new_classes = temp_classes + [[m_add, best_r]]
                        new_score = score(new_classes)
                        if new_score > current_score:
                            classes = new_classes
                            current_score = new_score
                            used_moduli = set(m for m, r in classes)
                            improved = True
                            break

            # Phase 3: Re-optimize residues of existing classes for flexibility
            if not improved:
                for i in range(len(classes)):
                    m_i, r_i = classes[i]
                    others = [c for j, c in enumerate(classes) if j != i]
                    best_r = None
                    best_flex = -1
                    for r in range(m_i):
                        if not can_add_residue(others, m_i, r):
                            continue
                        flex = 0
                        for m2 in moduli:
                            if m2 in used_moduli or m2 == m_i:
                                continue
                            g = gcd(m_i, m2)
                            if g == 1:
                                continue
                            r0 = r % g
                            if r0 < m2:
                                conflict_count = (m2 - 1 - r0) // g + 1
                            else:
                                conflict_count = 0
                            flex += m2 - conflict_count
                        if flex > best_flex:
                            best_flex = flex
                            best_r = r
                    if best_r is not None and best_r != r_i:
                        classes[i] = [m_i, best_r]
                        improved = True

        return classes, current_score

    # --- Branch and Bound Exact Solver ---
    # Formulation: We want to select a subset of moduli and assign residues to maximize sum(1/m)
    # subject to pairwise disjointness constraints.
    # 
    # Key insight: For a fixed set of selected moduli, we need to check if there exists
    # an assignment of residues that satisfies all pairwise constraints. This is a CSP.
    # 
    # For the B&B, we use a different approach:
    # - State: (selected_moduli_set, assigned_residues_dict)
    # - We build up the solution incrementally
    # - Upper bound: current_score + sum of 1/m for all remaining moduli that could potentially be added
    # - Pruning: if upper bound <= best_score, prune
    
    def upper_bound(assigned, used_moduli_set):
        """Compute upper bound on achievable score from current state."""
        current = Fraction(0, 1)
        for m, r in assigned:
            current += Fraction(1, m)
        
        # For each unused modulus, check if it's compatible with all assigned moduli
        # (i.e., gcd > 1 with all of them). If so, add 1/m to the bound.
        # This is an optimistic bound because we don't check residue compatibility.
        for m in moduli:
            if m in used_moduli_set:
                continue
            # Check if m shares gcd > 1 with all assigned moduli
            compatible = True
            for (m2, r2) in assigned:
                if gcd(m, m2) == 1:
                    compatible = False
                    break
            if compatible:
                current += Fraction(1, m)
        return current

    def get_valid_residues(assigned, m):
        """Get all valid residues for modulus m given current assignment."""
        valid = []
        for r in range(m):
            if can_add_residue(assigned, m, r):
                valid.append(r)
        return valid

    def branch_and_bound(assigned, used_moduli_set, best_score, best_classes, time_limit):
        """Branch and bound to find optimal solution."""
        # Upper bound check
        ub = upper_bound(assigned, used_moduli_set)
        if ub <= best_score:
            return best_score, best_classes, False
        
        # If we've used all compatible moduli, this is a complete solution
        # Find unused moduli that are compatible with all assigned
        candidates = []
        for m in moduli:
            if m in used_moduli_set:
                continue
            compatible = True
            for (m2, r2) in assigned:
                if gcd(m, m2) == 1:
                    compatible = False
                    break
            if compatible:
                candidates.append(m)
        
        if not candidates:
            # Complete solution
            current_score = score(assigned)
            if current_score > best_score:
                best_score = current_score
                best_classes = [list(c) for c in assigned]
            return best_score, best_classes, True
        
        # Try adding each candidate modulus with each valid residue
        # Sort candidates by 1/m descending (small moduli first) for better pruning
        candidates.sort(key=lambda m: -1.0/m)
        
        for m in candidates:
            valid_residues = get_valid_residues(assigned, m)
            if not valid_residues:
                continue
            
            # Try each valid residue
            for r in valid_residues:
                new_assigned = assigned + [[m, r]]
                new_used = used_moduli_set | {m}
                
                # Quick upper bound check
                ub2 = upper_bound(new_assigned, new_used)
                if ub2 <= best_score:
                    continue
                
                best_score, best_classes, complete = branch_and_bound(
                    new_assigned, new_used, best_score, best_classes, time_limit
                )
                
                if time_limit is not None and time.time() > time_limit:
                    return best_score, best_classes, False
        
        return best_score, best_classes, False

    def bnb_with_heuristic(assigned, used_moduli_set, best_score, best_classes, time_limit, max_depth=50):
        """Branch and bound with depth limit and heuristic ordering."""
        ub = upper_bound(assigned, used_moduli_set)
        if ub <= best_score:
            return best_score, best_classes, False
        
        candidates = []
        for m in moduli:
            if m in used_moduli_set:
                continue
            compatible = True
            for (m2, r2) in assigned:
                if gcd(m, m2) == 1:
                    compatible = False
                    break
            if compatible:
                candidates.append(m)
        
        if not candidates:
            current_score = score(assigned)
            if current_score > best_score:
                best_score = current_score
                best_classes = [list(c) for c in assigned]
            return best_score, best_classes, True
        
        if len(assigned) >= max_depth:
            current_score = score(assigned)
            if current_score > best_score:
                best_score = current_score
                best_classes = [list(c) for c in assigned]
            return best_score, best_classes, True
        
        candidates.sort(key=lambda m: -1.0/m)
        
        for m in candidates:
            valid_residues = get_valid_residues(assigned, m)
            if not valid_residues:
                continue
            
            # For each residue, compute a heuristic score and sort
            scored_residues = []
            for r in valid_residues:
                # Heuristic: count how many future moduli remain valid
                flex = 0
                for m2 in moduli:
                    if m2 in used_moduli_set or m2 == m:
                        continue
                    g = gcd(m, m2)
                    if g == 1:
                        continue
                    r0 = r % g
                    if r0 < m2:
                        conflict_count = (m2 - 1 - r0) // g + 1
                    else:
                        conflict_count = 0
                    flex += m2 - conflict_count
                scored_residues.append((flex, r))
            
            # Try residues in order of decreasing flexibility
            scored_residues.sort(key=lambda x: -x[0])
            
            for flex, r in scored_residues:
                new_assigned = assigned + [[m, r]]
                new_used = used_moduli_set | {m}
                
                ub2 = upper_bound(new_assigned, new_used)
                if ub2 <= best_score:
                    continue
                
                best_score, best_classes, complete = bnb_with_heuristic(
                    new_assigned, new_used, best_score, best_classes, time_limit, max_depth
                )
                
                if time_limit is not None and time.time() > time_limit:
                    return best_score, best_classes, False
        
        return best_score, best_classes, False

    # --- Main search ---
    best_classes = []
    best_score = Fraction(0, 1)

    # Initial greedy with ascending moduli
    classes = solve_greedy_flexible(sorted(moduli))
    classes, sc = local_search(classes, max_time=10.0)
    if sc > best_score:
        best_score = sc
        best_classes = [list(c) for c in classes]

    # Random restarts
    random.seed(42)
    for restart in range(30):
        if time.time() > deadline - 30.0:
            break

        order = moduli.copy()
        random.shuffle(order)
        order.sort(key=lambda m: random.random() * 10 + m / 10.0)

        classes = solve_greedy_flexible(order)
        classes, sc = local_search(classes, max_time=5.0)
        if sc > best_score:
            best_score = sc
            best_classes = [list(c) for c in classes]

    # Try B&B from the best solution found, extending it
    if best_classes and time.time() < deadline - 5.0:
        bnb_time = min(30.0, deadline - time.time() - 5.0)
        if bnb_time > 2.0:
            try:
                bnb_score, bnb_classes, complete = bnb_with_heuristic(
                    [list(c) for c in best_classes],
                    set(m for m, r in best_classes),
                    best_score, best_classes,
                    time.time() + bnb_time,
                    max_depth=60
                )
                if bnb_score > best_score:
                    best_score = bnb_score
                    best_classes = [list(c) for c in bnb_classes]
            except Exception:
                pass

    # Also try B&B from scratch with small moduli only (20-50) for exact solution on subset
    if time.time() < deadline - 5.0:
        small_moduli = list(range(20, 51))
        small_idx = {m: i for i, m in enumerate(small_moduli)}
        # B&B on small moduli only
        try:
            small_score, small_classes, _ = bnb_with_heuristic(
                [], set(), Fraction(0,1), [],
                time.time() + min(10.0, deadline - time.time() - 5.0),
                max_depth=30
            )
            if small_score > best_score:
                best_score = small_score
                best_classes = [list(c) for c in small_classes]
        except Exception:
            pass

    # Final local search polish
    if best_classes and time.time() < deadline:
        polish_time = min(10.0, deadline - time.time())
        if polish_time > 1.0:
            polished, ps = local_search(best_classes, max_time=polish_time)
            if ps > best_score:
                best_score = ps
                best_classes = [list(c) for c in polished]

    return best_classes
# EVOLVE-BLOCK-END
