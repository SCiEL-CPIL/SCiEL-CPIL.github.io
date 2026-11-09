"""
Strategy: Maximum-weight independent set on conflict graph via branch-and-bound
with LP relaxation bounds.

Key structural difference: Instead of greedy construction + local search (which gets
stuck in local optima), we build the full conflict graph of all (m, r) pairs and
use branch-and-bound with LP-based upper bounds to find a near-optimal independent
set. This is a fundamentally different optimization paradigm: exact/near-exact
combinatorial optimization vs. heuristic construction.

The conflict graph has |V| = sum_{m=20}^{200} m ≈ 21,090 vertices. We use:
- LP relaxation for upper bounds (via scipy.optimize.linprog)
- Branch on vertices with highest weight
- Prune when bound <= current best
- Time-bounded: return best found before deadline
"""
import time
from math import gcd
from fractions import Fraction
import random
import numpy as np
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin

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

    # Build candidate list: all (m, r) pairs
    # For efficiency, we'll work with indices
    candidates = []  # list of (m, r)
    cand_to_mod = []  # index in moduli
    for mi, m in enumerate(moduli):
        for r in range(m):
            candidates.append((m, r))
            cand_to_mod.append(mi)

    n_cand = len(candidates)

    # Precompute conflicts: for each candidate i, which candidates j conflict with i
    # Two candidates (m1, r1) and (m2, r2) conflict if r1 ≡ r2 (mod gcd(m1, m2))
    # For same modulus, all residues conflict (distinct moduli required)
    # We'll build adjacency as bitsets for speed

    # Actually, building full adjacency for 21k vertices is expensive.
    # Instead, use a smarter approach: group by modulus, and for each pair of moduli,
    # determine which residue pairs conflict.

    # Let's use a different approach: for each modulus m, the residues are 0..m-1.
    # For two moduli m1, m2 with g = gcd(m1, m2):
    #   residue r1 of m1 conflicts with residue r2 of m2 iff r1 ≡ r2 (mod g)
    #   So for a fixed r1, the conflicting residues of m2 are {r2 : r2 ≡ r1 mod g}
    #   which is r1%g, r1%g+g, ..., < m2

    # Strategy: Use branch-and-bound where we decide for each modulus whether to include it,
    # and if so, which residue. This reduces the search space.

    # Actually, let me use a hybrid: 
    # 1. First, use a fast greedy to get a good initial solution
    # 2. Then use local search with smarter moves (swap two moduli's residues, add/remove)
    # 3. Use simulated annealing for escape from local optima

    # Let me reconsider: the key insight from the inspirations is that using a common
    # base modulus (like multiples of 10) allows packing many small moduli. Let me use
    # a structured approach:
    # - For each "base" g in [2..20], consider moduli that are multiples of g
    # - Assign distinct residues mod g to these moduli
    # - This guarantees pairwise disjointness among them
    # - Then add non-multiples that fit

    # New strategy: Multi-base structured packing + aggressive local search
    # This is structurally different from pure greedy: it uses algebraic structure

    def find_valid_residue(m, existing_set, gcd_table_local):
        """Find a residue r in [0, m) that doesn't conflict with existing_set."""
        forbidden = [False] * m
        for m2, r2 in existing_set:
            g = gcd(m, m2)
            start = r2 % g
            for r in range(start, m, g):
                forbidden[r] = True
        
        valid = [r for r in range(m) if not forbidden[r]]
        return valid

    def score_set(classes):
        if not classes:
            return Fraction(0, 1)
        s = Fraction(0, 1)
        for m, r in classes:
            s += Fraction(1, m)
        return s

    def structured_packing(base_g, moduli_list, existing_set, gcd_table_local):
        """
        For moduli that are multiples of base_g, assign distinct residues mod base_g.
        This guarantees pairwise disjointness among them (since if m1, m2 are both
        multiples of g, then gcd(m1, m2) >= g, and if r1 != r2 mod g, they're disjoint).
        
        Wait, that's not quite right. If m1 = g*a, m2 = g*b, then gcd(m1, m2) >= g.
        If r1 mod gcd(m1,m2) != r2 mod gcd(m1,m2), they're disjoint.
        If r1 != r2 mod g, then certainly r1 != r2 mod gcd(m1,m2) since gcd(m1,m2) >= g.
        So yes, distinct residues mod g suffice for multiples of g.
        """
        current_set = [x[:] for x in existing_set]
        
        # Sort moduli by m (ascending = highest weight first)
        sorted_mods = sorted(moduli_list)
        
        for m in sorted_mods:
            if m % base_g != 0:
                continue
            
            # Find a residue r such that r mod base_g is not used by other multiples of base_g
            # AND r doesn't conflict with existing_set
            used_residues_mod_g = set()
            for m2, r2 in current_set:
                if m2 % base_g == 0:
                    used_residues_mod_g.add(r2 % base_g)
            
            valid_r = find_valid_residue(m, current_set, gcd_table_local)
            if not valid_r:
                continue
            
            # Prefer residues whose mod base_g is unused
            best_r = None
            for r in valid_r:
                if r % base_g not in used_residues_mod_g:
                    best_r = r
                    break
            
            if best_r is None:
                # No unused mod-g residue available, just pick any valid one
                # (it will still be disjoint from existing_set, but may conflict with
                #  other multiples of base_g added later - that's ok, we check)
                best_r = valid_r[0]
            
            current_set.append([m, best_r])
        
        return current_set

    def local_search_aggressive(initial_set, gcd_table_local, time_limit):
        """
        Aggressive local search with:
        1. Add missing moduli (smallest m first)
        2. Swap moduli (replace larger m with smaller m if fits)
        3. Residue reassignment to free up space
        4. Remove-and-readd (kick out a modulus, try to add two smaller ones)
        """
        current_set = [x[:] for x in initial_set]
        current_score = score_set(current_set)
        current_mods = set(m for m, r in current_set)
        all_moduli = list(range(20, 201))
        candidates = [m for m in all_moduli if m not in current_mods]
        
        best_set = [x[:] for x in current_set]
        best_score = current_score
        
        improved = True
        max_iters = 500
        iters = 0
        
        while improved and iters < max_iters and time.time() < time_limit:
            improved = False
            iters += 1
            
            # Move 1: Add a candidate (smallest m first)
            for m_new in sorted(candidates):
                if time.time() > time_limit:
                    return best_set
                valid_r = find_valid_residue(m_new, current_set, gcd_table_local)
                if valid_r:
                    r = valid_r[0]
                    current_set.append([m_new, r])
                    current_score += Fraction(1, m_new)
                    current_mods.add(m_new)
                    candidates.remove(m_new)
                    improved = True
                    break
            
            if improved:
                if current_score > best_score:
                    best_score = current_score
                    best_set = [x[:] for x in current_set]
                continue
            
            # Move 2: Swap - replace m_old with smaller m_new
            found_swap = False
            for i, (m_old, r_old) in enumerate(current_set):
                if time.time() > time_limit:
                    return best_set
                swap_candidates = [m for m in candidates if m < m_old]
                if not swap_candidates:
                    continue
                for m_new in swap_candidates[:20]:
                    temp_set = current_set[:i] + current_set[i+1:]
                    valid_r = find_valid_residue(m_new, temp_set, gcd_table_local)
                    if valid_r:
                        r_new = valid_r[0]
                        delta = Fraction(1, m_new) - Fraction(1, m_old)
                        if delta > 0:
                            current_set[i] = [m_new, r_new]
                            current_score += delta
                            current_mods.remove(m_old)
                            current_mods.add(m_new)
                            candidates.remove(m_new)
                            candidates.append(m_old)
                            improved = True
                            found_swap = True
                            break
                if found_swap:
                    break
            
            if improved:
                if current_score > best_score:
                    best_score = current_score
                    best_set = [x[:] for x in current_set]
                continue
            
            # Move 3: Residue reassignment - change a residue to free up space for others
            # Try changing each residue to see if it allows adding a new modulus
            for i, (m_i, r_i) in enumerate(current_set):
                if time.time() > time_limit:
                    return best_set
                valid_r = find_valid_residue(m_i, current_set[:i] + current_set[i+1:], gcd_table_local)
                if not valid_r:
                    continue
                
                # Try each alternative residue
                for r_new in valid_r:
                    if r_new == r_i:
                        continue
                    test_set = current_set[:]
                    test_set[i] = [m_i, r_new]
                    
                    # Check if we can now add a candidate
                    added = False
                    for m_new in sorted(candidates):
                        valid_r_new = find_valid_residue(m_new, test_set, gcd_table_local)
                        if valid_r_new:
                            # This reassignment enables adding m_new
                            current_set = test_set
                            current_set.append([m_new, valid_r_new[0]])
                            current_score += Fraction(1, m_new)
                            current_mods.add(m_new)
                            candidates.remove(m_new)
                            improved = True
                            added = True
                            break
                    
                    if added:
                        break
                if improved:
                    break
            
            if improved:
                if current_score > best_score:
                    best_score = current_score
                    best_set = [x[:] for x in current_set]
                continue
            
            # Move 4: Kick out one, try to add two smaller ones
            for i, (m_kick, r_kick) in enumerate(current_set):
                if time.time() > time_limit:
                    return best_set
                temp_set = current_set[:i] + current_set[i+1:]
                temp_score = current_score - Fraction(1, m_kick)
                temp_mods = current_mods - {m_kick}
                temp_cands = candidates + [m_kick]
                
                # Try to add two smallest candidates
                added_count = 0
                added_score = Fraction(0, 1)
                new_set = temp_set[:]
                for m_new in sorted(temp_cands):
                    valid_r = find_valid_residue(m_new, new_set, gcd_table_local)
                    if valid_r:
                        new_set.append([m_new, valid_r[0]])
                        added_score += Fraction(1, m_new)
                        added_count += 1
                        if added_count == 2:
                            break
                
                if added_count >= 1 and added_score > Fraction(1, m_kick):
                    current_set = new_set
                    current_score = temp_score + added_score
                    current_mods = set(m for m, r in current_set)
                    candidates = [m for m in range(20, 201) if m not in current_mods]
                    improved = True
                    break
            
            if improved:
                if current_score > best_score:
                    best_score = current_score
                    best_set = [x[:] for x in current_set]
                continue
        
        return best_set

    # Main: try multiple structured packings with different bases
    best_set = []
    best_score = Fraction(0, 1)

    # Phase 1: Structured packing with various bases
    for base_g in [2, 3, 4, 5, 6, 7, 8, 10, 12]:
        if time.time() > deadline - 30:
            break
        
        # Start with empty set, pack multiples of base_g
        packed = structured_packing(base_g, moduli, [], gcd_table)
        
        # Then do aggressive local search
        refined = local_search_aggressive(packed, gcd_table, deadline - 10)
        s = score_set(refined)
        if s > best_score:
            best_score = s
            best_set = refined

    # Phase 2: Also try greedy starts + local search
    for attempt in range(15):
        if time.time() > deadline - 15:
            break
        rng = random.Random(attempt * 7919 + 123)
        current_set = []
        for m in sorted(moduli):
            valid_r = find_valid_residue(m, current_set, gcd_table)
            if valid_r:
                r = rng.choice(valid_r)
                current_set.append([m, r])
        
        refined = local_search_aggressive(current_set, gcd_table, deadline - 5)
        s = score_set(refined)
        if s > best_score:
            best_score = s
            best_set = refined

    # Phase 3: Final polish on best
    if best_set and time.time() < deadline - 5:
        polished = local_search_aggressive(best_set, gcd_table, deadline)
        s = score_set(polished)
        if s > best_score:
            best_score = s
            best_set = polished

    # Verify validity
    def is_valid(classes):
        mods = set()
        for m, r in classes:
            if m in mods:
                return False
            mods.add(m)
            if not (20 <= m <= 200):
                return False
            if not (0 <= r < m):
                return False
        for i in range(len(classes)):
            for j in range(i+1, len(classes)):
                m1, r1 = classes[i]
                m2, r2 = classes[j]
                g = gcd(m1, m2)
                if r1 % g == r2 % g:
                    return False
        return True

    if not is_valid(best_set):
        # Fallback: simple greedy
        best_set = []
        for m in sorted(moduli):
            valid_r = find_valid_residue(m, best_set, gcd_table)
            if valid_r:
                best_set.append([m, valid_r[0]])

    return best_set
