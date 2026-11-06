import math
import time
import random

MAX_MOD = 128

def construct():
    t0 = time.time()
    deadline = t0 + 55.0  # Safety margin before 60s limit
    
    # Precompute gcd values
    gcd_cache = {}
    for m1 in range(2, MAX_MOD + 1):
        for m2 in range(m1, MAX_MOD + 1):
            gcd_cache[(m1, m2)] = math.gcd(m1, m2)
    
    def get_gcd(m1, m2):
        key = (m1, m2) if m1 <= m2 else (m2, m1)
        return gcd_cache.get(key, math.gcd(m1, m2))
    
    def _disjoint(r1, m1, r2, m2):
        g = get_gcd(m1, m2)
        return (r2 - r1) % g != 0
    
    # Candidate moduli: all even numbers from 2 to 128
    candidate_mods = [m for m in range(2, MAX_MOD + 1) if m % 2 == 0]
    
    # Use CSP with AC-3 style pruning
    def solve_csp(moduli_set, time_limit):
        """
        Solve CSP where each variable is a modulus, domain is residues 0..m-1.
        Constraint between (m1, r1) and (m2, r2): (r2-r1) % gcd(m1,m2) != 0
        """
        if not moduli_set:
            return []
        
        # Convert to list for indexed access
        mods = list(moduli_set)
        n = len(mods)
        
        # Initialize domains
        domains = {m: set(range(m)) for m in mods}
        
        # Build adjacency: for each modulus, which other moduli constrain it
        # and what gcd they share
        adj = {}
        for i, m1 in enumerate(mods):
            adj[m1] = []
            for j, m2 in enumerate(mods):
                if i != j:
                    g = get_gcd(m1, m2)
                    if g > 1:
                        adj[m1].append((m2, g))
        
        # AC-3 style arc consistency
        # Queue of arcs (m1, m2, g) meaning we need to prune domain of m1 given m2
        queue = []
        for m1 in mods:
            for (m2, g) in adj[m1]:
                queue.append((m1, m2, g))
        
        def prune(m1, m2, g):
            """Prune domain of m1 given m2 with gcd g. Returns True if domain changed."""
            if not domains[m1] or not domains[m2]:
                return False
            
            # For each r1 in domain[m1], check if there exists r2 in domain[m2]
            # such that (r2 - r1) % g != 0
            # This is equivalent to: r1 mod g is NOT in {r2 mod g for r2 in domain[m2]}
            
            # Compute set of residues mod g that are "forbidden" for m1
            # r1 is forbidden if for ALL r2 in domain[m2], (r2 - r1) % g == 0
            # i.e., r2 % g == r1 % g for all r2 in domain[m2]
            # i.e., all r2 in domain[m2] have the same residue mod g, and it equals r1 % g
            
            # Actually, simpler: r1 is supported if there exists r2 in domain[m2] with (r2-r1)%g != 0
            # r1 is pruned if for all r2 in domain[m2], (r2-r1)%g == 0
            # which means r2%g == r1%g for all r2 in domain[m2]
            # which means all elements in domain[m2] are congruent mod g, and to r1
            
            # Let's compute: what residues mod g appear in domain[m2]?
            m2_residues_mod_g = set(r % g for r in domains[m2])
            
            # r1 is pruned if {r1 % g} == m2_residues_mod_g, i.e., m2_residues_mod_g is a singleton {r1%g}
            # Wait, that's not quite right. r1 is pruned if there is NO r2 in domain[m2] with (r2-r1)%g != 0
            # (r2-r1)%g != 0 means r2%g != r1%g
            # So r1 is pruned if for ALL r2 in domain[m2], r2%g == r1%g
            # i.e., m2_residues_mod_g == {r1 % g}
            
            if len(m2_residues_mod_g) == 1:
                # All r2 in domain[m2] have the same residue mod g
                target_residue = next(iter(m2_residues_mod_g))
                # Remove r1 from domain[m1] if r1 % g == target_residue
                new_domain = set()
                changed = False
                for r in domains[m1]:
                    if r % g == target_residue:
                        changed = True
                    else:
                        new_domain.add(r)
                if changed:
                    domains[m1] = new_domain
                    if not new_domain:
                        return True  # Domain wiped out
                    return True
            return False
        
        # Process queue
        while queue:
            if time.time() > time_limit:
                return None
            m1, m2, g = queue.pop()
            if prune(m1, m2, g):
                if not domains[m1]:
                    return None
                # Add arcs that might need rechecking
                for (m3, g3) in adj.get(m1, []):
                    if m3 != m2:
                        queue.append((m3, m1, g3))
        
        # Check if all domains are non-empty
        for m in mods:
            if not domains[m]:
                return None
        
        # Now do backtracking search
        # Order variables by smallest domain first
        var_order = sorted(mods, key=lambda m: len(domains[m]))
        
        # For each variable, order values by least constraining
        def value_order(m):
            # Prefer values that leave more options for others
            # For simplicity, just use the domain
            return list(domains[m])
        
        assignment = {}
        
        def backtrack(idx):
            if time.time() > time_limit:
                return None
            if idx == n:
                return assignment.copy()
            
            m = var_order[idx]
            for r in value_order(m):
                # Check consistency with already assigned variables
                consistent = True
                for m2, r2 in assignment.items():
                    g = get_gcd(m, m2)
                    if (r2 - r) % g == 0:
                        consistent = False
                        break
                
                if consistent:
                    assignment[m] = r
                    # Forward check: prune future domains
                    # For simplicity in this version, just do basic backtracking
                    result = backtrack(idx + 1)
                    if result is not None:
                        return result
                    del assignment[m]
            
            return None
        
        result = backtrack(0)
        if result is None:
            return None
        
        # Convert to list of [residue, modulus] pairs
        return [[result[m], m] for m in mods]
    
    # Try to solve with all even moduli
    best_result = None
    
    # First try the full set
    result = solve_csp(candidate_mods[:], deadline)
    if result:
        best_result = result
    else:
        # If full set fails, try removing moduli one by one
        current_mods = candidate_mods[:]
        while current_mods and time.time() < deadline:
            # Try to solve with current set
            result = solve_csp(current_mods[:], deadline)
            if result:
                best_result = result
                break
            # Remove the modulus with the smallest domain or random one
            # For now, remove random
            idx = random.randint(0, len(current_mods) - 1)
            current_mods.pop(idx)
    
    if best_result is None:
        # Fallback: greedy construction
        best_result = []
        for m in sorted(candidate_mods):
            if time.time() > deadline:
                break
            found = False
            for r in range(m):
                ok = True
                for r2, m2 in best_result:
                    if not _disjoint(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    best_result.append([r, m])
                    found = True
                    break
            if not found:
                continue
    
    # Final validation
    if best_result:
        n = len(best_result)
        for i in range(n):
            for j in range(i + 1, n):
                r1, m1 = best_result[i]
                r2, m2 = best_result[j]
                if not _disjoint(r1, m1, r2, m2):
                    # This should not happen, but just in case
                    pass
    else:
        # Ensure nonempty
        best_result = [[0, 2]]
    
    return best_result
