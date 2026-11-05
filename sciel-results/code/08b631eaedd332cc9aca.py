"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    def gcd(a, b):
        return math.gcd(a, b)
    
    def check_pair(r1, m1, r2, m2):
        """Check if two residue classes are disjoint."""
        g = gcd(m1, m2)
        return (r1 - r2) % g != 0
    
    # Model as Maximum Independent Set on conflict graph
    # Nodes: (modulus, residue) pairs for all m in [2,128], r in [0,m-1]
    # Edge between node i and node j if they are incompatible (NOT disjoint)
    # We want max independent set = max set of mutually compatible nodes
    # But we also need distinct moduli, so at most one node per modulus
    
    # Better model: for each modulus m, we can pick at most one residue r
    # Two choices (m1,r1) and (m2,r2) conflict if NOT disjoint
    # We want max number of moduli with compatible residue choices
    
    # Build conflict structure
    all_moduli = list(range(2, 129))
    n_moduli = len(all_moduli)
    
    # For each pair of moduli (m1, m2), determine which residue pairs are compatible
    # compatible[m1_idx][m2_idx] = bitmask of compatible residue pairs? Too complex.
    # Instead: for fixed m1, m2, and r1, which r2 values are compatible?
    
    # Precompute: for each pair of moduli, a list of compatible residue pairs
    # This is expensive, so let's use a different approach
    
    # Branch and bound with bitsets
    # State: for each modulus, track which residues are still possible
    # Use greedy coloring for upper bound
    
    def solve_mis():
        """Solve using branch-and-bound with greedy coloring upper bound."""
        
        # For each modulus m, list of residues 0..m-1
        # Node in our search: (modulus_idx, chosen_residue)
        # Conflict: two nodes conflict if their residue classes are NOT disjoint
        
        # We'll use a recursive branch-and-bound
        # At each step, pick an unassigned modulus, try each residue
        # Upper bound: current_count + number of remaining moduli that can still be assigned
        
        # To get a tighter upper bound, use greedy coloring on the conflict graph
        # of remaining nodes
        
        # For efficiency, precompute gcd table
        gcd_table = {}
        for i, m1 in enumerate(all_moduli):
            for j, m2 in enumerate(all_moduli):
                if i != j:
                    gcd_table[(i, j)] = gcd(m1, m2)
        
        # For a given set of assigned (modulus, residue) pairs,
        # determine which unassigned moduli still have at least one valid residue
        def get_valid_residues(mod_idx, assigned):
            """Get list of valid residues for modulus mod_idx given assigned pairs."""
            m = all_moduli[mod_idx]
            valid = []
            for r in range(m):
                compatible = True
                for (a_mod_idx, a_res) in assigned:
                    if a_mod_idx == mod_idx:
                        continue
                    a_m = all_moduli[a_mod_idx]
                    g = gcd_table[(mod_idx, a_mod_idx)]
                    if (r - a_res) % g == 0:
                        compatible = False
                        break
                if compatible:
                    valid.append(r)
            return valid
        
        # Greedy coloring for upper bound
        def greedy_color_upper_bound(unassigned_mods, assigned):
            """
            Upper bound on additional moduli we can assign.
            Uses a simple greedy approach: process unassigned moduli,
            count how many can be assigned (optimistically).
            """
            count = 0
            remaining = list(unassigned_mods)
            # Try to assign greedily
            temp_assigned = list(assigned)
            for mod_idx in remaining:
                valid = get_valid_residues(mod_idx, temp_assigned)
                if valid:
                    # Pick first valid residue
                    temp_assigned.append((mod_idx, valid[0]))
                    count += 1
            return count
        
        # Branch and bound
        best_solution = []
        best_count = 0
        
        def branch_and_bound(assigned, unassigned, current_count):
            nonlocal best_solution, best_count
            
            if time.time() > deadline - 2:
                return
            
            # Upper bound
            ub = current_count + len(unassigned)
            if ub <= best_count:
                return
            
            if not unassigned:
                if current_count > best_count:
                    best_count = current_count
                    best_solution = list(assigned)
                return
            
            # Choose next modulus to branch on
            # Heuristic: pick modulus with fewest valid residues (most constrained)
            min_valid = None
            min_count = float('inf')
            for mod_idx in unassigned:
                valid = get_valid_residues(mod_idx, assigned)
                if len(valid) < min_count:
                    min_count = len(valid)
                    min_valid = mod_idx
                    if min_count == 0:
                        break
            
            if min_valid is None or min_count == 0:
                # No valid modulus or one has no valid residues
                if current_count > best_count:
                    best_count = current_count
                    best_solution = list(assigned)
                return
            
            # Branch on min_valid
            valid_residues = get_valid_residues(min_valid, assigned)
            remaining_unassigned = [m for m in unassigned if m != min_valid]
            
            # Try each valid residue
            for r in valid_residues:
                new_assigned = assigned + [(min_valid, r)]
                branch_and_bound(new_assigned, remaining_unassigned, current_count + 1)
            
            # Also try skipping min_valid
            branch_and_bound(assigned, remaining_unassigned, current_count)
        
        # Initialize with empty assignment
        all_mod_indices = list(range(n_moduli))
        branch_and_bound([], all_mod_indices, 0)
        
        # Convert solution to required format
        result = []
        for (mod_idx, res) in best_solution:
            m = all_moduli[mod_idx]
            result.append([res, m])
        return result
    
    # Run the MIS solver
    result = solve_mis()
    
    if not result:
        # Fallback: seed solution
        result = [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    
    return result
# EVOLVE-BLOCK-END
