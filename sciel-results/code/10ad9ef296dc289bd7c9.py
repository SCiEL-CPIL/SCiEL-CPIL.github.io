import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # 1. Define all potential nodes: (modulus, residue)
    # We only consider moduli in [20, 200]
    moduli = list(range(20, 201))
    n_moduli = len(moduli)
    
    # Precompute gcds for moduli
    gcds = [[0] * n_moduli for _ in range(n_moduli)]
    for i in range(n_moduli):
        for j in range(i, n_moduli):
            g = math.gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g

    # 2. Build the list of all valid (m, r) pairs and their indices
    # Node index is a unique ID for (m_idx, r)
    all_nodes = []
    node_id_map = {} # (m_idx, r) -> node_id
    weights = []     # 1/m for each node
    
    current_id = 0
    for m_idx, m in enumerate(moduli):
        for r in range(m):
            node_id_map[(m_idx, r)] = current_id
            all_nodes.append((m_idx, r))
            weights.append(1.0 / m)
            current_id += 1
            
    total_nodes = current_id

    # 3. Precompute compatibility bitmasks
    # compat_mask[i] is a bitmask where bit j is 1 if node i and node j are COMPATIBLE (disjoint)
    # Note: A node is always compatible with itself, so we set bit i for node i.
    # We use Python integers as bitsets.
    
    # Optimization: We only need to check compatibility if they might conflict.
    # Two nodes (m1, r1) and (m2, r2) conflict if (r1 - r2) % gcd(m1, m2) == 0.
    # If they don't conflict, they are compatible.
    
    compat_masks = [0] * total_nodes
    
    # This O(N^2) precomputation might be slow if N is large.
    # N = sum(m for m in 20..200) = sum(m) = 20*201/2 + ... wait, sum of 20..200 is (20+200)*181/2 = 220*90.5 = 19910.
    # 19910^2 is ~400 million operations. In Python, this is too slow (will take minutes).
    
    # We need a faster way to build the masks or a smarter heuristic that doesn't require the full N^2 matrix upfront.
    # Alternative: Build the masks on the fly? No, that defeats the purpose.
    # Alternative: Use NumPy for the conflict matrix?
    # Let's try to optimize the mask generation.
    
    # Actually, we can group by modulus.
    # For a fixed modulus m1, and residue r1, it conflicts with (m2, r2) if r2 == r1 mod gcd(m1, m2).
    
    # Let's try a hybrid approach:
    # 1. Generate a good initial solution using a fast greedy heuristic that checks conflicts dynamically.
    # 2. Use a local search that uses pre-computed *conflict* lists for specific moduli, or just re-checks.
    
    # Given the 120s limit, building the full 20k x 20k compatibility matrix in pure Python is risky.
    # 400M iterations * ~100ns = 40s just for precomputation. Plus the search.
    # Let's optimize the precomputation using NumPy if possible, or a smarter Python loop.
    
    # Let's use NumPy to generate the conflict matrix.
    import numpy as np
    
    # Create arrays for all nodes
    node_m_idx = np.array([n[0] for n in all_nodes])
    node_r = np.array([n[1] for n in all_nodes])
    node_m = np.array([moduli[idx] for idx in node_m_idx])
    
    # We want a matrix C[i, j] = 1 if compatible, 0 if conflict.
    # Conflict if (node_r[i] - node_r[j]) % gcd(node_m[i], node_m[j]) == 0.
    
    # This is still O(N^2) memory and time.
    # Memory: 20k x 20k bytes (if uint8) = 400MB. Might be too much for some environments, but usually okay.
    # Time: 400M ops in NumPy is fast (seconds).
    
    # Let's try to construct the compatibility matrix using broadcasting.
    # However, gcd is not a vectorized operation in standard NumPy.
    # We can precompute gcds for all pairs of moduli (181x181) and then map them.
    
    gcd_matrix = np.zeros((n_moduli, n_moduli), dtype=np.int32)
    for i in range(n_moduli):
        for j in range(i, n_moduli):
            g = math.gcd(moduli[i], moduli[j])
            gcd_matrix[i, j] = g
            gcd_matrix[j, i] = g
            
    # Map node indices to their modulus indices
    # We need gcd for each pair of nodes.
    # G = gcd_matrix[node_m_idx[:, None], node_m_idx[None, :]]
    # R_diff = node_r[:, None] - node_r[None, :]
    # Conflict = (R_diff % G == 0)
    # Compatible = ~Conflict
    
    # This creates a 20k x 20k array. 
    # 20000 * 20000 * 4 bytes = 1.6 GB. This is likely too much memory.
    
    # We need a different strategy for the "Graph" part.
    # We cannot store the full compatibility matrix.
    
    # Revised Strategy:
    # Use a greedy construction with a strong local search that uses *on-the-fly* conflict checks.
    # But the current program already does this and scores 3.28.
    # The prompt asks for a *structurally different* approach.
    
    # What if we use the property that we want to maximize sum(1/m)?
    # This is equivalent to minimizing sum(1 - 1/m)? No.
    
    # Let's look at the "Maximum Weight Independent Set" again.
    # If we can't build the full graph, can we build a *sparse* representation?
    # Or can we use a heuristic that doesn't need the full graph?
    
    # New Idea: **Modulus-First Greedy with Backtracking (DFS)**
    # The number of moduli is only 181.
    # The choice of residue is what creates the conflicts.
    # If we fix the set of moduli, the problem reduces to finding residues.
    # But we also choose the set of moduli.
    
    # Let's try a different structural approach: **Integer Linear Programming (ILP)** relaxation or **Lagrangian Relaxation**?
    # Too complex to implement from scratch reliably in 120s.
    
    # How about **Simulated Annealing with a better Move Operator**?
    # The current SA changes residues or adds/removes moduli.
    # What if we define moves based on *Swapping Moduli*?
    # If we swap modulus m1 with m2, we keep the set size same, but change weights.
    
    # Actually, the biggest win might be in the **Initialization**.
    # The current program uses a simple greedy.
    # What if we use a **Randomized Greedy** that considers the *future* impact?
    
    # Let's try a **Genetic Algorithm** approach.
    # Representation: A binary vector of length 181 indicating which moduli are selected.
    # For each selected modulus, we assign the *best* residue that minimizes conflicts with others.
    # This decouples the "which moduli" and "which residues" problems.
    
    def best_residue_for_modulus(m_idx, other_indices, other_residues):
        """Find the residue r for m_idx that minimizes conflicts with the rest."""
        m = moduli[m_idx]
        # For each candidate r, count how many other nodes it conflicts with.
        # But wait, if it conflicts, it's invalid.
        # We need a valid residue.
        # If multiple are valid, does it matter which one we pick?
        # Yes, because it affects *future* additions?
        # No, in this GA model, we fix the set of moduli first.
        # Then we just need *any* valid residue for each.
        # If a set of moduli has *any* valid assignment of residues, it's feasible.
        # The score depends ONLY on the set of moduli!
        
        # Wait, is that true?
        # Score = sum(1/m for m in selected).
        # Yes! The score does NOT depend on the residues, only on the moduli.
        # The residues only determine *feasibility*.
        
        # So the problem is:
        # Find a subset S of moduli such that there exist residues r_m for m in S
        # such that for all m1 != m2 in S, r_m1 != r_m2 mod gcd(m1, m2).
        # Maximize sum(1/m for m in S).
        
        # This is a much simpler structure!
        # We don't need to optimize residues for score. We just need to find a feasible set of moduli.
        # Among feasible sets, pick the one with max sum(1/m).
        
        # How to check if a set S is feasible?
        # This is a constraint satisfaction problem (CSP).
        # Variables: r_m for m in S.
        # Domains: 0..m-1.
        # Constraints: r_m1 != r_m2 mod gcd(m1, m2).
        
        # Since we want to maximize weight, we can try to add moduli in descending order of weight (ascending order of m).
        # Greedy: Start with empty set. Add smallest m. Then next smallest m if feasible, etc.
        # To check feasibility of adding a new modulus m_new to set S:
        # We need to check if the CSP is still solvable.
        # This is expensive to check exactly (NP-hard in general, but small here).
        
        # However, we can use a simple heuristic:
        # Try to assign residues to S. If we can assign residues, it's feasible.
        # Use a backtracking search for residue assignment.
        
        # Algorithm:
        # 1. Sort moduli by 1/m (descending).
        # 2. Iterate through moduli.
        # 3. For each modulus m, try to add it to the current set.
        # 4. Check if the new set is feasible by running a fast CSP solver (backtracking) for residues.
        # 5. If feasible, keep it. If not, skip it.
        
        # To improve this, we can use **Local Search on the Set of Moduli**.
        # Start with a greedy set.
        # Move: Remove one modulus, add another.
        # Check feasibility of the new set using the CSP solver.
        
        def is_feasible(indices):
            """Check if there exist residues for the given moduli indices."""
            if not indices:
                return True, []
            
            # Order indices by modulus size (smaller moduli have smaller domains, so constrain them first)
            # Actually, smaller moduli have fewer residues, so they are more constrained?
            # No, larger moduli have more residues, so they are more flexible.
            # We should assign residues to moduli with *largest* gcds with others first?
            # Or just sort by modulus size.
            
            sorted_indices = sorted(indices, key=lambda i: moduli[i])
            n = len(sorted_indices)
            
            # Backtracking search
            residues = [None] * n
            
            def backtrack(pos):
                if pos == n:
                    return True
                
                m_idx = sorted_indices[pos]
                m = moduli[m_idx]
                
                for r in range(m):
                    # Check compatibility with already assigned residues
                    compatible = True
                    for k in range(pos):
                        m_old_idx = sorted_indices[k]
                        r_old = residues[k]
                        g = gcds[m_idx][m_old_idx]
                        if (r - r_old) % g == 0:
                            compatible = False
                            break
                    if compatible:
                        residues[pos] = r
                        if backtrack(pos + 1):
                            return True
                        residues[pos] = None # Undo
                return False
            
            if backtrack(0):
                # Map residues back to original order
                res_map = {sorted_indices[i]: residues[i] for i in range(n)}
                return True, [res_map[i] for i in indices]
            return False, []

        # Phase 1: Greedy Construction
        # Sort moduli by 1/m (descending) i.e., m ascending.
        sorted_moduli_indices = sorted(range(n_moduli), key=lambda i: moduli[i])
        
        selected_indices = []
        for m_idx in sorted_moduli_indices:
            new_indices = selected_indices + [m_idx]
            feasible, _ = is_feasible(new_indices)
            if feasible:
                selected_indices.append(m_idx)
                
        # Get score
        current_score = sum(1.0 / moduli[i] for i in selected_indices)
        
        # Phase 2: Local Search on Moduli Set
        # Try to swap a selected modulus with an unselected one.
        # Or add a new one and remove a conflicting one.
        
        best_indices = list(selected_indices)
        best_score = current_score
        
        # Precompute list of unselected
        selected_set = set(selected_indices)
        
        # Use a time budget
        search_deadline = start_time + 110.0
        
        while time.time() < search_deadline:
            improved = False
            # Try adding a random unselected modulus and removing one selected if infeasible
            # Or just try swapping.
            
            # Random swap
            if best_indices:
                pos_to_remove = random.randint(0, len(best_indices) - 1)
                removed_idx = best_indices[pos_to_remove]
                remaining_indices = best_indices[:pos_to_remove] + best_indices[pos_to_remove+1:]
                
                # Try to add a new one
                candidates = [i for i in range(n_moduli) if i not in remaining_indices]
                random.shuffle(candidates)
                
                # Try top candidates (smallest moduli first for better score potential)
                for cand in candidates:
                    new_indices = remaining_indices + [cand]
                    feasible, _ = is_feasible(new_indices)
                    if feasible:
                        new_score = sum(1.0 / moduli[i] for i in new_indices)
                        if new_score > best_score:
                            best_score = new_score
                            best_indices = new_indices
                            selected_set = set(new_indices)
                            improved = True
                            break
            else:
                # If empty, just add smallest
                best_indices = [0]
                best_score = 1.0 / moduli[0]
                selected_set = {0}
                improved = True
            
            if not improved:
                # Try a pure addition (if there's room)
                # Check if we can add any unselected modulus without removing
                candidates = [i for i in range(n_moduli) if i not in selected_set]
                random.shuffle(candidates)
                added = False
                for cand in candidates:
                    new_indices = best_indices + [cand]
                    feasible, _ = is_feasible(new_indices)
                    if feasible:
                        new_score = sum(1.0 / moduli[i] for i in new_indices)
                        if new_score > best_score:
                            best_score = new_score
                            best_indices = new_indices
                            selected_set = set(new_indices)
                            added = True
                            break
                if not added:
                    # If no improvement, break or do a more complex move?
                    # For now, break to avoid infinite loop of no-op swaps
                    pass
                    
        # Final step: Get residues for the best set
        feasible, final_residues = is_feasible(best_indices)
        if not feasible:
            # Fallback to greedy result if something went wrong
            feasible, final_residues = is_feasible(selected_indices)
            best_indices = selected_indices
            
        # Construct result
        result_classes = []
        for i, r in zip(best_indices, final_residues):
            result_classes.append([moduli[i], r])
            
        return result_classes

construct()
