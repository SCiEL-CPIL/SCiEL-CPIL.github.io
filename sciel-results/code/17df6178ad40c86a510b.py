"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 285  # safety margin before 330s kill
    rng = random.Random(0)
    
    # Pre-compute clause literal indices for speed
    clause_indices = [[(abs(l) - 1, l > 0) for l in c] for c in clauses]
    num_clauses = len(clauses)
    
    # Build variable-to-clause index for efficient updates
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause_idxs in enumerate(clause_indices):
        for var_idx, is_pos in clause_idxs:
            var_to_clauses[var_idx].append((ci, is_pos))
    
    def is_satisfied(ci, assignment):
        for var_idx, is_pos in clause_indices[ci]:
            if assignment[var_idx] == is_pos:
                return True
        return False
    
    # Multi-restart WalkSAT with clause weighting
    best_assignment = None
    best_unsat = num_clauses + 1
    num_restarts = 0
    max_restarts = 100
    
    while time.time() < deadline and num_restarts < max_restarts:
        num_restarts += 1
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Compute initial unsat count and per-clause status
        unsat_clauses = []
        for ci in range(num_clauses):
            if not is_satisfied(ci, assignment):
                unsat_clauses.append(ci)
        
        current_unsat = len(unsat_clauses)
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment[:]
        
        if current_unsat == 0:
            best_assignment = assignment[:]
            best_unsat = 0
            break
        
        # WalkSAT loop
        steps_per_restart = 15000
        for step in range(steps_per_restart):
            if time.time() >= deadline:
                break
            
            if not unsat_clauses:
                best_assignment = assignment[:]
                best_unsat = 0
                break
            
            # Count false literals per variable in unsat clauses
            false_counts = [0] * n  # number of false literals in unsat clauses per var
            true_counts = [0] * n   # number of true literals in unsat clauses per var
            
            for ci in unsat_clauses:
                for var_idx, is_pos in clause_indices[ci]:
                    if not assignment[var_idx] == is_pos:
                        false_counts[var_idx] += 1
                    else:
                        true_counts[var_idx] += 1
            
            # Weighted clause selection: prefer clauses with fewer false literals
            # (closer to being satisfied)
            weights = [1.0 / (len(clause_indices[ci]) ** 2) for ci in unsat_clauses]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0.0
            chosen_idx = 0
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    chosen_idx = i
                    break
            
            chosen_ci = unsat_clauses[chosen_idx]
            false_lits = [(var_idx, is_pos) for var_idx, is_pos in clause_indices[chosen_ci]
                         if not assignment[var_idx] == is_pos]
            
            if not false_lits:
                # Should not happen, but skip if it does
                continue
            
            # Variable selection: weighted by (true_count / (true_count + false_count + 1))
            # Prefer flipping variables that would satisfy more other unsat clauses
            var_weights = []
            for var_idx, is_pos in false_lits:
                # If we flip this var, it becomes true for this clause
                # Benefit: it might satisfy other unsat clauses where it's false
                benefit = true_counts[var_idx]
                cost = false_counts[var_idx]
                # Soft weighting
                w = benefit / (benefit + cost + 1) + 0.1
                var_weights.append(w)
            
            total_vw = sum(var_weights)
            rv = rng.random() * total_vw
            cum = 0.0
            chosen_var_idx = false_lits[0][0]
            for j, vw in enumerate(var_weights):
                cum += vw
                if rv <= cum:
                    chosen_var_idx = false_lits[j][0]
                    break
            
            # Flip the variable
            old_val = assignment[chosen_var_idx]
            assignment[chosen_var_idx] = not old_val
            
            # Update unsat_clauses incrementally
            # Check clauses containing chosen_var_idx
            affected = var_to_clauses[chosen_var_idx]
            
            new_unsat = []
            for ci in unsat_clauses:
                # Check if this clause is affected by the flip
                is_affected = False
                for var_idx, is_pos in clause_indices[ci]:
                    if var_idx == chosen_var_idx:
                        is_affected = True
                        break
                
                if is_affected:
                    if is_satisfied(ci, assignment):
                        # Clause became satisfied, remove from unsat
                        pass
                    else:
                        new_unsat.append(ci)
                else:
                    new_unsat.append(ci)
            
            # Check if any previously satisfied clauses became unsatisfied
            for ci, is_pos in affected:
                if ci not in new_unsat and not is_satisfied(ci, assignment):
                    new_unsat.append(ci)
            
            unsat_clauses = new_unsat
            current_unsat = len(unsat_clauses)
            
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment[:]
            
            if current_unsat == 0:
                best_assignment = assignment[:]
                best_unsat = 0
                break
        
        if best_unsat == 0:
            break
    
    if best_assignment is not None:
        return best_assignment
    else:
        return [False] * n
# EVOLVE-BLOCK-END
