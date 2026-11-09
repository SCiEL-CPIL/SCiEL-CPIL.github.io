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
    # Also build a set for O(1) membership tests
    var_to_clauses = [[] for _ in range(n)]
    var_to_clause_set = [set() for _ in range(n)]
    for ci, clause_idxs in enumerate(clause_indices):
        for var_idx, is_pos in clause_idxs:
            var_to_clauses[var_idx].append((ci, is_pos))
            var_to_clause_set[var_idx].add(ci)
    
    def is_satisfied(ci, assignment):
        for var_idx, is_pos in clause_indices[ci]:
            if assignment[var_idx] == is_pos:
                return True
        return False
    
    # Multi-restart WalkSAT with clause weighting
    best_assignment = None
    best_unsat = num_clauses + 1
    num_restarts = 0
    max_restarts = 500
    
    # Effective noise parameter for WalkSAT
    # Standard WalkSAT uses noise ~ 0.3-0.5 for 3-SAT
    noise = 0.35
    
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
        steps_per_restart = 20000
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
            
            # Variable selection: standard WalkSAT heuristic
            # With probability (1-noise), pick the literal that maximizes true_counts
            # (i.e., flipping it satisfies the most other unsat clauses)
            # With probability noise, pick randomly
            if rng.random() < (1 - noise):
                # Greedy: pick the literal with max true_count
                best_val = -1
                best_var_idx = false_lits[0][0]
                for var_idx, is_pos in false_lits:
                    if true_counts[var_idx] > best_val:
                        best_val = true_counts[var_idx]
                        best_var_idx = var_idx
                chosen_var_idx = best_var_idx
            else:
                # Random selection from false literals
                chosen_var_idx = false_lits[rng.randint(0, len(false_lits) - 1)][0]
            
            # Flip the variable
            old_val = assignment[chosen_var_idx]
            assignment[chosen_var_idx] = not old_val
            
            # Update unsat_clauses incrementally
            # Check clauses containing chosen_var_idx
            affected = var_to_clauses[chosen_var_idx]
            
            # Build set of unsat clauses for O(1) lookup
            unsat_set = set(unsat_clauses)
            
            new_unsat = []
            for ci in unsat_clauses:
                # Check if this clause is affected by the flip using set lookup
                if chosen_var_idx in [v for v, _ in clause_indices[ci]]:
                    if is_satisfied(ci, assignment):
                        # Clause became satisfied, remove from unsat
                        unsat_set.discard(ci)
                    else:
                        new_unsat.append(ci)
                else:
                    new_unsat.append(ci)
            
            # Check if any previously satisfied clauses became unsatisfied
            for ci, is_pos in affected:
                if ci not in unsat_set and not is_satisfied(ci, assignment):
                    new_unsat.append(ci)
                    unsat_set.add(ci)
            
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
