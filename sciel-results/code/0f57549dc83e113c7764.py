# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 290.0  # safety margin before 300s kill

    # Preprocess clauses into numpy arrays for fast evaluation
    # Each clause: array of signed literals (1-based)
    # We'll work with 0-based variable indices and polarity
    var_idx = np.array([[abs(l) - 1 for l in c] for c in clauses], dtype=np.int32)
    polarity = np.array([[(l > 0) for l in c] for c in clauses], dtype=bool)
    # polarity[c][i] is True if literal is positive, False if negative

    # Build clause-to-variable and variable-to-clause incidence for unit propagation
    var_to_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(clauses):
        for l in c:
            var_to_clauses[abs(l) - 1].append(ci)

    def count_unsat_fast(assignment):
        """Count unsatisfied clauses using numpy."""
        # For each clause, check if any literal is satisfied
        # assignment[var_idx[c][i]] == polarity[c][i] means literal satisfied
        lit_vals = assignment[var_idx]  # shape (m, 3)
        sat = (lit_vals == polarity).any(axis=1)
        return (~sat).sum()

    def find_unsat_clauses(assignment):
        """Return list of unsatisfied clause indices."""
        lit_vals = assignment[var_idx]
        sat = (lit_vals == polarity).any(axis=1)
        return np.where(~sat)[0]

    def walksat_step(assignment, unsat_indices, rng, phase):
        """Perform one WalkSAT step. Returns new assignment."""
        if len(unsat_indices) == 0:
            return assignment
        # Choose an unsatisfied clause (weighted by number of satisfied literals for better heuristics)
        ci = unsat_indices[rng.randint(0, len(unsat_indices) - 1)]
        clause_vars = var_idx[ci]
        clause_pol = polarity[ci]
        
        # Compute score for each variable in clause:
        # - number of other clauses that would become unsatisfied if we flip
        # - prefer flipping a variable that satisfies this clause
        # Use a mix of random and greedy
        
        best_var = None
        best_score = None
        for vi in clause_vars:
            # If we flip this variable, does it satisfy the clause?
            current_val = assignment[vi]
            desired_val = clause_pol[np.where(clause_vars == vi)[0][0]]
            would_satisfy = (not current_val) == desired_val if desired_val else current_val
            # Simpler: after flip, is literal satisfied?
            new_val = not current_val
            lit_satisfied = (new_val == desired_val)
            
            # Count how many other clauses contain this variable and would be affected
            # Greedy: minimize number of newly unsatisfied clauses
            affected = 0
            for oc in var_to_clauses[vi]:
                if oc == ci:
                    continue
                # Check if this clause is currently satisfied and would become unsatisfied
                # Only check if this is the only satisfied literal in that clause
                oc_vars = var_idx[oc]
                oc_pol = polarity[oc]
                # Current state
                oc_sat = False
                for j, ov in enumerate(oc_vars):
                    if assignment[ov] == oc_pol[j]:
                        oc_sat = True
                        break
                if not oc_sat:
                    continue  # already unsatisfied, flipping might help or not
                # Would it become unsatisfied?
                # Check if this variable was the only satisfied one
                only_sat = True
                for j, ov in enumerate(oc_vars):
                    if ov == vi:
                        continue
                    if assignment[ov] == oc_pol[j]:
                        only_sat = False
                        break
                if only_sat:
                    affected += 1
            
            score = affected * 10 - (1 if lit_satisfied else 0) * 5 + rng.random()
            if best_score is None or score < best_score:
                best_score = score
                best_var = vi
        
        if best_var is not None:
            assignment[best_var] = not assignment[best_var]
        return assignment

    # Main search loop with restarts
    rng = random.Random(0)
    best_assignment = None
    best_unsat = float('inf')
    
    max_restarts = 50
    max_steps_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize assignment
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
        
        # Try to find a solution
        unsat_count = count_unsat_fast(assignment)
        if unsat_count == 0:
            return assignment.tolist()
        
        unsat_indices = find_unsat_clauses(assignment)
        local_best_unsat = unsat_count
        local_best_assign = assignment.copy()
        
        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break
            
            unsat_count = count_unsat_fast(assignment)
            if unsat_count == 0:
                return assignment.tolist()
            
            if unsat_count < local_best_unsat:
                local_best_unsat = unsat_count
                local_best_assign = assignment.copy()
                if local_best_unsat < best_unsat:
                    best_unsat = local_best_unsat
                    best_assignment = local_best_assign.copy()
            
            unsat_indices = find_unsat_clauses(assignment)
            assignment = walksat_step(assignment, unsat_indices, rng, restart)
        
        # If we found a better local minimum, keep it
        if local_best_unsat < best_unsat:
            best_unsat = local_best_unsat
            best_assignment = local_best_assign.copy()
        
        # If best is 0, we're done
        if best_unsat == 0:
            return best_assignment.tolist()
    
    # Return best found
    if best_assignment is not None:
        return best_assignment.tolist()
    return [False] * n
# EVOLVE-BLOCK-END
