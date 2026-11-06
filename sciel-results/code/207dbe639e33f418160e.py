"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard
    start_time = time.time()
    timeout = 320.0  # Leave margin for overhead (evaluator kills at 330s)
    
    m = len(clauses)
    if m == 0:
        return [True] * n
        
    # Preprocess clauses into numpy arrays for fast evaluation
    clause_data = np.empty((m, 3), dtype=np.int8)
    clause_vars = np.empty((m, 3), dtype=np.int32)
    
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_data[i, j] = 1 if lit > 0 else -1
            clause_vars[i, j] = abs(lit) - 1
    
    # Precompute sign_bool for faster satisfaction check
    sign_bool = (clause_data == 1)  # (m, 3) bool array
    
    # Build variable-to-clauses mapping for incremental updates
    var_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_clauses[v].append(i)
    
    # Precompute for each variable, the list of (clause_idx, position) pairs
    var_clause_pos = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_clause_pos[v].append((i, j))
    
    def compute_satisfaction(assignment):
        """Compute boolean array of satisfied clauses from scratch."""
        vals = assignment[clause_vars]  # (m, 3) bools
        literal_true = (vals == sign_bool)  # (m, 3)
        return np.any(literal_true, axis=1)  # (m,) bools
    
    def unit_propagate(assignment, sat):
        """Apply unit propagation to fix forced assignments. Returns (assignment, sat, changed)."""
        changed = False
        for _ in range(50):  # Limit iterations
            found_unit = False
            # Find unit clauses: exactly one unassigned/unsatisfied literal
            # For each unsatisfied clause, if exactly one literal can be made true by flipping, force it
            unsat_indices = np.where(~sat)[0]
            for ci in unsat_indices:
                # Check if this is a unit clause (only one literal not yet satisfied)
                # A clause is unit if exactly one literal is false and the other two are... 
                # Actually, for unit propagation in SAT: a clause with one unassigned variable
                # But here all variables are assigned, so we look for clauses where
                # flipping exactly one variable would satisfy it, and that variable
                # doesn't appear in any other unsatisfied clause in a conflicting way.
                # Simpler: find clauses where exactly one literal is false (the other two true)
                # Wait, if clause is unsatisfied, ALL literals are false.
                # Unit propagation applies when a clause has all but one literal false,
                # and the remaining literal's variable is unassigned.
                # Since all variables are assigned in local search, traditional unit prop
                # doesn't directly apply. Instead, we do a "forced flip" heuristic:
                # If a variable appears in many unsatisfied clauses and flipping it
                # would satisfy all of them without breaking satisfied ones, force it.
                pass
            # For now, skip unit propagation as it's complex in local search context
            break
        return assignment, sat, changed
    
    def walk_sat(restart_assignment, max_flips, seed):
        """Run WalkSAT with precomputed clause-position tracking for O(degree) flip evaluation."""
        rng = random.Random(seed)
        assignment = restart_assignment.copy()
        
        # Initialize satisfaction state
        sat = compute_satisfaction(assignment)
        unsat_count = int(np.sum(~sat))
        
        if unsat_count == 0:
            return assignment.tolist()
        
        # Track which clauses are unsatisfied for quick access
        unsat_indices = np.where(~sat)[0]
        
        best_unsat = unsat_count
        best_assignment = assignment.copy()
        
        # Precompute sole_support: for each variable v, for each clause ci it appears in,
        # sole_support[v, ci] = True if v is the ONLY literal that could satisfy ci
        # (i.e., the other two literals are currently false)
        # We'll compute this on-the-fly per flip for speed, but cache the structure.
        
        for step in range(max_flips):
            if time.time() - start_time > timeout - 2:
                break
            
            if unsat_count == 0:
                return assignment.tolist()
            
            # Track best state
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment.copy()
            
            # Pick a random unsatisfied clause
            if unsat_count == 0:
                return assignment.tolist()
            clause_idx = int(unsat_indices[rng.randint(0, len(unsat_indices) - 1)])
            
            # Choose variable to flip using score-based strategy
            clause_v = clause_vars[clause_idx]
            deltas = [0, 0, 0]
            
            for pos, v in enumerate(clause_v):
                # Compute impact of flipping variable v
                # For each (ci, j) in var_clause_pos[v]:
                #   if ci is unsatisfied: flipping v satisfies ci (delta += 1)
                #   if ci is satisfied: check if v is sole support (delta -= 1)
                becomes_sat = 0
                becomes_unsat = 0
                
                for ci, j in var_clause_pos[v]:
                    if not sat[ci]:
                        # Clause is unsatisfied. Flipping v makes literal (ci,j) true.
                        becomes_sat += 1
                    else:
                        # Clause is satisfied. Check if v is the sole supporting literal.
                        # i.e., the other two literals are false.
                        sole = True
                        for k in range(3):
                            if k == j:
                                continue
                            v2 = clause_vars[ci, k]
                            if v2 == v:
                                continue  # Same variable, skip
                            if assignment[v2] == sign_bool[ci, k]:
                                sole = False
                                break
                        if sole:
                            becomes_unsat += 1
                
                deltas[pos] = becomes_sat - becomes_unsat
            
            # Use soft-max for variable selection to smooth the objective
            # Convert deltas to probabilities using softmax with temperature
            temp = 2.0
            exp_deltas = np.exp(np.array(deltas, dtype=np.float64) / temp)
            probs = exp_deltas / exp_deltas.sum()
            
            # With probability 0.7, pick greedily (max delta); otherwise sample
            if rng.random() < 0.7:
                var_to_flip = clause_v[int(np.argmax(deltas))]
            else:
                var_to_flip = clause_v[rng.choices([0, 1, 2], weights=probs)[0]]
            
            # Flip the variable
            assignment[var_to_flip] = not assignment[var_to_flip]
            
            # Incrementally update satisfaction for affected clauses
            for ci, j in var_clause_pos[var_to_flip]:
                was_sat = sat[ci]
                # Recompute satisfaction for clause ci
                is_now_sat = False
                for k in range(3):
                    if assignment[clause_vars[ci, k]] == sign_bool[ci, k]:
                        is_now_sat = True
                        break
                sat[ci] = is_now_sat
                
                if was_sat and not is_now_sat:
                    unsat_count += 1
                    unsat_indices = np.where(~sat)[0]
                elif not was_sat and is_now_sat:
                    unsat_count -= 1
                    unsat_indices = np.where(~sat)[0]
                else:
                    # Still need to update unsat_indices if count changed
                    pass
            
            # Rebuild unsat_indices only when count changed to avoid O(m) scan each step
            # Actually, let's rebuild it less frequently
            if step % 100 == 0:
                unsat_indices = np.where(~sat)[0]
        
        # Final check
        if unsat_count == 0:
            return assignment.tolist()
        
        return None
    
    # Main search loop with restarts and phase saving
    best_result = None
    rng = random.Random(0)
    restart_count = 0
    phase_assignment = None
    best_unsat_ever = m + 1
    
    # Initial phase: try a few quick random walks to find a good starting point
    for _ in range(5):
        initial = np.array([rng.random() < 0.5 for _ in range(n)])
        sat = compute_satisfaction(initial)
        unsat = int(np.sum(~sat))
        if unsat < best_unsat_ever:
            best_unsat_ever = unsat
            phase_assignment = initial.copy()
    
    while time.time() - start_time < timeout - 10:
        restart_count += 1
        
        # Use phase assignment with some perturbation for better restarts
        if phase_assignment is not None and restart_count > 1:
            # Perturb the phase assignment slightly
            initial = phase_assignment.copy()
            num_flips = max(1, int(n * 0.1 * (1 + 0.1 * min(restart_count, 20))))
            flip_indices = rng.sample(range(n), min(num_flips, n))
            for idx in flip_indices:
                initial[idx] = not initial[idx]
        else:
            initial = np.array([rng.random() < 0.5 for _ in range(n)])
        
        # Adaptive flip budget
        base_flips = 10000
        adaptive_factor = 1.0 + 0.2 * min(restart_count, 15)
        max_flips = int(base_flips * adaptive_factor)
        
        result = walk_sat(initial, max_flips, seed=restart_count * 12345 + 67890)
        
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return best_result
    
    # If we didn't find a solution, return the best assignment we found
    if phase_assignment is not None:
        return phase_assignment.tolist()
    
    # Fallback: return a random assignment
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
