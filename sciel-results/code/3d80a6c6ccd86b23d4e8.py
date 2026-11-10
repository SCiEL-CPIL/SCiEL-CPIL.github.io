import time
import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Precompute variable to clause mapping
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for lit in clause:
            var_idx = abs(lit) - 1
            var_to_clauses[var_idx].append(i)
            
    def compute_unsat_counts(assignment):
        unsat_counts = [0] * len(clauses)
        for i, clause in enumerate(clauses):
            for lit in clause:
                if assignment[abs(lit)-1] != (lit > 0):
                    unsat_counts[i] += 1
        return unsat_counts

    start_time = time.time()
    # 280s limit to be safe
    deadline = start_time + 280
    
    restart_count = 0
    while time.time() < deadline:
        # Deterministic seed for reproducibility across runs of the same instance
        rng = random.Random(restart_count)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        unsat_counts = compute_unsat_counts(assignment)
        if all(c == 0 for c in unsat_counts):
            return assignment
            
        unsat_clauses = [i for i, c in enumerate(unsat_counts) if c > 0]
        
        # Local search loop
        while time.time() < deadline:
            if not unsat_clauses:
                return assignment
                
            # Pick random unsat clause
            clause_idx = rng.choice(unsat_clauses)
            clause = clauses[clause_idx]
            
            # Calculate cost for each literal in the clause
            # Cost = number of other clauses containing this var that are currently satisfied 
            # but would become unsatisfied if we flip this var.
            
            best_var = -1
            best_cost = float('inf')
            
            # We'll gather (cost, var_idx) pairs
            # To break ties randomly, we can add a small random noise or just pick first min
            # Standard GSAT picks min. WalkSAT picks random with prob p.
            
            candidates = []
            for lit in clause:
                var_idx = abs(lit) - 1
                cost = 0
                for c_idx in var_to_clauses[var_idx]:
                    c = clauses[c_idx]
                    # Check if clause c is satisfied
                    is_sat = False
                    for l in c:
                        if assignment[abs(l)-1] == (l > 0):
                            is_sat = True
                            break
                    
                    if is_sat:
                        # If we flip var_idx, does c remain satisfied?
                        # Find the literal in c corresponding to var_idx
                        lit_in_c = 0
                        for l in c:
                            if abs(l) == var_idx + 1:
                                lit_in_c = l
                                break
                                
                        is_true_before = assignment[var_idx] == (lit_in_c > 0)
                        
                        if is_true_before:
                            # It was True, becomes False.
                            # Check if any OTHER literal in c is True.
                            has_other_true = False
                            for l in c:
                                if l != lit_in_c and assignment[abs(l)-1] == (l > 0):
                                    has_other_true = True
                                    break
                            if not has_other_true:
                                cost += 1
                candidates.append((cost, var_idx))
                
            # Pick best
            # Noise parameter 0.2
            if rng.random() < 0.2:
                _, best_var = rng.choice(candidates)
            else:
                # Sort by cost, then by var_idx to be deterministic if costs are equal?
                # Or just pick the first one with min cost.
                # rng.choice among mins is better for diversity.
                min_cost = min(c for c, _ in candidates)
                best_vars = [v for c, v in candidates if c == min_cost]
                best_var = rng.choice(best_vars)
                
            # Flip
            assignment[best_var] = not assignment[best_var]
            
            # Update unsat_counts and unsat_clauses
            for c_idx in var_to_clauses[best_var]:
                # Recompute count for this clause
                new_count = 0
                for lit in clauses[c_idx]:
                    if assignment[abs(lit)-1] != (lit > 0):
                        new_count += 1
                old_count = unsat_counts[c_idx]
                
                if old_count > 0 and new_count == 0:
                    # Became satisfied
                    if c_idx in unsat_clauses:
                        unsat_clauses.remove(c_idx)
                elif old_count == 0 and new_count > 0:
                    # Became unsatisfied
                    if c_idx not in unsat_clauses:
                        unsat_clauses.append(c_idx)
                        
                unsat_counts[c_idx] = new_count
                
        restart_count += 1
        
    return assignment
