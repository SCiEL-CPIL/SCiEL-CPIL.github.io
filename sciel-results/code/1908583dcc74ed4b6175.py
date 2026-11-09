import time
import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    start_time = time.time()
    max_time = 280  # Safety margin
    
    # Precompute variable to clauses mapping
    var_clauses = [[] for _ in range(n + 1)]
    for ci, clause in enumerate(clauses):
        for l in clause:
            var_clauses[abs(l)].append(ci)
            
    # Track satisfied status of each clause
    # sat[ci] = True/False
    
    def is_satisfied(ci, assignment):
        for l in clauses[ci]:
            if assignment[abs(l)-1] == (l > 0):
                return True
        return False

    def walksat():
        best_assignment = None
        best_unsat_count = float('inf')
        
        max_restarts = 100
        base_budget = 500
        growth_factor = 1.2
        
        rng = random.Random(42) # Fixed seed for reproducibility in testing, but maybe random is better?
        # The task says "locally generated", so instances vary.
        # Using a fixed seed for the solver's internal randomness is fine.
        
        for restart in range(max_restarts):
            if time.time() - start_time > max_time:
                break
                
            assignment = [rng.random() < 0.5 for _ in range(n)]
            
            # Initialize sat status
            sat = [False] * len(clauses)
            unsat_count = 0
            for ci in range(len(clauses)):
                if is_satisfied(ci, assignment):
                    sat[ci] = True
                else:
                    sat[ci] = False
                    unsat_count += 1
                    
            if unsat_count == 0:
                return assignment
                
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment[:]
                
            budget = int(base_budget * (growth_factor ** restart))
            
            # DLS counts
            # We need to update counts when we flip.
            # Instead of recalculating all counts, we can maintain them.
            # But DLS requires counts of *unsat* clauses.
            # When a clause becomes sat/unsat, we update the counts of its variables.
            
            var_unsat_count = [0] * (n + 1)
            for ci in range(len(clauses)):
                if not sat[ci]:
                    for l in clauses[ci]:
                        var_unsat_count[abs(l)] += 1
                        
            for step in range(budget):
                if time.time() - start_time > max_time:
                    if best_assignment:
                        return best_assignment
                    return assignment
                    
                if unsat_count == 0:
                    return assignment
                    
                # Pick random unsat clause
                # To pick randomly, we can maintain a list of unsat clause indices?
                # Or just scan? Scanning is O(C).
                # Let's maintain a list of unsat clause indices for random selection.
                # This is complex to maintain.
                # Given C is small, scanning is fine.
                
                unsat_indices = [i for i in range(len(clauses)) if not sat[i]]
                ci = rng.choice(unsat_indices)
                clause = clauses[ci]
                
                # Pick variable with max var_unsat_count in this clause
                best_var = -1
                max_count = -1
                for l in clause:
                    v = abs(l)
                    if var_unsat_count[v] > max_count:
                        max_count = var_unsat_count[v]
                        best_var = v
                        
                # Flip best_var
                old_val = assignment[best_var - 1]
                new_val = not old_val
                assignment[best_var - 1] = new_val
                
                # Update sat status for clauses containing best_var
                for c_idx in var_clauses[best_var]:
                    old_sat = sat[c_idx]
                    new_sat = is_satisfied(c_idx, assignment)
                    
                    if old_sat != new_sat:
                        sat[c_idx] = new_sat
                        if new_sat:
                            unsat_count -= 1
                            # Remove from var_unsat_count
                            for l in clauses[c_idx]:
                                var_unsat_count[abs(l)] -= 1
                        else:
                            unsat_count += 1
                            # Add to var_unsat_count
                            for l in clauses[c_idx]:
                                var_unsat_count[abs(l)] += 1
                            
                if unsat_count < best_unsat_count:
                    best_unsat_count = unsat_count
                    best_assignment = assignment[:]
                    
            # End of budget
            
        if best_assignment:
            return best_assignment
        return assignment

    return walksat()
