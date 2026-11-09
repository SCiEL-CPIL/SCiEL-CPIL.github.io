import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Precompute variable to clause mapping
    # var_to_clauses[v] is a list of (clause_index, sign)
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_to_clauses[v].append((i, sign))
            
    m = len(clauses)
    
    def get_impact(v, assignment, clause_true_count):
        # Impact is the number of clauses that are currently satisfied 
        # and would become unsatisfied if we flip v.
        # This happens if clause_true_count[c] == 1 and v is the true literal in c.
        impact = 0
        for c_idx, sign in var_to_clauses[v]:
            if clause_true_count[c_idx] == 1:
                # Check if v is the true literal
                if (sign > 0 and assignment[v]) or (sign < 0 and not assignment[v]):
                    impact += 1
        return impact

    def solve_with_restart(max_restarts=200, max_steps_per_restart=10000):
        rng = random.Random(0)
        phase = [False] * n
        
        for restart in range(max_restarts):
            # Initialize assignment using phase saving
            assignment = phase[:]
            
            clause_true_count = [0] * m
            # Initial calculation
            for i in range(m):
                count = 0
                for lit in clauses[i]:
                    v = abs(lit) - 1
                    if (lit > 0 and assignment[v]) or (lit < 0 and not assignment[v]):
                        count += 1
                clause_true_count[i] = count
            
            # Check if already satisfied
            all_satisfied = True
            for i in range(m):
                if clause_true_count[i] == 0:
                    all_satisfied = False
                    break
            
            if all_satisfied:
                return assignment
            
            for step in range(max_steps_per_restart):
                # Find an unsatisfied clause
                c_idx = -1
                for i in range(m):
                    if clause_true_count[i] == 0:
                        c_idx = i
                        break
                        
                if c_idx == -1:
                    return assignment
                    
                clause = clauses[c_idx]
                
                # Find variable in clause with minimum impact
                min_impact = float('inf')
                best_candidates = []
                
                for lit in clause:
                    v = abs(lit) - 1
                    imp = get_impact(v, assignment, clause_true_count)
                    if imp < min_impact:
                        min_impact = imp
                        best_candidates = [v]
                    elif imp == min_impact:
                        best_candidates.append(v)
                        
                v = rng.choice(best_candidates)
                
                # Flip v
                assignment[v] = not assignment[v]
                phase[v] = assignment[v]
                
                # Update clause_true_count for clauses involving v
                for c_i, sign in var_to_clauses[v]:
                    old_val = not assignment[v]
                    was_true = (sign > 0 and old_val) or (sign < 0 and not old_val)
                    is_true_now = (sign > 0 and assignment[v]) or (sign < 0 and not assignment[v])
                    
                    if was_true and not is_true_now:
                        clause_true_count[c_i] -= 1
                    elif not was_true and is_true_now:
                        clause_true_count[c_i] += 1

        return assignment # Fallback

    return solve_with_restart()
