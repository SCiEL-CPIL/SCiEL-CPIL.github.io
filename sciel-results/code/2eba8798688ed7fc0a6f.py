import time
import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    MAX_TIME = 5.0
    start_time = time.time()
    rng = random.Random(123)
    
    # Precompute
    var_clauses = [[] for _ in range(n + 1)]
    for i, c in enumerate(clauses):
        for l in c:
            var_clauses[abs(l)].append(i)
            
    def run_walksat():
        # Init
        assignment = [rng.random() < 0.5 for _ in range(n)]
        var_activity = [1.0] * (n + 1)
        clause_weight = [1.0] * len(clauses)
        var_decay = 0.95
        clause_decay = 0.99
        
        best_unsat = float('inf')
        best_assign = assignment[:]
        
        no_improve_count = 0
        restart_threshold = 500 # Steps without improvement to restart
        
        while time.time() - start_time < MAX_TIME:
            # Check sat
            unsat_indices = []
            total_unsat = 0
            for i, c in enumerate(clauses):
                sat = False
                for l in c:
                    if assignment[abs(l)-1] == (l > 0):
                        sat = True
                        break
                if not sat:
                    unsat_indices.append(i)
                    total_unsat += 1
                    clause_weight[i] = min(clause_weight[i] * 1.05, 10.0)
                else:
                    clause_weight[i] = max(clause_weight[i] * clause_decay, 1.0)
                    
            if total_unsat == 0:
                return assignment[:]
                
            if total_unsat < best_unsat:
                best_unsat = total_unsat
                best_assign = assignment[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            if no_improve_count > restart_threshold:
                # Restart
                assignment = [rng.random() < 0.5 for _ in range(n)]
                var_activity = [1.0] * (n + 1)
                clause_weight = [1.0] * len(clauses)
                no_improve_count = 0
                continue
                
            if not unsat_indices:
                break
                
            # Pick clause
            # Weighted by clause_weight
            weights = [clause_weight[i] for i in unsat_indices]
            total_w = sum(weights)
            r = rng.random() * total_w
            acc = 0
            chosen_i = unsat_indices[0]
            for idx, w in zip(unsat_indices, weights):
                acc += w
                if r <= acc:
                    chosen_i = idx
                    break
                    
            c = clauses[chosen_i]
            
            # Pick variable
            # Score: var_activity[v] * (number of unsat clauses v is in)
            # To avoid O(N) scan for each var, we can just use var_activity.
            # Or compute unsat count for vars in c.
            
            # Let's use: Score = var_activity[v]
            # And boost if it's in many unsat clauses.
            
            # Simple: Pick var with highest activity in the clause.
            max_act = -1
            chosen_v = -1
            for l in c:
                v = abs(l)
                # Boost activity if this var is in many unsat clauses?
                # We can approximate by just using activity.
                if var_activity[v] > max_act:
                    max_act = var_activity[v]
                    chosen_v = v
                    
            if chosen_v == -1:
                chosen_v = abs(rng.choice(c))
                
            # Flip
            assignment[chosen_v - 1] = not assignment[chosen_v - 1]
            
            # Update activity
            # Increase activity of vars in unsat clauses
            for i in unsat_indices:
                for l in clauses[i]:
                    v = abs(l)
                    var_activity[v] = min(var_activity[v] * 1.01, 100.0)
                    
            # Decay all activity
            for v in range(1, n + 1):
                var_activity[v] *= var_decay

        return best_assign

    return run_walksat()
