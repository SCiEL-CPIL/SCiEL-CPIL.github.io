import random
import time
from collections import defaultdict

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    num_clauses = len(clauses)
    deadline = time.time() + 295.0
    
    fast_clauses = []
    for clause in clauses:
        fc = []
        for lit in clause:
            var_idx = abs(lit) - 1
            required = (lit > 0)
            fc.append((var_idx, required))
        fast_clauses.append(fc)
        
    var_to_clauses = defaultdict(list)
    for i, fc in enumerate(fast_clauses):
        for var_idx, _ in fc:
            var_to_clauses[var_idx].append(i)
            
    def is_satisfied(clause_idx, assignment):
        fc = fast_clauses[clause_idx]
        for var_idx, required in fc:
            if assignment[var_idx] == required:
                return True
        return False
        
    phases = [random.random() < 0.5 for _ in range(n)]
    clause_weights = [1.0] * num_clauses
    
    best_assignment = None
    best_unsat = float('inf')
    
    num_restarts = 500
    max_steps = 3000
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        assignment = []
        for i in range(n):
            if random.random() < 0.8:
                assignment.append(phases[i])
            else:
                assignment.append(random.random() < 0.5)
                
        unsat_set = set()
        for i in range(num_clauses):
            if not is_satisfied(i, assignment):
                unsat_set.add(i)
                
        var_unsat_count = [0] * n
        for c_idx in unsat_set:
            for var_idx, _ in fast_clauses[c_idx]:
                var_unsat_count[var_idx] += 1
                
        cur_unsat = len(unsat_set)
        if cur_unsat == 0:
            return assignment
            
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
            for i in range(n):
                phases[i] = assignment[i]
                
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if not unsat_set:
                return assignment
                
            if len(unsat_set) < best_unsat:
                best_unsat = len(unsat_set)
                best_assignment = assignment[:]
                for i in range(n):
                    phases[i] = assignment[i]
                    
            walk_prob = 0.2 + 0.3 * (step / max_steps)
            
            # Weighted clause selection
            # Approximate by picking a random unsat clause and checking if it's "heavy"
            # For speed, just pick random unsat clause.
            c_idx = random.choice(list(unsat_set))
            clause = fast_clauses[c_idx]
            
            if random.random() < walk_prob:
                var_idx, required = random.choice(clause)
            else:
                scores = [var_unsat_count[v] for v, _ in clause]
                total = sum(scores)
                if total == 0:
                    var_idx, required = random.choice(clause)
                else:
                    r = random.random() * total
                    cum = 0
                    var_idx, required = clause[0]
                    for i, s in enumerate(scores):
                        cum += s
                        if r <= cum:
                            var_idx, required = clause[i]
                            break
                            
            old_val = assignment[var_idx]
            assignment[var_idx] = not assignment[var_idx]
            
            for ci in var_to_clauses[var_idx]:
                was_unsat = ci in unsat_set
                is_sat = is_satisfied(ci, assignment)
                if was_unsat and is_sat:
                    unsat_set.discard(ci)
                    for v, _ in fast_clauses[ci]:
                        var_unsat_count[v] -= 1
                elif not was_unsat and not is_sat:
                    unsat_set.add(ci)
                    for v, _ in fast_clauses[ci]:
                        var_unsat_count[v] += 1
                            
            if not unsat_set:
                return assignment
                
        # Decay weights
        for i in range(num_clauses):
            clause_weights[i] *= 0.9
            if clause_weights[i] < 0.1:
                clause_weights[i] = 0.1
                
    if best_assignment:
        return best_assignment
    return [random.random() < 0.5 for _ in range(n)]
