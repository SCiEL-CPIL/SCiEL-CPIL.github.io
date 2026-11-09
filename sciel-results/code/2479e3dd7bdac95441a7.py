import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Fallback: n=9 modular coloring (valid)
    def mod_color(n, mod=3):
        return [0 if (a + b + c) % mod == 0 else 1 for a, b, c in combinations(range(n), 3)]

    best_n = 9
    best_colors = mod_color(9, 3)

    def is_valid(n, colors):
        """Verify a coloring is valid (no red K4, no blue K5)."""
        cm = {}
        for i, t in enumerate(combinations(range(n), 3)):
            cm[t] = colors[i]
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if cm[(a,b,c)] == 0 and cm[(a,b,d)] == 0 and cm[(a,c,d)] == 0 and cm[(b,c,d)] == 0:
                return False
        for quint in combinations(range(n), 5):
            all_blue = True
            for t in combinations(quint, 3):
                if cm[t] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False
        return True

    def run_walksat(target_n, deadline, initial_colors=None):
        """
        WalkSAT-style local search for Hypergraph Ramsey.
        Variables: Triples (0=Red, 1=Blue).
        Clauses:
          1. No Red K4: For any 4 vertices, not all 4 triples are Red (0).
             Clause: (~c1 OR ~c2 OR ~c3 OR ~c4) where ci is 1 if Red? 
             Wait, standard SAT: Variable x is True/False.
             Let's define x_i = 1 if Blue, 0 if Red.
             Red K4 violation: c1=0, c2=0, c3=0, c4=0.
             Constraint: NOT (c1=0 AND c2=0 AND c3=0 AND c4=0)
             This is equivalent to: (c1=1 OR c2=1 OR c3=1 OR c4=1).
             So the clause is (x1 OR x2 OR x3 OR x4).
             
          2. No Blue K5: For any 5 vertices, not all 10 triples are Blue (1).
             Blue K5 violation: all 10 x_i = 1.
             Constraint: NOT (x1=1 AND ... AND x10=1)
             This is equivalent to: (~x1 OR ~x2 OR ... OR ~x10).
             So the clause is (~x1 OR ... OR ~x10).
        
        We maintain the number of unsatisfied clauses for each variable.
        """
        triple_list = list(combinations(range(target_n), 3))
        num_triples = len(triple_list)
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}
        
        # Build clauses
        # Clause: (list of var indices, list of polarities)
        # Polarity: 1 means variable must be 1 (True/Blue), -1 means variable must be 0 (False/Red)
        # Actually, easier: Clause is satisfied if ANY literal is True.
        # Literal: (var_idx, is_positive)
        # If is_positive: satisfied if color[var_idx] == 1
        # If is_negative: satisfied if color[var_idx] == 0
        
        clauses = []
        # var_to_clauses: for each var, list of clause indices it belongs to
        
        # 1. Red K4 constraints
        # For each quad (a,b,c,d):
        # Triples: (a,b,c), (a,b,d), (a,c,d), (b,c,d)
        # Constraint: At least one is Blue (1).
        # Clause: (x_abc OR x_abd OR x_acd OR x_bcd)
        # Literals: (idx_abc, True), (idx_abd, True), (idx_acd, True), (idx_bcd, True)
        
        quad_clauses = []
        for quad in combinations(range(target_n), 4):
            a, b, c, d = quad
            t1 = triple_to_idx[(a,b,c)]
            t2 = triple_to_idx[(a,b,d)]
            t3 = triple_to_idx[(a,c,d)]
            t4 = triple_to_idx[(b,c,d)]
            clause = [(t1, 1), (t2, 1), (t3, 1), (t4, 1)]
            quad_clauses.append(clause)
            
        # 2. Blue K5 constraints
        # For each quint (a,b,c,d,e):
        # All 10 triples must NOT be all Blue.
        # Constraint: At least one is Red (0).
        # Clause: (~x_abc OR ~x_abd OR ...)
        # Literals: (idx_abc, -1), ...
        
        quint_clauses = []
        for quint in combinations(range(target_n), 5):
            lits = []
            for t in combinations(quint, 3):
                idx = triple_to_idx[t]
                lits.append((idx, -1))
            quint_clauses.append(lits)
            
        all_clauses = quad_clauses + quint_clauses
        num_clauses = len(all_clauses)
        
        # Map variable to clauses
        var_to_clauses = [[] for _ in range(num_triples)]
        for c_idx, clause in enumerate(all_clauses):
            for var_idx, pol in clause:
                var_to_clauses[var_idx].append(c_idx)
                
        # Initialize colors
        if initial_colors is None:
            color = [random.choice([0, 1]) for _ in range(num_triples)]
        else:
            color = list(initial_colors)
            
        # Initial clause satisfaction status
        # clause_sat[c_idx] = True if satisfied, False if not
        clause_sat = [False] * num_clauses
        # clause_unsat_count[c_idx] = number of unsatisfied literals in this clause
        # A clause is satisfied if unsat_count < len(clause)
        # Wait, if ANY literal is satisfied, the clause is satisfied.
        # So we track num_satisfied_literals. If > 0, it's satisfied.
        # Or track num_unsat_literals. If 0, it's satisfied.
        
        clause_num_unsat = [0] * num_clauses
        total_unsat_clauses = 0
        
        # var_unsat_clause_count[var_idx] = number of unsatisfied clauses this var is in
        # This is useful for heuristic: pick a var that is in many unsatisfied clauses.
        var_unsat_clause_count = [0] * num_triples
        
        for c_idx, clause in enumerate(all_clauses):
            unsat = 0
            for var_idx, pol in clause:
                # Check if literal is satisfied
                if pol == 1:
                    if color[var_idx] == 0: # Needs 1, has 0 -> Unsatisfied
                        unsat += 1
                else:
                    if color[var_idx] == 1: # Needs 0, has 1 -> Unsatisfied
                        unsat += 1
            clause_num_unsat[c_idx] = unsat
            if unsat == len(clause):
                # All literals unsatisfied -> Clause unsatisfied
                clause_sat[c_idx] = False
                total_unsat_clauses += 1
                for var_idx, pol in clause:
                    var_unsat_clause_count[var_idx] += 1
            else:
                clause_sat[c_idx] = True
                
        best_total_unsat = total_unsat_clauses
        best_color = list(color)
        
        # WalkSAT Parameters
        # Probability of picking a var from an unsatisfied clause vs random var
        p_walk = 0.5 
        # Number of steps
        max_steps = 1000000
        step = 0
        
        while time.time() < deadline - 1:
            step += 1
            
            if total_unsat_clauses == 0:
                return color, 0
                
            if total_unsat_clauses < best_total_unsat:
                best_total_unsat = total_unsat_clauses
                best_color = list(color)
                if best_total_unsat == 0:
                    return best_color, 0
                    
            # Heuristic: Pick a clause
            # Pick a random unsatisfied clause
            unsat_clauses = [i for i in range(num_clauses) if not clause_sat[i]]
            if not unsat_clauses:
                break
            c_idx = random.choice(unsat_clauses)
            clause = all_clauses[c_idx]
            
            # Pick a variable to flip
            if random.random() < p_walk:
                # Pick a random var from the clause
                var_idx, pol = random.choice(clause)
            else:
                # Pick a random var globally
                var_idx = random.randint(0, num_triples - 1)
                
            # Flip var_idx
            old_val = color[var_idx]
            new_val = 1 - old_val
            color[var_idx] = new_val
            
            # Update clauses containing var_idx
            # For each clause C in var_to_clauses[var_idx]:
            #   Check if C's status changes
            for c_idx2 in var_to_clauses[var_idx]:
                clause2 = all_clauses[c_idx2]
                was_sat = clause_sat[c_idx2]
                
                # Recalculate unsat count for this clause? 
                # No, we can update incrementally.
                # The only literal that changed is the one for var_idx.
                # Find its polarity in the clause.
                # Since var_idx is in the clause, we can just check the new value.
                
                # We need to know if the literal for var_idx was satisfied before and after.
                # We don't store polarity per var in the clause easily for O(1) lookup unless we map it.
                # Let's just recount this small clause. Size <= 10.
                
                unsat_count = 0
                for v, p in clause2:
                    if p == 1:
                        if color[v] == 0:
                            unsat_count += 1
                    else:
                        if color[v] == 1:
                            unsat_count += 1
                
                now_sat = (unsat_count < len(clause2))
                
                if was_sat == now_sat:
                    continue
                
                # Status changed
                if now_sat:
                    # Became satisfied
                    clause_sat[c_idx2] = True
                    total_unsat_clauses -= 1
                    # Decrement var_unsat_clause_count for all vars in this clause
                    for v, p in clause2:
                        var_unsat_clause_count[v] -= 1
                else:
                    # Became unsatisfied
                    clause_sat[c_idx2] = False
                    total_unsat_clauses += 1
                    # Increment var_unsat_clause_count for all vars in this clause
                    for v, p in clause2:
                        var_unsat_clause_count[v] += 1

        return best_color, best_total_unsat

    # Strategy: Try n=34 with multiple restarts
    targets = [34, 33, 32]
    time_allocations = [30, 20, 15]
    
    best_found = (9, mod_color(9, 3))
    
    for i, n in enumerate(targets):
        if time.time() > deadline - 15:
            break
            
        alloc = time_allocations[i] if i < len(time_allocations) else 10
        current_deadline = time.time() + alloc
        
        max_restarts = 20
        found_valid = False
        
        for r in range(max_restarts):
            if time.time() > current_deadline - 2:
                break
            
            # Different initialization strategies
            if r % 5 == 0:
                # Modular by 3
                init = [0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in combinations(range(n), 3)]
            elif r % 5 == 1:
                # Modular by 5
                init = [0 if (a+b+c) % 5 == 0 else 1 for (a,b,c) in combinations(range(n), 3)]
            elif r % 5 == 2:
                # Random with 40% red
                init = [0 if random.random() < 0.4 else 1 for _ in combinations(range(n), 3)]
            elif r % 5 == 3:
                # Random with 50% red
                init = [0 if random.random() < 0.5 else 1 for _ in combinations(range(n), 3)]
            else:
                # Modular by 7
                init = [0 if (a+b+c) % 7 == 0 else 1 for (a,b,c) in combinations(range(n), 3)]
            
            colors, cost = run_walksat(n, current_deadline, init)
            
            if cost == 0:
                if is_valid(n, colors):
                    best_found = (n, colors)
                    found_valid = True
                    break
        
        if found_valid:
            break

    return {'n': best_found[0], 'colors': best_found[1]}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
    if res['n'] > 9:
        print("Validating...")
        print(is_valid(res['n'], res['colors']))
