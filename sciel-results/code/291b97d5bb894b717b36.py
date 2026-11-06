import numpy as np
import random
import time
import math

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    C = len(clauses)
    
    # Build data structures
    # clause_vars[i] = list of (var_idx, sign) for clause i
    clause_vars = []
    var_to_clauses = [[] for _ in range(n)]
    
    for i, clause in enumerate(clauses):
        cv = []
        for j, lit in enumerate(clause):
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            cv.append((v, sign))
            var_to_clauses[v].append((i, j))
        clause_vars.append(cv)
    
    # Precompute for each variable and each clause it appears in,
    # the other two literals
    var_other_lits = [{} for _ in range(n)]
    for v in range(n):
        for (ci, pos) in var_to_clauses[v]:
            others = []
            for p in range(3):
                if p != pos:
                    others.append(clause_vars[ci][p])
            var_other_lits[v][ci] = (others[0][0], others[0][1], others[1][0], others[1][1])
    
    start_time = time.time()
    deadline = start_time + 295.0
    
    rng = random.Random(12345)
    
    # Preprocessing: Unit propagation and pure literal elimination
    def preprocess(assignment, unsat_set):
        """Apply unit propagation and pure literal elimination.
        Returns (assignment, unsat_set, changed) where changed indicates if any variable was fixed."""
        changed = False
        
        # Pure literal elimination: if a variable appears with only one sign, set it to satisfy that sign
        for v in range(n):
            if assignment[v] is None:
                pos_count = 0
                neg_count = 0
                for (ci, pos) in var_to_clauses[v]:
                    if ci in unsat_set:  # Only consider unsatisfied clauses
                        lit = clauses[ci][pos]
                        if lit > 0:
                            pos_count += 1
                        else:
                            neg_count += 1
                if pos_count > 0 and neg_count == 0:
                    assignment[v] = True
                    changed = True
                elif neg_count > 0 and pos_count == 0:
                    assignment[v] = False
                    changed = True
        
        # Unit propagation: if a clause has only one unassigned literal, assign it
        for ci in list(unsat_set):
            unassigned = []
            sat = False
            for (v, s) in clause_vars[ci]:
                if assignment[v] is not None:
                    val = assignment[v]
                    if (s == 1 and val) or (s == -1 and not val):
                        sat = True
                        break
                else:
                    unassigned.append((v, s))
            if sat:
                unsat_set.discard(ci)
            elif len(unassigned) == 1:
                v, s = unassigned[0]
                assignment[v] = (s == 1)
                changed = True
        
        return assignment, unsat_set, changed
    
    # Fast evaluation for a full boolean assignment
    def fast_evaluate(assignment):
        num_unsat = 0
        unsat_clauses = []
        unsat_set = set()
        for c in range(C):
            cv = clause_vars[c]
            sat = False
            for (v, s) in cv:
                val = assignment[v]
                if (s == 1 and val) or (s == -1 and not val):
                    sat = True
                    break
            if not sat:
                num_unsat += 1
                unsat_clauses.append(c)
                unsat_set.add(c)
        return num_unsat, unsat_clauses, unsat_set
    
    # Compute delta for flipping variable v given current assignment
    def compute_delta(v, assignment, unsat_set):
        delta = 0
        for (ci, pos) in var_to_clauses[v]:
            o1v, o1s, o2v, o2s = var_other_lits[v][ci]
            
            o1_sat = (o1s == 1 and assignment[o1v]) or (o1s == -1 and not assignment[o1v])
            o2_sat = (o2s == 1 and assignment[o2v]) or (o2s == -1 and not assignment[o2v])
            
            if o1_sat and o2_sat:
                if ci in unsat_set:
                    delta += 1
            else:
                if not o1_sat and not o2_sat:
                    if ci not in unsat_set:
                        delta -= 1
        return delta
    
    # WalkSAT with adaptive noise
    def walk_sat(assignment, num_unsat, unsat_clauses, unsat_set, rng, max_steps, deadline):
        for step in range(max_steps):
            if time.time() >= deadline:
                break
            if num_unsat == 0:
                return True
            
            # Pick a random unsatisfied clause
            c = unsat_clauses[rng.randrange(num_unsat)]
            cv = clause_vars[c]
            
            # Adaptive noise: decrease over time
            progress = step / max(1, max_steps)
            noise_scale = 0.4 * (1.0 - 0.5 * progress)
            
            # For each variable in this clause, compute delta
            best_v = -1
            best_score = -1e9
            for (v, s) in cv:
                delta = compute_delta(v, assignment, unsat_set)
                noise = rng.random() * noise_scale
                score = delta + noise
                if score > best_score:
                    best_score = score
                    best_v = v
            
            if best_v == -1:
                break
            
            # Flip best_v
            assignment[best_v] = not assignment[best_v]
            
            # Update num_unsat and unsat tracking incrementally
            affected = set()
            for (ci, pos) in var_to_clauses[best_v]:
                affected.add(ci)
            
            new_unsat_clauses = []
            new_unsat_set = set()
            
            for ci in range(C):
                cv2 = clause_vars[ci]
                sat = False
                for (v2, s2) in cv2:
                    val = assignment[v2]
                    if (s2 == 1 and val) or (s2 == -1 and not val):
                        sat = True
                        break
                if not sat:
                    new_unsat_clauses.append(ci)
                    new_unsat_set.add(ci)
            
            unsat_clauses = new_unsat_clauses
            unsat_set = new_unsat_set
            num_unsat = len(unsat_clauses)
            
            if num_unsat == 0:
                return True
        
        return False
    
    # Clause-weighted local search (Minisat-style)
    def weighted_walk_sat(assignment, num_unsat, unsat_clauses, unsat_set, rng, max_steps, deadline):
        weights = [1.0] * C
        boost = 1.0
        decay = 0.999
        
        for step in range(max_steps):
            if time.time() >= deadline:
                break
            if num_unsat == 0:
                return True
            
            # Weight unsatisfied clauses
            for ci in unsat_clauses:
                weights[ci] += boost
            
            # Pick a weighted random unsatisfied clause
            total_w = sum(weights[ci] for ci in unsat_clauses)
            if total_w <= 0:
                c = unsat_clauses[rng.randrange(num_unsat)]
            else:
                r = rng.random() * total_w
                cum = 0.0
                c = unsat_clauses[0]
                for ci in unsat_clauses:
                    cum += weights[ci]
                    if r <= cum:
                        c = ci
                        break
            
            cv = clause_vars[c]
            
            # For each variable in this clause, compute weighted delta
            best_v = -1
            best_score = -1e9
            for (v, s) in cv:
                delta = 0
                for (ci2, pos2) in var_to_clauses[v]:
                    o1v, o1s, o2v, o2s = var_other_lits[v][ci2]
                    o1_sat = (o1s == 1 and assignment[o1v]) or (o1s == -1 and not assignment[o1v])
                    o2_sat = (o2s == 1 and assignment[o2v]) or (o2s == -1 and not assignment[o2v])
                    
                    if o1_sat and o2_sat:
                        if ci2 in unsat_set:
                            delta += weights[ci2]
                    else:
                        if not o1_sat and not o2_sat:
                            if ci2 not in unsat_set:
                                delta -= weights[ci2]
                
                noise = rng.random() * 0.3
                score = delta + noise
                if score > best_score:
                    best_score = score
                    best_v = v
            
            if best_v == -1:
                break
            
            assignment[best_v] = not assignment[best_v]
            
            # Update
            new_unsat_clauses = []
            new_unsat_set = set()
            for ci in range(C):
                cv2 = clause_vars[ci]
                sat = False
                for (v2, s2) in cv2:
                    val = assignment[v2]
                    if (s2 == 1 and val) or (s2 == -1 and not val):
                        sat = True
                        break
                if not sat:
                    new_unsat_clauses.append(ci)
                    new_unsat_set.add(ci)
            
            unsat_clauses = new_unsat_clauses
            unsat_set = new_unsat_set
            num_unsat = len(unsat_clauses)
            
            # Decay weights
            for ci in range(C):
                weights[ci] *= decay
            
            if num_unsat == 0:
                return True
        
        return False
    
    best_num_unsat = C + 1
    best_assignment = None
    
    # Phase 1: Try preprocessing-based approach
    # Phase 2: Multiple WalkSAT restarts
    # Phase 3: Weighted local search
    
    restart = 0
    while time.time() < deadline:
        if time.time() >= deadline:
            break
        
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Evaluate
        num_unsat, unsat_clauses, unsat_set = fast_evaluate(assignment)
        
        if num_unsat == 0:
            return assignment
        
        if num_unsat < best_num_unsat:
            best_num_unsat = num_unsat
            best_assignment = assignment[:]
        
        # Adaptive step limit
        base_steps = 8000 + restart * 2000
        max_steps = min(base_steps, 200000)
        
        # Alternate between plain WalkSAT and weighted WalkSAT
        if restart % 3 == 0:
            solved = walk_sat(assignment, num_unsat, unsat_clauses, unsat_set, rng, max_steps, deadline)
        elif restart % 3 == 1:
            solved = weighted_walk_sat(assignment, num_unsat, unsat_clauses, unsat_set, rng, max_steps, deadline)
        else:
            # Try with different noise
            solved = walk_sat(assignment, num_unsat, unsat_clauses, unsat_set, rng, max_steps, deadline)
        
        if solved:
            return assignment
        
        restart += 1
        
        # Every 10 restarts, try a smarter initialization
        if restart % 10 == 0:
            # Greedy initialization: start all False, flip variables to satisfy more clauses
            assignment = [False] * n
            for v in range(n):
                # Try both values, pick the one that satisfies more clauses
                best_val = False
                best_sat = -1
                for val in [False, True]:
                    assignment[v] = val
                    sat_count = 0
                    for (ci, pos) in var_to_clauses[v]:
                        cv = clause_vars[ci]
                        clause_sat = False
                        for (v2, s2) in cv:
                            val2 = assignment[v2]
                            if (s2 == 1 and val2) or (s2 == -1 and not val2):
                                clause_sat = True
                                break
                        if clause_sat:
                            sat_count += 1
                    if sat_count > best_sat:
                        best_sat = sat_count
                        best_val = val
                assignment[v] = best_val
            
            num_unsat, unsat_clauses, unsat_set = fast_evaluate(assignment)
            if num_unsat == 0:
                return assignment
            if num_unsat < best_num_unsat:
                best_num_unsat = num_unsat
                best_assignment = assignment[:]
    
    # Return best found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
