def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    deadline = time.time() + 290.0 # Safety margin
    
    # Convert to 0-based
    clauses_0 = [[l - 1 if l > 0 else -l - 1 for l in clause] for clause in clauses]
    
    # Build var -> list of (clause_idx, lit_in_clause)
    # This allows O(1) access to the literal sign for variable v in clause ci
    var_clause_lits = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses_0):
        for lit in clause:
            v = lit if lit >= 0 else -lit - 1
            var_clause_lits[v].append((ci, lit))
            
    def initial_state(assignment):
        sat_counts = [0] * m
        unsat_set = set()
        for ci, clause in enumerate(clauses_0):
            cnt = 0
            for lit in clause:
                if (lit >= 0 and assignment[lit]) or (lit < 0 and not assignment[-lit-1]):
                    cnt += 1
            sat_counts[ci] = cnt
            if cnt == 0:
                unsat_set.add(ci)
        return sat_counts, unsat_set

    best_assignment = None
    best_unsat = m + 1
    
    max_restarts = 1000
    max_steps = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        rng = random.Random(restart * 12345 + 6789)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        sat_counts, unsat_set = initial_state(assignment)
        unsat_count = len(unsat_set)
        
        if unsat_count == 0:
            return assignment
            
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
            
        p = 0.5
        # Weighted clause selection? For now, random from unsat_set.
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            if not unsat_set:
                return assignment
                
            # Pick random unsat clause
            clause_idx = rng.choice(list(unsat_set))
            clause = clauses_0[clause_idx]
            
            # Score flips
            best_score = -1000
            best_lit = clause[0]
            
            for lit in clause:
                v = lit if lit >= 0 else -lit - 1
                score = 0
                for (ci, l_ci) in var_clause_lits[v]:
                    if ci == clause_idx:
                        continue
                    # Check current state of l_ci
                    if l_ci >= 0:
                        is_sat = assignment[l_ci]
                    else:
                        is_sat = not assignment[-l_ci-1]
                    
                    if is_sat:
                        # Currently satisfied. Flipping makes it unsat.
                        if sat_counts[ci] == 1:
                            score -= 1
                    else:
                        # Currently unsat. Flipping makes it sat.
                        score += 1
                        
                if score > best_score:
                    best_score = score
                    best_lit = lit
            
            # Flip
            if rng.random() < p:
                lit_to_flip = best_lit
            else:
                lit_to_flip = clause[rng.randint(0, 2)]
                
            v = lit_to_flip if lit_to_flip >= 0 else -lit_to_flip - 1
            assignment[v] = not assignment[v]
            
            # Update
            for (ci, l_ci) in var_clause_lits[v]:
                if l_ci >= 0:
                    new_sat = assignment[l_ci]
                else:
                    new_sat = not assignment[-l_ci-1]
                    
                # We need to know if it was sat before.
                # Since we flipped v, the state of l_ci changed.
                # old_sat = not new_sat
                old_sat = not new_sat
                
                if new_sat and not old_sat:
                    sat_counts[ci] += 1
                    if ci in unsat_set:
                        unsat_set.discard(ci)
                elif old_sat and not new_sat:
                    sat_counts[ci] -= 1
                    if sat_counts[ci] == 0:
                        unsat_set.add(ci)
                        
            unsat_count = len(unsat_set)
            if unsat_count == 0:
                return assignment
                
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
                
            # Adapt p
            if step % 100 == 99:
                p = max(0.2, 0.5 - 0.1 * (step // 100))
                
    if best_assignment:
        return best_assignment
    return [False] * n
