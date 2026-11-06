import random
import time

def solve(instance):
    """
    Solves a planted 3-SAT instance using an optimized WalkSAT algorithm.
    Combines the robust restart strategy of the seed program with the
    pre-computed neighbor structure and efficient delta evaluation of the inspirations.
    """
    n = instance["n"]
    clauses = instance["clauses"]
    C = len(clauses)
    
    # Safety margin: finish before 300s. We use 290s to be safe.
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Preprocessing: Build efficient data structures
    # clause_vars[i] = list of (var_idx, sign) for clause i
    # var_to_clauses[v] = list of (clause_idx, pos_in_clause) for variable v
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
    # the indices of the other two literals in that clause.
    # This allows us to check if flipping a variable satisfies/unsatisfies
    # a clause by only looking at the other 2 literals.
    # var_other_lits[v] = list of (clause_idx, other_pos1, other_pos2)
    var_other_lits = [[] for _ in range(n)]
    
    for v in range(n):
        for (ci, pos) in var_to_clauses[v]:
            # Get the other two positions
            others = []
            for p in range(3):
                if p != pos:
                    others.append(p)
            var_other_lits[v].append((ci, others[0], others[1]))

    def evaluate(assignment):
        """
        Evaluates the assignment.
        Returns:
        - num_unsat: number of unsatisfied clauses
        - clause_sat: list of bools, True if clause is satisfied
        """
        num_unsat = 0
        clause_sat = [False] * C
        for c in range(C):
            cv = clause_vars[c]
            sat = False
            for (v, s) in cv:
                val = assignment[v]
                if (s == 1 and val) or (s == -1 and not val):
                    sat = True
                    break
            clause_sat[c] = sat
            if not sat:
                num_unsat += 1
        return num_unsat, clause_sat

    def get_best_flip(unsat_clause_idx, clause_sat, assignment, rng):
        """
        Determines the best variable to flip given an unsatisfied clause.
        Uses a greedy heuristic with random tie-breaking.
        """
        clause = clause_vars[unsat_clause_idx]
        best_v = -1
        best_score = -float('inf')
        
        # Candidates are the variables in the unsatisfied clause
        for (v, s) in clause:
            # Calculate the delta (change in number of unsatisfied clauses)
            # if we flip variable v.
            # Delta = (# of clauses that become satisfied) - (# of clauses that become unsatisfied)
            
            # We iterate over all clauses containing v
            delta = 0
            for (ci, pos1, pos2) in var_other_lits[v]:
                # Check the state of the other two literals in clause ci
                # If clause ci is currently UNSAT:
                #   It becomes SAT if the other two literals are SAT.
                # If clause ci is currently SAT:
                #   It becomes UNSAT if the other two literals are UNSAT.
                
                v1, s1 = clause_vars[ci][pos1]
                v2, s2 = clause_vars[ci][pos2]
                
                val1 = assignment[v1]
                val2 = assignment[v2]
                
                # Check if literal 1 is satisfied
                lit1_sat = (s1 == 1 and val1) or (s1 == -1 and not val1)
                # Check if literal 2 is satisfied
                lit2_sat = (s2 == 1 and val2) or (s2 == -1 and not val2)
                
                if not clause_sat[ci]:
                    # Clause is unsat. Flipping v might satisfy it.
                    # It is satisfied if both other literals are satisfied.
                    if lit1_sat and lit2_sat:
                        delta += 1
                else:
                    # Clause is sat. Flipping v might unsatisfy it.
                    # It becomes unsat if both other literals are unsatisfied.
                    if not lit1_sat and not lit2_sat:
                        delta -= 1
            
            # Add random noise to break ties and add stochasticity
            # This helps escape local minima
            noise = rng.random()
            score = delta + noise
            
            if score > best_score:
                best_score = score
                best_v = v
                
        return best_v

    # Parameters
    # WalkSAT typically uses a weight parameter. Here we use pure greedy with noise.
    # We can also mix in random moves if needed, but the noise in scoring usually suffices.
    max_steps_per_restart = 1000 * n  # 5000 steps for n=50
    
    rng = random.Random()
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    while time.time() < deadline:
        # Initialize random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        num_unsat, clause_sat = evaluate(assignment)
        
        if num_unsat == 0:
            return assignment
        
        if num_unsat < best_unsat_count:
            best_unsat_count = num_unsat
            best_assignment = assignment[:]
        
        steps = 0
        while steps < max_steps_per_restart and time.time() < deadline:
            # Find an unsatisfied clause
            # We can pick a random one from the unsat list, or just scan.
            # Scanning is O(C). C is small.
            # To be slightly more efficient, we could maintain a list of unsat clauses,
            # but updating it is complex. Scanning is fine for n=50.
            
            unsat_clause_idx = -1
            for c in range(C):
                if not clause_sat[c]:
                    unsat_clause_idx = c
                    break
            
            if unsat_clause_idx == -1:
                return assignment
            
            # Get best flip
            v = get_best_flip(unsat_clause_idx, clause_sat, assignment, rng)
            
            if v == -1:
                # Should not happen, but break just in case
                break
            
            # Flip variable
            assignment[v] = not assignment[v]
            
            # Re-evaluate
            # Instead of full re-evaluation, we can update incrementally,
            # but full re-evaluation is O(C*3) which is very fast.
            # Let's stick to full re-evaluation for simplicity and correctness.
            num_unsat, clause_sat = evaluate(assignment)
            
            if num_unsat == 0:
                return assignment
            
            if num_unsat < best_unsat_count:
                best_unsat_count = num_unsat
                best_assignment = assignment[:]
                
            steps += 1
    
    # If we run out of time, return the best assignment found
    if best_assignment is not None:
        return best_assignment
    else:
        return [False] * n
