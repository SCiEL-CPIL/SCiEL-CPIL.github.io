"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n = instance["n"]
    clauses_raw = instance["clauses"]
    
    # Convert to 0-based: store literal as signed int (positive = var+1, negative = -(var+1))
    clause_literals = []  # list of lists of (var_idx, is_pos)
    for c in clauses_raw:
        cv = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            cv.append((var_idx, is_pos))
        clause_literals.append(cv)
        
    m = len(clause_literals)
    
    # Build variable to clause index with polarity
    var_to_clauses = [[] for _ in range(n)]
    for i, cv in enumerate(clause_literals):
        for var_idx, is_pos in cv:
            var_to_clauses[var_idx].append((i, is_pos))
    
    rng = random.Random(42)
    deadline = time.monotonic() + 290.0
    
    def init_clause_counts(assignment):
        counts = [0] * m
        for i, cv in enumerate(clause_literals):
            cnt = 0
            for var_idx, is_pos in cv:
                if assignment[var_idx] == is_pos:
                    cnt += 1
            counts[i] = cnt
        return counts
    
    def get_unsat_from_counts(counts):
        return [i for i in range(m) if counts[i] == 0]
    
    def belief_propagation_init(num_iterations=100):
        """
        Run Belief Propagation to get variable biases for planted 3-SAT.
        Returns a list of booleans representing the most likely assignment.
        """
        # For each variable v, track the log-odds of v=True vs v=False
        # We'll use a simplified BP that works with the factor graph structure
        
        # Precompute: for each clause, which variables it contains and their polarity
        # clause_vars[c] = [(var_idx, is_pos), ...]
        
        # Messages from clauses to variables: log-odds favoring the value that satisfies
        # Initialize all to 0
        var_bias = [0.0] * n  # log-odds that var is True
        
        for it in range(num_iterations):
            new_var_bias = [0.0] * n
            
            # Update: for each clause, compute its contribution to each variable's bias
            for c_idx, cv in enumerate(clause_literals):
                # For a 3-SAT clause with literals (v1,p1), (v2,p2), (v3,p3):
                # The clause is satisfied if any literal is true.
                # For variable v_j with polarity p_j:
                #   P(clause sat | v_j = p_j) = 1.0 (literal satisfied)
                #   P(clause sat | v_j = not p_j) = 1 - prod_{k != j} P(v_k != p_k)
                #   where P(v_k != p_k) depends on the current belief about v_k
                #
                # The message from clause to var v_j should be:
                #   log(P(clause sat | v_j = p_j) / P(clause sat | v_j = not p_j))
                # = log(1.0 / (1 - prod_{k != j} P(v_k != p_k)))
                #
                # But this is the log-odds in favor of v_j = p_j over v_j = not p_j.
                # To convert to log-odds of v_j = True:
                #   if p_j is True: add the message
                #   if p_j is False: subtract the message (since p_j=False means we favor v_j=False)
                
                for j, (var_idx, is_pos) in enumerate(cv):
                    # Compute prod_{k != j} P(v_k != p_k)
                    prod_complement = 1.0
                    for k, (v2, p2) in enumerate(cv):
                        if k == j:
                            continue
                        # Get current belief about v2
                        # var_bias[v2] is log-odds that v2 is True
                        # P(v2 = True) = sigmoid(var_bias[v2])
                        # P(v2 = False) = 1 - sigmoid(var_bias[v2])
                        # We need P(v2 != p2):
                        #   if p2 is True: P(v2 != True) = P(v2 = False) = 1 - sigmoid(var_bias[v2])
                        #   if p2 is False: P(v2 != False) = P(v2 = True) = sigmoid(var_bias[v2])
                        bias = var_bias[v2]
                        if p2:  # p2 is True, need P(v2 = False)
                            if bias > 0:
                                p_false = 1.0 / (1.0 + math.exp(bias))
                            else:
                                p_false = math.exp(bias) / (1.0 + math.exp(bias))
                        else:  # p2 is False, need P(v2 = True)
                            if bias > 0:
                                p_true = 1.0 / (1.0 + math.exp(-bias))
                            else:
                                p_true = 1.0 / (1.0 + math.exp(bias))
                            p_false = p_true  # We need P(v2 != False) = P(v2 = True)
                        
                        if p_false > 1.0:
                            p_false = 1.0
                        elif p_false < 0.0:
                            p_false = 0.0
                        prod_complement *= p_false
                    
                    # P(clause sat | v_j = not p_j) = 1 - prod_complement
                    p_sat_given_false = 1.0 - prod_complement
                    if p_sat_given_false < 1e-15:
                        p_sat_given_false = 1e-15
                    
                    # Message: log(1.0 / p_sat_given_false) = -log(p_sat_given_false)
                    msg = -math.log(p_sat_given_false)
                    
                    # Convert to log-odds of var_idx = True
                    if is_pos:
                        # We favor var_idx = True
                        new_var_bias[var_idx] += msg
                    else:
                        # We favor var_idx = False, so subtract
                        new_var_bias[var_idx] -= msg
            
            var_bias = new_var_bias
        
        # Convert biases to boolean assignment
        assignment = [bias > 0 for bias in var_bias]
        return assignment
    
    def optimized_walksat(assignment, max_steps, sa_temp=1.0):
        counts = init_clause_counts(assignment)
        unsat_set = set(get_unsat_from_counts(counts))
        
        for step in range(max_steps):
            if time.monotonic() > deadline - 2.0:
                return False, assignment, unsat_set, counts
                
            if not unsat_set:
                return True, assignment, unsat_set, counts
            
            # Select unsatisfied clause
            c_idx = rng.choice(list(unsat_set))
            cv = clause_literals[c_idx]
            
            # Compute break count for each variable in the clause
            best_var = None
            best_score = float('inf')
            
            for j, (var_idx, is_pos) in enumerate(cv):
                # Break count: number of satisfied clauses containing var_idx
                # that would become unsatisfied if we flip var_idx
                break_count = 0
                for c_aff, is_pos_aff in var_to_clauses[var_idx]:
                    if counts[c_aff] == 0:
                        continue  # already unsat, flipping can't make it worse
                    cv_aff = clause_literals[c_aff]
                    # Check if any OTHER literal in c_aff is satisfied
                    # For 3-SAT, check the other 2 literals
                    still_sat = False
                    for k2, (v2, p2) in enumerate(cv_aff):
                        if v2 == var_idx:
                            continue
                        if assignment[v2] == p2:
                            still_sat = True
                            break
                    if not still_sat:
                        break_count += 1
                
                # WalkSAT: minimize break count, random tiebreak
                score = break_count + rng.random() * 0.5
                
                if score < best_score:
                    best_score = score
                    best_var = var_idx
            
            if best_var is None:
                best_var = cv[0][0]
            
            var_idx = best_var
            
            old_unsat = len(unsat_set)
            
            # Flip variable
            assignment[var_idx] = not assignment[var_idx]
            
            # Update counts incrementally
            for c_aff, is_pos_aff in var_to_clauses[var_idx]:
                if assignment[var_idx] == is_pos_aff:
                    if counts[c_aff] == 0:
                        unsat_set.discard(c_aff)
                    counts[c_aff] += 1
                else:
                    counts[c_aff] -= 1
                    if counts[c_aff] == 0:
                        unsat_set.add(c_aff)
            
            new_unsat = len(unsat_set)
            
            # SA: accept worse moves with decreasing probability
            if new_unsat > old_unsat:
                delta = new_unsat - old_unsat
                T = sa_temp * max(0.01, 1.0 - step / max_steps)
                if rng.random() > math.exp(-delta / max(T, 0.001)):
                    # Reject: flip back
                    assignment[var_idx] = not assignment[var_idx]
                    for c_aff, is_pos_aff in var_to_clauses[var_idx]:
                        if assignment[var_idx] == is_pos_aff:
                            if counts[c_aff] == 0:
                                unsat_set.discard(c_aff)
                            counts[c_aff] += 1
                        else:
                            counts[c_aff] -= 1
                            if counts[c_aff] == 0:
                                unsat_set.add(c_aff)
        
        return False, assignment, unsat_set, counts
    
    best_assignment = [False] * n
    best_unsat_count = m + 1
    
    # Multi-restart
    restart_budget = 2000
    base_steps = 5000
    
    for restart in range(restart_budget):
        if time.monotonic() > deadline - 5.0:
            break
        
        # Initialize: use BP for the first few restarts, then random
        if restart < 5:
            # BP-informed initialization
            assignment = belief_propagation_init(num_iterations=50 + restart * 10)
        else:
            # Random initialization
            assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Vary temperature for diversity
        sa_temp = 0.5 + rng.random() * 1.5
        
        remaining = deadline - time.monotonic()
        if remaining < 10.0:
            break
        elif remaining < 30.0:
            steps = 2000
        elif remaining < 60.0:
            steps = 4000
        else:
            steps = base_steps
        
        solved, assignment, unsat_set, counts = optimized_walksat(assignment, steps, sa_temp)
        
        if solved:
            return assignment
        
        current_unsat = len(unsat_set)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
    
    return best_assignment
# EVOLVE-BLOCK-END
