"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
from collections import defaultdict


def solve(instance):
    """
    Solve a planted 3-SAT instance with n variables and clauses.
    Returns a list of booleans (length n) that satisfies all clauses if possible.
    
    Strategy: Use Belief Propagation to generate high-confidence variable assignments,
    then use local search (WalkSAT) from those starting points.
    """
    n, clauses = instance["n"], instance["clauses"]
    num_clauses = len(clauses)
    
    # Deadline: leave a safety margin before the 330s kill
    deadline = time.time() + 325.0
    
    # Pre-process clauses for faster access
    # Each clause is a list of signed literals (1-based).
    # Convert to 0-based indices and signs.
    fast_clauses = []
    for clause in clauses:
        fast_clause = []
        for lit in clause:
            var_idx = abs(lit) - 1
            required = (lit > 0)
            fast_clause.append((var_idx, required))
        fast_clauses.append(fast_clause)
    
    # Build variable-to-clauses index for efficient updates
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
    
    def count_unsat(assignment):
        count = 0
        for i in range(num_clauses):
            if not is_satisfied(i, assignment):
                count += 1
        return count
    
    # Belief Propagation to get high-confidence assignments
    def run_belief_propagation(max_iterations=50, beta=1.0):
        """
        Run Belief Propagation on the 3-SAT factor graph.
        Returns a list of booleans representing the most likely assignment.
        """
        # Initialize beliefs: log-odds that variable is True
        # belief[i] = log(P(x_i=True) / P(x_i=False))
        beliefs = [0.0] * n  # Start with uniform beliefs
        
        # For each clause, track messages from variable nodes to clause nodes
        # and from clause nodes to variable nodes
        # msg_var_to_clause[c][v] = message from variable v to clause c
        # msg_clause_to_var[c][v] = message from clause c to variable v
        
        # Initialize messages to 0 (uniform)
        msg_var_to_clause = [[0.0 for _ in fc] for fc in fast_clauses]
        msg_clause_to_var = [[0.0 for _ in fc] for fc in fast_clauses]
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Update messages from variables to clauses
            for c_idx, fc in enumerate(fast_clauses):
                for v_pos, (var_idx, required) in enumerate(fc):
                    # Collect all messages from other clauses to this variable
                    # that are not from this clause
                    sum_msg = 0.0
                    for other_c_idx in var_to_clauses[var_idx]:
                        if other_c_idx != c_idx:
                            # Find position of this variable in the other clause
                            for other_pos, (other_var, _) in enumerate(fast_clauses[other_c_idx]):
                                if other_var == var_idx:
                                    sum_msg += msg_clause_to_var[other_c_idx][other_pos]
                                    break
                    
                    # The message from variable to clause depends on the sign
                    # If required is True (positive literal), message is sum_msg
                    # If required is False (negative literal), message is -sum_msg
                    if required:
                        msg_var_to_clause[c_idx][v_pos] = sum_msg
                    else:
                        msg_var_to_clause[c_idx][v_pos] = -sum_msg
            
            # Update messages from clauses to variables
            for c_idx, fc in enumerate(fast_clauses):
                # For a 3-CNF clause (l1 OR l2 OR l3), the message from clause to variable vi
                # is: -log(1 - exp(-msg1) * exp(-msg2) * exp(-msg3))
                # where msgj are the messages from the other variables to this clause
                # This is the soft-min of the other messages
                
                for v_pos, (var_idx, required) in enumerate(fc):
                    # Get messages from other variables in this clause
                    other_msgs = []
                    for other_pos, (other_var, _) in enumerate(fc):
                        if other_pos != v_pos:
                            other_msgs.append(msg_var_to_clause[c_idx][other_pos])
                    
                    if not other_msgs:
                        msg_clause_to_var[c_idx][v_pos] = 0.0
                    elif len(other_msgs) == 1:
                        # Single other variable: message is -log(1 - exp(-m))
                        m = other_msgs[0]
                        # Numerically stable computation
                        if m > 50:
                            msg_clause_to_var[c_idx][v_pos] = 0.0
                        elif m < -50:
                            msg_clause_to_var[c_idx][v_pos] = -m
                        else:
                            msg_clause_to_var[c_idx][v_pos] = -math.log1p(-math.exp(-m))
                    else:
                        # Multiple other variables: product of exp(-mj), then -log(1 - product)
                        # Use log-sum-exp for numerical stability
                        log_terms = [-m for m in other_msgs]
                        max_log = max(log_terms) if log_terms else 0
                        if max_log > 50:
                            msg_clause_to_var[c_idx][v_pos] = 0.0
                        elif max_log < -50:
                            msg_clause_to_var[c_idx][v_pos] = -max_log
                        else:
                            sum_exp = sum(math.exp(lt - max_log) for lt in log_terms)
                            log_product = max_log + math.log(sum_exp)
                            # -log(1 - exp(log_product))
                            if log_product > 0:
                                # This shouldn't happen for valid BP, but guard anyway
                                msg_clause_to_var[c_idx][v_pos] = 0.0
                            else:
                                msg_clause_to_var[c_idx][v_pos] = -math.log1p(-math.exp(log_product))
            
            # Update beliefs
            for var_idx in range(n):
                sum_msg = 0.0
                for c_idx in var_to_clauses[var_idx]:
                    # Find position of this variable in the clause
                    for v_pos, (v, _) in enumerate(fast_clauses[c_idx]):
                        if v == var_idx:
                            sum_msg += msg_clause_to_var[c_idx][v_pos]
                            break
                
                # Belief is the sum of all messages to this variable
                # Apply beta scaling for annealing
                beliefs[var_idx] = beta * sum_msg
        
        # Convert beliefs to assignment
        assignment = [b > 0 for b in beliefs]
        
        # Compute confidence (how strongly each variable is determined)
        confidences = [abs(b) for b in beliefs]
        
        return assignment, confidences, beliefs
    
    # Local search from a given starting assignment
    def local_search(start_assignment, max_steps=10000, min_confidence=0.0):
        """
        Run WalkSAT from the given starting assignment.
        Returns the best assignment found.
        """
        assignment = start_assignment[:]
        
        # Track current unsat count and unsat clause set
        unsat_set = set()
        for i in range(num_clauses):
            if not is_satisfied(i, assignment):
                unsat_set.add(i)
        
        current_unsat = len(unsat_set)
        
        if current_unsat == 0:
            return assignment, 0
        
        best_local = assignment[:]
        best_local_unsat = current_unsat
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if not unsat_set:
                return assignment, 0
            
            if len(unsat_set) < best_local_unsat:
                best_local_unsat = len(unsat_set)
                best_local = assignment[:]
            
            # Choose a clause from unsat_set
            clause_idx = random.choice(list(unsat_set))
            clause = fast_clauses[clause_idx]
            
            # Heuristic: prefer literals that appear in more unsat clauses
            scores = []
            for var_idx, required in clause:
                score = 0
                for c_idx in var_to_clauses[var_idx]:
                    if c_idx in unsat_set:
                        score += 1
                scores.append(score)
            
            # Weighted random choice based on scores
            total_score = sum(scores)
            if total_score == 0:
                var_idx, required = random.choice(clause)
            else:
                r = random.random() * total_score
                cumulative = 0
                var_idx, required = clause[0]
                for i, score in enumerate(scores):
                    cumulative += score
                    if r <= cumulative:
                        var_idx, required = clause[i]
                        break
            
            # Flip the variable
            assignment[var_idx] = not assignment[var_idx]
            
            # Update unsat_set: check clauses that contain this variable
            for c_idx in var_to_clauses[var_idx]:
                if is_satisfied(c_idx, assignment):
                    unsat_set.discard(c_idx)
                else:
                    unsat_set.add(c_idx)
            
            if not unsat_set:
                return assignment, 0
        
        return best_local, best_local_unsat
    
    # Main solver: BP-guided local search
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Phase 1: Run BP to get high-confidence starting points
    bp_assignment, bp_confidences, bp_beliefs = run_belief_propagation(max_iterations=30, beta=1.0)
    
    # Check if BP solution is already satisfying
    unsat_bp = count_unsat(bp_assignment)
    if unsat_bp == 0:
        return bp_assignment
    
    # Run local search from BP solution
    ls_assignment, ls_unsat = local_search(bp_assignment, max_steps=5000)
    if ls_unsat == 0:
        return ls_assignment
    
    if ls_unsat < best_unsat_count:
        best_unsat_count = ls_unsat
        best_assignment = ls_assignment
    
    # Phase 2: Generate multiple starting points by perturbing the BP solution
    # Focus on low-confidence variables first
    sorted_vars = sorted(range(n), key=lambda i: bp_confidences[i])
    
    num_perturbed_restarts = 150
    for restart in range(num_perturbed_restarts):
        if time.time() > deadline:
            break
        
        # Create a perturbed version of the BP solution
        assignment = bp_assignment[:]
        
        # Flip some low-confidence variables
        num_flips = random.randint(1, min(5, n))
        for _ in range(num_flips):
            # Preferentially flip low-confidence variables
            if random.random() < 0.7 and sorted_vars:
                var_idx = sorted_vars[random.randint(0, min(10, len(sorted_vars)-1))]
            else:
                var_idx = random.randint(0, n-1)
            assignment[var_idx] = not assignment[var_idx]
        
        # Run local search
        ls_assignment, ls_unsat = local_search(assignment, max_steps=5000)
        if ls_unsat == 0:
            return ls_assignment
        
        if ls_unsat < best_unsat_count:
            best_unsat_count = ls_unsat
            best_assignment = ls_assignment[:]
    
    # Phase 3: If still not solved, try pure random restarts with WalkSAT
    num_random_restarts = 50
    for restart in range(num_random_restarts):
        if time.time() > deadline:
            break
        
        assignment = [random.random() < 0.5 for _ in range(n)]
        ls_assignment, ls_unsat = local_search(assignment, max_steps=5000)
        if ls_unsat == 0:
            return ls_assignment
        
        if ls_unsat < best_unsat_count:
            best_unsat_count = ls_unsat
            best_assignment = ls_assignment[:]
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    else:
        return [random.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
