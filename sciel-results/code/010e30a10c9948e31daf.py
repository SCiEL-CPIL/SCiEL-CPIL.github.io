"""Synthetic planted 3-SAT solver (150 variables) with incremental WalkSAT and adaptive restarts."""
import time
import random
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 285  # safety margin before 300s kill

    # Precompute clause literals as (var_idx, polarity) for faster access
    clauses_opt = [[(abs(l) - 1, 1 if l > 0 else 0) for l in c] for c in clauses]
    num_clauses = len(clauses_opt)
    
    # Build variable to clause index map for incremental updates
    var_to_clauses = [[] for _ in range(n)]
    # Also store polarity lookup: var_to_clause_pol[var_idx] = list of (clause_idx, polarity)
    var_to_clause_pol = [[] for _ in range(n)]
    for i, clause in enumerate(clauses_opt):
        for var_idx, polarity in clause:
            var_to_clauses[var_idx].append(i)
            var_to_clause_pol[var_idx].append((i, polarity))

    # State:
    # assignment: list of booleans (0/1)
    # clause_weights: list of ints, number of literals satisfied in each clause
    # unsat_set: set of indices of unsatisfied clauses
    
    def init_state(assignment):
        """Initialize clause weights and unsat set from a given assignment."""
        clause_weights = [0] * num_clauses
        unsat_set = set()
        for i, clause in enumerate(clauses_opt):
            w = 0
            for var_idx, polarity in clause:
                if assignment[var_idx] == polarity:
                    w += 1
            clause_weights[i] = w
            if w == 0:
                unsat_set.add(i)
        return clause_weights, unsat_set

    def walksat(assignment, rng, max_steps=5000, p_noise=0.3, tabu_size=50):
        """
        Efficient WalkSAT with incremental clause weight updates and tabu list.
        Returns True if a solution is found, False otherwise.
        Modifies assignment in place.
        """
        clause_weights, unsat_set = init_state(assignment)
        
        if not unsat_set:
            return True

        # Tabu list: recently flipped variables
        tabu = {}  # var_idx -> step when it was flipped
        current_step = 0
        
        for step in range(max_steps):
            if not unsat_set:
                return True
            
            # Check deadline
            if time.time() > deadline:
                return False

            # Pick a random unsatisfied clause
            ci = rng.choice(list(unsat_set))
            clause = clauses_opt[ci]
            
            # Choose variable to flip using gain heuristic
            # For each variable in the clause, compute the gain (change in total unsat count)
            best_var = None
            best_gain = -100  # We want to maximize gain (decrease in unsat count)
            best_ties = []
            
            for var_idx, polarity in clause:
                # Compute gain of flipping var_idx
                gain = 0
                # This clause will become satisfied (if it wasn't already, but it is unsat)
                # Actually, since the clause is unsat (weight 0), flipping any var in it will satisfy it
                # So gain += 1 (this clause leaves unsat set)
                gain += 1
                
                # Check other clauses containing var_idx
                for j, pol_j in var_to_clause_pol[var_idx]:
                    if j == ci:
                        continue
                    
                    old_val = assignment[var_idx]
                    new_val = not old_val
                    
                    old_lit_sat = (old_val == pol_j)
                    new_lit_sat = (new_val == pol_j)
                    
                    if old_lit_sat and not new_lit_sat:
                        # Literal goes from True to False
                        # Weight decreases by 1
                        new_w = clause_weights[j] - 1
                        if new_w == 0:
                            # Clause becomes unsat
                            gain -= 1
                    elif not old_lit_sat and new_lit_sat:
                        # Literal goes from False to True
                        # Weight increases by 1
                        # If clause was unsat (weight 0), it becomes sat
                        if clause_weights[j] == 0:
                            gain += 1
                    # else: no change in sat/unsat status
                
                # Apply tabu penalty
                if var_idx in tabu and (current_step - tabu[var_idx]) < tabu_size:
                    gain -= 2  # Penalty for tabu variable
                
                if gain > best_gain:
                    best_gain = gain
                    best_ties = [var_idx]
                elif gain == best_gain:
                    best_ties.append(var_idx)
            
            # Choose variable: with probability p_noise, pick random; else pick best
            if rng.random() < p_noise:
                var_idx = rng.choice(clause)[0]
            else:
                var_idx = rng.choice(best_ties) if best_ties else rng.choice(clause)[0]
            
            # Perform flip
            old_val = assignment[var_idx]
            assignment[var_idx] = not old_val
            
            # Update tabu
            tabu[var_idx] = current_step
            current_step += 1
            
            # Update state incrementally
            # 1. Update clause ci (was unsat, now sat)
            clause_weights[ci] = 1
            unsat_set.discard(ci)
            
            # 2. Update other clauses containing var_idx
            for j, pol_j in var_to_clause_pol[var_idx]:
                if j == ci:
                    continue
                
                old_lit_sat = (old_val == pol_j)
                new_lit_sat = (assignment[var_idx] == pol_j)
                
                if old_lit_sat and not new_lit_sat:
                    # Weight decreases by 1
                    clause_weights[j] -= 1
                    if clause_weights[j] == 0:
                        # Becomes unsat
                        unsat_set.add(j)
                elif not old_lit_sat and new_lit_sat:
                    # Weight increases by 1
                    clause_weights[j] += 1
                    if clause_weights[j] == 1:
                        # Was unsat, now sat
                        unsat_set.discard(j)

            if not unsat_set:
                return True

        return False

    def polish(assignment, rng, max_steps=1000):
        """
        Final polish pass: try targeted flips to fix remaining unsat clauses.
        Returns True if solution found.
        """
        clause_weights, unsat_set = init_state(assignment)
        
        if not unsat_set:
            return True
        
        for step in range(max_steps):
            if not unsat_set:
                return True
            
            if time.time() > deadline:
                return False
            
            # Pick an unsat clause
            ci = rng.choice(list(unsat_set))
            clause = clauses_opt[ci]
            
            # Try each variable in the clause
            best_var = None
            best_gain = -100
            best_ties = []
            
            for var_idx, polarity in clause:
                gain = 1  # This clause becomes sat
                
                for j, pol_j in var_to_clause_pol[var_idx]:
                    if j == ci:
                        continue
                    
                    old_val = assignment[var_idx]
                    new_val = not old_val
                    
                    old_lit_sat = (old_val == pol_j)
                    new_lit_sat = (new_val == pol_j)
                    
                    if old_lit_sat and not new_lit_sat:
                        new_w = clause_weights[j] - 1
                        if new_w == 0:
                            gain -= 1
                    elif not old_lit_sat and new_lit_sat:
                        if clause_weights[j] == 0:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_ties = [var_idx]
                elif gain == best_gain:
                    best_ties.append(var_idx)
            
            var_idx = rng.choice(best_ties) if best_ties else rng.choice(clause)[0]
            old_val = assignment[var_idx]
            assignment[var_idx] = not old_val
            
            # Update state
            clause_weights[ci] = 1
            unsat_set.discard(ci)
            
            for j, pol_j in var_to_clause_pol[var_idx]:
                if j == ci:
                    continue
                
                old_lit_sat = (old_val == pol_j)
                new_lit_sat = (assignment[var_idx] == pol_j)
                
                if old_lit_sat and not new_lit_sat:
                    clause_weights[j] -= 1
                    if clause_weights[j] == 0:
                        unsat_set.add(j)
                elif not old_lit_sat and new_lit_sat:
                    clause_weights[j] += 1
                    if clause_weights[j] == 1:
                        unsat_set.discard(j)
            
            if not unsat_set:
                return True
        
        return False

    # Strategy: Adaptive restarts with progress-based scheduling
    best_assignment = None
    best_unsat = float('inf')
    
    rng = random.Random(0)
    
    # Initial random assignment
    assignment = [rng.random() < 0.5 for _ in range(n)]
    _, unsat_set = init_state(assignment)
    best_unsat = len(unsat_set)
    best_assignment = assignment[:]
    
    if best_unsat == 0:
        return assignment

    # Adaptive restart parameters
    base_steps = 2000
    max_restarts = 500
    no_improve_count = 0
    max_no_improve = 10
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Adaptive step count: increase if making progress, decrease if stuck
        if no_improve_count > 0:
            steps = max(500, base_steps // (no_improve_count + 1))
        else:
            steps = base_steps
        
        # New random assignment with slight bias toward best
        if best_unsat < 10 and rng.random() < 0.3:
            # Start from best assignment with some random flips
            assignment = best_assignment[:]
            num_flips = max(1, n // 10)
            for _ in range(num_flips):
                assignment[rng.randrange(n)] = not assignment[rng.randrange(n)]
        else:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Run WalkSAT
        success = walksat(assignment, rng, max_steps=steps)
        
        if success:
            return assignment
        
        # Track best
        _, unsat_set = init_state(assignment)
        current_unsat = len(unsat_set)
        
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment[:]
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # If close to solution, try polish
        if best_unsat <= 5 and time.time() < deadline - 5:
            polish_assignment = best_assignment[:]
            if polish(polish_assignment, rng, max_steps=500):
                return polish_assignment
            _, unsat_set2 = init_state(polish_assignment)
            if len(unsat_set2) < best_unsat:
                best_unsat = len(unsat_set2)
                best_assignment = polish_assignment[:]

    # Final attempt: polish the best assignment
    if best_assignment and time.time() < deadline - 2:
        if polish(best_assignment, rng, max_steps=1000):
            return best_assignment

    return best_assignment if best_assignment else [False] * n
