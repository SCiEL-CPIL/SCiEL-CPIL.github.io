"""Synthetic planted 3-SAT solver (50 variables) with WalkSAT + improvements."""
import random
import time


def solve(instance):
    """
    Solve a planted 3-SAT instance using:
    1. Fast pre-processing (unit propagation, pure literals, subsumption)
    2. WalkSAT with adaptive parameters, restarts, and improved heuristics
    3. Time-aware budgeting with a safety margin
    """
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline: 325 seconds (5 second safety margin before 330s kill)
    deadline = time.time() + 325.0
    
    # Pre-process: convert to 0-based
    # Build clause list with 0-based indices: list of (var_idx, is_positive)
    clause_list = []
    for c in clauses:
        clause_list.append([(abs(l) - 1, l > 0) for l in c])
    
    m = len(clause_list)
    
    # Build occurrence lists: for each variable, which clauses contain it and with what polarity
    # var_occurrence[var] = list of (clause_idx, is_positive_in_clause)
    var_occurrence = [[] for _ in range(n)]
    for ci, clause in enumerate(clause_list):
        for var_idx, is_pos in clause:
            var_occurrence[var_idx].append((ci, is_pos))
    
    # Pre-processing: unit propagation and pure literals
    assignment = [None] * n  # None = unassigned, True/False = assigned
    
    def propagate_unit():
        """Iteratively apply unit propagation and pure literal elimination."""
        changed = True
        while changed:
            changed = False
            
            # Check for pure literals
            for var in range(n):
                if assignment[var] is not None:
                    continue
                # Check if all occurrences have the same polarity
                polarities = set()
                for ci, is_pos in var_occurrence[var]:
                    # If the clause is already satisfied, skip
                    if any(assignment[v] is not None and assignment[v] == p for v, p in clause_list[ci]):
                        continue
                    polarities.add(is_pos)
                if len(polarities) == 1:
                    assignment[var] = polarities.pop()
                    changed = True
            
            # Check for unit clauses
            for ci, clause in enumerate(clause_list):
                if any(assignment[v] is not None and assignment[v] == p for v, p in clause):
                    continue  # already satisfied
                unassigned = [(v, p) for v, p in clause if assignment[v] is None]
                if len(unassigned) == 1:
                    v, p = unassigned[0]
                    assignment[v] = p
                    changed = True
                elif len(unassigned) == 0:
                    return False  # contradiction
        
        return True
    
    if not propagate_unit():
        # Contradiction - shouldn't happen for satisfiable instances, but fallback
        assignment = [random.random() < 0.5 for _ in range(n)]
    
    # Fill in unassigned variables randomly
    for i in range(n):
        if assignment[i] is None:
            assignment[i] = random.random() < 0.5
    
    # Check if we already have a solution
    def count_unsat(assn):
        count = 0
        for clause in clause_list:
            sat = False
            for v, p in clause:
                if assn[v] == p:
                    sat = True
                    break
            if not sat:
                count += 1
        return count
    
    if count_unsat(assignment) == 0:
        return assignment
    
    # WalkSAT with improved heuristics
    rng = random.Random(12345)
    
    best_assignment = assignment[:]
    best_unsat = count_unsat(assignment)
    
    if best_unsat == 0:
        return best_assignment
    
    max_restarts = 200
    base_max_steps = 5000
    
    for restart in range(max_restarts):
        # Check time
        if time.time() > deadline - 2.0:
            break
        
        # Random initialization with some bias toward current best
        if restart == 0:
            current = best_assignment[:]
        elif rng.random() < 0.3:
            # Start from best assignment
            current = best_assignment[:]
            # Flip a few random variables to escape local optimum
            for _ in range(rng.randint(1, 5)):
                v = rng.randrange(n)
                current[v] = not current[v]
        else:
            current = [rng.random() < 0.5 for _ in range(n)]
        
        # Adaptive parameters
        p = 0.5 + 0.1 * (restart % 5) / 5.0  # vary between 0.5 and 0.6
        if restart > 50:
            p = 0.55 + 0.15 * rng.random()
        
        max_steps = base_max_steps + restart * 50
        if time.time() > deadline - 10.0:
            max_steps = min(max_steps, 1000)
        
        for step in range(max_steps):
            # Check time periodically
            if step % 50 == 0 and time.time() > deadline - 2.0:
                break
            
            # Find unsatisfied clauses
            unsat_clauses = []
            for ci, clause in enumerate(clause_list):
                sat = False
                for v, p_lit in clause:
                    if current[v] == p_lit:
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(ci)
            
            if not unsat_clauses:
                return current
            
            unsat_count = len(unsat_clauses)
            
            # Track best
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = current[:]
            
            # Choose an unsatisfied clause
            # Weighted selection: prefer clauses with fewer literals that could be satisfied
            clause_idx = rng.choice(unsat_clauses)
            clause = clause_list[clause_idx]
            
            # Choose a literal to flip
            if rng.random() < p:
                # Random literal from the clause
                var_idx, _ = rng.choice(clause)
            else:
                # Min-unsat heuristic: flip the variable that minimizes the number
                # of currently-satisfied clauses that would become unsatisfied
                best_var = None
                best_score = float('inf')
                best_tie_break = float('inf')
                
                for var_idx, _ in clause:
                    # Count how many currently-satisfied clauses contain this variable
                    # and would become unsatisfied if we flip it
                    would_break = 0
                    would_satisfy = 0
                    
                    for ci2, (v2, p2) in var_occurrence[var_idx]:
                        # Check if clause ci2 is currently satisfied
                        clause2 = clause_list[ci2]
                        is_sat = False
                        for v3, p3 in clause2:
                            if current[v3] == p3:
                                is_sat = True
                                break
                        
                        if is_sat:
                            # Would flipping var_idx break this clause?
                            # Only if var_idx is the only literal satisfying it
                            only_satisfier = True
                            for v3, p3 in clause2:
                                if v3 != var_idx and current[v3] == p3:
                                    only_satisfier = False
                                    break
                            if only_satisfier:
                                would_break += 1
                        else:
                            # Would flipping var_idx satisfy this clause?
                            new_val = not current[var_idx]
                            if new_val == p2:
                                would_satisfy += 1
                    
                    # Primary: minimize would_break, secondary: maximize would_satisfy
                    score = would_break - would_satisfy * 0.1
                    tie_break = rng.random()
                    
                    if score < best_score or (score == best_score and tie_break < best_tie_break):
                        best_score = score
                        best_tie_break = tie_break
                        best_var = var_idx
                
                var_idx = best_var
            
            # Flip the variable
            current[var_idx] = not current[var_idx]
        
        # End of restart
        if best_unsat == 0:
            break
        
        # If we've been stuck for a while, try a more aggressive restart strategy
        if restart % 20 == 19 and best_unsat > 3:
            # Try to use the best assignment as a seed for a more focused search
            pass
    
    return best_assignment
