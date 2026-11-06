"""Planted 3-SAT solver with robust local search and time management."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280.0  # 20s safety margin before 300s kill, evaluator kills at 330s
    
    # Preprocess: extract variable indices and clause structure for speed
    # clauses is list of lists of signed 1-based literals
    num_clauses = len(clauses)
    
    # For each clause, store: list of (var_idx, is_positive)
    clause_data = []
    for c in clauses:
        cd = [(abs(l) - 1, l > 0) for l in c]
        clause_data.append(cd)
    
    def is_satisfied(assignment, cd):
        for var_idx, is_pos in cd:
            if assignment[var_idx] == is_pos:
                return True
        return False
    
    def count_unsatisfied(assignment):
        count = 0
        for cd in clause_data:
            if not is_satisfied(assignment, cd):
                count += 1
        return count
    
    def flip_variable(assignment, var_idx):
        assignment[var_idx] = not assignment[var_idx]
    
    def walk_sat_step(assignment, cd, rng, p=0.55):
        """Pick a random unsatisfied clause, flip a literal in it."""
        # For each literal in the clause, count how many other clauses would become satisfied
        # Use a simple heuristic: flip the literal that maximizes satisfied clauses
        best_var = -1
        best_score = -1
        for var_idx, is_pos in cd:
            # Current state: this literal is False (since clause is unsatisfied)
            # If we flip it, it becomes True, satisfying this clause
            # Count how many other clauses this flip affects
            # Simple approach: just pick randomly with some bias
            # More sophisticated: count clauses where this variable appears
            pass
        
        # Use a weighted random selection
        # For each literal in the clause, compute a weight
        weights = []
        for var_idx, is_pos in cd:
            # Count how many clauses have this variable
            # This is a precomputation opportunity
            w = 1.0
            weights.append(w)
        
        # Pick a literal with some probability
        idx = rng.choices(range(len(cd)), weights=weights, k=1)[0]
        var_idx, is_pos = cd[idx]
        flip_variable(assignment, var_idx)
    
    # Precompute: for each variable, which clauses contain it
    var_to_clauses = [[] for _ in range(n)]
    for i, cd in enumerate(clause_data):
        for var_idx, is_pos in cd:
            var_to_clauses[var_idx].append(i)
    
    def walk_sat_with_heuristic(assignment, cd, rng, p=0.55):
        """WalkSAT step with min-conflicts heuristic."""
        # For each literal in the clause, compute the number of clauses that would be satisfied
        best_var = -1
        best_count = -1
        for var_idx, is_pos in cd:
            # This literal is currently False (clause is unsatisfied)
            # Flipping it makes it True
            # Count how many other clauses this variable appears in where it's the only false literal
            count = 0
            for clause_idx in var_to_clauses[var_idx]:
                if clause_idx == clause_data.index(cd) if cd in clause_data else -1:
                    continue
                other_cd = clause_data[clause_idx]
                # Check if this clause is currently unsatisfied
                sat = False
                for v, pos in other_cd:
                    if assignment[v] == pos:
                        sat = True
                        break
                if not sat:
                    # Check if flipping var_idx would satisfy this clause
                    # i.e., all other literals in this clause are already satisfied
                    all_others_sat = True
                    for v, pos in other_cd:
                        if v == var_idx:
                            continue
                        if not assignment[v] == pos:
                            all_others_sat = False
                            break
                    if all_others_sat:
                        count += 1
            
            if count > best_count:
                best_count = count
                best_var = var_idx
        
        # With probability p, flip the best variable; otherwise flip randomly
        if best_var >= 0 and rng.random() < p:
            flip_variable(assignment, best_var)
        else:
            var_idx, is_pos = cd[rng.randint(0, len(cd) - 1)]
            flip_variable(assignment, var_idx)
    
    # Multi-restart WalkSAT with adaptive parameters
    rng = random.Random(0)
    
    best_assignment = None
    best_unsat_count = num_clauses + 1
    
    num_restarts = 0
    max_restarts = 50
    
    while time.time() < deadline and num_restarts < max_restarts:
        num_restarts += 1
        
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initial evaluation
        unsat_count = count_unsatisfied(assignment)
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        
        # WalkSAT with restarts
        max_steps = 10000
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if unsat_count == 0:
                return assignment
            
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment[:]
            
            # Find unsatisfied clauses
            unsat_clauses = []
            for i, cd in enumerate(clause_data):
                if not is_satisfied(assignment, cd):
                    unsat_clauses.append(i)
            
            if not unsat_clauses:
                return assignment
            
            # Pick a random unsatisfied clause
            clause_idx = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
            cd = clause_data[clause_idx]
            
            # WalkSAT step with min-conflicts heuristic
            # For each literal in the clause, compute the number of clauses that would be satisfied
            best_var = -1
            best_count = -1
            for var_idx, is_pos in cd:
                count = 0
                for other_clause_idx in var_to_clauses[var_idx]:
                    if other_clause_idx == clause_idx:
                        continue
                    other_cd = clause_data[other_clause_idx]
                    # Check if this clause is currently unsatisfied
                    sat = False
                    for v, pos in other_cd:
                        if assignment[v] == pos:
                            sat = True
                            break
                    if not sat:
                        # Check if flipping var_idx would satisfy this clause
                        all_others_sat = True
                        for v, pos in other_cd:
                            if v == var_idx:
                                continue
                            if not assignment[v] == pos:
                                all_others_sat = False
                                break
                        if all_others_sat:
                            count += 1
                
                if count > best_count:
                    best_count = count
                    best_var = var_idx
            
            # With probability p, flip the best variable; otherwise flip randomly
            p = 0.55
            if best_var >= 0 and rng.random() < p:
                flip_variable(assignment, best_var)
            else:
                var_idx, is_pos = cd[rng.randint(0, len(cd) - 1)]
                flip_variable(assignment, var_idx)
            
            # Update unsat_count incrementally
            # Re-evaluate only affected clauses
            # For simplicity, recompute periodically
            if step % 100 == 0:
                unsat_count = count_unsatisfied(assignment)
        
        if unsat_count == 0:
            return assignment
    
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
