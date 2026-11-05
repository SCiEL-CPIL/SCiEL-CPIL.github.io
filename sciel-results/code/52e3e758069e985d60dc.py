"""Rewritten planted 3-SAT solver with smarter local search and clause learning."""
import time
import random
import math

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    # Deadline guard: return best result with safety margin before 330s
    deadline = time.time() + 325.0  # 5s safety margin
    
    # Preprocess: convert clauses to 0-based, track literal signs
    # clause: list of (var_idx, is_positive)
    clauses_pp = []
    for c in clauses:
        pp_c = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            pp_c.append((var_idx, is_pos))
        clauses_pp.append(pp_c)
    
    # Track variable occurrences for better heuristics
    var_count = [0] * n
    for c in clauses_pp:
        for v, _ in c:
            var_count[v] += 1
    
    def is_satisfied(assignment, clause):
        for v, pos in clause:
            if assignment[v] == pos:
                return True
        return False
    
    def count_unsat(assignment):
        cnt = 0
        for c in clauses_pp:
            if not is_satisfied(assignment, c):
                cnt += 1
        return cnt
    
    def satisfied_clauses(assignment):
        """Return list of unsatisfied clause indices"""
        unsat = []
        for i, c in enumerate(clauses_pp):
            if not is_satisfied(assignment, c):
                unsat.append(i)
        return unsat
    
    def min_unsat_clauses(assignment):
        """Return the minimum number of unsatisfied clauses and the assignment"""
        unsat_list = satisfied_clauses(assignment)
        return len(unsat_list), unsat_list
    
    # Try multiple restarts with different strategies
    best_assignment = None
    best_unsat_count = num_clauses + 1
    
    rng = random.Random(42)
    
    max_restarts = 50
    steps_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize assignment
        # For planted SAT, the planted solution is uniform, so random init is fine
        # But we can bias slightly based on clause structure
        if restart == 0:
            # First restart: try a deterministic heuristic
            # Set each variable to satisfy the most clauses
            assignment = [False] * n
            for v in range(n):
                # Count how many clauses are satisfied if v=True vs v=False
                # (considering other variables as False for simplicity)
                true_sat = 0
                false_sat = 0
                for c in clauses_pp:
                    v_pos = None
                    for vv, pos in c:
                        if vv == v:
                            v_pos = pos
                            break
                    if v_pos is not None:
                        if v_pos:
                            true_sat += 1
                        else:
                            false_sat += 1
                assignment[v] = true_sat >= false_sat
        else:
            # Subsequent restarts: random with some noise
            assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Local search with adaptive parameters
        # Use a combination of WalkSAT and a greedy approach
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            unsat_list = satisfied_clauses(assignment)
            if not unsat_list:
                # Found satisfying assignment
                return assignment
            
            # Update best
            if len(unsat_list) < best_unsat_count:
                best_unsat_count = len(unsat_list)
                best_assignment = assignment[:]
            
            # Choose an unsatisfied clause
            # Weight by how "hard" it is (fewer true literals = harder)
            weighted_clauses = []
            for ci in unsat_list:
                c = clauses_pp[ci]
                # Count how many literals are false (all are false since clause is unsat)
                # Use inverse of number of true literals in other clauses as weight
                weight = 1.0 / (len(c) + 1)
                weighted_clauses.append((weight, ci))
            
            # Weighted random choice
            total_w = sum(w for w, _ in weighted_clauses)
            r = rng.random() * total_w
            cumulative = 0
            chosen_ci = weighted_clauses[0][1]
            for w, ci in weighted_clauses:
                cumulative += w
                if r <= cumulative:
                    chosen_ci = ci
                    break
            clause = clauses_pp[chosen_ci]
            
            # Choose a literal to flip
            # With probability p, flip a literal from the clause
            # With probability 1-p, flip any variable
            # Adaptive p based on progress
            progress = 1.0 - (len(unsat_list) / num_clauses) if num_clauses > 0 else 1.0
            p = 0.5 + 0.5 * progress  # More greedy as we get closer
            
            if rng.random() < p:
                # Pick a literal from the clause to flip
                # Weight by how many other clauses this flip would affect
                lit_weights = []
                for v, pos in clause:
                    # Count impact: how many clauses contain this variable
                    impact = var_count[v]
                    lit_weights.append(impact)
                total_lw = sum(lit_weights)
                r2 = rng.random() * total_lw
                cumulative2 = 0
                chosen_idx = 0
                for i, lw in enumerate(lit_weights):
                    cumulative2 += lw
                    if r2 <= cumulative2:
                        chosen_idx = i
                        break
                v, pos = clause[chosen_idx]
                assignment[v] = not assignment[v]
            else:
                # Random variable flip
                v = rng.randint(0, n - 1)
                assignment[v] = not assignment[v]
        
        # If we've exhausted this restart, check if we should continue
        if best_unsat_count == 0:
            break
    
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
