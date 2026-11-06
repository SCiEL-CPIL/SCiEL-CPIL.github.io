"""Synthetic planted 3-SAT solver (150 variables)."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)

    # Deadline guard: finish well before 330s wall-clock kill.
    start_time = time.time()
    deadline = start_time + 280.0  # 50s safety margin

    # Pre-process clauses into arrays for fast evaluation
    # Each clause: list of signed literals (1-based)
    # Convert to 0-based indices with sign
    clause_vars = []  # list of lists of (var_idx, sign) where sign is +1 or -1
    for c in clauses:
        cv = []
        for lit in c:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            cv.append((v, s))
        clause_vars.append(cv)

    # Build variable-to-clause incidence for fast updates
    var_to_clauses = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for v, s in cv:
            var_to_clauses[v].append(ci)

    def make_initial_assignment(seed):
        rng = random.Random(seed)
        return [rng.random() < 0.5 for _ in range(n)]

    def eval_clause(assignment, ci):
        cv = clause_vars[ci]
        for v, s in cv:
            if assignment[v] == (s == 1):
                return True
        return False

    def count_unsat(assignment):
        count = 0
        for ci in range(num_clauses):
            if not eval_clause(assignment, ci):
                count += 1
        return count

    def get_unsat_clauses(assignment):
        unsat = []
        for ci in range(num_clauses):
            if not eval_clause(assignment, ci):
                unsat.append(ci)
        return unsat

    # Phase 1: Gaussian Elimination / parity-based initialization
    # For planted SAT with low density, try to find a consistent assignment
    # by treating clauses as soft constraints and using a linear relaxation heuristic.
    # We'll use a simple approach: initialize with a random assignment, then
    # use a "majority vote" per variable based on clause structure.
    
    # Build literal frequency: for each variable, count how many times it appears
    # positively vs negatively in clauses. Use this as a bias for initialization.
    pos_count = [0] * n
    neg_count = [0] * n
    for ci, cv in enumerate(clause_vars):
        for v, s in cv:
            if s == 1:
                pos_count[v] += 1
            else:
                neg_count[v] += 1

    # Initialize: set variable to True if positive appearances > negative, else False
    # This is a simple heuristic that may help for planted instances.
    def initial_assignment_biased(seed):
        rng = random.Random(seed)
        assign = []
        for v in range(n):
            if pos_count[v] > neg_count[v]:
                assign.append(True)
            elif neg_count[v] > pos_count[v]:
                assign.append(False)
            else:
                assign.append(rng.random() < 0.5)
        return assign

    # Phase 2: WalkSAT with improved heuristics
    def walksat(assignment, seed, max_steps=50000):
        rng = random.Random(seed)
        # Track clause satisfaction state
        clause_sat = [eval_clause(assignment, ci) for ci in range(num_clauses)]
        num_unsat = sum(1 for s in clause_sat if not s)
        
        if num_unsat == 0:
            return assignment, 0

        # Precompute for each clause, which variables are in it (for fast flip evaluation)
        # We'll use a simple approach: when flipping a variable, update only affected clauses.
        
        # For each clause, track how many literals are satisfied
        clause_sat_count = [0] * num_clauses
        for ci, cv in enumerate(clause_vars):
            cnt = 0
            for v, s in cv:
                if assignment[v] == (s == 1):
                    cnt += 1
            clause_sat_count[ci] = cnt

        num_unsat = sum(1 for c in clause_sat_count if c == 0)
        
        if num_unsat == 0:
            return assignment, 0

        step = 0
        while num_unsat > 0 and step < max_steps:
            if time.time() > deadline:
                break

            # Pick an unsatisfied clause
            unsat_clauses = [ci for ci in range(num_clauses) if clause_sat_count[ci] == 0]
            if not unsat_clauses:
                break
            ci = rng.choice(unsat_clauses)
            cv = clause_vars[ci]
            
            # Choose a literal to flip using a weighted heuristic
            # Prefer literals that appear in many other unsatisfied clauses
            # (min-unsat heuristic)
            weights = []
            for v, s in cv:
                # Count how many other unsatisfied clauses contain this variable
                w = 0
                for other_ci in var_to_clauses[v]:
                    if other_ci != ci and clause_sat_count[other_ci] == 0:
                        w += 1
                weights.append(w)
            
            # Weighted selection
            total_w = sum(weights)
            if total_w == 0:
                idx = rng.randrange(len(cv))
            else:
                r = rng.random() * total_w
                cum = 0
                idx = 0
                for i, w in enumerate(weights):
                    cum += w
                    if r <= cum:
                        idx = i
                        break
                else:
                    idx = len(cv) - 1
            
            v, s = cv[idx]
            
            # Flip variable v
            old_val = assignment[v]
            assignment[v] = not old_val
            
            # Update clause satisfaction counts for all clauses containing v
            for other_ci in var_to_clauses[v]:
                oc = clause_vars[other_ci]
                # Determine if this variable's literal is satisfied in this clause
                old_sat = old_val == (next(s2 for v2, s2 in oc if v2 == v))
                new_sat = (not old_val) == (next(s2 for v2, s2 in oc if v2 == v))
                if old_sat and not new_sat:
                    clause_sat_count[other_ci] -= 1
                    if clause_sat_count[other_ci] == 0:
                        num_unsat += 1
                elif not old_sat and new_sat:
                    clause_sat_count[other_ci] += 1
                    if clause_sat_count[other_ci] == 1:
                        num_unsat -= 1
            
            step += 1

        return assignment, num_unsat

    # Phase 3: Targeted repair - only flip variables in unsatisfied clauses
    def targeted_repair(assignment, seed, max_rounds=100):
        rng = random.Random(seed)
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            unsat = get_unsat_clauses(assignment)
            if not unsat:
                break
            
            # Collect all variables in unsatisfied clauses
            var_set = set()
            for ci in unsat:
                for v, s in clause_vars[ci]:
                    var_set.add(v)
            
            # Try flipping each variable and see if it reduces unsat count
            best_var = None
            best_unsat = len(unsat)
            for v in var_set:
                # Flip v
                old_val = assignment[v]
                assignment[v] = not old_val
                new_unsat = count_unsat(assignment)
                assignment[v] = old_val
                if new_unsat < best_unsat:
                    best_unsat = new_unsat
                    best_var = v
            
            if best_var is not None:
                assignment[best_var] = not assignment[best_var]
            else:
                # No single flip helps; try random flip in unsat clauses
                ci = rng.choice(unsat)
                v, s = rng.choice(clause_vars[ci])
                assignment[v] = not assignment[v]
        
        return assignment

    # Main search: try multiple restarts with different seeds
    best_assignment = None
    best_unsat = float('inf')
    
    num_restarts = 0
    while time.time() < deadline:
        seed = num_restarts
        # Use biased initialization for some restarts, random for others
        if seed % 3 == 0:
            assignment = initial_assignment_biased(seed)
        else:
            assignment = make_initial_assignment(seed)
        
        # Run WalkSAT
        assignment, num_unsat = walksat(assignment, seed)
        
        if num_unsat < best_unsat:
            best_unsat = num_unsat
            best_assignment = assignment[:]
        
        if num_unsat == 0:
            break
        
        # If we have some unsat, try targeted repair
        if num_unsat <= 5 and time.time() < deadline:
            repaired = targeted_repair(assignment, seed + 1000)
            final_unsat = count_unsat(repaired)
            if final_unsat < best_unsat:
                best_unsat = final_unsat
                best_assignment = repaired[:]
            if final_unsat == 0:
                break
        
        num_restarts += 1
    
    if best_assignment is None:
        # Fallback: return a random assignment
        best_assignment = make_initial_assignment(0)
    
    return best_assignment
# EVOLVE-BLOCK-END
