import random
import time
import math

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Preprocessing: remove tautologies and duplicates
    clean_clauses = []
    seen = set()
    for c in clauses:
        # Check for tautology (x and ~x)
        lits = set(c)
        if any(-l in lits for l in lits):
            continue
        # Normalize for duplicate check
        # Sort literals to have a canonical form for set comparison
        # Note: literal sign matters, so (1, 2) != (1, -2)
        # But (1, 2) == (2, 1)
        key = tuple(sorted(lits))
        if key not in seen:
            seen.add(key)
            clean_clauses.append(c)
            
    if not clean_clauses:
        return [False] * n

    # DPLL with Unit Propagation
    # We will implement a simple recursive DPLL.
    # For n=50, recursion depth is manageable.
    
    # Data structures for DPLL
    # We'll use a list of clauses.
    # Unit propagation: if a clause has one literal, that literal must be true.
    
    def unit_propagate(clauses, assignment):
        """
        assignment: list of None, True, False.
        Returns (new_clauses, new_assignment, satisfied)
        If satisfied is True, all clauses are satisfied.
        If conflict is detected, returns None for new_clauses.
        """
        changed = True
        new_clauses = []
        for c in clauses:
            # Evaluate clause
            unassigned_lits = []
            is_satisfied = False
            for l in c:
                val = assignment[abs(l)-1]
                if val is None:
                    unassigned_lits.append(l)
                else:
                    if val == (l > 0):
                        is_satisfied = True
                        break
            if is_satisfied:
                continue # Clause satisfied, drop it
            elif len(unassigned_lits) == 0:
                return None, assignment, False # Conflict: clause unsatisfied, no unassigned lits
            else:
                new_clauses.append(unassigned_lits)
        
        # Check for unit clauses and propagate
        changed = True
        while changed:
            changed = False
            next_clauses = []
            for c in new_clauses:
                if len(c) == 1:
                    l = c[0]
                    var_idx = abs(l) - 1
                    val = l > 0
                    if assignment[var_idx] is None:
                        assignment[var_idx] = val
                        changed = True
                    elif assignment[var_idx] != val:
                        return None, assignment, False # Conflict
                    # Clause is satisfied by this unit, drop it
                else:
                    next_clauses.append(c)
            new_clauses = next_clauses
            
        # Check if all clauses satisfied
        if not new_clauses:
            return new_clauses, assignment, True
            
        return new_clauses, assignment, False

    def dpll(clauses, assignment):
        """
        Returns assignment (list of bool) if solved, None if unsat.
        """
        # Check for empty clauses (conflict)
        if any(len(c) == 0 for c in clauses):
            return None
            
        # Unit propagation
        prop_clauses, prop_assignment, solved = unit_propagate(clauses, assignment)
        if prop_clauses is None:
            return None
        if solved:
            # Fill remaining unassigned with False (arbitrary)
            final_assignment = []
            for i in range(n):
                if prop_assignment[i] is None:
                    final_assignment.append(False)
                else:
                    final_assignment.append(prop_assignment[i])
            return final_assignment

        # Choose a variable
        # Heuristic: pick variable with highest frequency in remaining clauses
        var_counts = [0] * n
        for c in prop_clauses:
            for l in c:
                var_counts[abs(l)-1] += 1
        
        # Pick variable with max count
        max_count = max(var_counts)
        candidates = [i for i in range(n) if var_counts[i] == max_count and prop_assignment[i] is None]
        
        if not candidates:
            # All assigned, but not solved? Shouldn't happen if UP is correct.
            # Check if satisfied
            for c in prop_clauses:
                sat = False
                for l in c:
                    if prop_assignment[abs(l)-1] == (l > 0):
                        sat = True
                        break
                if not sat:
                    return None
            final_assignment = []
            for i in range(n):
                if prop_assignment[i] is None:
                    final_assignment.append(False)
                else:
                    final_assignment.append(prop_assignment[i])
            return final_assignment

        var = random.choice(candidates)
        
        # Try True
        assignment[var] = True
        res = dpll(prop_clauses, assignment)
        if res is not None:
            return res
        
        # Try False
        assignment[var] = False
        res = dpll(prop_clauses, assignment)
        if res is not None:
            return res
            
        assignment[var] = None
        return None

    # Try DPLL first
    initial_assignment = [None] * n
    dpll_result = dpll(clean_clauses, initial_assignment)
    if dpll_result is not None:
        return dpll_result

    # If DPLL fails, use Simulated Annealing / WalkSAT
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0 # 10s margin
    
    # SA Parameters
    T = 1.0
    cooling = 0.99
    p_walk = 0.5 # Probability to pick a literal that satisfies the clause
    
    best_assignment = None
    best_unsat = float('inf')
    
    # Luby restart sequence
    def luby_sequence():
        i = 0
        while True:
            yield 1
            i += 1
            # Luby sequence: 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, ...
            # Simplified: just use fixed restarts or simple doubling
            # Let's use a simple restart every 1000 steps or if stuck
            pass

    # We'll implement a simple restart mechanism
    # State: assignment, unsat_counts, var_counts
    
    def init_state():
        assign = [random.random() < 0.5 for _ in range(n)]
        unsat_counts = [0] * len(clean_clauses)
        var_unsat = [[0, 0] for _ in range(n)] # [count_if_true_unsat, count_if_false_unsat]
        # Actually, for WalkSAT we just need to know which clauses are unsat.
        # For SA, we need energy change.
        
        # Let's use a simpler structure for speed:
        # assign: list of bool
        # clause_unsat: list of int (0 if sat, 1 if unsat)
        # var_impact: for each var, list of (clause_idx, sign)
        
        # Precompute var_impact
        # var_impact[i] = list of (clause_idx, literal_sign)
        # literal_sign: 1 if l > 0, -1 if l < 0
        # If assign[i] changes, check clauses in var_impact[i]
        
        return assign, clause_unsat, var_impact

    # Precompute var_impact once
    var_impact = [[] for _ in range(n)]
    for i, c in enumerate(clean_clauses):
        for l in c:
            var_idx = abs(l) - 1
            sign = 1 if l > 0 else -1
            var_impact[var_idx].append((i, sign))

    def eval_assignment(assign):
        """Calculate unsat counts for all clauses."""
        unsat_counts = [0] * len(clean_clauses)
        for i, c in enumerate(clean_clauses):
            sat = False
            for l in c:
                if assign[abs(l)-1] == (l > 0):
                    sat = True
                    break
            if not sat:
                unsat_counts[i] = 1
        return unsat_counts

    def get_unsat_clauses(unsat_counts):
        return [i for i, u in enumerate(unsat_counts) if u > 0]

    # Main SA Loop
    # We will run multiple restarts
    max_steps_per_restart = 1000
    restart_count = 0
    
    while time.time() < deadline:
        # Initialize
        assign = [random.random() < 0.5 for _ in range(n)]
        unsat_counts = eval_assignment(assign)
        current_unsat = sum(unsat_counts)
        
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assign.copy()
            if best_unsat == 0:
                return best_assignment

        T = 1.0
        steps = 0
        stuck_counter = 0
        
        while steps < max_steps_per_restart and time.time() < deadline:
            if current_unsat == 0:
                return assign
            
            # Pick a random unsatisfied clause
            unsat_clauses = get_unsat_clauses(unsat_counts)
            if not unsat_clauses:
                return assign
                
            clause_idx = random.choice(unsat_clauses)
            clause = clean_clauses[clause_idx]
            
            # Pick a variable to flip
            # WalkSAT heuristic:
            # 1. Count literals in clause that would become satisfied if flipped (i.e., currently unsatisfied)
            #    Actually, in WalkSAT, we pick a literal.
            #    Let's stick to standard WalkSAT:
            #    Pick a literal from the clause.
            #    With prob p, pick one that satisfies the clause (if any exist).
            #    With prob 1-p, pick randomly.
            
            # Identify literals in clause
            # If we flip var i, does it satisfy clause?
            # Literal l is satisfied if assign[abs(l)-1] == (l>0).
            # If we flip assign[abs(l)-1], it becomes not assign...
            # So literal l becomes satisfied if it was previously unsatisfied.
            
            satisfying_lits = []
            for l in clause:
                var_idx = abs(l) - 1
                # Currently, is this literal satisfied?
                if not (assign[var_idx] == (l > 0)):
                    satisfying_lits.append(var_idx)
            
            if satisfying_lits and random.random() < p_walk:
                var_to_flip = random.choice(satisfying_lits)
            else:
                # Pick any variable in clause
                var_to_flip = random.choice([abs(l)-1 for l in clause])
            
            # Calculate energy change
            # Flip var_to_flip
            old_val = assign[var_to_flip]
            new_val = not old_val
            
            delta = 0
            for c_idx, sign in var_impact[var_to_flip]:
                # Check if this clause was satisfied before
                # And if it will be satisfied after
                clause = clean_clauses[c_idx]
                
                # Current status
                was_sat = False
                for l in clause:
                    v_idx = abs(l) - 1
                    if v_idx == var_to_flip:
                        # This literal's value changes
                        if old_val == (l > 0):
                            was_sat = True
                            break
                    else:
                        if assign[v_idx] == (l > 0):
                            was_sat = True
                            break
                
                # New status
                is_sat = False
                for l in clause:
                    v_idx = abs(l) - 1
                    if v_idx == var_to_flip:
                        if new_val == (l > 0):
                            is_sat = True
                            break
                    else:
                        if assign[v_idx] == (l > 0):
                            is_sat = True
                            break
                
                if was_sat and not is_sat:
                    delta += 1
                elif not was_sat and is_sat:
                    delta -= 1
            
            # Acceptance criterion
            if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                assign[var_to_flip] = new_val
                current_unsat += delta
                
                if current_unsat < best_unsat:
                    best_unsat = current_unsat
                    best_assignment = assign.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
                    
                # Update unsat_counts for the affected clauses
                # We need to update unsat_counts for all clauses involving var_to_flip
                for c_idx, sign in var_impact[var_to_flip]:
                    # Re-evaluate this clause
                    clause = clean_clauses[c_idx]
                    sat = False
                    for l in clause:
                        if assign[abs(l)-1] == (l > 0):
                            sat = True
                            break
                    if sat:
                        if unsat_counts[c_idx] > 0:
                            current_unsat -= 1 # This is redundant with delta, but unsat_counts needs update
                        unsat_counts[c_idx] = 0
                    else:
                        if unsat_counts[c_idx] == 0:
                            current_unsat += 1
                        unsat_counts[c_idx] = 1

            else:
                stuck_counter += 1
                
            steps += 1
            T *= cooling
            
            # If stuck for too long, restart
            if stuck_counter > 50:
                break

    return best_assignment
