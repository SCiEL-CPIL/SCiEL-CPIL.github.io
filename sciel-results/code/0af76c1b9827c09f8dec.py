import time
import random
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline guard: 300s limit, use 295s for safety
    start_time = time.time()
    deadline = start_time + 295.0
    
    # Preprocess
    # clause_vars[c] = list of 0-based var indices
    # clause_sigs[c] = list of +1/-1 signs
    clause_vars = []
    clause_sigs = []
    var_to_clauses = [[] for _ in range(n)]
    
    for c_idx, clause in enumerate(clauses):
        vars_list = []
        sigs_list = []
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            vars_list.append(v)
            sigs_list.append(s)
            var_to_clauses[v].append(c_idx)
        clause_vars.append(vars_list)
        clause_sigs.append(sigs_list)
        
    rng = random.Random(42)
    
    def check_clause(c_idx, assignment):
        """Check if clause c_idx is satisfied."""
        vs = clause_vars[c_idx]
        ss = clause_sigs[c_idx]
        for i in range(len(vs)):
            if assignment[vs[i]] == (ss[i] > 0):
                return True
        return False

    def get_unsat_clauses(assignment):
        """Return list of unsatisfied clause indices."""
        unsat = []
        for c_idx in range(m):
            if not check_clause(c_idx, assignment):
                unsat.append(c_idx)
        return unsat

    def update_unsat_status(assignment, var_idx, old_val, unsat_clauses_set, clause_sat_list):
        """
        Update unsat status for clauses containing var_idx after flip.
        var_idx: variable that was flipped
        old_val: value before flip
        unsat_clauses_set: set of currently unsatisfied clause indices
        clause_sat_list: list of bools, True if satisfied
        """
        new_val = not old_val
        for c_idx in var_to_clauses[var_idx]:
            # Determine if this clause is affected
            # If it was unsat, it might become sat.
            # If it was sat, it might become unsat (if this was the only satisfying literal).
            
            # Simple approach: re-evaluate the clause.
            # Since n=150, m is likely small (e.g., 450-1000), re-evaluating 3 literals is fast.
            is_sat = check_clause(c_idx, assignment)
            
            if is_sat:
                if c_idx in unsat_clauses_set:
                    unsat_clauses_set.remove(c_idx)
                clause_sat_list[c_idx] = True
            else:
                if c_idx not in unsat_clauses_set:
                    unsat_clauses_set.add(c_idx)
                clause_sat_list[c_idx] = False

    def compute_gains(assignment, unsat_clauses_list, chosen_clause_idx):
        """
        Compute gain for each variable in the chosen clause.
        Gain = number of other unsatisfied clauses that would be satisfied by flipping this variable.
        """
        gains = {}
        vs = clause_vars[chosen_clause_idx]
        ss = clause_sigs[chosen_clause_idx]
        
        # We need to know which unsatisfied clauses contain which variables with which signs.
        # To make this efficient, we can just iterate through the unsat clauses for each var?
        # That might be slow if unsat is large.
        # Alternative: Pre-compute for each variable, the list of (clause_idx, sign) for unsat clauses?
        # That requires updating on every flip.
        
        # Let's use a simpler heuristic that is still effective:
        # Gain = (number of unsat clauses this var is in) 
        # This is fast to compute if we maintain a count of unsat appearances per var.
        
        # Let's try the "count of unsat clauses containing v" heuristic.
        # We need to maintain `unsat_count_per_var`.
        pass # We will implement a different gain calculation in the main loop for simplicity first, then optimize if needed.
        
        # For now, let's just return random weights or simple counts.
        # Actually, let's implement the proper gain:
        # For var v in chosen_clause:
        #   For each other clause c' in unsat_clauses_list:
        #     If v is in c' and the literal sign in c' matches the flip direction, gain++.
        
        # This is O(|unsat| * 3) per step. If |unsat| is 100, that's 300 ops. 5000 steps * 300 = 1.5M ops. Fine.
        
        for i in range(len(vs)):
            v = vs[i]
            s = ss[i]
            # Flip direction: if current assignment[v] is False and s==1, flip to True.
            # If current assignment[v] is True and s==-1, flip to False.
            # Basically, we want to set assignment[v] to (s > 0).
            target_val = (s > 0)
            current_val = assignment[v]
            
            if current_val == target_val:
                # Flipping would make this literal False, so it wouldn't help this clause?
                # Wait, the clause is unsat, so all literals are False.
                # If current_val == target_val, then the literal is currently TRUE?
                # No, if clause is unsat, all literals are False.
                # So if s==1, literal is +v. If v is True, literal is True. Clause would be Sat.
                # So if clause is Unsat, and s==1, v must be False.
                # So current_val (False) != target_val (True).
                # So this branch shouldn't happen for an unsat clause.
                gains[v] = 0
                continue
            
            # We are going to flip v to target_val.
            gain = 0
            # Check other unsat clauses
            for c_idx in unsat_clauses_list:
                if c_idx == chosen_clause_idx:
                    continue
                c_vs = clause_vars[c_idx]
                c_ss = clause_sigs[c_idx]
                for j in range(len(c_vs)):
                    if c_vs[j] == v:
                        c_s = c_ss[j]
                        # If we flip v to target_val, does this literal become True?
                        # Literal is +v if c_s==1, -v if c_s==-1.
                        # Literal becomes True if (c_s==1 and target_val==True) or (c_s==-1 and target_val==False)
                        if (c_s == 1 and target_val) or (c_s == -1 and not target_val):
                            gain += 1
                        break # Var v appears at most once in a clause (standard 3-SAT)
            gains[v] = gain
        return gains

    def walksat_attempt(assignment, max_steps, rng):
        unsat_clauses = get_unsat_clauses(assignment)
        unsat_set = set(unsat_clauses)
        
        if not unsat_set:
            return True, assignment
            
        best_unsat_in_attempt = len(unsat_set)
        best_assignment = assignment[:]
        
        for step in range(max_steps):
            if time.time() >= deadline:
                break
            
            if not unsat_set:
                return True, assignment
            
            # Pick random unsat clause
            c_idx = rng.choice(list(unsat_set))
            
            # Compute gains
            gains = compute_gains(assignment, list(unsat_set), c_idx)
            
            # Pick var to flip
            vs = clause_vars[c_idx]
            # Weight by gain + 1 (to avoid zero weights)
            weights = [max(1, gains[v]) for v in vs]
            total_w = sum(weights)
            r = rng.random() * total_w
            cumulative = 0
            chosen_v = vs[0]
            for i, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    chosen_v = vs[i]
                    break
            
            # Flip
            old_val = assignment[chosen_v]
            assignment[chosen_v] = not assignment[chosen_v]
            
            # Update unsat status
            update_unsat_status(assignment, chosen_v, old_val, unsat_set, [False]*m) # Re-init sat list? No, we don't use it in update.
            # Wait, my update function signature was a bit messy. Let's simplify update.
            
            # Let's re-implement the update logic inline for clarity and speed.
            # Actually, let's just re-calculate unsat_set from scratch if it's small?
            # No, incremental is better.
            
            # Let's refine the update function.
            # We need to know which clauses changed status.
            # Clauses containing chosen_v are the only ones that can change.
            
            # Re-evaluate clauses containing chosen_v
            for c_idx_2 in var_to_clauses[chosen_v]:
                is_sat = check_clause(c_idx_2, assignment)
                if is_sat:
                    unsat_set.discard(c_idx_2)
                else:
                    unsat_set.add(c_idx_2)
                    
            if len(unsat_set) < best_unsat_in_attempt:
                best_unsat_in_attempt = len(unsat_set)
                best_assignment = assignment[:]
                
        return False, best_assignment

    # Main Loop with Restarts
    best_assignment_overall = None
    best_unsat_overall = m + 1
    
    max_restarts = 100
    base_steps = 2000
    
    for restart in range(max_restarts):
        if time.time() >= deadline:
            break
            
        # Initialization
        if best_assignment_overall is not None and rng.random() < 0.5:
            assignment = best_assignment_overall[:]
            # Perturb slightly
            for _ in range(3):
                v = rng.randint(0, n-1)
                assignment[v] = not assignment[v]
        else:
            assignment = [rng.random() < 0.5 for _ in range(n)]
            
        solved, assignment = walksat_attempt(assignment, base_steps, rng)
        
        if solved:
            return assignment
            
        unsat_count = len(get_unsat_clauses(assignment))
        if unsat_count < best_unsat_overall:
            best_unsat_overall = unsat_count
            best_assignment_overall = assignment[:]
            
        # Increase steps
        base_steps = int(base_steps * 1.2)
        
    if best_assignment_overall:
        return best_assignment_overall
    return assignment
