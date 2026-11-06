import time
import sys
import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline: 290 seconds to be safe (330s limit, leave 40s margin)
    deadline = time.time() + 290
    
    # Pre-process clauses to use 0-based internal indices
    # Store as (var_idx, is_positive) for faster comparison
    internal_clauses = []
    for clause in clauses:
        int_clause = []
        for lit in clause:
            v = abs(lit) - 1
            p = lit > 0
            int_clause.append((v, p))
        internal_clauses.append(int_clause)
        
    m = len(internal_clauses)
    
    # Build variable to clause incidence for faster lookups
    var_clauses = [[] for _ in range(n)]
    for c_idx, clause in enumerate(internal_clauses):
        for v, p in clause:
            var_clauses[v].append(c_idx)
    
    # Helper to check if a clause is satisfied
    def is_satisfied(clause, assignment):
        for v, p in clause:
            val = assignment[v]
            if val is not None:
                if (val == True and p == True) or (val == False and p == False):
                    return True
        return False
        
    # Helper to check if a clause is falsified
    def is_falsified(clause, assignment):
        for v, p in clause:
            val = assignment[v]
            if val is None:
                return False
            if (val == True and p == False) or (val == False and p == True):
                continue
            else:
                return False
        return True

    # Fast unit propagation using worklist
    def propagate_fast(assignment, clauses):
        """
        Returns (conflict_bool). Modifies assignment in place.
        Uses a queue of clauses to check.
        """
        # Track unassigned count per clause
        # For each clause, count how many literals are unassigned
        clause_unassigned_count = [0] * len(clauses)
        clause_satisfied = [False] * len(clauses)
        
        # Initialize counts
        for c_idx, clause in enumerate(clauses):
            for v, p in clause:
                if assignment[v] is None:
                    clause_unassigned_count[c_idx] += 1
                else:
                    val = assignment[v]
                    if (val == True and p == True) or (val == False and p == False):
                        clause_satisfied[c_idx] = True
                        break
        
        # Queue of clauses to process (those with 0 or 1 unassigned)
        from collections import deque
        queue = deque()
        for c_idx in range(len(clauses)):
            if clause_satisfied[c_idx]:
                continue
            if clause_unassigned_count[c_idx] <= 1:
                queue.append(c_idx)
        
        while queue:
            c_idx = queue.popleft()
            if clause_satisfied[c_idx]:
                continue
                
            clause = clauses[c_idx]
            
            # Count unassigned and find the single unassigned literal
            unassigned_count = 0
            single_unassigned = None
            for v, p in clause:
                val = assignment[v]
                if val is None:
                    unassigned_count += 1
                    single_unassigned = (v, p)
                else:
                    if (val == True and p == True) or (val == False and p == False):
                        clause_satisfied[c_idx] = True
                        break
            
            if clause_satisfied[c_idx]:
                continue
                
            if unassigned_count == 0:
                return True  # Conflict
            
            if unassigned_count == 1:
                v, p = single_unassigned
                new_val = p  # True if positive literal, False if negative
                
                # If already assigned, check for conflict
                if assignment[v] is not None:
                    if assignment[v] != new_val:
                        return True  # Conflict
                    else:
                        continue
                
                # Assign
                assignment[v] = new_val
                
                # Update all clauses containing v
                for other_c_idx in var_clauses[v]:
                    if clause_satisfied[other_c_idx]:
                        continue
                    # Check if this clause is now satisfied
                    other_clause = clauses[other_c_idx]
                    for vv, pp in other_clause:
                        if vv == v:
                            val = assignment[v]
                            if (val == True and pp == True) or (val == False and pp == False):
                                clause_satisfied[other_c_idx] = True
                                break
                    if clause_satisfied[other_c_idx]:
                        continue
                    
                    # Count unassigned in this clause
                    unassigned = 0
                    for vv, pp in other_clause:
                        if assignment[vv] is None:
                            unassigned += 1
                        else:
                            val = assignment[vv]
                            if (val == True and pp == True) or (val == False and pp == False):
                                clause_satisfied[other_c_idx] = True
                                break
                    if clause_satisfied[other_c_idx]:
                        continue
                    if unassigned <= 1:
                        queue.append(other_c_idx)
        
        return False  # No conflict

    # Pure literal elimination (simplified, only do once at start)
    def pure_literal_elimination(assignment, clauses):
        changed = True
        while changed:
            changed = False
            for v in range(n):
                if assignment[v] is not None:
                    continue
                pos = False
                neg = False
                for c_idx in var_clauses[v]:
                    clause = clauses[c_idx]
                    # Check if any other literal in this clause is already satisfied
                    already_sat = False
                    for vv, pp in clause:
                        if vv == v:
                            continue
                        val = assignment[vv]
                        if val is not None:
                            if (val == True and pp == True) or (val == False and pp == False):
                                already_sat = True
                                break
                    if already_sat:
                        continue
                    for vv, pp in clause:
                        if vv == v:
                            if pp:
                                pos = True
                            else:
                                neg = True
                            break
                if pos and not neg:
                    assignment[v] = True
                    changed = True
                elif neg and not pos:
                    assignment[v] = False
                    changed = True
    
    # Variable selection: pick from shortest unsatisfied clause
    def select_variable(assignment, clauses):
        best_clause_idx = -1
        best_len = float('inf')
        for c_idx, clause in enumerate(clauses):
            if is_satisfied(clause, assignment):
                continue
            unassigned_count = 0
            for v, p in clause:
                if assignment[v] is None:
                    unassigned_count += 1
            if unassigned_count < best_len:
                best_len = unassigned_count
                best_clause_idx = c_idx
                if best_len == 1:
                    break
        if best_clause_idx == -1:
            return None
        clause = clauses[best_clause_idx]
        for v, p in clause:
            if assignment[v] is None:
                return v
        return None

    # DPLL with fast propagation
    sys.setrecursionlimit(10000)
    
    def dpll(assignment, clauses):
        if time.time() > deadline:
            return None
            
        # Propagate
        conflict = propagate_fast(assignment, clauses)
        if conflict:
            return None
            
        # Check if all satisfied
        all_sat = True
        for c in clauses:
            if not is_satisfied(c, assignment):
                if is_falsified(c, assignment):
                    return None
                all_sat = False
                break
        
        if all_sat:
            for i in range(n):
                if assignment[i] is None:
                    assignment[i] = False
            return assignment
            
        # Select variable
        v = select_variable(assignment, clauses)
        if v is None:
            return None

        # Branch 1: v = True
        old_val = assignment[v]
        assignment[v] = True
        result = dpll(assignment, clauses)
        if result is not None:
            return result
        
        # Branch 2: v = False
        assignment[v] = False
        result = dpll(assignment, clauses)
        if result is not None:
            return result
            
        # Backtrack
        assignment[v] = old_val
        return None

    # Initial Assignment
    initial_assignment = [None] * n
    
    # Try DPLL with time limit
    dpll_deadline = time.time() + 60  # Give DPLL 60 seconds
    old_deadline = deadline
    deadline = dpll_deadline
    
    result = dpll(initial_assignment, internal_clauses)
    
    deadline = old_deadline
    
    if result is not None:
        return result
        
    # Fallback: Improved WalkSAT
    rng = random.Random(42)
    
    def walksat_improved(assignment, clauses, max_steps):
        """WalkSAT with frequency-based variable selection"""
        # Precompute clause satisfaction status
        clause_sat = [False] * len(clauses)
        for c_idx, clause in enumerate(clauses):
            clause_sat[c_idx] = is_satisfied(clause, assignment)
        
        unsat_count = sum(1 for s in clause_sat if not s)
        
        for step in range(max_steps):
            if time.time() > deadline:
                return assignment, unsat_count
            
            if unsat_count == 0:
                return assignment, 0
            
            # Find unsatisfied clauses
            unsat_indices = [i for i, s in enumerate(clause_sat) if not s]
            clause_idx = rng.choice(unsat_indices)
            clause = clauses[clause_idx]
            
            # Calculate frequency of each variable in unsat clauses
            freq = {}
            for u_idx in unsat_indices:
                for v, p in clauses[u_idx]:
                    freq[v] = freq.get(v, 0) + 1
            
            # Score each variable in the clause
            best_var = None
            best_score = -float('inf')
            
            for v, p in clause:
                # Current contribution: this variable is in unsat clause
                # If we flip, how many clauses become satisfied?
                # How many become unsatisfied?
                
                # Clauses containing v that are currently unsat
                gain = 0
                loss = 0
                
                for c_idx in var_clauses[v]:
                    if clause_sat[c_idx]:
                        # Currently satisfied, might become unsat
                        c = clauses[c_idx]
                        # Check if v is the only satisfying literal
                        only_satisfying = True
                        for vv, pp in c:
                            if vv == v:
                                continue
                            val = assignment[vv]
                            if val is not None:
                                if (val == True and pp == True) or (val == False and pp == False):
                                    only_satisfying = False
                                    break
                        if only_satisfying:
                            loss += 1
                    else:
                        # Currently unsat, might become sat
                        c = clauses[c_idx]
                        # Check if flipping v would satisfy this clause
                        # i.e., v is currently falsifying its literal
                        for vv, pp in c:
                            if vv == v:
                                val = assignment[v]
                                # If current val makes literal false, flipping makes it true
                                if (val == True and pp == False) or (val == False and pp == True):
                                    gain += 1
                                break
                
                score = gain - loss
                
                # Add small random noise
                score += rng.gauss(0, 0.1)
                
                if score > best_score:
                    best_score = score
                    best_var = v
            
            if best_var is None:
                # Fallback: random
                v, p = rng.choice(clause)
                best_var = v
            
            # Flip
            assignment[best_var] = not assignment[best_var]
            
            # Update clause satisfaction
            for c_idx in var_clauses[best_var]:
                clause_sat[c_idx] = is_satisfied(clauses[c_idx], assignment)
            
            # Recount unsat
            old_unsat = unsat_count
            unsat_count = sum(1 for s in clause_sat if not s)
            
            # Early exit if solved
            if unsat_count == 0:
                return assignment, 0
                
        return assignment, unsat_count

    # Try WalkSAT with multiple restarts
    best_result = None
    best_unsat = m
    
    for restart in range(500):
        if time.time() > deadline - 5:
            break
            
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Quick check
        if all(is_satisfied(c, assignment) for c in internal_clauses):
            return assignment
            
        result, unsat = walksat_improved(assignment, internal_clauses, 5000)
        
        if unsat == 0:
            return result
            
        if unsat < best_unsat:
            best_unsat = unsat
            best_result = result
            
        # If we're making progress, continue; else restart sooner
        if restart > 50 and unsat >= best_unsat:
            continue
            
    # Return best found
    if best_result is not None:
        return best_result
        
    # Last resort: return random assignment
    return [random.random() < 0.5 for _ in range(n)]
