import time
import random
import sys

sys.setrecursionlimit(10000)

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard: 280 seconds to be safe
    deadline = time.time() + 280
    
    # DPLL Solver with Watched Literals and Heuristics
    
    # Clause representation: list of literals (signed ints)
    # We will use a list of clauses, each clause is a list of literals.
    # We maintain a list of "watched" literals for each clause.
    
    # assignment: -1 (unassigned), 0 (false), 1 (true)
    assignment = [-1] * n
    
    # To support backtracking, we need to be able to undo assignments.
    # We will use a trail of assignments.
    trail = []
    
    # Watched literals structure:
    # For each literal, we need to know which clauses watch it.
    # However, standard watched literals implementation is complex to backtrack.
    # Given n=50 and small m, a simpler DPLL with unit propagation and pure literal
    # heuristics, rebuilding the formula or using a simpler clause list might be faster to implement correctly
    # and sufficient.
    
    # Let's implement a clean DPLL with:
    # 1. Unit Propagation
    # 2. Pure Literal Heuristic
    # 3. Heuristic variable selection (Degree or VSIDS-like)
    
    # To make it fast and robust for n=50, we can use a recursive DPLL that copies the state.
    # Since n is small, copying lists of length m and n is cheap.
    
    def simplify(clauses, assignment):
        """
        Applies unit propagation and pure literal elimination.
        Returns (new_clauses, new_assignment, satisfied, unsat)
        satisfied: True if all clauses are satisfied
        unsat: True if a contradiction is found
        """
        # We work with a list of clauses.
        # A clause is a list of literals.
        # If a clause becomes empty, it's unsat.
        # If a clause has a true literal, it's satisfied and can be removed.
        # If a clause has one unassigned literal, it's a unit clause.
        
        changed = True
        while changed:
            changed = False
            new_clauses = []
            unit_clauses = []
            
            # Filter clauses based on current assignment
            for clause in clauses:
                new_clause = []
                satisfied = False
                for lit in clause:
                    var = abs(lit) - 1
                    val = assignment[var]
                    if val == 1:
                        if lit > 0:
                            satisfied = True
                            break
                    elif val == 0:
                        if lit < 0:
                            satisfied = True
                            break
                    # If unassigned, keep it
                    new_clause.append(lit)
                
                if satisfied:
                    continue
                
                if len(new_clause) == 0:
                    return None, None, False, True # UNSAT
                
                if len(new_clause) == 1:
                    unit_clauses.append(new_clause[0])
                else:
                    new_clauses.append(new_clause)
            
            if not unit_clauses:
                break
                
            # Process unit clauses
            # Note: Multiple unit clauses might assign the same variable or conflicting variables.
            # We need to handle them carefully.
            # Since we are in a loop, we can just assign one, add to assignment, and restart the loop.
            # But if there are multiple units, we should assign all non-conflicting ones.
            
            assigned_in_this_round = {}
            conflict = False
            for lit in unit_clauses:
                var = abs(lit) - 1
                # Determine required value
                if lit > 0:
                    required_val = 1
                else:
                    required_val = 0
                    
                if var in assigned_in_this_round:
                    if assigned_in_this_round[var] != required_val:
                        conflict = True
                        break
                else:
                    if assignment[var] != -1:
                        if assignment[var] != required_val:
                            conflict = True
                            break
                    assigned_in_this_round[var] = required_val
            
            if conflict:
                return None, None, False, True
                
            if assigned_in_this_round:
                changed = True
                for var, val in assigned_in_this_round.items():
                    assignment[var] = val
                    trail.append(var)
            
            clauses = new_clauses
            
        # Check for Pure Literals
        # A literal is pure if its negation does not appear in any clause.
        # We can identify pure literals and assign them.
        
        # Build a set of all literals present in the current clauses
        lits_present = set()
        for clause in clauses:
            for lit in clause:
                lits_present.add(lit)
                
        pure_lits = []
        for lit in lits_present:
            if -lit not in lits_present:
                pure_lits.append(lit)
                
        if pure_lits:
            changed = True
            # Assign all pure literals
            # Check for conflicts among pure literals (shouldn't happen if logic is sound, 
            # but a var might appear as pure positive and another clause forces it? 
            # No, pure literal heuristic is safe. If a var is pure, assigning it true 
            # satisfies all clauses containing it and doesn't falsify any clause containing its negation 
            # because its negation doesn't exist in the formula.
            # So we can assign all of them.
            for lit in pure_lits:
                var = abs(lit) - 1
                val = 1 if lit > 0 else 0
                if assignment[var] == -1:
                    assignment[var] = val
                    trail.append(var)
            # We need to re-simplify because assignments changed clause status
            return simplify(clauses, assignment)
            
        # No more simplification
        # Check if all clauses are satisfied (i.e., clauses list is empty)
        if not clauses:
            return clauses, assignment, True, False
            
        return clauses, assignment, False, False

    def select_variable(clauses, assignment):
        """
        Select a variable to branch on.
        Heuristic: Choose the variable appearing in the most clauses (Degree Heuristic).
        Or, for 3-SAT, a simple random selection from unassigned variables in clauses is often good.
        Let's use a weighted random selection based on frequency.
        """
        var_freq = [0] * n
        for clause in clauses:
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == -1:
                    var_freq[var] += 1
        
        total_freq = sum(var_freq)
        if total_freq == 0:
            return -1 # No unassigned variables in clauses? Should be satisfied.
            
        # Randomized selection weighted by frequency
        # To avoid bias, we can just pick a random one from those with max freq, or weighted.
        # Let's pick randomly from unassigned variables appearing in clauses.
        candidates = [i for i in range(n) if var_freq[i] > 0]
        if not candidates:
            return -1
            
        # Simple: pick the one with max freq, break ties randomly
        max_freq = max(var_freq[i] for i in candidates)
        top_candidates = [i for i in candidates if var_freq[i] == max_freq]
        return random.choice(top_candidates)

    def dpll(clauses, assignment, trail_len):
        """
        Recursive DPLL.
        Returns assignment list if solved, None if UNSAT.
        """
        if time.time() > deadline:
            return None # Timeout, let caller handle best-effort if needed, but for exact solver we just fail.
            
        # 1. Simplify (Unit Prop + Pure Literal)
        simplified_clauses, simplified_assignment, is_sat, is_unsat = simplify(clauses, assignment)
        
        if is_unsat:
            return None
            
        if is_sat:
            return simplified_assignment
            
        clauses = simplified_clauses
        assignment = simplified_assignment
        
        # 2. Select variable
        var = select_variable(clauses, assignment)
        if var == -1:
            # No variables left? If clauses are not empty, this is a contradiction.
            # But simplify should have handled units.
            # If clauses remain and no vars, it's UNSAT.
            return None
            
        # 3. Branch: Try True then False (or random order)
        
        # Try assignment[var] = 1
        assignment[var] = 1
        trail.append(var)
        
        # Copy clauses for the recursive call? 
        # Since we are modifying 'assignment' in place, and 'simplify' returns a new list of clauses,
        # we can pass the current state.
        # But 'simplify' modifies 'assignment' in place too.
        # We need to be careful with backtracking.
        # The 'simplify' function above modifies 'assignment' and 'trail'.
        # We saved 'trail_len' to restore.
        
        res = dpll(clauses, assignment, trail_len + 1)
        if res is not None:
            return res
            
        # Backtrack
        # Restore assignment for vars assigned in this branch
        # The 'trail' contains all assignments made since the start of this branch.
        # We need to undo assignments made after 'trail_len'.
        # Note: 'simplify' in the recursive call might have added to trail.
        # We need to pop from trail until len(trail) == trail_len.
        while len(trail) > trail_len:
            v = trail.pop()
            assignment[v] = -1
            
        # Try assignment[var] = 0
        assignment[var] = 0
        trail.append(var)
        
        res = dpll(clauses, assignment, trail_len + 1)
        if res is not None:
            return res
            
        # Backtrack again
        while len(trail) > trail_len:
            v = trail.pop()
            assignment[v] = -1
            
        return None

    # Initial call
    # Make a copy of clauses to avoid modifying the input if we were to retry, though we only run once.
    init_clauses = [list(c) for c in clauses]
    init_assignment = [-1] * n
    init_trail = []
    
    result = dpll(init_clauses, init_assignment, 0)
    
    if result is not None:
        # Convert to list of booleans
        return [bool(x) for x in result]
    else:
        # If DPLL fails (shouldn't for SAT instances, but maybe timeout or bug), 
        # fall back to a random assignment or the best found?
        # Since the task says "satisfiable", DPLL should find it.
        # If it times out, we return a random guess to avoid crashing.
        return [random.random() < 0.5 for _ in range(n)]
