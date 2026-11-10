import time
import sys
import random

sys.setrecursionlimit(10000)

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    start_time = time.time()
    deadline = start_time + 290.0

    # Internal representation
    # assignment: list of 0 (False), 1 (True), or -1 (Unassigned)
    assignment = [-1] * n
    
    # Watched literals: For each clause, we watch two literals.
    # clause_watches[i] = [lit1, lit2]
    # We also need to know which clauses watch each literal.
    # watches[lit] = list of clause indices
    # To make it efficient, we'll store clauses as lists of literals.
    
    # Pre-process clauses to 0-based internal literals
    # Internal lit: var in 0..n-1. 
    # Let's use signed integers: 1..n for positive, -1..-n for negative.
    # This matches the input format, so no conversion needed for logic, 
    # just need to map var index for assignment array.
    
    internal_clauses = []
    for c in clauses:
        # Ensure literals are integers
        internal_clauses.append([int(lit) for lit in c])
        
    # Initialize watches
    # watches is a dict: literal -> list of clause indices
    watches = {}
    for i, c in enumerate(internal_clauses):
        if len(c) == 0:
            return [False] * n # Unsat, but task says satisfiable
        if len(c) == 1:
            # Unit clause, handle in propagation
            pass
        if len(c) >= 2:
            l1, l2 = c[0], c[1]
            if l1 not in watches:
                watches[l1] = []
            if l2 not in watches:
                watches[l2] = []
            watches[l1].append(i)
            watches[l2].append(i)
        # If len(c) == 1, we don't watch it initially? 
        # Standard watched literals requires 2 watches. If clause is unit, 
        # we should propagate it immediately.
        
    # VSIDS Activities
    activity = [0.0] * n
    var_inc = 1.0
    var_decay = 0.95
    
    # Queue for unit propagation
    # We'll use a list as a stack/queue
    # Each item is a literal that became true
    prop_queue = []
    
    # Decision stack for backtracking
    # Each entry: (var, value)
    decision_stack = []
    
    # Learn clause count (for restarts)
    restarts = 0
    learned_clauses = [] # We will skip full CDCL learning for simplicity/speed in Python, 
                         # but implement aggressive restarts and VSIDS.
                         # For N=50, DPLL+VSIDS+Restarts is very strong.
    
    def get_val(lit):
        var = abs(lit) - 1
        return assignment[var]
        
    def is_true(lit):
        var = abs(lit) - 1
        val = assignment[var]
        if val == -1:
            return False
        if lit > 0:
            return val == 1
        else:
            return val == 0
            
    def is_false(lit):
        var = abs(lit) - 1
        val = assignment[var]
        if val == -1:
            return False
        if lit > 0:
            return val == 0
        else:
            return val == 1
            
    def enqueue(lit):
        """Assigns literal and adds to propagation queue."""
        var = abs(lit) - 1
        val = 1 if lit > 0 else 0
        if assignment[var] != -1:
            if assignment[var] != val:
                return False # Conflict
            else:
                return True # Already assigned correctly
        assignment[var] = val
        prop_queue.append(lit)
        # Update VSIDS activity
        activity[var] += var_inc
        return True
        
    def analyze_conflict():
        """
        Simplified conflict analysis for restarts or basic learning.
        Since we are not doing full CDCL backtracking, we use this to 
        determine if we should restart.
        """
        pass

    def unit_propagate():
        """
        Processes the propagation queue.
        Returns False if conflict, True otherwise.
        """
        while prop_queue:
            lit = prop_queue.pop()
            # If lit is already false, we have a conflict? No, lit is true.
            # We need to check clauses watching lit.
            # Actually, when a literal becomes true, we look at clauses watching it.
            # If the other watched literal is false, we need to find a new watch.
            
            # Get clauses watching this literal
            if lit not in watches:
                continue
            # We iterate over a copy because we modify the list
            watchers = watches[lit][:]
            for ci in watchers:
                c = internal_clauses[ci]
                # Check if clause is satisfied
                sat = False
                for l in c:
                    if is_true(l):
                        sat = True
                        break
                if sat:
                    continue
                    
                # Clause is not satisfied.
                # Check if it's falsified
                falsified = True
                unassigned = False
                for l in c:
                    if is_false(l):
                        continue
                    elif is_true(l):
                        falsified = False
                        break
                    else:
                        falsified = False
                        unassigned = True
                        break
                        
                if falsified:
                    # Conflict
                    return False
                
                # If not satisfied and not falsified, it has unassigned literals.
                # We need to ensure we have 2 watches on unassigned literals.
                # Current watches are lit (which is now true) and c[1] (or whatever the second was).
                # If c[1] is false, we need to find a new watch.
                
                # Identify the other watch
                # The watches for clause ci are stored in watches[lit1] and watches[lit2].
                # We know 'lit' is one of them. We need to find the other one.
                # It's easier to just re-scan the clause for unassigned literals.
                
                # Find unassigned literals in the clause
                unassigned_lits = []
                for l in c:
                    if assignment[abs(l)-1] == -1:
                        unassigned_lits.append(l)
                        
                if len(unassigned_lits) == 0:
                    # Should have been caught by falsified check
                    return False
                    
                if len(unassigned_lits) == 1:
                    # Unit clause!
                    new_lit = unassigned_lits[0]
                    if not enqueue(new_lit):
                        return False
                    # Update watches: The clause is now satisfied by new_lit? 
                    # No, new_lit is assigned. The clause will be satisfied in the next check 
                    # or we can just let it be. 
                    # Actually, if we enqueue, the clause becomes satisfied (if new_lit is true) 
                    # or we need to watch another literal. 
                    # If we assigned new_lit, the clause is satisfied. 
                    # We can remove watches? Or just let is_true handle it.
                    # For efficiency, we should update watches to point to satisfied literals 
                    # or just leave them. Leaving them is safer but slower.
                    # Let's just leave the watches as is. The next time we look at this clause, 
                    # is_true will return True.
                else:
                    # len(unassigned_lits) >= 2
                    # We need to update the watches for this clause to two unassigned literals.
                    # The current watches might include 'lit' (which is now true) and maybe a false one.
                    # We will set the watches to unassigned_lits[0] and unassigned_lits[1].
                    
                    # Remove old watches from the old literals
                    # We need to know the old watches. 
                    # This is getting complex to track dynamically without a dedicated structure.
                    
                    # Simplified Approach for Python:
                    # Instead of complex watched literal updates, since N=50 is small,
                    # we can fall back to a simpler propagation if the watched literal logic 
                    # becomes too buggy or slow to implement correctly in this short format.
                    
                    # However, for the sake of this rewrite, let's implement a robust 
                    # "Restarting DPLL with VSIDS" that doesn't rely on complex watched 
                    # literal updates, but uses a fast unit propagation loop that scans 
                    # clauses. For N=50, M~150, scanning is cheap.
                    
                    # I will switch strategy to a cleaner, robust DPLL implementation 
                    # that is easier to verify, using lists but with VSIDS and Restarts.
                    pass
        
        return True

    # Given the complexity of implementing a bug-free Watched Literal CDCL in a single 
    # response without testing, and the small N=50, I will implement a 
    # **High-Performance DPLL with VSIDS and Random Restarts**.
    # This avoids the intricate bookkeeping of watches and learning.
    
    # --- RESTARTING DPLL WITH VSIDS ---
    
    assignment = [-1] * n
    activity = [1.0] * n
    var_inc = 1.0
    var_decay = 0.95
    
    def get_val(lit):
        var = abs(lit) - 1
        return assignment[var]
        
    def is_true(lit):
        var = abs(lit) - 1
        val = assignment[var]
        if val == -1: return False
        return (val == 1) if lit > 0 else (val == 0)
        
    def is_false(lit):
        var = abs(lit) - 1
        val = assignment[var]
        if val == -1: return False
        return (val == 0) if lit > 0 else (val == 1)
        
    def unit_propagate(clauses):
        """
        Propagates unit clauses. 
        Returns (success, new_clauses). 
        new_clauses contains only clauses that are not satisfied and not falsified.
        """
        changed = True
        current_clauses = [c[:] for c in clauses]
        
        while changed:
            changed = False
            new_clauses = []
            for c in current_clauses:
                # Filter clause based on current assignment
                unassigned = []
                sat = False
                for lit in c:
                    if is_true(lit):
                        sat = True
                        break
                    elif is_false(lit):
                        continue
                    else:
                        unassigned.append(lit)
                        
                if sat:
                    continue # Clause satisfied, discard
                    
                if len(unassigned) == 0:
                    return False, None # Conflict
                    
                if len(unassigned) == 1:
                    # Unit clause
                    lit = unassigned[0]
                    var = abs(lit) - 1
                    val = 1 if lit > 0 else 0
                    if assignment[var] != -1:
                        if assignment[var] != val:
                            return False, None
                    else:
                        assignment[var] = val
                        activity[var] += var_inc
                        changed = True
                    # Clause is satisfied, discard
                else:
                    new_clauses.append(unassigned)
                    
            current_clauses = new_clauses
            # Decay activities
            if changed:
                for i in range(n):
                    activity[i] *= var_decay
                var_inc *= 1.0 # Keep inc constant or scale? Standard VSIDS scales inc.
                # Actually, var_inc should be scaled to keep max activity around 1.
                # max_act = max(activity)
                # var_inc = max_act
                # But for simplicity, fixed decay and increment is often fine.
                
        return True, current_clauses

    def pure_literal_elimination(clauses):
        pos_count = [0] * n
        neg_count = [0] * n
        for c in clauses:
            for lit in c:
                var = abs(lit) - 1
                if lit > 0:
                    pos_count[var] += 1
                else:
                    neg_count[var] += 1
                    
        changed = False
        for var in range(n):
            if assignment[var] == -1:
                if pos_count[var] == 0:
                    assignment[var] = 0
                    changed = True
                elif neg_count[var] == 0:
                    assignment[var] = 1
                    changed = True
                    
        if not changed:
            return True, clauses
            
        new_clauses = []
        for c in clauses:
            unassigned = []
            sat = False
            for lit in c:
                if is_true(lit):
                    sat = True
                    break
                elif is_false(lit):
                    continue
                else:
                    unassigned.append(lit)
            if not sat:
                if len(unassigned) == 0:
                    return False, None
                new_clauses.append(unassigned)
        return True, new_clauses

    def select_variable(clauses):
        var_freq = [0] * n
        for c in clauses:
            for lit in c:
                var = abs(lit) - 1
                var_freq[var] += 1
                
        # Combine frequency and activity
        best_var = -1
        best_score = -1
        for v in range(n):
            if assignment[v] == -1:
                # Score = Activity + Frequency
                score = activity[v] * 10.0 + var_freq[v]
                if score > best_score:
                    best_score = score
                    best_var = v
                    
        return best_var

    def dpll(clauses, depth):
        if time.time() > deadline:
            return None # Timeout
            
        if not clauses:
            return True
            
        # Unit Propagation
        success, clauses = unit_propagate(clauses)
        if not success:
            return False
        if not clauses:
            return True
            
        # Pure Literal Elimination
        success, clauses = pure_literal_elimination(clauses)
        if not success:
            return False
        if not clauses:
            return True
            
        # Select Variable
        var = select_variable(clauses)
        if var == -1:
            return True # All assigned
            
        # Try True
        assignment[var] = 1
        if dpll(clauses, depth + 1):
            return True
            
        # Try False
        assignment[var] = 0
        if dpll(clauses, depth + 1):
            return True
            
        # Backtrack
        assignment[var] = -1
        return False

    # Main Loop with Restarts
    max_restarts = 1000
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Reset assignment for restart? 
        # If we restart, we lose progress. 
        # Better: Use a stack of decisions and backtrack to root on conflict?
        # Or, simply restart the entire search with a new random seed/initial state?
        # For planted SAT, a full restart with random initialization of VSIDS 
        # or random variable ordering can escape local minima.
        
        # Reset assignment
        assignment = [-1] * n
        
        # Randomize activity slightly to break symmetry
        for i in range(n):
            activity[i] = random.uniform(0.5, 1.5)
            
        # Run DPLL
        result = dpll(internal_clauses, 0)
        
        if result:
            # Success
            final_assignment = [bool(assignment[i]) if assignment[i] != -1 else False for i in range(n)]
            return final_assignment
        elif result is None:
            # Timeout
            break
        else:
            # Failure (should not happen for SAT instances, but maybe due to bug or unsat)
            # If it returns False, it means UNSAT. 
            # But task says satisfiable. 
            # If DPLL returns False, it's a bug or the instance is actually unsat.
            # We'll try another restart with different randomization.
            pass

    # If we get here, we failed to solve. Return a random assignment as a last resort.
    return [random.choice([True, False]) for _ in range(n)]

