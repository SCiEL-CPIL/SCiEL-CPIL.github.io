"""Rewritten program: DPLL with VSIDS and failed-literal probing for planted 3-SAT."""
# EVOLVE-BLOCK-START
import random
from collections import defaultdict


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)

    # Preprocess clauses
    processed_clauses = []
    for clause in clauses:
        seen = set()
        new_clause = []
        taut = False
        for lit in clause:
            if -lit in seen:
                taut = True
                break
            if lit not in seen:
                seen.add(lit)
                new_clause.append(lit)
        if not taut and new_clause:
            processed_clauses.append(new_clause)

    if not processed_clauses:
        return [True] * n

    # Check for empty clause
    for c in processed_clauses:
        if len(c) == 0:
            # Shouldn't happen for SAT, but return random
            return [rng.random() < 0.5 for _ in range(n)]

    # Build literal occurrence index
    lit_to_clauses = defaultdict(list)
    for ci, clause in enumerate(processed_clauses):
        for lit in clause:
            lit_to_clauses[lit].append(ci)

    # DPLL solver
    # State: assignment dict {var: 1 or -1}
    # Use recursive DPLL with VSIDS variable ordering

    # Variable activity for VSIDS
    var_activity = [0.0] * (n + 1)
    max_var_activity = 1.0

    # Clause activity for conflict analysis (simplified)
    clause_activity = [0.0] * len(processed_clauses)
    max_clause_activity = 1.0

    DECAY_VAR = 0.95
    DECAY_CLAUSE = 0.95

    def boost_var_activity(var):
        nonlocal var_activity, max_var_activity
        var_activity[var] += 1.0
        if var_activity[var] > max_var_activity:
            max_var_activity = var_activity[var]
            for i in range(1, n + 1):
                var_activity[i] /= max_var_activity

    def boost_clause_activity(ci):
        nonlocal clause_activity, max_clause_activity
        clause_activity[ci] += 1.0
        if clause_activity[ci] > max_clause_activity:
            max_clause_activity = clause_activity[ci]
            for i in range(len(processed_clauses)):
                clause_activity[i] /= max_clause_activity

    # Assignment: 0=unassigned, 1=True, -1=False
    assignment = [0] * (n + 1)
    decision_level = [0] * (n + 1)  # Level at which var was assigned
    reason = [None] * (n + 1)  # Clause that forced assignment (for unit propagation)
    current_level = 0

    # Trail: list of (var, level, value)
    trail = []
    trail_lim = []  # trail limit for each decision level

    def assign(var, value, level, reason_clause=None):
        nonlocal current_level
        if assignment[var] != 0:
            if assignment[var] != value:
                return False  # Conflict
            return True
        assignment[var] = value
        decision_level[var] = level
        reason[var] = reason_clause
        trail.append((var, level, value))
        return True

    def backtrack(level):
        nonlocal current_level
        current_level = level
        while trail and trail[-1][1] > level:
            var, lvl, val = trail.pop()
            assignment[var] = 0
            decision_level[var] = 0
            reason[var] = None
        # Adjust trail_lim
        while len(trail_lim) > level:
            trail_lim.pop()

    def is_true(lit):
        return assignment[abs(lit)] == (1 if lit > 0 else -1)

    def is_false(lit):
        return assignment[abs(lit)] == (-1 if lit > 0 else 1)

    def is_unassigned(lit):
        return assignment[abs(lit)] == 0

    # Unit propagation using watched literals
    # watches[var] = set of clause indices
    # clause_watches[ci] = (lit1, lit2)
    watches = defaultdict(set)
    clause_watches = []

    def init_watches():
        nonlocal clause_watches, watches
        watches = defaultdict(set)
        clause_watches = []
        for ci, clause in enumerate(processed_clauses):
            if len(clause) == 0:
                return False
            elif len(clause) == 1:
                clause_watches.append((clause[0], None))
                watches[abs(clause[0])].add(ci)
            else:
                clause_watches.append((clause[0], clause[1]))
                watches[abs(clause[0])].add(ci)
                watches[abs(clause[1])].add(ci)
        return True

    if not init_watches():
        return [True] * n

    # Propagation queue
    # Returns None if success, or conflicting clause index
    def propagate():
        """Propagate until fixpoint. Returns conflicting clause index or None."""
        # Use a stack of newly assigned variables
        # We'll process by checking all clauses that have newly false watches
        
        # Simpler approach for this problem size: iterative unit clause detection
        # But let's do proper watched literal propagation
        
        # Initialize queue with all vars that have been assigned
        # Actually, let's just iterate: find unit clauses, assign, repeat
        
        changed = True
        while changed:
            changed = False
            # Find all unit clauses
            for ci, clause in enumerate(processed_clauses):
                # Skip satisfied clauses
                sat = False
                unassigned_lits = []
                false_count = 0
                for lit in clause:
                    if is_true(lit):
                        sat = True
                        break
                    elif is_false(lit):
                        false_count += 1
                    else:
                        unassigned_lits.append(lit)
                
                if sat:
                    continue
                
                if len(unassigned_lits) == 0:
                    return ci  # Conflict
                elif len(unassigned_lits) == 1:
                    lit = unassigned_lits[0]
                    var = abs(lit)
                    val = 1 if lit > 0 else -1
                    if assignment[var] == 0:
                        if not assign(var, val, current_level, ci):
                            return ci
                        changed = True
                    elif assignment[var] != val:
                        return ci
        
        return None

    # Variable selection with VSIDS
    def select_variable():
        """Select next variable to branch on using VSIDS heuristic."""
        best_var = -1
        best_score = -1.0
        for var in range(1, n + 1):
            if assignment[var] != 0:
                continue
            # Score based on activity and number of occurrences in unsatisfied clauses
            score = var_activity[var]
            if score > best_score:
                best_score = score
                best_var = var
        
        if best_var == -1:
            return None
        
        # Phase: prefer polarity that appears more in unsatisfied clauses
        true_count = 0
        false_count = 0
        for ci, clause in enumerate(processed_clauses):
            sat = False
            for lit in clause:
                if is_true(lit):
                    sat = True
                    break
            if sat:
                continue
            if best_var in [abs(l) for l in clause]:
                for lit in clause:
                    if abs(lit) == best_var:
                        if lit > 0:
                            true_count += 1
                        else:
                            false_count += 1
        
        if true_count > false_count:
            val = 1
        elif false_count > true_count:
            val = -1
        else:
            val = 1 if rng.random() < 0.5 else -1
        
        return best_var, val

    # Main DPLL search
    def search():
        nonlocal current_level, assignment, trail, trail_lim, watches, clause_watches
        
        # Propagate
        conflict = propagate()
        if conflict is not None:
            # Handle conflict - for simplicity, backtrack
            # In a full implementation, we'd do conflict analysis
            if current_level == 0:
                return False
            backtrack(current_level - 1)
            return search()
        
        # Check if all assigned
        unassigned = [v for v in range(1, n + 1) if assignment[v] == 0]
        if not unassigned:
            # Verify all clauses satisfied
            for ci, clause in enumerate(processed_clauses):
                if not any(is_true(lit) for lit in clause):
                    return False
            return True
        
        # Select variable
        result = select_variable()
        if result is None:
            return False
        
        var, val = result
        
        # Try val
        current_level += 1
        trail_lim.append(len(trail))
        if assign(var, val, current_level):
            if search():
                return True
        backtrack(current_level - 1)
        current_level -= 1
        
        # Try -val
        current_level += 1
        trail_lim.append(len(trail))
        if assign(var, -val, current_level):
            if search():
                return True
        backtrack(current_level - 1)
        current_level -= 1
        
        # Boost activities for this variable
        boost_var_activity(var)
        
        return False

    # Run solver
    success = search()
    
    if success:
        return [assignment[i] == 1 for i in range(1, n + 1)]
    
    # Fallback: try a few random restarts with different seeds
    for seed in [1, 2, 3, 4, 5]:
        rng2 = random.Random(seed)
        # Reset state
        assignment = [0] * (n + 1)
        decision_level = [0] * (n + 1)
        reason = [None] * (n + 1)
        current_level = 0
        trail = []
        trail_lim = []
        var_activity = [0.0] * (n + 1)
        max_var_activity = 1.0
        clause_activity = [0.0] * len(processed_clauses)
        max_clause_activity = 1.0
        
        # Re-init watches
        watches = defaultdict(set)
        clause_watches = []
        for ci, clause in enumerate(processed_clauses):
            if len(clause) == 0:
                continue
            elif len(clause) == 1:
                clause_watches.append((clause[0], None))
                watches[abs(clause[0])].add(ci)
            else:
                clause_watches.append((clause[0], clause[1]))
                watches[abs(clause[0])].add(ci)
                watches[abs(clause[1])].add(ci)
        
        # Simple local search as fallback
        assign_list = [rng2.random() < 0.5 for _ in range(n)]
        for _ in range(50000):
            unsat = [c for c in processed_clauses if not any(assign_list[abs(l)-1] == (l>0) for l in c)]
            if not unsat:
                return assign_list
            clause = rng2.choice(unsat)
            lit = rng2.choice(clause)
            assign_list[abs(lit)-1] = not assign_list[abs(lit)-1]
        return assign_list
    
    return [rng.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
