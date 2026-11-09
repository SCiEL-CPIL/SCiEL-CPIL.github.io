import random
import time
import sys

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290.0  # Safety margin
    
    # Convert to 0-based signed literals
    # Literal l: 1-based signed. 
    # Internal: (var, pos) where var is 0..n-1, pos is True/False
    # Or just use integers: var + 1 if positive, -(var+1) if negative? 
    # Let's stick to (var, pos) for clarity in logic, or just signed integers for speed.
    # Signed integers: 1..n is positive, -1..-n is negative.
    # Let's use signed integers for internal representation to make watched literals easier to index?
    # No, watched literals need to map literal -> clauses.
    # Literal index: 0..2n-1. 
    # var v (0-indexed), pos p (0/1).
    # lit_index = v * 2 + p. 
    # neg_lit_index = v * 2 + (1-p).
    
    internal_clauses = []
    for c in clauses:
        ic = []
        for l in c:
            v = abs(l) - 1
            p = 1 if l > 0 else 0
            ic.append(v * 2 + p)
        internal_clauses.append(ic)
        
    # --- DPLL with Watched Literals ---
    
    def dpll():
        # assignment: list of size 2n? No, assignment per variable.
        # assign[v] = -1 (unset), 0 (False), 1 (True)
        assign = [-1] * n
        trail = [] # List of (var, value) in order of assignment
        
        # watched: list of size 2n, each element is a list of clause indices
        # lit_idx = v * 2 + pos
        watched = [[] for _ in range(2 * n)]
        
        # Initialize watched literals
        # For each clause, pick two literals to watch
        # If clause has < 2 literals, handle specially? 3-SAT so always 3.
        for i, clause in enumerate(internal_clauses):
            if len(clause) >= 2:
                watched[clause[0]].append(i)
                watched[clause[1]].append(i)
            elif len(clause) == 1:
                # Unit clause
                l = clause[0]
                v = l // 2
                p = l % 2
                if assign[v] == -1:
                    assign[v] = p
                    trail.append((v, p))
                    # Need to propagate this!
                    # For now, just add to a queue
                elif assign[v] != p:
                    return False, None # Conflict
            else:
                return False, None # Empty clause
        
        # We need a proper propagation loop. 
        # Let's restructure DPLL to use a queue.
        
        def propagate(queue):
            # queue contains literal indices that were just assigned
            # Returns (success, conflict, new_queue)
            # Actually, we can just loop while queue is not empty
            while queue:
                if time.time() > deadline - 10: # Reserve time for WalkSAT
                    return None, None # Timeout signal
                lit = queue.pop(0)
                v = lit // 2
                p = lit % 2
                
                # Check if assignment is consistent
                if assign[v] != -1 and assign[v] != p:
                    return False, None # Conflict
                    
                if assign[v] != -1:
                    # Already assigned, skip? 
                    # If it was assigned, it should have been processed.
                    continue
                    
                assign[v] = p
                trail.append((v, p))
                
                neg_lit = v * 2 + (1 - p)
                
                # Process clauses watching neg_lit
                # We need to iterate over a copy because we modify the list
                w_list = watched[neg_lit][:]
                for i in w_list:
                    clause = internal_clauses[i]
                    
                    # Check if clause is satisfied by some other literal
                    sat = False
                    for l in clause:
                        if l == neg_lit:
                            continue
                        cv = l // 2
                        cp = l % 2
                        if assign[cv] == cp:
                            sat = True
                            break
                    
                    if sat:
                        # Clause is satisfied, remove neg_lit from watch list
                        # We don't need to add anything else
                        watched[neg_lit].remove(i)
                    else:
                        # Clause is not satisfied
                        # Check if it becomes unit
                        unassigned = []
                        for l in clause:
                            if l == neg_lit:
                                continue
                            cv = l // 2
                            cp = l % 2
                            if assign[cv] == -1:
                                unassigned.append(l)
                        
                        if len(unassigned) == 0:
                            # Conflict
                            return False, None
                        elif len(unassigned) == 1:
                            # Unit clause
                            new_lit = unassigned[0]
                            # Remove neg_lit from watch list
                            watched[neg_lit].remove(i)
                            # Add new_lit to watch list? 
                            # The clause now watches new_lit and one other unassigned?
                            # No, we just need to ensure it's watched by two.
                            # Currently it was watched by neg_lit and one other.
                            # We removed neg_lit. It is now watched by the other one.
                            # We should add new_lit to the watch list of the clause?
                            # Actually, the standard algorithm says:
                            # If we remove a watch and the clause is not satisfied, 
                            # we must find a new literal to watch.
                            # Here, the new literal is the one that makes it unit?
                            # No, if it's unit, we assign it.
                            # We add the new unit literal to the queue.
                            # We don't need to update the watch list for the clause immediately?
                            # Yes we do. The clause must have 2 watchers.
                            # Currently it has 1 (the other original watcher).
                            # We need to add a new watcher.
                            # Any unassigned literal in the clause can be a watcher.
                            # But there are no other unassigned literals!
                            # So the clause is unit. We assign the literal.
                            # Once assigned, the clause will be satisfied or conflict.
                            # So we just push to queue.
                            queue.append(new_lit)
                        else:
                            # Still has > 1 unassigned
                            # We removed neg_lit. We need to add a new watcher.
                            # Pick any unassigned literal from the clause.
                            new_watcher = unassigned[0]
                            watched[neg_lit].remove(i)
                            watched[new_watcher].append(i)
                            
            return True, None
        
        # Initialize queue with any unit clauses? 
        # In 3-SAT, initial clauses are size 3. No units.
        # So queue is empty initially.
        
        # Heuristic: Pick variable with highest frequency in unsatisfied clauses
        def pick_var():
            freq = [0] * n
            for i, clause in enumerate(internal_clauses):
                # Check if clause is satisfied
                sat = False
                for l in clause:
                    v = l // 2
                    p = l % 2
                    if assign[v] == p:
                        sat = True
                        break
                if sat:
                    continue
                for l in clause:
                    v = l // 2
                    if assign[v] == -1:
                        freq[v] += 1
            max_freq = 0
            best_var = -1
            for v in range(n):
                if assign[v] == -1 and freq[v] > max_freq:
                    max_freq = freq[v]
                    best_var = v
            return best_var

        # Main DPLL loop
        # We need to handle backtracking.
        # Since N=50, we can use a simple recursive approach with state copying?
        # No, state copying is expensive. Use trail.
        
        # Let's implement a recursive DPLL with explicit backtracking using the trail.
        # But recursion depth 50 is fine.
        
        def dpll_recursive():
            # Check for conflicts/units? 
            # We rely on propagate to be called before branching.
            
            # Pick a variable
            var = pick_var()
            if var == -1:
                # All variables assigned
                # Check if all clauses satisfied
                for clause in internal_clauses:
                    sat = False
                    for l in clause:
                        v = l // 2
                        p = l % 2
                        if assign[v] == p:
                            sat = True
                            break
                    if not sat:
                        return False
                return True
            
            # Try True
            lit_true = var * 2 + 1
            queue = [lit_true]
            trail_start = len(trail)
            
            ok, conflict = propagate(queue)
            if ok is None:
                # Timeout
                return None
            if ok:
                res = dpll_recursive()
                if res is True:
                    return True
                elif res is None:
                    return None
                # res is False, backtrack
            
            # Backtrack
            while len(trail) > trail_start:
                v, p = trail.pop()
                assign[v] = -1
                # Note: Watched literals are not restored! 
                # This is a problem. Watched literals are modified in place.
                # To support backtracking, we need to undo the watch list changes.
                # This is complex.
                
                # Alternative: Don't use in-place watch lists for backtracking?
                # Or, since N=50, we can just copy the entire state?
                # State: assign (50 ints), watched (100 lists of ints).
                # Copying is O(N * M). M is number of clauses.
                # For N=50, M is likely a few hundred.
                # Copying is fast enough.
                
            # So, let's abandon the in-place watch list DPLL with backtracking 
            # and use a simpler approach:
            # 1. Copy state.
            # 2. Assign.
            # 3. Propagate (using watch lists on the copied state).
            # 4. Recurse.
            
            # Let's rewrite DPLL to use state copying.
            pass
        
        # Given the complexity of implementing a correct CDCL/DPLL with watch lists and backtracking 
        # in a short time, and N=50, a simpler DPLL might be sufficient.
        # The original program's DPLL was O(M) per propagation step. 
        # With N=50, even O(N*M) per node is fine if the tree is small.
        # Planted instances usually have small search trees.
        
        # Let's stick to the original DPLL structure but optimize it slightly.
        # The original DPLL was:
        # 1. Propagate (O(M))
        # 2. Pure literal (O(M))
        # 3. Branch (O(M))
        # This is O(M) per node. If the tree has 2^50 nodes, it's too slow.
        # But planted instances are easy.
        
        # I will implement a better heuristic for variable selection: 
        # "Most Constrained Variable" (highest degree in remaining clauses).
        
        # And I will add a timeout to DPLL.
        
        return None, None # Placeholder

    # Let's just use the original DPLL but with a timeout, and a better WalkSAT.
    # The original DPLL code was actually decent. I will reuse it but add a timeout.
    
    def dpll_simple(clauses, assign, time_limit):
        # clauses: list of lists of (var, pos)
        # assign: list of -1, 0, 1
        # Returns (success, assign) or (None, None) on timeout
        
        def propagate(clauses, assign):
            changed = True
            while changed:
                changed = False
                new_clauses = []
                for clause in clauses:
                    sat = False
                    unassigned = []
                    for var, pos in clause:
                        if assign[var] == 1:
                            if pos:
                                sat = True
                                break
                        elif assign[var] == 0:
                            if not pos:
                                sat = True
                                break
                        else:
                            unassigned.append((var, pos))
                    if sat:
                        continue
                    if not unassigned:
                        return False, None, None, True
                    if len(unassigned) == 1:
                        var, pos = unassigned[0]
                        if assign[var] == -1:
                            assign[var] = 1 if pos else 0
                            changed = True
                        elif assign[var] != (1 if pos else 0):
                            return False, None, None, True
                    else:
                        new_clauses.append(unassigned)
            return True, new_clauses, assign, False

        def pure_literal_elim(clauses, assign):
            if not clauses:
                return clauses, assign
            pos_count = [0] * n
            neg_count = [0] * n
            for clause in clauses:
                for var, pos in clause:
                    if assign[var] != -1:
                        continue
                    if pos:
                        pos_count[var] += 1
                    else:
                        neg_count[var] += 1

            pure = {}
            for var in range(n):
                if assign[var] != -1:
                    continue
                if pos_count[var] > 0 and neg_count[var] == 0:
                    pure[var] = 1
                elif neg_count[var] > 0 and pos_count[var] == 0:
                    pure[var] = 0

            if not pure:
                return clauses, assign

            for var, val in pure.items():
                assign[var] = val

            new_clauses = []
            for clause in clauses:
                sat = False
                remaining = []
                for var, pos in clause:
                    if assign[var] == 1:
                        if pos:
                            sat = True
                            break
                    elif assign[var] == 0:
                        if not pos:
                            sat = True
                            break
                    else:
                        remaining.append((var, pos))
                if sat:
                    continue
                if not remaining:
                    return None, None
                new_clauses.append(remaining)
            return new_clauses, assign

        while True:
            if time.time() > time_limit:
                return None, None

            ok, clauses, assign, conflict = propagate(clauses, assign)
            if conflict:
                return False, None
            if not clauses:
                return True, assign

            clauses, assign = pure_literal_elim(clauses, assign)
            if clauses is None:
                return False, None
            if not clauses:
                return True, assign

            var_freq = {}
            for clause in clauses:
                for var, pos in clause:
                    if assign[var] == -1:
                        var_freq[var] = var_freq.get(var, 0) + 1

            if not var_freq:
                return True, assign

            # Heuristic: pick variable with highest frequency
            var = max(var_freq, key=lambda v: var_freq[v])

            for val in [1, 0]:
                new_assign = assign[:]
                new_assign[var] = val
                new_clauses = []
                conflict_found = False
                for clause in clauses:
                    sat = False
                    remaining = []
                    for v, pos in clause:
                        if v == var:
                            if (val == 1 and pos) or (val == 0 and not pos):
                                sat = True
                                break
                        elif new_assign[v] == -1:
                            remaining.append((v, pos))
                        else:
                            if (new_assign[v] == 1 and pos) or (new_assign[v] == 0 and not pos):
                                sat = True
                                break
                    if sat:
                        continue
                    if not remaining:
                        conflict_found = True
                        break
                    new_clauses.append(remaining)

                if not conflict_found:
                    result, new_assign2 = dpll_simple(new_clauses, new_assign, time_limit)
                    if result is not None:
                        return result, new_assign2
                    if result is False:
                        continue
                    return None, None

            return False, None

    # Convert to internal format for DPLL
    def to_internal(clause):
        return [(abs(l) - 1, l > 0) for l in clause]
    
    internal_clauses_dpll = [to_internal(c) for c in clauses]
    initial_assign = [-1] * n
    
    # Run DPLL with a short budget (e.g., 2 seconds)
    dpll_time_limit = time.time() + 2.0
    success, dpll_assign = dpll_simple(internal_clauses_dpll, initial_assign, dpll_time_limit)
    
    if success and dpll_assign is not None:
        result = []
        for v in range(n):
            if dpll_assign[v] == -1:
                result.append(random.random() < 0.5)
            else:
                result.append(dpll_assign[v] == 1)
        return result

    # Fallback to WalkSAT
    # Use a fixed seed for reproducibility in testing, but random for actual runs?
    # The task says "locally generated", so we can use random.
    rng = random.Random()
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    def is_satisfied(clause, assign):
        return any(assign[abs(l) - 1] == (l > 0) for l in clause)

    def count_unsat(clause, assign):
        return sum(1 for l in clause if not (assign[abs(l) - 1] == (l > 0)))

    # WalkSAT parameters
    max_restarts = 200
    max_steps_per_restart = 5000
    walk_prob = 0.2  # Probability of random walk

    current_time = time.time()
    for restart in range(max_restarts):
        if time.time() > deadline - 1.0:
            break

        if restart > 0:
            assignment = [rng.random() < 0.5 for _ in range(n)]

        for step in range(max_steps_per_restart):
            if time.time() > deadline - 1.0:
                return assignment

            # Find unsatisfied clauses
            unsat_clauses = [c for c in clauses if not is_satisfied(c, assignment)]
            
            if not unsat_clauses:
                return assignment

            # Pick the clause with the fewest false literals (min-conflicts)
            # If multiple, pick one randomly among them? Or just the first?
            # Standard WalkSAT picks one at random from the unsatisfied clauses,
            # but picking the "most constrained" (fewest false literals) can help.
            # Let's pick the clause with the minimum number of false literals.
            min_unsat_count = min(count_unsat(c, assignment) for c in unsat_clauses)
            candidate_clauses = [c for c in unsat_clauses if count_unsat(c, assignment) == min_unsat_count]
            
            if not candidate_clauses:
                break # Should not happen

            clause = rng.choice(candidate_clauses)

            # Pick a literal to flip
            # With probability walk_prob, pick a random literal from the clause
            # Otherwise, pick the literal that minimizes the number of newly unsatisfied clauses
            if rng.random() < walk_prob:
                lit = rng.choice(clause)
            else:
                best_lit = None
                best_score = float('inf')
                for lit in clause:
                    var = abs(lit) - 1
                    # Calculate delta: how many clauses become unsatisfied if we flip this var?
                    # Actually, we want to minimize the number of clauses that become unsatisfied.
                    # Or maximize the number of clauses that become satisfied.
                    # Let's count how many clauses are currently unsatisfied and contain this var.
                    # If we flip, some might become satisfied, some might become unsatisfied.
                    
                    # Simulate flip
                    old_val = assignment[var]
                    assignment[var] = not old_val
                    
                    # Count unsatisfied clauses
                    unsat_count = 0
                    for c in clauses:
                        if not is_satisfied(c, assignment):
                            unsat_count += 1
                            
                    # Revert flip
                    assignment[var] = old_val
                    
                    if unsat_count < best_score:
                        best_score = unsat_count
                        best_lit = lit
                
                if best_lit is None:
                    best_lit = rng.choice(clause)
                lit = best_lit

            # Flip the variable
            var = abs(lit) - 1
            assignment[var] = not assignment[var]

    return assignment
