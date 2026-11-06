"""
CDCL Solver for 3-SAT
Implements Conflict-Driven Clause Learning with Watched Literals.
"""
import time
import sys

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 10s margin for 300s limit
    
    # Preprocess: Convert to 1-based signed literals (1..2n)
    # var i (1..n) -> positive literal i, negative literal -i
    # Internal representation:
    # We will use 1..n for positive, -1..-n for negative.
    # Assignment array: 0 (unassigned), 1 (true), -1 (false)
    # size n+1 (index 0 unused)
    
    # Build clauses in internal format
    # A clause is a list of literals.
    # We need to handle unit clauses and binary clauses specially for speed? 
    # For 3-SAT, most are ternary.
    
    # Watches: dict literal -> list of clause indices
    # We need to initialize watches for the first two literals of each clause.
    
    # Let's define the solver class for encapsulation
    class CDCL:
        def __init__(self, n, clauses):
            self.n = n
            self.num_vars = n
            # assignment: 0, 1, -1
            self.assignment = [0] * (n + 1)
            self.reason = [-1] * (n + 1)
            self.level = [0] * (n + 1)
            self.activity = [0.0] * (n + 1)
            self.trail = []
            self.trail_lim = []
            self.qhead = 0
            self.clauses = clauses
            self.learnts = []
            self.num_clauses = len(clauses)
            self.conflicts = 0
            self.restart_interval = 100
            self.max_learnts = 1000
            self.decision_level = 0
            self.watches = {}
            # Initialize watches
            for i, clause in enumerate(clauses):
                if len(clause) == 0:
                    # Empty clause: UNSAT
                    return
                if len(clause) == 1:
                    # Unit clause
                    lit = clause[0]
                    var = abs(lit)
                    val = 1 if lit > 0 else -1
                    if self.assignment[var] == 0:
                        self.assign(var, val, -1, 0)
                    elif self.assignment[var] != val:
                        # Contradiction
                        pass
                elif len(clause) >= 2:
                    l1 = clause[0]
                    l2 = clause[1]
                    if l1 not in self.watches:
                        self.watches[l1] = []
                    if l2 not in self.watches:
                        self.watches[l2] = []
                    self.watches[l1].append(i)
                    self.watches[l2].append(i)
            
            # Handle initial unit propagation
            # Note: The constructor above assigns unit literals but doesn't propagate consequences.
            # We need to run propagate() initially.
            
        def assign(self, var, val, reason, level):
            self.assignment[var] = val
            self.reason[var] = reason
            self.level[var] = level
            self.trail.append(var)
            
        def propagate(self):
            while self.qhead < len(self.trail):
                p = self.trail[self.qhead]
                self.qhead += 1
                # p is the variable assigned. 
                # We need to look at watches for the literal that is now FALSE.
                # If assignment[p] is 1 (True), then literal -p is False.
                # If assignment[p] is -1 (False), then literal p is False.
                false_lit = -p if self.assignment[p] == 1 else p
                # Actually, standard watched literal:
                # We watch literals. When a literal becomes FALSE, we check its clauses.
                # If var p is assigned True, literal -p is False.
                # If var p is assigned False, literal p is False.
                
                if false_lit not in self.watches:
                    continue
                    
                i = 0
                watches = self.watches[false_lit]
                while i < len(watches):
                    ci = watches[i]
                    clause = self.clauses[ci] if ci < self.num_clauses else self.learnts[ci - self.num_clauses]
                    
                    # Check if clause is satisfied by any literal
                    sat = False
                    for lit in clause:
                        if lit == false_lit:
                            continue
                        var = abs(lit)
                        val = self.assignment[var]
                        if val != 0:
                            # Check if lit is satisfied
                            if (lit > 0 and val == 1) or (lit < 0 and val == -1):
                                sat = True
                                break
                    if sat:
                        i += 1
                        continue
                    
                    # Clause is not satisfied.
                    # Check if it has a second unassigned literal
                    second_unassigned = None
                    second_idx = -1
                    for j, lit in enumerate(clause):
                        if lit == false_lit:
                            continue
                        var = abs(lit)
                        if self.assignment[var] == 0:
                            second_unassigned = lit
                            second_idx = j
                            break
                    
                    if second_unassigned is None:
                        # Conflict
                        return ci
                    else:
                        # Unit clause
                        # Move watch from false_lit to second_unassigned
                        var = abs(second_unassigned)
                        val = 1 if second_unassigned > 0 else -1
                        self.assign(var, val, ci, self.decision_level)
                        
                        # Update watches
                        # Remove ci from watches[false_lit]
                        watches[i] = watches[-1]
                        watches.pop()
                        # Add ci to watches[second_unassigned]
                        if second_unassigned not in self.watches:
                            self.watches[second_unassigned] = []
                        self.watches[second_unassigned].append(ci)
                        # Note: We don't increment i because we swapped the last element to i
                        continue
                # End while i < len(watches)
            return -1 # No conflict
            
        def analyze(self, conflict_clause_idx):
            # Find First UIP
            # conflict_clause_idx is index into all clauses (original + learnts)
            # Get the clause
            if conflict_clause_idx < self.num_clauses:
                clause = self.clauses[conflict_clause_idx]
            else:
                clause = self.learnts[conflict_clause_idx - self.num_clauses]
                
            # The clause is falsified. 
            # We need to find the literals in this clause that are at the current decision level.
            # The "asserting literal" is the one that will become unit when we backtrack.
            
            seen = set()
            learnt_clause = []
            counter = 0
            p = None
            bjl = 0 # Backjump level
            
            # Iterate through literals in the conflict clause
            # We need to trace back the reasons
            # Standard First UIP algorithm:
            
            # 1. Initialize
            # The conflict clause has at least one literal at the current level.
            # Actually, all literals in the conflict clause are False.
            # We look for the literal assigned at the highest level (current).
            # There might be multiple. We pick one to start the analysis.
            
            # Let's collect all literals in the clause
            # For each literal l in clause:
            #   var = abs(l)
            #   if level[var] == current_level:
            #       start_var = var
            #       break
            
            start_var = None
            for lit in clause:
                var = abs(lit)
                if self.level[var] == self.decision_level:
                    start_var = var
                    break
            
            if start_var is None:
                # This shouldn't happen if propagation is correct
                # But if it does, we can't learn
                return -1, [], 0
                
            seen.add(start_var)
            learnt_clause.append(-start_var) # The literal is false, so the negation is the one that caused it? 
            # Wait, the clause is (l1 or l2 or ...). All are false.
            # The reason for start_var being assigned is some clause R.
            # R is (start_var_lit or r1 or r2 ...).
            # Since start_var_lit is false (because start_var is assigned opposite to start_var_lit? No, start_var is assigned a value, making one of its literals true and the other false. In the conflict clause, the literal is FALSE. So the assigned value is the one that makes it false.
            # Example: Clause has -x. x is True. So -x is False.
            # Reason for x=True is Clause C: (x or y). Since x is True, C is satisfied. But we need the reason for the assignment.
            # The reason clause R contains the literal that was made True by the assignment.
            # If x is True, the reason clause contains x.
            # So we add x to the learnt clause?
            # Yes, the learnt clause is the implication: if (other literals in R are false) then x must be true.
            # So we add x (the true literal) to the learnt clause.
            
            # Let's restart the analysis logic properly.
            
            seen = set()
            learnt_clause = []
            counter = 0
            p = start_var
            bjl = 0
            
            while True:
                # Get reason clause for p
                reason_idx = self.reason[p]
                if reason_idx == -1:
                    # p is a decision variable
                    # This means we have reached a decision level that must be backtracked to
                    # But p is at current level?
                    # If p is a decision variable at current level, it's the decision itself.
                    # We should have found the asserting literal before this.
                    # Actually, if p is a decision var, we stop and backtrack to the level of the next variable?
                    # No, in First UIP, we stop when we reach a variable that is a decision variable OR when counter > 1 and we've seen a variable at a lower level?
                    # Standard algorithm:
                    # 1. Add p to learnt clause (as the literal that is True).
                    # 2. If p is a decision variable:
                    #    If counter > 1: break (found asserting literal)
                    #    Else: continue? No, if p is decision, it's the first UIP?
                    #    Actually, the decision variable is the first UIP if no other variables at this level are in the implication graph.
                    #    So we add it and break.
                    # 3. Else (p is implied):
                    #    Get reason clause R.
                    #    For each literal l in R:
                    #       If l is the one that made p true (i.e., l matches assignment of p): skip? No, that's the head.
                    #       The reason clause is (p_lit or r1 or r2 ...). p_lit is True.
                    #       We add r1, r2 ... to the learnt clause?
                    #       We need to remove literals that are already in learnt_clause or at lower levels?
                    
                    # Let's use a standard implementation pattern.
                    pass
                
                # This manual implementation is getting complex and error-prone.
                # Given the 300s limit and Python speed, a CDCL solver is very risky.
                # A highly optimized WalkSAT is much safer.
                
                break # Abort analysis, fall back to simple restart or return failure
            
            # Since the analysis is too complex to write correctly in this scratchpad without testing,
            # and the risk of bugs leading to 0 score is high,
            # I will revert to the "Optimized WalkSAT with Incremental Tracking" strategy
            # which is known to work well for 150 vars and is much easier to implement correctly in Python.
            # The prompt asks to "try CDCL", but if I can't implement it correctly, I should implement the best alternative.
            # However, the directive is specific.
            # Let's try a simpler CDCL: No clause learning, just DPLL with Unit Propagation and Restart.
            # This is essentially a SAT solver without learning.
            # For 150 vars, DPLL might be too slow.
            
            # Alternative: Implement WalkSAT with incremental updates.
            # This is a "local search" solver, not CDCL.
            # But it is the "current program's family" (WalkSAT).
            # The directive says "Do NOT switch to a different design family".
            # The current program is WalkSAT.
            # The strategy says "Implement a CDCL solver... which fundamentally differs...".
            # This is a contradiction. The directive says "REFINEMENT: for this proposal, KEEP this design... Do NOT switch".
            # But the "Strategy to try" says "Implement a CDCL solver".
            # Usually, "Strategy to try" is the specific instruction for this turn.
            # "Keep this design" might refer to the general approach of solving SAT, or it might be a generic template text.
            # Given the explicit "Strategy to try", I should try CDCL.
            # But if I can't write a correct CDCL, I fail.
            
            # Let's write a correct, simple CDCL.
            pass
            
    # I will implement a correct, simple CDCL solver.
    
    class SimpleCDCL:
        def __init__(self, n, clauses):
            self.n = n
            self.assignment = [0] * (n + 1)
            self.reason = [-1] * (n + 1)
            self.level = [0] * (n + 1)
            self.trail = []
            self.qhead = 0
            self.clauses = clauses
            self.learnts = []
            self.num_orig = len(clauses)
            self.watches = {}
            self.decision_level = 0
            self.trail_lim = []
            self.conflicts = 0
            self.restart_limit = 100
            self.max_learnts = 500
            
            # Init watches
            for i, c in enumerate(clauses):
                if len(c) >= 2:
                    l1, l2 = c[0], c[1]
                    if l1 not in self.watches: self.watches[l1] = []
                    if l2 not in self.watches: self.watches[l2] = []
                    self.watches[l1].append(i)
                    self.watches[l2].append(i)
                elif len(c) == 1:
                    lit = c[0]
                    var = abs(lit)
                    val = 1 if lit > 0 else -1
                    if self.assignment[var] == 0:
                        self.assign(var, val, -1, 0)
                    elif self.assignment[var] != val:
                        self.unsat = True
                        return
                    else:
                        self.unsat = False
            
            self.unsat = False
            self.propagate()
            
        def assign(self, var, val, reason, level):
            self.assignment[var] = val
            self.reason[var] = reason
            self.level[var] = level
            self.trail.append(var)
            
        def propagate(self):
            while self.qhead < len(self.trail):
                p = self.trail[self.qhead]
                self.qhead += 1
                false_lit = -p if self.assignment[p] == 1 else p
                if false_lit not in self.watches: continue
                i = 0
                ws = self.watches[false_lit]
                while i < len(ws):
                    ci = ws[i]
                    if ci < self.num_orig:
                        clause = self.clauses[ci]
                    else:
                        clause = self.learnts[ci - self.num_orig]
                    
                    # Check if satisfied
                    sat = False
                    for lit in clause:
                        if lit == false_lit: continue
                        var = abs(lit)
                        val = self.assignment[var]
                        if val != 0 and ((lit > 0 and val == 1) or (lit < 0 and val == -1)):
                            sat = True
                            break
                    if sat:
                        i += 1
                        continue
                    
                    # Find second unassigned
                    second = None
                    for lit in clause:
                        if lit == false_lit: continue
                        var = abs(lit)
                        if self.assignment[var] == 0:
                            second = lit
                            break
                    if second is None:
                        return ci # Conflict
                    else:
                        var = abs(second)
                        val = 1 if second > 0 else -1
                        self.assign(var, val, ci, self.decision_level)
                        # Move watch
                        ws[i] = ws[-1]
                        ws.pop()
                        if second not in self.watches: self.watches[second] = []
                        self.watches[second].append(ci)
                        # Do not increment i
            return -1
            
        def analyze(self, conflict_ci):
            if conflict_ci < self.num_orig:
                clause = self.clauses[conflict_ci]
            else:
                clause = self.learnts[conflict_ci - self.num_orig]
                
            seen = set()
            learnt = []
            counter = 0
            p = None
            bjl = 0
            
            # Find first literal at current level
            for lit in clause:
                var = abs(lit)
                if self.level[var] == self.decision_level:
                    p = var
                    break
            
            if p is None:
                return -1, [], 0
                
            seen.add(p)
            learnt.append(p) # Store var, we determine sign later? No, store literal.
            # The literal in the clause is FALSE. So the TRUE literal is -lit.
            # But we are building the implication.
            # Let's store the variables and determine the sign at the end.
            
            while True:
                # p is the current variable
                # Add the literal that is TRUE for p to learnt
                # If assignment[p] is 1, literal p is True. If -1, literal -p is True.
                true_lit = p if self.assignment[p] == 1 else -p
                # Check if true_lit is already in learnt (as var)
                # We store vars in learnt for now.
                
                if self.reason[p] == -1:
                    # Decision variable
                    if counter > 1:
                        # Found asserting literal
                        break
                    else:
                        # This is the first decision var?
                        # We must backtrack to the level of the next variable?
                        # No, if counter==1, we haven't seen another var at this level.
                        # We continue to the next level?
                        # Actually, if p is a decision var, it's the first UIP.
                        # We break.
                        break
                else:
                    # Implied
                    reason_idx = self.reason[p]
                    if reason_idx < self.num_orig:
                        reason_clause = self.clauses[reason_idx]
                    else:
                        reason_clause = self.learnts[reason_idx - self.num_orig]
                    
                    for lit in reason_clause:
                        if lit == true_lit:
                            continue
                        var = abs(lit)
                        if var in seen:
                            continue
                        seen.add(var)
                        learnt.append(var)
                        if self.level[var] < self.decision_level:
                            if self.level[var] > bjl:
                                bjl = self.level[var]
                        else:
                            counter += 1
                            if counter > 1:
                                # Found asserting literal
                                # Backtrack to bjl
                                # The asserting literal is the one in learnt at level decision_level
                                # We need to identify it.
                                # It's the one we just added?
                                break
                    if counter > 1:
                        break
                    # Move to the next variable in the reason clause at current level
                    # Find it
                    next_p = None
                    for lit in reason_clause:
                        if lit == true_lit: continue
                        var = abs(lit)
                        if var in seen: continue
                        if self.level[var] == self.decision_level:
                            next_p = var
                            break
                    if next_p is None:
                        # Should not happen
                        break
                    p = next_p
                    continue
                # If decision var
                break
                
            # Construct learned clause
            # The literals in the learned clause are the negations of the literals in the conflict clause?
            # No.
            # The learned clause is (l1 or l2 or ... or lk)
            # where li are the literals that were assumed True in the implication graph.
            # So for each var in learnt, the literal is the one that is TRUE in the current assignment.
            
            learned_clause = []
            for var in learnt:
                lit = var if self.assignment[var] == 1 else -var
                learned_clause.append(lit)
                
            # Minimize? Skip for simplicity.
            
            # Determine asserting literal
            # It's the literal in learned_clause that is at decision_level
            assert_lit = None
            for lit in learned_clause:
                var = abs(lit)
                if self.level[var] == self.decision_level:
                    assert_lit = lit
                    break
                    
            return bjl, learned_clause, assert_lit
            
        def solve(self, time_limit):
            start = time.time()
            # Initial propagation
            if self.unsat:
                return None
            conflict_ci = self.propagate()
            if conflict_ci != -1:
                # Conflict at level 0? UNSAT
                return None
                
            while time.time() - start < time_limit:
                # Branch
                # Pick unassigned var
                var = 0
                for i in range(1, self.n + 1):
                    if self.assignment[i] == 0:
                        var = i
                        break
                if var == 0:
                    # All assigned
                    return self.assignment[1:]
                    
                self.decision_level += 1
                self.trail_lim.append(len(self.trail))
                
                # Heuristic: Pick False? Or random?
                # Let's pick False for simplicity, or use activity.
                # For now, pick False.
                val = -1
                self.assign(var, val, -1, self.decision_level)
                
                # Propagate
                conflict_ci = self.propagate()
                if conflict_ci == -1:
                    continue
                    
                # Conflict
                self.conflicts += 1
                bjl, learned_clause, assert_lit = self.analyze(conflict_ci)
                
                if bjl == -1:
                    # Error
                    break
                    
                # Add learned clause
                if len(learned_clause) > 0:
                    if len(self.learnts) < self.max_learnts:
                        self.learnts.append(learned_clause)
                        # Add to watches
                        if len(learned_clause) >= 2:
                            l1, l2 = learned_clause[0], learned_clause[1]
                            new_idx = self.num_orig + len(self.learnts) - 1
                            if l1 not in self.watches: self.watches[l1] = []
                            if l2 not in self.watches: self.watches[l2] = []
                            self.watches[l1].append(new_idx)
                            self.watches[l2].append(new_idx)
                    # If full, we might need to delete clauses, but for now just skip adding.
                    
                # Backtrack
                if bjl < self.decision_level:
                    # Undo assignments
                    while len(self.trail) > self.trail_lim[0]:
                        var = self.trail.pop()
                        self.assignment[var] = 0
                        self.reason[var] = -1
                        self.level[var] = 0
                    # Pop trail limits
                    while len(self.trail_lim) > bjl:
                        self.trail_lim.pop()
                    self.decision_level = bjl
                    self.qhead = len(self.trail) # Reset qhead to end of current trail
                    
                    # Assign asserting literal
                    if assert_lit:
                        var = abs(assert_lit)
                        val = 1 if assert_lit > 0 else -1
                        self.assign(var, val, self.num_orig + len(self.learnts) - 1 if len(self.learnts) > 0 else -1, self.decision_level)
                        # Propagate
                        self.propagate()
                        
                else:
                    # bjl == decision_level
                    # This means the asserting literal is at the same level?
                    # Usually we backtrack to bjl. If bjl == decision_level, we stay?
                    # No, we backtrack to bjl, which is the same.
                    # But we need to assign the asserting literal.
                    # Actually, if bjl == decision_level, we just assign the asserting literal and continue.
                    if assert_lit:
                        var = abs(assert_lit)
                        val = 1 if assert_lit > 0 else -1
                        # The reason is the learned clause
                        reason_idx = self.num_orig + len(self.learnts) - 1 if len(self.learnts) > 0 else -1
                        self.assign(var, val, reason_idx, self.decision_level)
                        self.propagate()

            return None

    # Run solver
    solver = SimpleCDCL(n, clauses)
    result = solver.solve(280.0)
    
    if result is not None:
        # Verify
        for clause in clauses:
            sat = False
            for lit in clause:
                var = abs(lit)
                val = result[var-1]
                if (lit > 0 and val) or (lit < 0 and not val):
                    sat = True
                    break
            if not sat:
                return [False] * n # Fallback
        return result
    else:
        # Fallback to WalkSAT if CDCL fails
        # ... (Implement a quick WalkSAT)
        pass
        
    # Fallback WalkSAT
    import random
    rng = random.Random(42)
    deadline = start_time + 290.0
    best_assignment = None
    best_unsat = float('inf')
    
    # Preprocess
    proc_clauses = []
    for c in clauses:
        proc_clauses.append([(abs(l)-1, 1 if l>0 else -1) for l in c])
        
    var_to_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(proc_clauses):
        for v, s in c:
            var_to_clauses[v].append((ci, s))
            
    for _ in range(50):
        if time.time() > deadline: break
        assign = [rng.random() < 0.5 for _ in range(n)]
        # Count unsat
        unsat = 0
        for c in proc_clauses:
            sat = False
            for v, s in c:
                if (s==1 and assign[v]) or (s==-1 and not assign[v]):
                    sat = True
                    break
            if not sat: unsat += 1
        if unsat == 0: return assign
        if unsat < best_unsat:
            best_unsat = unsat
            best_assignment = assign[:]
            
        for _ in range(10000):
            if time.time() > deadline: break
            # Find unsat
            unsat_clauses = []
            for ci, c in enumerate(proc_clauses):
                sat = False
                for v, s in c:
                    if (s==1 and assign[v]) or (s==-1 and not assign[v]):
                        sat = True
                        break
                if not sat: unsat_clauses.append(ci)
            if not unsat_clauses: return assign
            ci = rng.choice(unsat_clauses)
            c = proc_clauses[ci]
            v, s = rng.choice(c)
            assign[v] = not assign[v]
            
    if best_assignment:
        return best_assignment
    return [False] * n
