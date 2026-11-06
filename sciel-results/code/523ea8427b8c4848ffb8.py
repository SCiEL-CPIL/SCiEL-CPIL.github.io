"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
import random

class DPLL:
    def __init__(self, n, clauses):
        self.n = n
        self.clauses = clauses
        # Watched literals: for each variable, list of (clause_index, literal_sign)
        # literal_sign is True if the literal is positive (var), False if negative (-var)
        # We need to track which literals are watched for each clause.
        # Standard watched literal scheme:
        # clause_watches[clause_idx] = [lit1, lit2]
        # var_watches[var_idx] = list of (clause_idx, watch_index) where watch_index is 0 or 1
        
        self.clause_watches = []
        self.var_watches = [[] for _ in range(n)]
        
        for i, clause in enumerate(clauses):
            # Pick two literals to watch. Usually the first two.
            # If clause has only 1 literal, watch it twice? No, standard is 2.
            # If clause has 0, it's unsat.
            if len(clause) < 2:
                # Handle trivial cases, but 3-SAT implies 3.
                # If len 1, it's a unit clause.
                self.clause_watches.append([clause[0], clause[0]])
            else:
                self.clause_watches.append([clause[0], clause[1]])
                
        # Build var_watches
        for ci, (w0, w1) in enumerate(self.clause_watches):
            v0 = abs(w0) - 1
            v1 = abs(w1) - 1
            self.var_watches[v0].append((ci, 0))
            self.var_watches[v1].append((ci, 1))
            
        # Assignment: -1 unknown, 0 false, 1 true
        self.assignment = [-1] * n
        self.assign_level = [-1] * n
        
        # VSIDS activities
        self.activity = [1.0] * n
        self.decay = 0.95
        self.activity_scale = 1.0
        
        # Trail for backtracking
        self.trail = []
        self.trail_lim = []
        
        # Clause satisfaction cache? 
        # With watched literals, we don't need to check all clauses.
        # We just need to know if a clause is satisfied.
        # A clause is satisfied if any literal is true.
        # We can check this quickly during propagation.
        
        self.start_time = time.time()
        self.timeout = 250.0 # Leave buffer for overhead

    def is_sat(self, clause_idx):
        """Check if clause is satisfied by current assignment."""
        w0 = self.clause_watches[clause_idx][0]
        w1 = self.clause_watches[clause_idx][1]
        
        # Check watched literals
        if self.assignment[abs(w0)-1] != -1:
            if (self.assignment[abs(w0)-1] == 1 and w0 > 0) or (self.assignment[abs(w0)-1] == 0 and w0 < 0):
                return True
        if self.assignment[abs(w1)-1] != -1:
            if (self.assignment[abs(w1)-1] == 1 and w1 > 0) or (self.assignment[abs(w1)-1] == 0 and w1 < 0):
                return True
        
        # If neither watched literal is satisfied, check the rest of the clause
        # This is inefficient if done often, but with watched literals, 
        # we only need to do this when swapping watches.
        # For simplicity in this implementation, if watched literals are not satisfied,
        # we assume the clause might be satisfied by other literals.
        # However, standard watched literals invariant: if a clause is satisfied,
        # at least one of the watched literals is satisfied.
        # So if neither watched literal is satisfied, the clause is NOT satisfied 
        # ONLY IF no other literal is satisfied.
        # Wait, the invariant is: if a clause is satisfied, at least one watch is satisfied.
        # This is maintained by the swap logic.
        # So if neither watch is satisfied, the clause is unsatisfied?
        # No, that's not true. The invariant is:
        # "A clause is satisfied iff at least one of its watched literals is satisfied."
        # This is only true if we maintain the invariant correctly.
        # Let's stick to a simpler check for correctness first, then optimize if needed.
        # Given n=50, m is likely small enough that checking the whole clause is fast.
        
        clause = self.clauses[clause_idx]
        for lit in clause:
            var = abs(lit) - 1
            if self.assignment[var] != -1:
                if (self.assignment[var] == 1 and lit > 0) or (self.assignment[var] == 0 and lit < 0):
                    return True
        return False

    def propagate(self):
        """Perform unit propagation. Returns False if conflict (empty clause), True otherwise."""
        while self.trail:
            # Check time
            if time.time() - self.start_time > self.timeout:
                return "timeout"
            
            # Get the last assigned literal
            # We need to know which variable was assigned.
            # The trail stores variable indices.
            # We need to iterate through var_watches for that variable.
            
            # Actually, standard watched literal propagation works on a queue of unit clauses.
            # Let's use a queue of (clause_idx, unit_lit)
            
            # Re-initializing queue logic inside loop is messy.
            # Let's use a list as a queue.
            
            # To keep it simple and correct, I'll implement a basic propagation
            # that scans for unit clauses. For n=50, this is O(m*n) per step,
            # which is fine.
            
            # Find unit clauses
            unit_clauses = []
            for i, clause in enumerate(self.clauses):
                if self.is_sat(i):
                    continue
                    
                # Count unassigned and satisfied
                unassigned = 0
                last_unassigned = None
                for lit in clause:
                    var = abs(lit) - 1
                    if self.assignment[var] == -1:
                        unassigned += 1
                        last_unassigned = lit
                    else:
                        # If assigned and false, it doesn't help.
                        # If assigned and true, clause is sat (handled above).
                        pass
                
                if unassigned == 0:
                    return False # Conflict
                elif unassigned == 1:
                    unit_clauses.append((i, last_unassigned))
            
            if not unit_clauses:
                return True
            
            # Pick the first unit clause and assign
            ci, lit = unit_clauses[0]
            var = abs(lit) - 1
            
            # Assign
            self.assignment[var] = 1 if lit > 0 else 0
            self.trail.append(var)
            
            # Boost activity
            self.activity[var] *= self.decay
            self.activity[var] += self.activity_scale
            if self.activity[var] > 1e100:
                for i in range(self.n):
                    self.activity[i] /= 1e100
                self.activity_scale /= 1e100
        
        return True

    def pick_var(self):
        """Pick variable with highest activity."""
        max_act = -1
        max_var = -1
        for i in range(self.n):
            if self.assignment[i] == -1 and self.activity[i] > max_act:
                max_act = self.activity[i]
                max_var = i
        return max_var

    def solve_dpll(self):
        """Main DPLL solver."""
        # Initial propagation
        prop_res = self.propagate()
        if prop_res == False:
            return None
        if prop_res == "timeout":
            return None
            
        if self.is_satisfied_all():
            return self.assignment
        
        var = self.pick_var()
        if var == -1:
            # All assigned, check if satisfied
            if self.is_satisfied_all():
                return self.assignment
            else:
                return None

        # Try True
        self.trail_lim.append(len(self.trail))
        self.assignment[var] = 1
        self.trail.append(var)
        
        res = self.solve_dpll()
        if res is not None:
            return res
            
        # Backtrack
        while len(self.trail) > self.trail_lim[-1]:
            v = self.trail.pop()
            self.assignment[v] = -1
        self.trail_lim.pop()
        
        # Try False
        self.trail_lim.append(len(self.trail))
        self.assignment[var] = 0
        self.trail.append(var)
        
        res = self.solve_dpll()
        if res is not None:
            return res
            
        # Backtrack
        while len(self.trail) > self.trail_lim[-1]:
            v = self.trail.pop()
            self.assignment[v] = -1
        self.trail_lim.pop()
        
        return None

    def is_satisfied_all(self):
        for i, clause in enumerate(self.clauses):
            if not self.is_sat(i):
                return False
        return True

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Check for trivial satisfiability (empty clause)
    for c in clauses:
        if len(c) == 0:
            return [False] * n # Should not happen, but safe
            
    # Try DPLL
    dpll = DPLL(n, clauses)
    
    # Recursion limit might be hit if deep, but n=50 is small.
    sys.setrecursionlimit(10000)
    
    result = dpll.solve_dpll()
    
    if result is not None:
        # Convert -1/0/1 to False/True
        # 0 is False, 1 is True
        return [val == 1 for val in result]
    else:
        # If DPLL fails (timeout or UNSAT), fall back to WalkSAT-like or Random
        # Since task says "planted satisfiable", it should be SAT.
        # If DPLL times out, try a quick WalkSAT.
        
        # Quick WalkSAT
        rng = random.Random(42)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Pre-compute clause data for WalkSAT
        clause_data = []
        for clause in clauses:
            cdata = []
            for lit in clause:
                var_idx = abs(lit) - 1
                sign = lit > 0
                cdata.append((var_idx, sign))
            clause_data.append(cdata)
            
        m = len(clauses)
        
        def count_unsat(assignment):
            cnt = 0
            for cdata in clause_data:
                sat = False
                for var_idx, sign in cdata:
                    if assignment[var_idx] == sign:
                        sat = True
                        break
                if not sat:
                    cnt += 1
            return cnt
            
        best_assign = assignment[:]
        best_score = count_unsat(assignment)
        
        # Simple WalkSAT for 5000 steps
        for _ in range(5000):
            unsat_clauses = []
            for i, cdata in enumerate(clause_data):
                sat = False
                for var_idx, sign in cdata:
                    if assignment[var_idx] == sign:
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(i)
                    
            if not unsat_clauses:
                return assignment
                
            ci = rng.choice(unsat_clauses)
            cdata = clause_data[ci]
            
            # Pick variable to flip
            # Simple: random literal from clause
            var_idx, sign = rng.choice(cdata)
            assignment[var_idx] = not assignment[var_idx]
            
            score = count_unsat(assignment)
            if score < best_score:
                best_score = score
                best_assign = assignment[:]
                
        return best_assign

# EVOLVE-BLOCK-END
