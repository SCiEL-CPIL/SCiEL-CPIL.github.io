# EVOLVE-BLOCK-START
import time
import random
import numpy as np
import heapq
from collections import defaultdict

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin

    # Convert to 0-based signed literals
    # Literal: var (0 to n-1), sign (1 for True, -1 for False)
    # We represent literals as integers: 
    # Positive literal (var is True): 2*var
    # Negative literal (var is False): 2*var + 1
    # This makes it easy to check satisfaction and negation.
    
    def lit_to_idx(lit):
        # lit is 1-based signed from input
        var = abs(lit) - 1
        if lit > 0:
            return 2 * var
        else:
            return 2 * var + 1

    def idx_to_var(idx):
        return idx // 2

    def idx_to_sign(idx):
        return 1 if idx % 2 == 0 else -1

    def neg_idx(idx):
        return idx ^ 1

    # Convert clauses to internal representation
    internal_clauses = []
    for c in clauses:
        ic = [lit_to_idx(l) for l in c]
        internal_clauses.append(ic)
    
    m = len(internal_clauses)

    # CDCL Solver Class
    class CDCLSolver:
        def __init__(self, n, clauses):
            self.n = n
            self.clauses = clauses
            self.m = len(clauses)
            
            # Values: 0 = False, 1 = True, -1 = Unassigned
            # Using a list of ints for speed
            self.values = [-1] * n
            
            # Trail: list of variables assigned in order
            self.trail = []
            
            # Trail position for each variable (for backtracking)
            self.trail_pos = [-1] * n
            
            # Reason clause for each variable assignment
            self.reason = [-1] * n
            
            # Watched literals: dict from literal idx to list of clause indices
            # We only watch 2 literals per clause
            self.watched = defaultdict(list)
            
            # Initialize watches
            for ci, clause in enumerate(self.clauses):
                # Pick first two literals to watch
                l1, l2 = clause[0], clause[1]
                self.watched[l1].append(ci)
                self.watched[l2].append(ci)
                
                # Store the watched literals for this clause
                # We need to know which literals are watched to update them
                # We'll store the clause index in the watched lists, and when we need to update,
                # we'll find the other watched literal in the clause.
                # To make this efficient, we can store the pair of watched literals in the clause itself?
                # No, standard approach: 
                # When a literal l becomes false, we look at all clauses watching l.
                # For each clause, if the other watched literal is also false, we need to find a new one.
                
            # VSIDS
            self.activity = [1.0] * n
            self.decay = 0.95
            self.bump = 1.0
            self.heap = [( -self.activity[i], i) for i in range(n)]
            heapq.heapify(self.heap)
            
            # Conflict count for restarts
            self.conflicts = 0
            self.restart_interval = 100
            self.next_restart = self.restart_interval
            
            # Learned clauses
            self.learned_clauses = []
            self.learned_count = 0
            
            # Decision level
            self.decision_level = 0
            self.trail_lim = [] # Trail limit for each decision level
            
        def is_true(self, lit_idx):
            v = idx_to_var(lit_idx)
            s = idx_to_sign(lit_idx)
            return self.values[v] == s
            
        def is_false(self, lit_idx):
            v = idx_to_var(lit_idx)
            s = idx_to_sign(lit_idx)
            return self.values[v] == -s
            
        def is_unassigned(self, lit_idx):
            v = idx_to_var(lit_idx)
            return self.values[v] == -1
            
        def assign(self, var, val, reason_clause_idx, level):
            self.values[var] = val
            self.trail.append(var)
            self.trail_pos[var] = len(self.trail) - 1
            self.reason[var] = reason_clause_idx
            
            # Update heap? VSIDS usually updates activity on conflict.
            # For now, we just assign.
            
            # Check watches
            # We need to propagate. 
            # Instead of calling propagate immediately, we push to a queue.
            # But standard CDCL does propagation immediately after assignment.
            
            # Let's return the list of literals to propagate?
            # No, let's do the propagation logic here.
            
            # Actually, it's cleaner to have a separate propagate function that processes the trail.
            # But watched literals require checking watches on assignment.
            
            # Let's use a queue for unit propagation.
            # When we assign a variable, we check its watches.
            
            # This method is getting complex. Let's use a simpler recursive/iterative propagation.
            
            pass
            
        def propagate(self):
            """
            Perform unit propagation.
            Returns:
            - True if no conflict
            - False if conflict, and sets self.conflict_clause
            """
            # We need a queue of literals to process?
            # No, standard watched literal propagation:
            # When a variable is assigned, we check the clauses watching the literal that became false.
            
            # Since we are doing this in Python, a queue is efficient.
            # But we need to know which literals became false.
            # Let's maintain a list of "to_check" literals.
            
            # Actually, the standard algorithm is:
            # 1. Assign variable.
            # 2. For each clause watching the false literal, check if the other is false.
            # 3. If so, find a new literal. If none, conflict. If one, it's unit, assign it.
            # 4. Repeat for newly assigned variables.
            
            # We'll implement this iteratively.
            # We need a stack of variables to process.
            
            # This is tricky to implement cleanly in one go.
            # Let's use a simplified approach:
            # Maintain a list of "active" clauses that need checking? No.
            
            # Let's try a different structure:
            # self.to_propagate = []
            # When a variable is assigned, add its literals to to_propagate?
            
            # Given the complexity, let's use a simpler DPLL with VSIDS but NO clause learning first, 
            # and see if it's fast enough. If not, I will add clause learning.
            
            # Actually, I will implement the full CDCL properly.
            
            pass
            
    # Given the complexity of writing a bug-free CDCL in one shot, 
    # and the risk of timeout/bugs, let's use a robust WalkSAT with improvements.
    # The previous WalkSAT was okay but maybe not optimized enough.
    # Let's try **MinConflicts** with random restarts and a good heuristic.
    # Or **WalkSAT** with adaptive noise and better variable selection.
    
    # Let's stick to the portfolio but improve the DPLL to be iterative and use VSIDS.
    # I will implement a simpler DPLL with VSIDS and Unit Propagation, no clause learning.
    # This is often called "DPLL with VSIDS".
    
    class SimpleDPLL:
        def __init__(self, n, clauses):
            self.n = n
            self.clauses = clauses
            self.values = [-1] * n
            self.trail = []
            self.trail_pos = [-1] * n
            self.reason = [-1] * n
            self.activity = [1.0] * n
            self.decay = 0.95
            self.bump = 1.0
            self.heap = [(-self.activity[i], i) for i in range(n)]
            heapq.heapify(self.heap)
            self.decision_level = 0
            self.trail_lim = []
            self.conflicts = 0
            self.restart_interval = 50
            self.next_restart = self.restart_interval
            self.watched = defaultdict(list)
            
            # Initialize watches
            for ci, clause in enumerate(self.clauses):
                if len(clause) >= 2:
                    l1, l2 = clause[0], clause[1]
                    self.watched[l1].append(ci)
                    self.watched[l2].append(ci)
                elif len(clause) == 1:
                    self.watched[clause[0]].append(ci)
                    
            # Store watched literals for each clause
            # We need to know which literals are watched to update them efficiently
            self.watched_lits = {}
            for ci, clause in enumerate(self.clauses):
                if len(clause) >= 2:
                    self.watched_lits[ci] = [clause[0], clause[1]]
                elif len(clause) == 1:
                    self.watched_lits[ci] = [clause[0]]
                    
        def is_true(self, lit_idx):
            v = idx_to_var(lit_idx)
            s = idx_to_sign(lit_idx)
            return self.values[v] == s
            
        def is_false(self, lit_idx):
            v = idx_to_var(lit_idx)
            s = idx_to_sign(lit_idx)
            return self.values[v] == -s
            
        def is_unassigned(self, lit_idx):
            v = idx_to_var(lit_idx)
            return self.values[v] == -1
            
        def assign(self, var, val, reason_clause_idx):
            self.values[var] = val
            self.trail.append(var)
            self.trail_pos[var] = len(self.trail) - 1
            self.reason[var] = reason_clause_idx
            
        def unassign(self, var):
            self.values[var] = -1
            self.trail_pos[var] = -1
            self.reason[var] = -1
            
        def backjump(self, level):
            # Unassign all variables assigned after 'level'
            while len(self.trail) > self.trail_lim[level]:
                var = self.trail.pop()
                self.unassign(var)
            self.decision_level = level
            if self.decision_level > 0:
                self.trail_lim.pop()
                
        def bump_activity(self, clause):
            for lit in clause:
                var = idx_to_var(lit)
                self.activity[var] += self.bump
                # Update heap? 
                # For simplicity, we can rebuild the heap occasionally or use a lazy heap.
                # Let's use a lazy approach: just update activity, and when selecting, 
                # we pop until we find a valid one.
                
        def select_variable(self):
            # Pop from heap until we find an unassigned variable
            while self.heap:
                _, var = heapq.heappop(self.heap)
                if self.values[var] == -1:
                    return var
            return -1
            
        def propagate(self):
            """
            Perform unit propagation using watched literals.
            Returns True if no conflict, False if conflict.
            If conflict, self.conflict_clause is set.
            """
            # We need to process the trail from the end?
            # No, watched literals require checking watches when a literal becomes false.
            # We'll maintain a list of literals to check.
            
            # Actually, the standard way is:
            # When we assign a variable, we check the watches for the false literal.
            # Let's do that in the assign step? No, assign is just state change.
            
            # Let's use a queue.
            # When a variable is assigned, we add the false literal to a queue.
            # Then we process the queue.
            
            # But we need to know which literal became false.
            # If we assign var to True, the negative literal becomes false.
            # If we assign var to False, the positive literal becomes false.
            
            # Let's modify assign to return the false literal?
            # Or just check in propagate.
            
            # For simplicity in this implementation, I will iterate through the trail 
            # and check watches for newly assigned variables.
            
            # This is inefficient. Let's do it properly.
            
            # I will implement a function that takes a variable assignment and processes watches.
            
            pass
            
    # Given the time constraints and risk of bugs in a complex CDCL implementation,
    # I will use a **highly optimized WalkSAT** with multiple restarts and adaptive parameters.
    # WalkSAT is often very effective for planted SAT instances.
    
    def walksat(rng, max_restarts=1000, max_steps=10000):
        best_assignment = None
        best_unsat = m + 1
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            assignment = [rng.random() < 0.5 for _ in range(n)]
            
            # Compute initial unsat count
            clause_unsat = np.zeros(m, dtype=bool)
            for ci, clause in enumerate(internal_clauses):
                sat = False
                for lit in clause:
                    var = idx_to_var(lit)
                    sign = idx_to_sign(lit)
                    if assignment[var] == (sign == 1):
                        sat = True
                        break
                clause_unsat[ci] = not sat
                
            unsat_count = int(np.sum(clause_unsat))
            if unsat_count == 0:
                return assignment
                
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
                
            # Adaptive noise
            noise = 0.5
            for step in range(max_steps):
                if time.time() > deadline:
                    return best_assignment
                    
                if unsat_count == 0:
                    return assignment
                    
                # Pick a random unsatisfied clause
                unsat_indices = np.where(clause_unsat)[0]
                ci = int(rng.choice(unsat_indices))
                clause = internal_clauses[ci]
                
                # Pick a literal to flip
                # Heuristic: Min damage
                best_var = -1
                best_score = -float('inf')
                
                for lit in clause:
                    var = idx_to_var(lit)
                    score = 0
                    
                    # Calculate damage
                    # We need to know which other clauses contain this variable
                    # Precompute var_clauses
                    for other_ci, other_lit in var_clauses[var]:
                        if other_ci == ci:
                            continue
                        if clause_unsat[other_ci]:
                            # If other clause is unsat, and this literal would satisfy it, good?
                            # No, we are flipping var. 
                            # If other clause is unsat, and this literal is the only one that could satisfy it, 
                            # flipping might break it? No, it's already unsat.
                            # If we flip var to satisfy this literal, does it satisfy the other clause?
                            other_var = idx_to_var(other_lit)
                            other_sign = idx_to_sign(other_lit)
                            if var == other_var:
                                # Same variable
                                if (sign == 1) != (other_sign == 1):
                                    # Contradictory literals
                                    # If we flip to satisfy current, we break other?
                                    # Current lit: (var == sign). Other lit: (var == other_sign).
                                    # If sign != other_sign, then satisfying one breaks the other.
                                    # But other clause is unsat, so it doesn't matter if we break it?
                                    # Actually, if we flip to satisfy current, we make other lit false.
                                    # Since other clause is already unsat, making one more literal false doesn't change its status.
                                    pass
                        else:
                            # Other clause is satisfied.
                            # Check if this literal is the ONLY satisfying literal in the other clause.
                            only_sat = True
                            for other_lit2 in internal_clauses[other_ci]:
                                if other_lit2 == other_lit:
                                    continue
                                v2 = idx_to_var(other_lit2)
                                s2 = idx_to_sign(other_lit2)
                                if assignment[v2] == (s2 == 1):
                                    only_sat = False
                                    break
                            if only_sat:
                                # Flipping var will break this clause
                                score -= 1
                                
                    score += rng.random() * noise
                    if score > best_score:
                        best_score = score
                        best_var = var
                        
                if best_var == -1:
                    # Random pick
                    lit = clause[rng.randrange(len(clause))]
                    best_var = idx_to_var(lit)
                    
                # Flip
                old_val = assignment[best_var]
                assignment[best_var] = not old_val
                
                # Update clause_unsat for clauses containing best_var
                for other_ci, other_lit in var_clauses[best_var]:
                    # Recompute satisfaction for this clause
                    sat = False
                    for lit2 in internal_clauses[other_ci]:
                        v2 = idx_to_var(lit2)
                        s2 = idx_to_sign(lit2)
                        if assignment[v2] == (s2 == 1):
                            sat = True
                            break
                    was_unsat = clause_unsat[other_ci]
                    is_unsat = not sat
                    if was_unsat and not is_unsat:
                        unsat_count -= 1
                    elif not was_unsat and is_unsat:
                        unsat_count += 1
                    clause_unsat[other_ci] = is_unsat
                    
                if unsat_count == 0:
                    return assignment
                    
                if unsat_count < best_unsat:
                    best_unsat = unsat_count
                    best_assignment = assignment[:]
                    
                if step % 1000 == 0 and step > 0:
                    noise = max(0.1, noise * 0.9)
                    
        return best_assignment

    # Precompute var_clauses for WalkSAT
    var_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(internal_clauses):
        for lit in clause:
            var = idx_to_var(lit)
            var_clauses[var].append((ci, lit))

    # Run WalkSAT with multiple random seeds and parameters
    best_result = None
    best_unsat = m + 1
    
    # Strategy 1: Quick WalkSAT
    rng1 = random.Random(42)
    result1 = walksat(rng1, max_restarts=50, max_steps=5000)
    def count_unsat(assignment):
        unsat = 0
        for clause in internal_clauses:
            sat = False
            for lit in clause:
                var = idx_to_var(lit)
                sign = idx_to_sign(lit)
                if assignment[var] == (sign == 1):
                    sat = True
                    break
            if not sat:
                unsat += 1
        return unsat

    if result1 is not None:
        u = count_unsat(result1)
        if u < best_unsat:
            best_unsat = u
            best_result = result1
        if u == 0:
            return result1

    # Strategy 2: Extended WalkSAT
    if time.time() < deadline - 10:
        rng2 = random.Random(123)
        result2 = walksat(rng2, max_restarts=200, max_steps=20000)
        if result2 is not None:
            u = count_unsat(result2)
            if u < best_unsat:
                best_unsat = u
                best_result = result2
            if u == 0:
                return result2

    # Strategy 3: Another WalkSAT with different noise
    if time.time() < deadline - 5:
        rng3 = random.Random(777)
        result3 = walksat(rng3, max_restarts=100, max_steps=10000)
        if result3 is not None:
            u = count_unsat(result3)
            if u < best_unsat:
                best_unsat = u
                best_result = result3
            if u == 0:
                return result3

    return best_result

# EVOLVE-BLOCK-END
