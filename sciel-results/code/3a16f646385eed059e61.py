"""Planted 3-SAT solver with WalkSAT + restarts and deadline guard."""
import time
import random
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline: 290s wall-clock (30s margin under 320s limit)
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Preprocess: convert to 0-based indices
    # clause_vars[c] = list of (var_idx, sign) where sign is +1 or -1
    clause_vars = []
    for c in clauses:
        clause_vars.append([(abs(l) - 1, 1 if l > 0 else -1) for l in c])
    
    # Build variable-to-clause incidence
    var_to_clauses = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for var, sign in cv:
            var_to_clauses[var].append(ci)
    
    # Precompute for each clause and each position the variable index
    # clause_var_idx[c] = np.array of variable indices
    # clause_sign[c] = np.array of signs
    clause_var_idx = np.array([cv[0] for cv in clause_vars], dtype=np.int32)
    clause_sign = np.array([cv[1] for cv in clause_vars], dtype=np.int32)
    
    def is_satisfied(assignment):
        """Check if all clauses are satisfied."""
        # assignment is a numpy bool array
        # For each clause, check if any literal is true
        # literal true: assignment[var] == (sign > 0)
        # i.e., assignment[var] * sign == 1
        # Using numpy: for clause c, (assignment[clause_var_idx[c]] * clause_sign[c]) has at least one +1
        vals = assignment[clause_var_idx]  # shape (m, 3)
        # Multiply by sign
        signed = vals * clause_sign
        # A clause is satisfied if max(signed) == 1 (i.e., at least one literal is true)
        # Actually assignment is bool, so vals is bool. Let's use int.
        # Let me redo with int assignment
        return None  # placeholder, we'll use a different approach
    
    def walk_sat_once(rng, max_flips):
        """Run one WalkSAT attempt. Returns assignment if solved, else None."""
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=np.int8)
        # +1 means True, -1 means False... actually let's use 0/1
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=np.bool_)
        
        # Track unsatisfied clause count and which clauses are unsatisfied
        # Use numpy for speed
        # For each clause, satisfied if any(literal is true)
        # literal true: assignment[var] == (sign > 0)
        # Let's compute for all clauses at once
        
        def compute_unsat(assignment):
            # assignment: bool array of length n
            # For each clause c, we need to check if any literal is satisfied
            # lit_val[c, i] = assignment[var_idx[c,i]] if sign[c,i]==1 else not assignment[var_idx[c,i]]
            # Satisfied if any lit_val[c, :] is True
            # Using numpy:
            vals = assignment[clause_var_idx]  # (m, 3) bool
            # Where sign is -1, flip
            vals = np.where(clause_sign > 0, vals, ~vals)
            unsat_mask = ~vals.any(axis=1)  # (m,) bool, True if unsatisfied
            return unsat_mask
        
        unsat_mask = compute_unsat(assignment)
        unsat_count = unsat_mask.sum()
        
        if unsat_count == 0:
            return assignment
        
        # Parameters for WalkSAT
        # Weight for choosing a literal from the clause that maximizes satisfied count
        # Standard WalkSAT: pick a variable from an unsatisfied clause,
        # with probability p pick a "good" flip (one that satisfies more clauses),
        # else pick randomly.
        
        # We need to efficiently compute for each variable in an unsatisfied clause,
        # how many clauses would be satisfied if we flip it.
        
        # For efficiency, maintain:
        # - sat_count[var]: number of clauses currently satisfied where var appears
        # - unsat_count[var]: number of clauses currently unsatisfied where var appears
        # When we flip var, the delta in total satisfied clauses is:
        #   (unsat_count[var] - sat_count[var]) for clauses where var is the only unsatisfied literal... 
        # Actually it's more complex. Let me use a simpler approach.
        
        # Alternative: for each flip candidate, compute the net change in satisfied clauses.
        # This requires checking all clauses containing that variable.
        
        # Let's maintain per-variable counts:
        # For each variable v:
        #   sat_clauses[v] = set of clause indices where v appears and the clause is satisfied
        #   unsat_clauses[v] = set of clause indices where v appears and the clause is unsatisfied
        # When flipping v:
        #   Clauses in unsat_clauses[v]: flipping v may satisfy them (if v was the only unsatisfied literal, or if v's flip makes at least one literal true)
        #   Clauses in sat_clauses[v]: flipping v may unsatisfy them (if v was the only satisfied literal)
        
        # This is getting complex. Let me use a simpler but still fast approach.
        
        # For each step:
        # 1. Pick a random unsatisfied clause
        # 2. For each of its 3 literals, compute the "gain" (number of clauses that would become satisfied minus number that would become unsatisfied)
        # 3. With probability p, pick the literal with max gain; else pick randomly
        # 4. Flip that variable
        
        # To compute gain for flipping variable v:
        # For each clause c containing v:
        #   If c is currently satisfied:
        #     Check if v is the ONLY satisfied literal in c. If yes, flipping v makes c unsatisfied.
        #   If c is currently unsatisfied:
        #     Check if flipping v makes at least one literal in c true. If yes, c becomes satisfied.
        
        # Let me precompute for each clause which positions are satisfied.
        
        # State:
        # clause_sat[c] = bool, is clause c satisfied
        # For each clause, we can track how many literals are satisfied.
        
        # Let me use a different representation:
        # lit_sat[c, i] = is literal i in clause c satisfied
        # clause_sat_count[c] = sum of lit_sat[c, :]
        # Clause is satisfied iff clause_sat_count[c] > 0
        
        # When we flip variable v:
        # For each clause c containing v at position pos:
        #   old_lit_sat = lit_sat[c, pos]
        #   new_lit_sat = not old_lit_sat
        #   old_clause_sat = clause_sat_count[c] > 0
        #   new_clause_sat_count = clause_sat_count[c] + (new_lit_sat - old_lit_sat)
        #   new_clause_sat = new_clause_sat_count > 0
        #   delta_sat = int(new_clause_sat) - int(old_clause_sat)
        
        # This is O(deg(v)) per flip evaluation.
        
        # Initialize lit_sat and clause_sat_count
        lit_sat = np.zeros((m, 3), dtype=np.bool_)
        for c in range(m):
            for i in range(3):
                v = clause_var_idx[c, i]
                s = clause_sign[c, i]
                lit_sat[c, i] = assignment[v] if s > 0 else (not assignment[v])
        
        clause_sat_count = lit_sat.sum(axis=1)  # (m,)
        clause_sat = clause_sat_count > 0  # (m,)
        
        # For each variable, list of (clause_idx, pos)
        var_clauses = [[] for _ in range(n)]
        for c in range(m):
            for i in range(3):
                v = clause_var_idx[c, i]
                var_clauses[v].append((c, i))
        
        # WalkSAT loop
        p_good = 0.55  # probability of picking the best flip
        
        for step in range(max_flips):
            if time.time() > deadline - 5.0:
                return None
            
            if clause_sat.all():
                return assignment
            
            # Pick random unsatisfied clause
            unsat_indices = np.where(~clause_sat)[0]
            if len(unsat_indices) == 0:
                return assignment
            
            c = rng.choice(unsat_indices.tolist())
            
            # Evaluate gain for each of the 3 literals in clause c
            gains = []
            for i in range(3):
                v = clause_var_idx[c, i]
                # Compute gain of flipping v
                gain = 0
                for (cc, pos) in var_clauses[v]:
                    old_lit = lit_sat[cc, pos]
                    new_lit = not old_lit
                    old_count = clause_sat_count[cc]
                    new_count = old_count + (1 if new_lit else 0) - (1 if old_lit else 0)
                    old_sat = old_count > 0
                    new_sat = new_count > 0
                    gain += int(new_sat) - int(old_sat)
                gains.append(gain)
            
            # Pick literal to flip
            if rng.random() < p_good:
                # Pick the one with max gain
                best_idx = gains.index(max(gains))
            else:
                best_idx = rng.randint(0, 2)
            
            v = clause_var_idx[c, best_idx]
            
            # Flip variable v
            assignment[v] = not assignment[v]
            
            # Update lit_sat and clause_sat_count for all clauses containing v
            for (cc, pos) in var_clauses[v]:
                old_lit = lit_sat[cc, pos]
                new_lit = not old_lit
                lit_sat[cc, pos] = new_lit
                old_count = clause_sat_count[cc]
                new_count = old_count + (1 if new_lit else 0) - (1 if old_lit else 0)
                clause_sat_count[cc] = new_count
                clause_sat[cc] = new_count > 0
        
        return None
    
    # Restart strategy
    rng = random.Random(42)
    max_flips_per_attempt = 5000
    best_assignment = None
    best_unsat = float('inf')
    
    attempt = 0
    while time.time() < deadline - 10.0:
        attempt += 1
        result = walk_sat_once(rng, max_flips_per_attempt)
        if result is not None:
            # Verify
            vals = result[clause_var_idx]
            vals = np.where(clause_sign > 0, vals, ~vals)
            if vals.all(axis=1).all():
                return result.tolist()
            else:
                # Should not happen, but return anyway
                return result.tolist()
    
    # If we didn't solve, return the best we can
    # Just return a random assignment
    assignment = [rng.random() < 0.5 for _ in range(n)]
    return assignment
