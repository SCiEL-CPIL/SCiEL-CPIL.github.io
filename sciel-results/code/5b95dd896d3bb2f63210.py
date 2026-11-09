import random
import time
from collections import defaultdict

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 280

    # Precompute variable to clauses mapping
    # var_clauses[v] = list of (clause_idx, sign) where sign is True if var is positive
    var_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            var_clauses[v].append((ci, lit > 0))

    def get_sat_count(assignment):
        """Calculate number of satisfied literals for each clause."""
        sat_counts = [0] * m
        for ci, clause in enumerate(clauses):
            count = 0
            for lit in clause:
                v = abs(lit) - 1
                is_pos = lit > 0
                if assignment[v] == is_pos:
                    count += 1
            sat_counts[ci] = count
        return sat_counts

    def find_unsat(sat_counts):
        """Return list of unsatisfied clause indices."""
        return [i for i, c in enumerate(sat_counts) if c == 0]

    def flip_var(assignment, v, sat_counts, unsat_set):
        """
        Flip variable v. Update sat_counts and unsat_set.
        Returns None.
        """
        old_val = assignment[v]
        new_val = not old_val
        assignment[v] = new_val
        
        for ci, is_pos in var_clauses[v]:
            old_sat = 1 if (old_val == is_pos) else 0
            new_sat = 1 if (new_val == is_pos) else 0
            
            delta = new_sat - old_sat
            old_count = sat_counts[ci]
            new_count = old_count + delta
            sat_counts[ci] = new_count
            
            if old_count == 0 and new_count > 0:
                # Clause became satisfied
                unsat_set.discard(ci)
            elif old_count > 0 and new_count == 0:
                # Clause became unsatisfied
                unsat_set.add(ci)
            # If old_count > 1 and new_count >= 1, it stays satisfied (not in unsat_set)
            # If old_count == 0 and new_count == 0, it stays unsatisfied (in unsat_set)

    def walksat_step(assignment, sat_counts, unsat_set, rng, p_walk):
        """
        Perform one WalkSAT step.
        """
        if not unsat_set:
            return True
        
        # Pick a random unsatisfied clause
        # Converting set to list is O(U). U is usually small.
        unsat_list = list(unsat_set)
        ci = rng.choice(unsat_list)
        clause = clauses[ci]
        
        # Determine if we should use WalkSAT heuristic or random flip
        if rng.random() < p_walk:
            best_var = -1
            best_score = -1000
            
            for lit in clause:
                v = abs(lit) - 1
                is_pos = lit > 0
                
                # Calculate gains and losses for flipping v
                gains = 0
                losses = 0
                
                for c2, is_pos2 in var_clauses[v]:
                    if c2 == ci:
                        continue
                    
                    old_val = assignment[v]
                    new_val = not old_val
                    
                    # Check status of clause c2 before and after flip
                    # We need to know if c2 is currently satisfied and if it will be.
                    # sat_counts[c2] > 0 means satisfied.
                    
                    # Literal contribution change
                    old_lit_sat = 1 if (old_val == is_pos2) else 0
                    new_lit_sat = 1 if (new_val == is_pos2) else 0
                    
                    if old_lit_sat == 0 and new_lit_sat == 1:
                        # This literal becomes satisfied.
                        # If the clause was unsat (count 0), it becomes sat.
                        # If the clause was sat (count > 0), it stays sat.
                        if sat_counts[c2] == 0:
                            gains += 1
                    elif old_lit_sat == 1 and new_lit_sat == 0:
                        # This literal becomes unsatisfied.
                        # If the clause was sat with count 1, it becomes unsat.
                        # If the clause was sat with count > 1, it stays sat.
                        if sat_counts[c2] == 1:
                            losses += 1
                            
                score = gains - losses
                if score > best_score:
                    best_score = score
                    best_var = v
            
            if best_var >= 0:
                flip_var(assignment, best_var, sat_counts, unsat_set)
            else:
                # Fallback to random
                lit = rng.choice(clause)
                flip_var(assignment, abs(lit) - 1, sat_counts, unsat_set)
        else:
            lit = rng.choice(clause)
            flip_var(assignment, abs(lit) - 1, sat_counts, unsat_set)
            
        return len(unsat_set) == 0

    def polish(assignment, sat_counts, unsat_set, rng, max_steps=1000):
        """
        Try to solve remaining unsat clauses using pairwise flips if single flips fail.
        """
        if not unsat_set:
            return True
            
        for _ in range(max_steps):
            if time.time() > deadline:
                break
            if not unsat_set:
                return True
                
            # Try single flips first (WalkSAT)
            if walksat_step(assignment, sat_counts, unsat_set, rng, 1.0):
                return True
                
            # If stuck, try pairwise flips?
            # For 3-SAT, pairwise flips can be expensive.
            # Instead, let's just do more WalkSAT with lower p_walk to explore.
            # Or restart.
            
        return len(unsat_set) == 0

    best_assignment = None
    best_unsat_count = m + 1

    for restart in range(1000):
        if time.time() > deadline:
            break
            
        rng = random.Random(restart * 104729 + 1)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        sat_counts = get_sat_count(assignment)
        unsat_set = set(find_unsat(sat_counts))
        cur_unsat = len(unsat_set)
        
        if cur_unsat < best_unsat_count:
            best_unsat_count = cur_unsat
            best_assignment = assignment[:]
            
        if cur_unsat == 0:
            return assignment

        # Adaptive parameters
        p_walk = 0.5
        steps = 0
        max_steps = 5000
        stall_count = 0
        prev_unsat = cur_unsat
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            if not unsat_set:
                return assignment
                
            walksat_step(assignment, sat_counts, unsat_set, rng, p_walk)
            steps += 1
            
            # Check progress every 50 steps
            if step % 50 == 0:
                cur_unsat = len(unsat_set)
                if cur_unsat < best_unsat_count:
                    best_unsat_count = cur_unsat
                    best_assignment = assignment[:]
                    
                if cur_unsat < prev_unsat:
                    stall_count = 0
                    # Make progress: slightly reduce p_walk to be more greedy?
                    # Or keep it?
                    p_walk = max(0.2, p_walk - 0.05)
                else:
                    stall_count += 1
                    # Stall: increase randomness
                    p_walk = min(0.8, p_walk + 0.05)
                    
                prev_unsat = cur_unsat
                
                if stall_count > 30:
                    break
                    
        if len(unsat_set) == 0:
            return assignment

    return best_assignment if best_assignment else [False] * n
