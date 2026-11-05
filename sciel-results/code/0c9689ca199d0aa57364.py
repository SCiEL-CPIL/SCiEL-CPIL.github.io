import random
import time
import math

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline: 315 seconds to allow for overhead and safety margin against 330s kill
    deadline = time.time() + 315
    
    # Pre-process clauses for faster satisfaction check
    # Store as list of tuples (var_index, is_positive)
    # 1-based literals -> 0-based indices
    processed_clauses = []
    for c in clauses:
        pc = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            pc.append((var_idx, is_pos))
        processed_clauses.append(pc)
        
    m = len(processed_clauses)
    
    # Precompute variable to clause mapping for efficient cost calculation
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(processed_clauses):
        for var_idx, _ in c:
            var_to_clauses[var_idx].append(i)
    
    # Helper to check if a clause is satisfied
    def is_clause_satisfied(clause_idx, assignment):
        for var_idx, is_pos in processed_clauses[clause_idx]:
            if assignment[var_idx] == is_pos:
                return True
        return False

    # Helper to count satisfied clauses (for initial state and potential debugging, but we track incrementally)
    def count_satisfied(assignment):
        count = 0
        for i in range(m):
            if is_clause_satisfied(i, assignment):
                count += 1
        return count

    # Simulated Annealing / WalkSAT Hybrid
    # We maintain a count of unsatisfied clauses.
    # When we flip a variable, we update the count efficiently by checking only clauses containing that variable.
    
    def solve_sa_run(rng, time_limit):
        """
        Runs a single Simulated Annealing / WalkSAT hybrid search.
        Returns (assignment, success_flag).
        """
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initialize clause satisfaction status
        clause_sat = [False] * m
        unsat_count = 0
        for i in range(m):
            if is_clause_satisfied(i, assignment):
                clause_sat[i] = True
            else:
                unsat_count += 1
                
        if unsat_count == 0:
            return assignment, True

        # Temperature schedule
        # Start high to allow exploration, decay to allow exploitation.
        # For n=50, typical path length to solution might be 50-200 flips.
        # Let's run for a fixed number of steps or time.
        
        start_time = time.time()
        T_initial = 1.0
        T_final = 0.01
        # Decay rate: we want T to reach T_final after roughly 'steps' iterations.
        # Let's define a step budget based on time, but we need to estimate steps per second.
        # Instead, let's use a fixed number of steps for the SA part, then switch to pure greedy/WalkSAT if stuck?
        # Or just run until time limit.
        
        # Let's use a step limit that adapts to time.
        # Estimate: ~10k-50k steps per second in Python for this n=50 case.
        # Let's just loop until time limit, with a max step cap per "segment" to allow restarts?
        # No, the outer loop handles restarts. This function runs for a specific time slice.
        
        # To make SA effective, we need a good step count.
        # Let's run for up to 50,000 steps or until time_limit is reached.
        max_steps = 50000
        
        for step in range(max_steps):
            if time.time() > time_limit:
                return assignment, False
            
            # Check if solved
            if unsat_count == 0:
                return assignment, True
            
            # Temperature decay
            # Linear decay or exponential? Exponential is common.
            # T = T_initial * (T_final/T_initial)^step
            # But we don't know total steps.
            # Let's use a fixed decay per step, or restart temperature periodically?
            # Simpler: Use a fixed temperature that decays slowly, or use WalkSAT's noise parameter analogy.
            
            # In SA, we accept worse moves with probability exp(-delta/T).
            # Here, "delta" is the change in unsat_count.
            # If we flip a var, unsat_count can change by -3, -2, -1, 0, 1, 2, 3.
            
            # Heuristic: Pick a clause to focus on.
            # Standard WalkSAT picks a random unsat clause.
            # SA might pick a clause proportional to its "weight" or just random.
            # Let's stick to picking a random unsat clause to guide the search.
            
            # Find an unsat clause.
            # For efficiency, we could maintain a list of unsat clause indices.
            # But updating that list on every flip is complex.
            # For n=50, m~250, scanning for an unsat clause is O(m) = 250 ops.
            # 50k steps * 250 ops = 12.5M ops. ~0.5s. Acceptable.
            
            # Pick random unsat clause
            unsat_clause_idx = -1
            # We can pick a random index and check if it's unsat, or scan.
            # Scanning is deterministic in cost. Random probing is variable.
            # Let's scan to find the first unsat, or pick a random index from 0..m-1 and check.
            # Actually, picking a random clause from the *set* of unsat clauses is better for diversity.
            # Let's just pick a random index 0..m-1. If it's sat, pick another?
            # No, that biases towards high indices if many are sat.
            # Better: Scan to collect unsat indices? Too slow if done every step.
            # Compromise: Pick a random index. If sat, try a few more times, else scan.
            
            # Simple approach: Pick random index. If sat, continue to next step? No, we must flip something.
            # Let's pick a random index. If it's sat, we don't have an unsat clause to "target".
            # But SA flips variables based on global energy.
            
            # Alternative SA approach:
            # Pick a random variable. Flip it. Calculate delta E. Accept with prob.
            # This is pure SA. It might be slow to converge to SAT.
            # Hybrid: Pick a variable from an unsat clause (like WalkSAT) to ensure we are "working on" a problem.
            
            # Let's find an unsat clause efficiently.
            # We can maintain a count of unsat clauses.
            # To pick a random unsat clause without a list, we can use rejection sampling?
            # Or just scan.
            
            # Let's scan for the first unsat clause. This might bias towards lower indices,
            # but combined with random variable selection within the clause, it's okay.
            # To reduce bias, we can pick a random start index for the scan.
            
            start_scan = rng.randint(0, m - 1)
            found_unsat = False
            for i in range(m):
                idx = (start_scan + i) % m
                if not clause_sat[idx]:
                    unsat_clause_idx = idx
                    found_unsat = True
                    break
            
            if not found_unsat:
                return assignment, True # Should have been caught by unsat_count check

            clause = processed_clauses[unsat_clause_idx]
            
            # Choose a variable from this clause to flip
            var_idx, is_pos = rng.choice(clause)
            
            # Calculate delta in unsat_count if we flip var_idx
            # Current state: assignment[var_idx]
            # New state: not assignment[var_idx]
            
            delta = 0
            # Check all clauses containing var_idx
            for c_idx in var_to_clauses[var_idx]:
                c = processed_clauses[c_idx]
                
                # Count how many literals in this clause are satisfied currently
                # And how many would be satisfied after flip
                # Since it's 3-SAT, we can just check the 3 literals.
                
                # Current satisfaction
                cur_sat = 0
                new_sat = 0
                for v, p in c:
                    if v == var_idx:
                        # This literal's truth value changes
                        if assignment[v] == p:
                            cur_sat += 1
                        # After flip, assignment[v] becomes not assignment[v]
                        # So if it was True (satisfied), it becomes False (unsatisfied) and vice versa
                        # Wait, p is the required polarity.
                        # If assignment[v] == p, literal is True.
                        # After flip, assignment[v] != p, so literal is False.
                        # So if cur_lit_sat: new_lit_sat = 0, else 1.
                    else:
                        if assignment[v] == p:
                            cur_sat += 1
                            new_sat += 1
                        # else both 0
                
                # Determine if clause is satisfied currently
                cur_clause_sat = (cur_sat > 0)
                # Determine if clause would be satisfied after flip
                # We need to know the new_sat count.
                # Let's re-calculate new_sat properly.
                
                # Optimization: 
                # If cur_clause_sat is False (unsat), then cur_sat == 0.
                # If we flip a var in this clause, and that var's literal becomes True, new_sat >= 1 -> Sat.
                # If that var's literal becomes False, new_sat == 0 -> Unsat.
                
                # If cur_clause_sat is True, it might become False only if the ONLY satisfied literal was the one we flipped.
                
                # Let's just compute new_sat for the clause containing var_idx.
                # Actually, we are iterating over ALL clauses containing var_idx.
                
                # Let's do a direct check for new satisfaction.
                # This is O(3) per clause.
                
                # Current state of this clause
                if clause_sat[c_idx]:
                    # It is currently satisfied.
                    # It will become unsatisfied ONLY if the literal for var_idx was the ONLY one satisfied.
                    # Check if any other literal is satisfied.
                    other_sat = False
                    for v, p in c:
                        if v != var_idx and assignment[v] == p:
                            other_sat = True
                            break
                    if not other_sat:
                        # Only var_idx's literal was satisfying it.
                        # If we flip var_idx, does its literal become unsatisfied?
                        # Yes, flipping always changes the truth value of the literal.
                        # So if it was satisfied, it becomes unsatisfied.
                        # So clause becomes unsatisfied.
                        delta -= 1 # We are losing a satisfied clause
                    else:
                        delta += 0 # Still satisfied
                else:
                    # It is currently unsatisfied.
                    # It will become satisfied ONLY if the literal for var_idx becomes satisfied after flip.
                    # Currently, assignment[var_idx] != is_pos (since clause is unsat, no literal is sat).
                    # Wait, if clause is unsat, NO literal is satisfied.
                    # So assignment[var_idx] != is_pos is TRUE for the var_idx literal.
                    # After flip, assignment[var_idx] becomes not assignment[var_idx].
                    # So it becomes == is_pos.
                    # So the literal for var_idx WILL be satisfied.
                    # Therefore, the clause WILL become satisfied.
                    delta += 1 # We are gaining a satisfied clause
            
            # delta is the change in number of satisfied clauses.
            # We want to maximize satisfied clauses.
            # If delta > 0, we improve. Always accept.
            # If delta < 0, we worsen. Accept with probability exp(delta / T).
            # If delta == 0, accept with prob 1 (or 0.5 to add noise?). Let's accept 1.
            
            # Temperature T
            # We need a T that allows some worsening moves early, and fewer later.
            # Let's use a step-based decay.
            # T = T_init * (decay_rate)^step
            # Let's say T_init = 1.0, decay such that T ~ 0.1 after 1000 steps.
            # 0.1 = 1.0 * d^1000 => d = 0.1^(1/1000) ~ 0.9977
            T = 1.0 * (0.9977 ** step)
            
            accept = False
            if delta >= 0:
                accept = True
            else:
                # delta is negative
                # prob = exp(delta / T)
                # delta is at most -3. T is >= 0.01.
                # exp(-3/0.01) = exp(-300) ~ 0.
                # exp(-3/1.0) = exp(-3) ~ 0.05.
                try:
                    prob = math.exp(delta / T)
                except OverflowError:
                    prob = 0.0
                accept = (rng.random() < prob)
            
            if accept:
                assignment[var_idx] = not assignment[var_idx]
                
                # Update clause_sat and unsat_count
                unsat_count -= delta # Wait, delta is change in SAT count.
                # If delta = +1 (one more sat), unsat_count decreases by 1.
                # If delta = -1 (one less sat), unsat_count increases by 1.
                # So unsat_count_new = unsat_count_old - delta.
                # Yes.
                
                # Update clause_sat flags for affected clauses
                for c_idx in var_to_clauses[var_idx]:
                    # Re-evaluate satisfaction for this clause
                    # Since we only flipped one var, and we know the delta logic, 
                    # we can update the flag directly if we tracked which ones changed?
                    # Easier: just re-check satisfaction for these few clauses.
                    # It's O(3) per clause.
                    if is_clause_satisfied(c_idx, assignment):
                        clause_sat[c_idx] = True
                    else:
                        clause_sat[c_idx] = False

        return assignment, False

    # Main loop: Restarts
    # Each restart runs SA for a fixed time slice or step count.
    # Let's use time slices of 0.5 seconds.
    
    rng_master = random.Random()
    
    while time.time() < deadline - 2:
        # Create a new RNG for this run
        run_rng = random.Random(rng_master.randint(0, 2**31))
        
        # Run SA for a short time slice
        # If it solves, return.
        # If not, restart.
        
        # How long per run?
        # If we run for 0.5s, we can do ~600 runs in 300s.
        # Each run does ~50k steps? No, 50k steps in 0.5s is 100k steps/s.
        # Is that feasible?
        # 50k steps * (scan 250 clauses + update 3 clauses) ~ 50k * 300 ops = 15M ops.
        # 15M ops in 0.5s is 30M ops/s. Python is slower than that for complex logic.
        # So 50k steps might take 1-2 seconds.
        # Let's limit steps to 10,000 per run to ensure we fit in 0.5s?
        # Or just let it run until time slice ends.
        
        run_deadline = time.time() + 0.5
        assignment, success = solve_sa_run(run_rng, run_deadline)
        
        if success:
            return assignment

    # Fallback
    return [False] * n
