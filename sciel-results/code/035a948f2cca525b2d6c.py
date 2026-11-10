import random
import time
import math
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n
        
    deadline = time.time() + 290

    # 1. Preprocessing: Build Factor Graph Structure
    # clause_vars[c] = list of (var_idx, sign)
    # var_occurrences[v] = list of (clause_idx, sign)
    clause_vars = []
    var_occurrences = [[] for _ in range(n)]
    
    for ci, clause in enumerate(clauses):
        lits = []
        for l in clause:
            vi = abs(l) - 1
            sign = 1 if l > 0 else -1
            lits.append((vi, sign))
            var_occurrences[vi].append((ci, sign))
        clause_vars.append(lits)

    # 2. Belief Propagation / Gibbs Sampling for Initialization
    # We run a short BP-like process to estimate variable biases.
    # This exploits the "planted" nature where the solution is a fixed point of the dynamics.
    
    def run_belief_propagation(max_iters=100, rng=None):
        """
        Runs a simplified Belief Propagation / Gibbs sampling to estimate
        the probability of each variable being True.
        Returns a list of probabilities [0.0, 1.0] for each variable.
        """
        if rng is None:
            rng = random.Random(42)
            
        # Initialize messages or biases randomly
        # bias[v] represents log-odds or a simple probability estimate
        # Let's use a simple iterative message passing or Gibbs update.
        # For 3-SAT, a common heuristic is to look at the fraction of satisfied literals.
        
        # Simple Gibbs Sampling approach to estimate marginals
        # We run this for a few iterations to let the "planted" solution attract the state.
        
        # State: assignment (boolean)
        # We track how often each variable is True in a set of samples
        # Or, we can do a "soft" assignment update.
        
        # Let's try a simple local consistency propagation:
        # 1. Start with random assignment.
        # 2. Iterate: For each variable, look at unsatisfied clauses. 
        #    If a variable appears in many unsatisfied clauses with a specific sign, 
        #    it's likely that setting it to satisfy them is correct.
        
        # Actually, a more robust BP for SAT involves messages from clauses to variables.
        # Message from clause C to variable v: "How likely is v to be set to satisfy C?"
        
        # Given time constraints, a fast heuristic is often better than full BP.
        # Let's use a "Counting" heuristic which is effectively 1-step BP.
        
        # Estimate: For each variable v, count how many clauses are UNSATISFIED 
        # if v is True vs False, assuming other variables are random (0.5).
        # This is computationally expensive if done naively, but we can approximate.
        
        # Alternative: Run a very short WalkSAT and observe the trajectory?
        # No, let's stick to a static analysis + short dynamic phase.
        
        # Let's implement a lightweight BP:
        # msg[c][v] = message from clause c to variable v
        # For 3-SAT, the clause is satisfied if ANY literal is true.
        
        # To keep it fast, we will do a few steps of "Gibbs Sampling" on a 
        # "soft" assignment (probabilities) and average them.
        
        # Initialize probabilities to 0.5
        probs = [0.5] * n
        
        # We will run a few iterations of updating probs based on clause satisfaction.
        # If a clause is likely unsatisfied, push its variables towards satisfying it.
        
        for _ in range(max_iters):
            new_probs = probs[:]
            
            # Sample a few random assignments to estimate clause states? 
            # Too slow. 
            # Let's use a deterministic update based on current probs.
            
            # For each variable v, calculate the "pressure" to be True or False.
            # pressure_true[v] = sum over clauses c containing v with positive sign 
            #                    of P(clause c is unsatisfied given v=True and others random)
            # This is complex.
            
            # Simpler Heuristic: 
            # Count occurrences in "hard" clauses.
            # Let's just do a fast random walk and count frequencies.
            pass
        
        # Since full BP is complex to implement correctly and fast in Python for this timeout,
        # we will use a "Weighted Random Initialization" based on variable frequency.
        # Variables that appear in many clauses are more constrained.
        # However, for PLANTED instances, the solution is uniform.
        
        # Let's try a different structural idea: 
        # "Phase Transition" aware restarts.
        # We will use the BP idea to *guide* the first few flips of the local search,
        # rather than just initializing.
        
        return probs

    # The BP above was a placeholder. Let's implement a concrete, fast "Bias" estimator.
    # In planted SAT, the solution is often correlated with the clause structure.
    # A simple effective method: 
    # 1. Compute degree of each variable.
    # 2. Initialize assignment randomly.
    # 3. Run 100 steps of "MinConflicts" (flip var in most unsat clauses).
    # 4. Use the result of this short run as the "seed" for the main search.
    
    def get_smart_initialization(rng):
        """
        Runs a very short, aggressive local search to find a 'good' starting point.
        This acts as a coarse-grained BP/Gibbs sampler.
        """
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Compute counts
        counts = [0] * m
        for ci in range(m):
            cnt = 0
            for vi, sign in clause_vars[ci]:
                if assignment[vi] == (sign == 1):
                    cnt += 1
            counts[ci] = cnt
            
        # Quick MinConflicts for 200 steps
        for _ in range(200):
            # Find unsatisfied clauses
            unsat = [ci for ci in range(m) if counts[ci] == 0]
            if not unsat:
                break
                
            # Pick random unsat clause
            ci = rng.choice(unsat)
            
            # Find variable in this clause that, if flipped, breaks the fewest other clauses
            # (Standard WalkSAT heuristic, but we just do it a few times)
            best_vi = None
            min_breaks = float('inf')
            
            for vi, sign in clause_vars[ci]:
                # Calculate breaks if we flip vi
                breaks = 0
                for oci, osign in var_occurrences[vi]:
                    if oci == ci:
                        continue
                    # If the literal in oci is currently satisfied, and it's the only one,
                    # flipping vi breaks oci.
                    if assignment[vi] == (osign == 1):
                        if counts[oci] == 1:
                            breaks += 1
                            
                if breaks < min_breaks:
                    min_breaks = breaks
                    best_vi = vi
                    
            if best_vi is not None:
                # Flip
                vi = best_vi
                old_val = assignment[vi]
                new_val = not old_val
                assignment[vi] = new_val
                for oci, osign in var_occurrences[vi]:
                    was_sat = old_val == (osign == 1)
                    now_sat = new_val == (osign == 1)
                    if was_sat and not now_sat:
                        counts[oci] -= 1
                    elif not was_sat and now_sat:
                        counts[oci] += 1
                        
        return assignment, counts

    # 3. Main Solver Loop
    # We alternate between:
    # A) Smart Initialization (MinConflicts warm-up) + WalkSAT
    # B) Pure Random Initialization + WalkSAT (for diversity)
    
    best_unsat = m + 1
    best_assignment = None
    rng = random.Random(12345)
    
    def walk_sat(assignment, counts, max_steps, rng, deadline):
        """
        Standard WalkSAT with incremental updates.
        """
        p = 0.5 # Probability of choosing a variable from the unsat clause (vs random)
        
        for step in range(max_steps):
            if time.time() > deadline:
                return False, assignment, counts
                
            # Find unsatisfied clauses
            unsat = [ci for ci in range(m) if counts[ci] == 0]
            if not unsat:
                return True, assignment, counts
                
            # Pick a random unsatisfied clause
            ci = rng.choice(unsat)
            
            # Decide which variable to flip
            if rng.random() < p:
                # Choose from literals in the unsatisfied clause
                # Heuristic: Flip the variable that breaks the fewest other clauses
                best_vi = None
                min_breaks = float('inf')
                
                for vi, sign in clause_vars[ci]:
                    breaks = 0
                    for oci, osign in var_occurrences[vi]:
                        if oci == ci:
                            continue
                        # If literal in oci is satisfied, and it's the only one,
                        # flipping vi breaks oci.
                        if assignment[vi] == (osign == 1):
                            if counts[oci] == 1:
                                breaks += 1
                                
                    if breaks < min_breaks:
                        min_breaks = breaks
                        best_vi = vi
                vi = best_vi
            else:
                # Random variable
                vi = rng.randrange(n)
                
            # Flip variable vi
            old_val = assignment[vi]
            new_val = not old_val
            assignment[vi] = new_val
            
            # Update counts
            for oci, osign in var_occurrences[vi]:
                was_sat = old_val == (osign == 1)
                now_sat = new_val == (osign == 1)
                if was_sat and not now_sat:
                    counts[oci] -= 1
                elif not was_sat and now_sat:
                    counts[oci] += 1
                    
        return False, assignment, counts

    max_restarts = 1000
    steps_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 2:
            break
            
        # Alternate between Smart and Random init
        if restart % 3 != 2: # 2/3 Smart, 1/3 Random
            assignment, counts = get_smart_initialization(rng)
        else:
            assignment = [rng.random() < 0.5 for _ in range(n)]
            counts = [0] * m
            for ci in range(m):
                cnt = 0
                for vi, sign in clause_vars[ci]:
                    if assignment[vi] == (sign == 1):
                        cnt += 1
                counts[ci] = cnt
        
        # Check if solved
        cur_unsat = sum(1 for c in counts if c == 0)
        if cur_unsat == 0:
            return assignment
            
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
            
        # Run WalkSAT
        solved, assignment, counts = walk_sat(assignment, counts, steps_per_restart, rng, deadline)
        
        if solved:
            return assignment
            
        cur_unsat = sum(1 for c in counts if c == 0)
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]

    if best_assignment is not None:
        return best_assignment
    return [False] * n
