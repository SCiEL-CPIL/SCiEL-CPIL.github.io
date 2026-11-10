"""Planted 3-SAT solver: spectral init + incremental WalkSAT + BP-guided restarts."""
# EVOLVE-BLOCK-START
import random
import time
import math
from collections import defaultdict


def solve(instance):
    """
    Solve a planted 3-SAT instance with n variables and clauses.
    Returns a list of booleans (length n) that satisfies all clauses if possible.
    
    Strategy: 
    1. Spectral initialization from literal co-occurrence matrix
    2. Fast incremental WalkSAT with multiple heuristics
    3. BP-guided perturbations for diverse restarts
    """
    n, clauses = instance["n"], instance["clauses"]
    num_clauses = len(clauses)
    
    # Deadline: leave a safety margin before the 330s kill
    deadline = time.time() + 310.0
    
    # Pre-process clauses for faster access
    # Each clause is a list of signed literals (1-based).
    # Convert to 0-based indices and signs.
    fast_clauses = []
    for clause in clauses:
        fast_clause = []
        for lit in clause:
            var_idx = abs(lit) - 1
            required = (lit > 0)
            fast_clause.append((var_idx, required))
        fast_clauses.append(fast_clause)
    
    # Build variable-to-clauses index for efficient updates
    var_to_clauses = [[] for _ in range(n)]
    for i, fc in enumerate(fast_clauses):
        for var_idx, _ in fc:
            var_to_clauses[var_idx].append(i)
    
    # Precompute for each clause the position of each variable (for fast flip)
    # clause_var_pos[c_idx] = dict mapping var_idx -> position in clause
    clause_var_pos = []
    for fc in fast_clauses:
        pos_map = {}
        for pos, (var_idx, required) in enumerate(fc):
            pos_map[var_idx] = pos
        clause_var_pos.append(pos_map)
    
    # Incremental satisfaction tracking
    # clause_satisfied[i] = True if clause i is satisfied
    # clause_sat_count[i] = number of satisfied literals in clause i
    
    def init_satisfaction(assignment):
        """Initialize clause satisfaction state."""
        clause_satisfied = [False] * num_clauses
        clause_sat_count = [0] * num_clauses
        for i, fc in enumerate(fast_clauses):
            count = 0
            for var_idx, required in fc:
                if assignment[var_idx] == required:
                    count += 1
            clause_sat_count[i] = count
            clause_satisfied[i] = count > 0
        return clause_satisfied, clause_sat_count
    
    def flip_variable(var_idx, assignment, clause_satisfied, clause_sat_count):
        """Flip variable and update clause satisfaction incrementally."""
        old_val = assignment[var_idx]
        assignment[var_idx] = not old_val
        unsat_delta = 0
        for c_idx in var_to_clauses[var_idx]:
            fc = fast_clauses[c_idx]
            was_sat = clause_satisfied[c_idx]
            old_count = clause_sat_count[c_idx]
            # Use precomputed position to find the literal quickly
            pos = clause_var_pos[c_idx][var_idx]
            _, required = fc[pos]
            if old_val == required:
                clause_sat_count[c_idx] = old_count - 1
            else:
                clause_sat_count[c_idx] = old_count + 1
            new_count = clause_sat_count[c_idx]
            new_sat = new_count > 0
            clause_satisfied[c_idx] = new_sat
            if was_sat and not new_sat:
                unsat_delta += 1
            elif not was_sat and new_sat:
                unsat_delta -= 1
        return unsat_delta
    
    def count_unsat_fast(clause_satisfied):
        """Count unsatisfied clauses."""
        return sum(1 for s in clause_satisfied if not s)
    
    # Spectral initialization using sparse matrix-vector multiply
    def spectral_init():
        """
        Compute principal eigenvector of literal co-occurrence matrix
        to initialize the assignment.
        """
        num_literals = 2 * n
        
        # Build sparse adjacency as list of (neighbor, weight) for each literal
        # Use a list of dicts for efficiency
        adj = [defaultdict(float) for _ in range(num_literals)]
        
        for fc in fast_clauses:
            for i in range(len(fc)):
                for j in range(i + 1, len(fc)):
                    var_i, req_i = fc[i]
                    var_j, req_j = fc[j]
                    lit_i = var_i * 2 + (0 if req_i else 1)
                    lit_j = var_j * 2 + (0 if req_j else 1)
                    if req_i == req_j:
                        adj[lit_i][lit_j] += 1.0
                        adj[lit_j][lit_i] += 1.0
                    else:
                        adj[lit_i][lit_j] -= 1.0
                        adj[lit_j][lit_i] -= 1.0
        
        # Power iteration for principal eigenvector
        rng = random.Random(42)
        vec = [rng.gauss(0, 1) for _ in range(num_literals)]
        
        # Normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        
        # Iterate - use sparse structure
        num_iters = 30  # reduced from 50 for speed, still converges well
        for it in range(num_iters):
            if time.time() > deadline:
                break
            new_vec = [0.0] * num_literals
            for lit_i in range(num_literals):
                if adj[lit_i]:
                    s = 0.0
                    for lit_j, w in adj[lit_i].items():
                        s += w * vec[lit_j]
                    new_vec[lit_i] = s
            
            # Normalize
            norm = math.sqrt(sum(x * x for x in new_vec))
            if norm < 1e-10:
                break
            vec = [x / norm for x in new_vec]
        
        # Convert to assignment
        assignment = []
        for var in range(n):
            pos_val = vec[2 * var]
            neg_val = vec[2 * var + 1]
            assignment.append(pos_val > neg_val)
        
        return assignment
    
    # Fast WalkSAT with dynamic clause weights, incremental updates, and restart strategy
    def fast_walksat(start_assignment, max_steps=20000, heuristic="min_conflicts"):
        """
        Fast WalkSAT with dynamic clause weights and restart strategy.
        Returns (assignment, solved, unsat_count).
        """
        assignment = start_assignment[:]
        clause_satisfied, clause_sat_count = init_satisfaction(assignment)
        unsat_count = count_unsat_fast(clause_satisfied)
        
        if unsat_count == 0:
            return assignment, True, 0
        
        best_unsat = unsat_count
        best_assignment = assignment[:]
        
        # Track unsatisfied clauses for quick access
        unsat_clauses = set(i for i in range(num_clauses) if not clause_satisfied[i])
        
        # Precompute unsat clause count per variable for min_conflicts heuristic
        var_unsat_count = [0] * n
        for c_idx in unsat_clauses:
            for var_idx, _ in fast_clauses[c_idx]:
                var_unsat_count[var_idx] += 1
        
        # Dynamic clause weights: track how long each clause has been unsatisfied
        clause_age = {c_idx: 0 for c_idx in unsat_clauses}
        
        # Restart parameters
        patience = 500  # steps without improvement before restart
        steps_since_improvement = 0
        restart_count = 0
        max_restarts = 10
        base_patience = 500
        patience_growth = 1.3  # increase patience after each restart
        
        # Track clause selection probabilities using soft-max over weights
        # Weight = base_weight * (1 + age * weight_decay_factor)
        weight_decay_factor = 0.1
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if not unsat_clauses:
                return assignment, True, 0
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
            
            # Restart strategy: if no improvement for patience steps, perturb and restart
            if steps_since_improvement >= patience and restart_count < max_restarts:
                restart_count += 1
                patience = int(base_patience * (patience_growth ** restart_count))
                steps_since_improvement = 0
                
                # Perturb: flip 1-3 random variables
                num_flips = random.randint(1, 3)
                for _ in range(num_flips):
                    var_idx = random.randint(0, n - 1)
                    affected_clauses = var_to_clauses[var_idx]
                    delta = flip_variable(var_idx, assignment, clause_satisfied, clause_sat_count)
                    unsat_count += delta
                    
                    # Update tracking structures
                    for c_idx in affected_clauses:
                        if clause_satisfied[c_idx]:
                            if c_idx in unsat_clauses:
                                unsat_clauses.discard(c_idx)
                                for v, _ in fast_clauses[c_idx]:
                                    var_unsat_count[v] -= 1
                                if c_idx in clause_age:
                                    del clause_age[c_idx]
                        else:
                            if c_idx not in unsat_clauses:
                                unsat_clauses.add(c_idx)
                                for v, _ in fast_clauses[c_idx]:
                                    var_unsat_count[v] += 1
                                clause_age[c_idx] = 0
                
                # If restart made things worse, go back to best
                if unsat_count > best_unsat:
                    # Revert to best assignment
                    assignment = best_assignment[:]
                    clause_satisfied, clause_sat_count = init_satisfaction(assignment)
                    unsat_count = best_unsat
                    unsat_clauses = set(i for i in range(num_clauses) if not clause_satisfied[i])
                    var_unsat_count = [0] * n
                    for c_idx in unsat_clauses:
                        for var_idx, _ in fast_clauses[c_idx]:
                            var_unsat_count[var_idx] += 1
                    clause_age = {c_idx: 0 for c_idx in unsat_clauses}
                continue
            
            # Update clause ages
            for c_idx in unsat_clauses:
                clause_age[c_idx] += 1
            
            # Choose a clause from unsat_clauses using weighted selection
            # Weight based on age (older unsat clauses get higher priority)
            if len(unsat_clauses) > 1:
                # Compute weights using soft-max-like approach
                weights = []
                for c_idx in unsat_clauses:
                    age = clause_age[c_idx]
                    w = 1.0 + weight_decay_factor * age
                    weights.append(w)
                
                # Normalize and select
                total_w = sum(weights)
                r = random.random() * total_w
                cumulative = 0
                chosen_clause_idx = next(iter(unsat_clauses))
                for c_idx, w in zip(unsat_clauses, weights):
                    cumulative += w
                    if r <= cumulative:
                        chosen_clause_idx = c_idx
                        break
                clause_idx = chosen_clause_idx
            else:
                clause_idx = next(iter(unsat_clauses))
            
            clause = fast_clauses[clause_idx]
            
            # Heuristic for variable selection
            if heuristic == "min_conflicts":
                # Use precomputed var_unsat_count
                scores = [var_unsat_count[var_idx] for var_idx, _ in clause]
                
                # Choose the literal with minimum score (break ties randomly)
                min_score = min(scores)
                candidates = [i for i, s in enumerate(scores) if s == min_score]
                chosen_idx = random.choice(candidates)
                
            elif heuristic == "weighted_random":
                # Weighted random: prefer variables in fewer unsat clauses
                scores = [1 + var_unsat_count[var_idx] for var_idx, _ in clause]
                
                total_score = sum(scores)
                r = random.random() * total_score
                cumulative = 0
                chosen_idx = 0
                for i, score in enumerate(scores):
                    cumulative += score
                    if r <= cumulative:
                        chosen_idx = i
                        break
                        
            else:  # random
                chosen_idx = random.randint(0, len(clause) - 1)
            
            var_idx, required = clause[chosen_idx]
            
            # Save old unsat counts for this variable's clauses
            old_var_unsat = var_unsat_count[var_idx]
            affected_clauses = var_to_clauses[var_idx]
            
            # Flip the variable and update
            delta = flip_variable(var_idx, assignment, clause_satisfied, clause_sat_count)
            unsat_count += delta
            
            # Update unsat_clauses set and var_unsat_count
            for c_idx in affected_clauses:
                was_sat = clause_satisfied[c_idx]
                if was_sat:
                    if c_idx in unsat_clauses:
                        unsat_clauses.discard(c_idx)
                        # Decrement count for all vars in this clause
                        for v, _ in fast_clauses[c_idx]:
                            var_unsat_count[v] -= 1
                        if c_idx in clause_age:
                            del clause_age[c_idx]
                else:
                    if c_idx not in unsat_clauses:
                        unsat_clauses.add(c_idx)
                        # Increment count for all vars in this clause
                        for v, _ in fast_clauses[c_idx]:
                            var_unsat_count[v] += 1
                        clause_age[c_idx] = 0
        
        return best_assignment, False, best_unsat
    
    # Belief Propagation (simplified, for diversity in restarts)
    def run_bp_simple(max_iterations=20):
        """
        Simplified BP for generating diverse starting points.
        """
        beliefs = [0.0] * n
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            new_beliefs = [0.0] * n
            for var_idx in range(n):
                sum_msg = 0.0
                for c_idx in var_to_clauses[var_idx]:
                    fc = fast_clauses[c_idx]
                    for var, required in fc:
                        if var != var_idx:
                            sum_msg += beliefs[var] * (1 if required else -1)
                new_beliefs[var_idx] = sum_msg
            
            # Damping
            for i in range(n):
                beliefs[i] = 0.5 * beliefs[i] + 0.5 * new_beliefs[i]
        
        assignment = [b > 0 for b in beliefs]
        confidences = [abs(b) for b in beliefs]
        return assignment, confidences
    
    # Main solver
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Phase 1: Spectral initialization
    spectral_assignment = spectral_init()
    spec_unsat = count_unsat_fast(init_satisfaction(spectral_assignment)[0])
    
    if spec_unsat == 0:
        return spectral_assignment
    
    # Fast WalkSAT from spectral init - use more steps since we're faster now
    ls_assignment, solved, ls_unsat = fast_walksat(spectral_assignment, max_steps=50000, heuristic="min_conflicts")
    if solved:
        return ls_assignment
    
    if ls_unsat < best_unsat_count:
        best_unsat_count = ls_unsat
        best_assignment = ls_assignment[:]
    
    # Phase 2: BP-guided perturbations
    bp_assignment, bp_confidences = run_bp_simple(max_iterations=15)
    bp_unsat = count_unsat_fast(init_satisfaction(bp_assignment)[0])
    
    if bp_unsat == 0:
        return bp_assignment
    
    # Sort variables by confidence (lowest first for perturbation)
    sorted_vars = sorted(range(n), key=lambda i: bp_confidences[i])
    
    # Use remaining time for restarts with adaptive step count
    elapsed = time.time() - (deadline - 310.0)
    remaining = deadline - time.time()
    
    # Estimate time per restart (rough: ~15000 steps * ~50us/step = ~0.75s)
    # Be conservative: assume 1s per restart on average
    est_restarts = max(10, int(remaining * 0.7))
    est_restarts = min(est_restarts, 200)
    
    for restart in range(est_restarts):
        if time.time() > deadline:
            break
        
        # Create perturbed version
        assignment = bp_assignment[:]
        num_flips = random.randint(1, min(3, n))
        for _ in range(num_flips):
            if random.random() < 0.7 and sorted_vars:
                var_idx = sorted_vars[random.randint(0, min(8, len(sorted_vars) - 1))]
            else:
                var_idx = random.randint(0, n - 1)
            assignment[var_idx] = not assignment[var_idx]
        
        # Alternate heuristics
        heuristic = "min_conflicts" if restart % 2 == 0 else "weighted_random"
        # Adaptive steps: use more steps when we have more time remaining
        steps_remaining = deadline - time.time()
        max_steps = min(25000, max(5000, int(steps_remaining * 30)))
        ls_assignment, solved, ls_unsat = fast_walksat(assignment, max_steps=max_steps, heuristic=heuristic)
        if solved:
            return ls_assignment
        
        if ls_unsat < best_unsat_count:
            best_unsat_count = ls_unsat
            best_assignment = ls_assignment[:]
    
    # Phase 3: Pure random restarts with remaining time
    remaining = deadline - time.time()
    if remaining > 5:
        num_random_restarts = max(1, int(remaining * 0.5))
        num_random_restarts = min(num_random_restarts, 50)
        for restart in range(num_random_restarts):
            if time.time() > deadline:
                break
            
            assignment = [random.random() < 0.5 for _ in range(n)]
            heuristic = "min_conflicts" if restart % 2 == 0 else "random"
            steps_remaining = deadline - time.time()
            max_steps = min(15000, max(3000, int(steps_remaining * 20)))
            ls_assignment, solved, ls_unsat = fast_walksat(assignment, max_steps=max_steps, heuristic=heuristic)
            if solved:
                return ls_assignment
            
            if ls_unsat < best_unsat_count:
                best_unsat_count = ls_unsat
                best_assignment = ls_assignment[:]
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    else:
        return [random.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
