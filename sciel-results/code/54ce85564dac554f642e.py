"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
import time
from collections import defaultdict


def solve(instance):
    """
    Solve a planted 3-SAT instance with n variables and clauses.
    Returns a list of booleans (length n) that satisfies all clauses if possible.
    """
    n, clauses = instance["n"], instance["clauses"]
    num_clauses = len(clauses)

    # Deadline: leave a safety margin before the 330s kill
    deadline = time.time() + 325.0

    # Pre-process clauses for faster access
    fast_clauses = []
    for clause in clauses:
        fast_clause = []
        for lit in clause:
            var_idx = abs(lit) - 1
            required = (lit > 0)
            fast_clause.append((var_idx, required))
        fast_clauses.append(fast_clause)

    # Build variable-to-clauses index for efficient updates
    var_to_clauses = defaultdict(list)
    for i, fc in enumerate(fast_clauses):
        for var_idx, _ in fc:
            var_to_clauses[var_idx].append(i)

    def is_satisfied(clause_idx, assignment):
        fc = fast_clauses[clause_idx]
        for var_idx, required in fc:
            if assignment[var_idx] == required:
                return True
        return False

    def count_unsat(assignment):
        count = 0
        for i in range(num_clauses):
            if not is_satisfied(i, assignment):
                count += 1
        return count

    # Initialize mutable state containers in the enclosing scope
    unsat_set = set()
    var_unsat_count = [0] * n

    # Build phase-savings index: for each (var, required_value), the list of
    # clause indices in which that literal appears.
    var_req_to_clauses = defaultdict(list)
    for i, fc in enumerate(fast_clauses):
        for var_idx, required in fc:
            var_req_to_clauses[(var_idx, required)].append(i)

    def phase_savings(var_idx, required):
        """Number of other unsat clauses satisfied by setting var_idx to `required`."""
        total = 0
        for c_idx in var_req_to_clauses[(var_idx, required)]:
            if c_idx in unsat_set:
                total += 1
        return total

    def pick_literal(clause, clause_idx, walk_prob):
        """Choose which literal to flip. Returns (var_idx, required).
        Uses a combined heuristic: phase savings + inverse frequency (prefer
        variables with fewer unsat clauses, i.e., 'easier' to flip)."""
        if random.random() < walk_prob:
            return random.choice(clause)

        scores = []
        for var_idx, required in clause:
            freq = var_unsat_count[var_idx]
            ps = phase_savings(var_idx, required)
            # Prefer literals with high phase savings but low frequency
            # (flipping a variable that's in many unsat clauses is risky)
            score = ps * 2.0 + max(0, 10 - freq) * 0.5
            scores.append(score)

        total_score = sum(scores)
        if total_score == 0:
            return random.choice(clause)

        r = random.random() * total_score
        cumulative = 0
        for i, score in enumerate(scores):
            cumulative += score
            if r <= cumulative:
                return clause[i]
        return clause[-1]

    def apply_flip(var_idx):
        """Flip variable var_idx and update unsat_set / var_unsat_count incrementally.
        Returns the net change in unsat count (positive = got worse)."""
        assignment[var_idx] = not assignment[var_idx]
        delta = 0
        for c_idx in var_to_clauses[var_idx]:
            was_unsat = c_idx in unsat_set
            is_sat = is_satisfied(c_idx, assignment)
            if was_unsat and is_sat:
                unsat_set.discard(c_idx)
                delta -= 1
                for v_idx, _ in fast_clauses[c_idx]:
                    var_unsat_count[v_idx] -= 1
            elif not was_unsat and not is_sat:
                unsat_set.add(c_idx)
                delta += 1
                for v_idx, _ in fast_clauses[c_idx]:
                    var_unsat_count[v_idx] += 1
        return delta

    def init_state(assignment):
        nonlocal unsat_set, var_unsat_count
        unsat_set = set()
        for i in range(num_clauses):
            if not is_satisfied(i, assignment):
                unsat_set.add(i)
        var_unsat_count = [0] * n
        for c_idx in unsat_set:
            for v_idx, _ in fast_clauses[c_idx]:
                var_unsat_count[v_idx] += 1
        return len(unsat_set)

    # --- LP relaxation for smart initialization ---
    def lp_relaxation_init(lr_base=0.05, num_iters=100, init_prob=0.5):
        """
        Compute a continuous probability vector via iterative LP relaxation
        using Lagrangian multipliers (soft-min smoothing), then round to binary.
        
        The LP relaxation: minimize sum_i max(0, 1 - sum_{l in c_i} x_l)
        where x_l is the probability literal l is true.
        
        We use subgradient method with adaptive learning rate.
        """
        # Initialize probabilities
        prob = [init_prob] * n
        
        # Lagrangian multipliers for each clause (dual variables)
        multipliers = [0.0] * num_clauses
        
        # Iterate with adaptive learning rate
        lr = lr_base
        for iteration in range(num_iters):
            if time.time() > deadline - 300.0:
                break
            
            # Compute subgradient for each variable
            grad = [0.0] * n
            
            for i, fc in enumerate(fast_clauses):
                # Compute probability clause is unsatisfied
                p_unsat = 1.0
                for var_idx, required in fc:
                    if required:
                        p_unsat *= (1.0 - prob[var_idx])
                    else:
                        p_unsat *= prob[var_idx]
                
                # Update multiplier
                multipliers[i] = max(0.0, multipliers[i] + lr * p_unsat)
                
                # Accumulate subgradient
                if p_unsat > 0.01:
                    for var_idx, required in fc:
                        if required:
                            grad[var_idx] += multipliers[i] * (1.0 - prob[var_idx])
                        else:
                            grad[var_idx] += multipliers[i] * prob[var_idx]
            
            # Update probabilities using projected gradient descent
            for v in range(n):
                prob[v] -= lr * grad[v]
                # Project to [0.02, 0.98]
                prob[v] = max(0.02, min(0.98, prob[v]))
            
            # Decay learning rate
            lr *= 0.995
        
        # Round to binary assignment
        assignment = [p > 0.5 for p in prob]
        return assignment

    best_assignment = None
    best_unsat_count = float('inf')

    base_walk_prob = 0.15
    max_walk_prob = 0.45
    progress_threshold = 300

    # Simulated annealing parameters
    T0 = 4.0          # initial temperature (accepts moderate worsening)
    T_floor = 0.02    # below this, effectively only improving moves accepted
    max_steps = 20000

    # Try multiple LP relaxation initializations with different parameters
    if time.time() < deadline - 10.0:
        # Try several LP variants to find a good starting point
        lp_configs = [
            (0.05, 100, 0.5),   # default
            (0.10, 80, 0.5),    # higher LR, fewer iterations
            (0.03, 150, 0.5),   # lower LR, more iterations
            (0.05, 100, 0.6),   # biased initial probability
            (0.05, 100, 0.4),   # opposite bias
        ]
        
        for lr_b, iters, init_p in lp_configs:
            if time.time() > deadline - 10.0:
                break
            lp_assignment = lp_relaxation_init(lr_b, iters, init_p)
            current_unsat = init_state(lp_assignment)
            if current_unsat == 0:
                return lp_assignment
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = lp_assignment[:]

    for restart in range(500):
        if time.time() > deadline:
            break

        # Use LP relaxation for first few restarts with varied params, then mix
        lp_restart_configs = [
            (0.05, 100, 0.5),
            (0.10, 80, 0.5),
            (0.03, 150, 0.5),
            (0.05, 100, 0.6),
            (0.05, 100, 0.4),
        ]
        
        if restart < 5 and time.time() < deadline - 5.0:
            cfg = lp_restart_configs[restart % len(lp_restart_configs)]
            assignment = lp_relaxation_init(*cfg)
        elif restart % 6 == 0 and time.time() < deadline - 5.0:
            # Every 6th restart, try a random LP config
            cfg = random.choice(lp_restart_configs)
            assignment = lp_relaxation_init(*cfg)
        else:
            assignment = [random.random() < 0.5 for _ in range(n)]
        current_unsat = init_state(assignment)

        if current_unsat == 0:
            return assignment

        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]

        # Annealing schedule for this restart
        T = T0
        cooling_rate = 0.99997
        steps_since_improvement = 0
        local_best_unsat = current_unsat
        walk_prob = base_walk_prob

        for step in range(max_steps):
            if time.time() > deadline:
                break

            if not unsat_set:
                return assignment

            if len(unsat_set) < best_unsat_count:
                best_unsat_count = len(unsat_set)
                best_assignment = assignment[:]

            if len(unsat_set) < local_best_unsat:
                local_best_unsat = len(unsat_set)
                steps_since_improvement = 0
                walk_prob = base_walk_prob
            else:
                steps_since_improvement += 1
                if steps_since_improvement > progress_threshold:
                    walk_prob = min(max_walk_prob,
                                    base_walk_prob + (steps_since_improvement - progress_threshold) * 0.001)

            # Choose a clause weighted toward harder clauses (fewer options)
            # A clause with fewer unassigned/unfalsified literals is "harder"
            # We weight by 1/sqrt(num_unsat_literals) to prefer constrained clauses
            unsat_list = list(unsat_set)
            weights = []
            for c_idx in unsat_list:
                # Count how many literals in this clause are currently falsified
                # (all are falsified since clause is unsat, so count = 3 always)
                # Instead, use the number of variables in this clause that have
                # low unsat count (more flexible) vs high (more constrained)
                fc = fast_clauses[c_idx]
                min_freq = min(var_unsat_count[v_idx] for v_idx, _ in fc)
                # Harder clauses have higher min_freq (all vars are heavily involved)
                weights.append(1.0 / (1.0 + 0.1 * min_freq))
            
            total_w = sum(weights)
            r = random.random() * total_w
            cum = 0
            chosen_idx = 0
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    chosen_idx = i
                    break
            clause_idx = unsat_list[chosen_idx]
            clause = fast_clauses[clause_idx]

            var_idx, required = pick_literal(clause, clause_idx, walk_prob)
            delta = apply_flip(var_idx)

            # Simulated annealing acceptance:
            # delta < 0 -> improvement, always accept.
            # delta > 0 -> worsening, accept with prob exp(-delta / T).
            if delta > 0:
                if T > T_floor:
                    if random.random() > math.exp(-delta / T):
                        # Reject: revert the flip
                        apply_flip(var_idx)
                else:
                    # Too cold to accept any worsening move: revert
                    apply_flip(var_idx)

            # Cool the temperature
            T *= cooling_rate
            if T < T_floor:
                T = T_floor

            if not unsat_set:
                return assignment

        # End of restart

    # Final high-precision polish pass on the best assignment found
    if best_assignment is not None and time.time() < deadline - 5.0:
        assignment = best_assignment[:]
        current_unsat = init_state(assignment)
        if current_unsat > 0:
            # Two-phase polish: first reduce with moderate walk prob, then tighten
            phase1_steps = 30000
            phase2_steps = 30000
            
            # Phase 1: moderate exploration
            for step in range(phase1_steps):
                if time.time() > deadline - 10.0:
                    break
                if not unsat_set:
                    break
                if len(unsat_set) < best_unsat_count:
                    best_unsat_count = len(unsat_set)
                    best_assignment = assignment[:]

                clause_idx = random.choice(list(unsat_set))
                clause = fast_clauses[clause_idx]
                var_idx, required = pick_literal(clause, clause_idx, 0.10)
                delta = apply_flip(var_idx)
                if delta > 2:
                    apply_flip(var_idx)  # revert large worsening

            # Phase 2: tight exploitation
            if time.time() < deadline - 3.0 and unsat_set:
                for step in range(phase2_steps):
                    if time.time() > deadline - 1.0:
                        break
                    if not unsat_set:
                        break
                    if len(unsat_set) < best_unsat_count:
                        best_unsat_count = len(unsat_set)
                        best_assignment = assignment[:]

                    clause_idx = random.choice(list(unsat_set))
                    clause = fast_clauses[clause_idx]
                    var_idx, required = pick_literal(clause, clause_idx, 0.03)
                    delta = apply_flip(var_idx)
                    if delta > 1:
                        apply_flip(var_idx)  # revert any worsening

        if best_unsat_count == 0:
            return best_assignment

    if best_assignment is not None:
        return best_assignment
    else:
        return [random.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
