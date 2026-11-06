"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 285.0  # safety margin before the 330s kill

    # Pre-compute clauses as 0-based (idx, pos) tuples and build a variable->clauses index
    clauses_idx = [[(abs(l) - 1, l > 0) for l in c] for c in clauses]
    m = len(clauses_idx)
    var_to_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(clauses_idx):
        for idx, _pos in c:
            var_to_clauses[idx].append(ci)

    # Precompute for each clause and each literal position:
    # - the variable index and polarity
    # - which OTHER clauses contain this variable (for impact estimation)
    clause_var_impact = []  # clause_var_impact[ci][k] = list of other clause indices
    for ci, c in enumerate(clauses_idx):
        impacts = []
        for k, (idx, pos) in enumerate(c):
            # Other clauses containing this variable
            others = [cj for cj in var_to_clauses[idx] if cj != ci]
            impacts.append(others)
        clause_var_impact.append(impacts)

    # Precompute for fast mask updates: for each clause, store (idx, pos) pairs
    # and also precompute the bit position for each literal in the clause
    clause_literals = clauses_idx  # clauses_idx[ci][k] = (idx, pos)

    def compute_mask(assignment, ci):
        c = clauses_idx[ci]
        mask = 0
        for k, (idx, pos) in enumerate(c):
            if assignment[idx] == pos:
                mask |= (1 << k)
        return mask

    rng = random.Random(12345)

    best_assignment = None
    best_unsat = m + 1

    # Adaptive noise parameters
    base_noise = 0.2
    max_noise = 0.7
    noise_increase = 0.03
    noise_decay = 0.95
    max_flips_per_run = 6000
    min_improvement_window = 100

    # Phase tracking: count how often each variable polarity appears in satisfied clauses
    phase_count = [0, 0]  # [count_true, count_false]
    phase_total = 0

    while time.time() < deadline:
        # Use phase-informed initial assignment
        if phase_total > 100:
            p_true = phase_count[0] / phase_total
            assignment = [rng.random() < p_true for _ in range(n)]
        else:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        
        masks = [compute_mask(assignment, ci) for ci in range(m)]
        unsat = set(ci for ci in range(m) if masks[ci] == 0)

        if not unsat:
            return assignment

        if len(unsat) < best_unsat:
            best_unsat = len(unsat)
            best_assignment = assignment[:]
            # Update phase counts from this good assignment
            for i in range(n):
                if assignment[i]:
                    phase_count[0] += 1
                else:
                    phase_count[1] += 1
            phase_total += n

        noise = base_noise
        last_improvement_flip = 0
        flip_count = 0

        for _flip in range(max_flips_per_run):
            if time.time() > deadline:
                break
            if not unsat:
                return assignment

            flip_count += 1
            if len(unsat) < best_unsat:
                best_unsat = len(unsat)
                best_assignment = assignment[:]
                last_improvement_flip = flip_count
                noise = max(base_noise, noise * noise_decay)

            # Pick an unsatisfied clause - weight by clause "difficulty"
            # Use a simple heuristic: prefer clauses with higher impact (more other clauses affected)
            unsat_list = list(unsat)
            if len(unsat_list) > 1:
                # Weight by total impact of the clause's variables
                weights = []
                for ci in unsat_list:
                    total_impact = sum(len(clause_var_impact[ci][k]) for k in range(3))
                    weights.append(1.0 / (total_impact + 1))
                total_w = sum(weights)
                r = rng.random() * total_w
                cum = 0
                ci = unsat_list[0]
                for i, w in enumerate(weights):
                    cum += w
                    if r <= cum:
                        ci = unsat_list[i]
                        break
            else:
                ci = unsat_list[0]
            
            c = clauses_idx[ci]

            # Score each literal flip by:
            # 1. Does it satisfy this clause? (primary)
            # 2. Impact on other clauses (secondary)
            best_k = -1
            best_score = -float('inf')
            
            for k in range(3):
                idx, pos = c[k]
                currently_satisfies = (assignment[idx] == pos)
                
                # If this literal already satisfies the clause, flipping it would break it
                if currently_satisfies:
                    # Only consider this if noise is high
                    score = -10.0 + rng.random() * 2.0
                else:
                    # Flipping this satisfies the clause
                    score = 10.0
                    
                    # Estimate impact on other clauses
                    others = clause_var_impact[ci][k]
                    if others:
                        # Count how many other clauses would be affected negatively
                        neg_impact = 0
                        for cj in others:
                            # Check if this variable currently satisfies cj
                            cj_vars = clauses_idx[cj]
                            for (other_idx, other_pos) in cj_vars:
                                if other_idx == idx and assignment[idx] == other_pos:
                                    # This variable currently satisfies cj, flipping would break it
                                    neg_impact += 1
                                    break
                        score -= neg_impact * 2.0
                    
                    # Add some randomness
                    score += rng.random() * 1.0
                
                if score > best_score:
                    best_score = score
                    best_k = k
            
            # With probability noise, flip a random literal instead
            if rng.random() < noise:
                k = rng.randrange(3)
            else:
                k = best_k
            
            idx, pos = c[k]
            assignment[idx] = not assignment[idx]

            # Incrementally update masks and unsat for clauses touching idx
            for cj in var_to_clauses[idx]:
                old_mask = masks[cj]
                # Fast update: only need to check if the flipped variable's literal in cj changes
                # Find the position of idx in clause cj
                new_mask = compute_mask(assignment, cj)
                masks[cj] = new_mask
                
                if old_mask == 0 and new_mask != 0:
                    unsat.discard(cj)
                elif old_mask != 0 and new_mask == 0:
                    unsat.add(cj)

            # Adaptive noise: increase if no improvement for a while
            if flip_count - last_improvement_flip > min_improvement_window:
                noise = min(max_noise, noise + noise_increase)

        # End of this run; continue with a fresh restart if time remains
        if time.time() > deadline:
            break

    if best_assignment is not None:
        return best_assignment
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
