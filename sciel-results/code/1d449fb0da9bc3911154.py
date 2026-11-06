"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)

    start_time = time.time()
    deadline = start_time + 315.0

    # Convert clauses to 0-based indices with sign
    # clause_vars[i] = list of (var_idx, sign) where sign = +1 or -1
    clause_vars = []
    for c in clauses:
        cv = []
        for l in c:
            var_idx = abs(l) - 1
            sign = 1 if l > 0 else -1
            cv.append((var_idx, sign))
        clause_vars.append(cv)

    # Build variable-to-clause index
    # var_clauses[var_idx] = list of (clause_idx, sign_in_clause)
    var_clauses = [[] for _ in range(n)]
    for i, cv in enumerate(clause_vars):
        for var_idx, sign in cv:
            var_clauses[var_idx].append((i, sign))

    # Convert clauses to bitmasks for fast evaluation
    # Use Python integers as bitsets
    clause_masks = []
    for cv in clause_vars:
        pos_mask = 0
        neg_mask = 0
        for var_idx, sign in cv:
            bit = 1 << var_idx
            if sign == 1:
                pos_mask |= bit
            else:
                neg_mask |= bit
        clause_masks.append((pos_mask, neg_mask))

    # Count occurrences of each literal across all clauses (for weighting)
    pos_occ = [0] * n
    neg_occ = [0] * n
    for cv in clause_vars:
        for var_idx, sign in cv:
            if sign == 1:
                pos_occ[var_idx] += 1
            else:
                neg_occ[var_idx] += 1

    def is_satisfied(assignment):
        """Check if assignment satisfies all clauses using bitmasks."""
        # Build positive and negative bitmasks for the assignment
        pos_mask = 0
        neg_mask = 0
        for i in range(n):
            if assignment[i]:
                pos_mask |= (1 << i)
            else:
                neg_mask |= (1 << i)
        # For each clause, check if any literal is satisfied
        for pm, nm in clause_masks:
            if (pm & pos_mask) | (nm & neg_mask) == 0:
                return False
        return True

    def count_unsat(assignment):
        """Count number of unsatisfied clauses."""
        pos_mask = 0
        neg_mask = 0
        for i in range(n):
            if assignment[i]:
                pos_mask |= (1 << i)
            else:
                neg_mask |= (1 << i)
        count = 0
        for pm, nm in clause_masks:
            if (pm & pos_mask) | (nm & neg_mask) == 0:
                count += 1
        return count

    def get_unsat_indices(assignment):
        """Return list of indices of unsatisfied clauses."""
        pos_mask = 0
        neg_mask = 0
        for i in range(n):
            if assignment[i]:
                pos_mask |= (1 << i)
            else:
                neg_mask |= (1 << i)
        unsat = []
        for i, (pm, nm) in enumerate(clause_masks):
            if (pm & pos_mask) | (nm & neg_mask) == 0:
                unsat.append(i)
        return unsat

    def walk_sat_glare(assignment, rng, max_flips, deadline):
        """
        GLARE-style WalkSAT:
        - Pick random unsat clause
        - Prefer flips that reduce unsat count (with noise)
        - Adaptive noise: starts high, decreases over flips
        - Occurrence-based weighting for tie-breaking
        """
        # Initialize: find unsat clauses
        unsat_clauses = get_unsat_indices(assignment)
        if not unsat_clauses:
            return True

        # Precompute literal satisfaction status for current assignment
        # lit_sat[var_idx][sign] = True if literal (var_idx, sign) is satisfied
        # Use two arrays: sat_pos[var] = assignment[var], sat_neg[var] = not assignment[var]
        # So literal (var, +1) is sat if assignment[var] == True
        #    literal (var, -1) is sat if assignment[var] == False

        flip_count = 0
        noise_start = 1.5
        noise_end = 0.05

        # Track consecutive flips without improvement for adaptive restart within phase
        no_improve = 0
        best_unsat_in_phase = len(unsat_clauses)

        while flip_count < max_flips:
            if not unsat_clauses:
                return True

            if time.time() > deadline:
                break

            # Adaptive noise
            progress = flip_count / max_flips
            noise = noise_end + (noise_start - noise_end) * (1 - progress) ** 2

            # Pick random unsat clause
            ci = rng.choice(unsat_clauses)
            cv = clause_vars[ci]

            # Evaluate each variable in the clause
            best_var = -1
            best_delta = float('inf')

            for var_idx, sign in cv:
                # Calculate delta: change in number of unsatisfied clauses
                delta = 0

                # This variable is in clause ci (which is unsat)
                # Flipping will satisfy the literal in ci if sign matches new value
                # Currently assignment[var_idx] != (sign==1) since ci is unsat
                # After flip: assignment becomes not assignment
                # New value satisfies literal if (sign==1 and new_val==True) or (sign==-1 and new_val==False)
                # new_val = not assignment[var_idx]
                # If sign==1: satisfied if not assignment[var_idx] == True, i.e., assignment[var_idx] == False
                # If sign==-1: satisfied if not assignment[var_idx] == False, i.e., assignment[var_idx] == True
                # So the literal in ci will be satisfied after flip iff assignment[var_idx] == (sign == -1)
                # i.e., assignment[var_idx] == (sign < 0)
                # Actually: literal (v, +1) is sat when v=True. Literal (v, -1) is sat when v=False.
                # After flip, v becomes not v.
                # So (v, +1) after flip is sat when not v = True, i.e., v = False.
                # And (v, -1) after flip is sat when not v = False, i.e., v = True.
                # So the literal in ci is satisfied after flip iff:
                #   sign==1 and assignment[var_idx]==False  -> delta -= 1 (ci becomes sat)
                #   sign==-1 and assignment[var_idx]==True  -> delta -= 1 (ci becomes sat)
                # Which is: assignment[var_idx] == (sign < 0)
                # i.e., the literal is currently UNSAT, so flipping makes it SAT.
                # Since ci is unsat, all its literals are currently unsat, so flipping any var in ci
                # will satisfy that literal in ci. So delta always gets -1 from ci.

                delta = -1  # ci becomes satisfied

                # Check all other clauses containing var_idx
                for ci2, sign2 in var_clauses[var_idx]:
                    if ci2 == ci:
                        continue

                    # Determine if the literal (var_idx, sign2) is currently satisfied
                    # Literal (v, +1) is sat if assignment[v] == True
                    # Literal (v, -1) is sat if assignment[v] == False
                    cur_sat = (assignment[var_idx] == (sign2 == 1))

                    # After flip, assignment[var_idx] becomes not assignment[var_idx]
                    new_sat = ((not assignment[var_idx]) == (sign2 == 1))

                    if cur_sat and not new_sat:
                        # Was satisfied, now not -> this clause might become unsat
                        # But we need to check if other literals in ci2 are satisfied
                        # For simplicity in WalkSAT, we approximate:
                        # If this was the only satisfied literal, delta += 1
                        # We can't easily check without scanning, so use occurrence weighting
                        # Instead, let's just count: if cur_sat and not new_sat, potential +1
                        # But this is an overestimate. Use a softer approach.
                        delta += 1
                    elif not cur_sat and new_sat:
                        # Was not satisfied, now is -> this clause might become sat
                        # delta -= 1 (potential)
                        delta -= 1

                # Add noise
                delta += rng.gauss(0, noise)

                # Occurrence-based tie-breaking bonus
                # Slightly prefer flipping variables that appear less (to diversify)
                occ = pos_occ[var_idx] if sign == 1 else neg_occ[var_idx]
                delta += occ * 0.01

                if delta < best_delta:
                    best_delta = delta
                    best_var = var_idx

            if best_var < 0:
                best_var = cv[rng.randrange(len(cv))][0]

            # Perform the flip
            assignment[best_var] = not assignment[best_var]

            # Update unsat_clauses incrementally
            # Recompute which clauses containing best_var are now unsat
            # This is O(deg(best_var)) checks
            affected_clauses = [ci2 for ci2, _ in var_clauses[best_var]]

            # Build new pos/neg masks for quick check
            pos_mask = 0
            neg_mask = 0
            for i in range(n):
                if assignment[i]:
                    pos_mask |= (1 << i)
                else:
                    neg_mask |= (1 << i)

            new_unsat_set = set()
            for ci2 in affected_clauses:
                pm, nm = clause_masks[ci2]
                if (pm & pos_mask) | (nm & neg_mask) == 0:
                    new_unsat_set.add(ci2)

            # Update unsat_clauses list
            # Remove clauses that are now satisfied
            unsat_clauses = [c for c in unsat_clauses if c not in affected_clauses or c in new_unsat_set]
            # Add newly unsatisfied clauses
            for c in new_unsat_set:
                if c not in unsat_clauses:
                    unsat_clauses.append(c)

            flip_count += 1
            no_improve += 1

            # If we've had too many flips without improvement, do a mini-restart
            # (this is handled by the outer loop via max_flips, but we can add internal reset)
            if no_improve > 200:
                # Reset noise temporarily by breaking to outer restart
                break

        return len(unsat_clauses) == 0

    # Main loop: multiple restarts
    best_assignment = None
    best_unsat = float('inf')

    restart = 0
    base_max_flips = 3000

    while time.time() < deadline:
        rng = random.Random(1000 + restart * 7)
        assignment = [rng.random() < 0.5 for _ in range(n)]

        max_flips = base_max_flips + restart * 300
        if max_flips > 50000:
            max_flips = 50000

        if walk_sat_glare(assignment, rng, max_flips, deadline):
            if is_satisfied(assignment):
                return assignment

        unsat_count = count_unsat(assignment)
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]

        restart += 1

        if restart > 2000:
            break

    # Final polish: try a short refinement from the best assignment
    if best_assignment is not None and time.time() < deadline - 10:
        rng = random.Random(99999)
        max_polish_flips = 5000
        unsat_clauses = get_unsat_indices(best_assignment)
        if unsat_clauses:
            # Do a few rounds of best-improvement local search
            for _ in range(50):
                if not unsat_clauses:
                    break
                if time.time() > deadline - 5:
                    break
                ci = rng.choice(unsat_clauses)
                cv = clause_vars[ci]
                best_var = -1
                best_delta = float('inf')
                for var_idx, sign in cv:
                    delta = -1
                    for ci2, sign2 in var_clauses[var_idx]:
                        if ci2 == ci:
                            continue
                        cur_sat = (best_assignment[var_idx] == (sign2 == 1))
                        new_sat = ((not best_assignment[var_idx]) == (sign2 == 1))
                        if cur_sat and not new_sat:
                            delta += 1
                        elif not cur_sat and new_sat:
                            delta -= 1
                    if delta < best_delta:
                        best_delta = delta
                        best_var = var_idx
                if best_var >= 0:
                    best_assignment[best_var] = not best_assignment[best_var]
                    # Update unsat_clauses
                    affected = [ci2 for ci2, _ in var_clauses[best_var]]
                    pos_mask = 0
                    neg_mask = 0
                    for i in range(n):
                        if best_assignment[i]:
                            pos_mask |= (1 << i)
                        else:
                            neg_mask |= (1 << i)
                    new_unsat = set()
                    for ci2 in affected:
                        pm, nm = clause_masks[ci2]
                        if (pm & pos_mask) | (nm & neg_mask) == 0:
                            new_unsat.add(ci2)
                    unsat_clauses = [c for c in unsat_clauses if c not in affected or c in new_unsat]
                    for c in new_unsat:
                        if c not in unsat_clauses:
                            unsat_clauses.append(c)

        if is_satisfied(best_assignment):
            return best_assignment

    if best_assignment is not None:
        return best_assignment

    return [True] * n
# EVOLVE-BLOCK-END
