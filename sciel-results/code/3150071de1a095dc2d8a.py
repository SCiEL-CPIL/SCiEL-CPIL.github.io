"""WalkSAT with adaptive clause-weight scoring, restarts, and a time budget."""
import random
import time
import math


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Pre-compute: for each variable, list of (clause_index, sign) where sign is +1 if lit == var, -1 if lit == -var
    var_occurs = [[] for _ in range(n)]
    for ci, cl in enumerate(clauses):
        for lit in cl:
            var_occurs[abs(lit) - 1].append((ci, 1 if lit > 0 else -1))

    # We'll use numpy-free approach but with lists for speed
    # Represent assignment as list of 0/1 (False/True)
    # clause_sat[ci] = number of satisfied literals in clause ci

    deadline = time.time() + 280  # 300s limit, keep 20s safety margin

    best_assignment = None
    best_unsat_count = m + 1

    rng = random.Random()

    max_steps_per_restart = 50000
    restart_budget = 5000  # max total restarts (will be limited by time)

    for restart in range(restart_budget):
        if time.time() > deadline - 2:
            break

        # Random initial assignment
        assignment = [random.getrandbits(1) for _ in range(n)]

        # Compute initial clause satisfaction counts
        clause_sat = [0] * m
        for ci, cl in enumerate(clauses):
            s = 0
            for lit in cl:
                if assignment[abs(lit) - 1] == (1 if lit > 0 else 0):
                    s += 1
            clause_sat[ci] = s

        unsat_count = sum(1 for s in clause_sat if s == 0)
        if unsat_count == 0:
            return assignment

        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = list(assignment)

        # Adaptive parameters: WalkSAT with clause weights
        # We use a "noisy min" strategy: flip a variable in an unsat clause,
        # biased by a score that considers how many clauses it affects.
        # Score = (clauses made sat) - (clauses broken) + random noise

        alpha = 0.5  # noise level (number of random flips before best)
        # Decay alpha over time within a restart to reduce randomness
        steps = 0
        max_steps = max_steps_per_restart

        # Pre-build clause list for faster access
        # Use local vars for speed
        assign = assignment
        c_sat = clause_sat
        n_vars = n
        var_occ = var_occurs

        while steps < max_steps:
            if time.time() > deadline - 2:
                break

            # Find an unsatisfied clause
            # To avoid O(m) scan every step, we can pick a random clause and check
            # But for 150 vars, m is typically ~450-1000, so O(m) is fine per step
            # However, we can be smarter: keep a list of unsat clause indices
            # For simplicity and speed, let's sample a few random clauses

            # Actually, let's just find unsat clauses efficiently
            # For m up to ~1000, scanning is fast enough in Python? Maybe not for 50k steps.
            # Let's use a different approach: pick a random clause, if sat, pick another.

            # Better: maintain unsat_clause_list
            # But updating it is expensive. Let's just scan.
            # With m ~ 500, scanning takes ~500 operations, times 50000 steps = 25M ops. Might be tight.
            # Let's try a hybrid: keep a set of unsat clauses, update incrementally.

            # Actually, let's just use a simple approach: pick random clause index, check if unsat.
            # If sat, pick another. Expected number of tries = m/unsat_count.
            # This is O(1) expected per selection.

            # Pick an unsat clause
            tries = 0
            while True:
                ci = random.randrange(m)
                if c_sat[ci] == 0:
                    break
                tries += 1
                if tries > 100:
                    # Fallback: scan
                    unsat_list = [i for i in range(m) if c_sat[i] == 0]
                    if not unsat_list:
                        break
                    ci = random.choice(unsat_list)
                    break

            if c_sat[ci] > 0:
                # All clauses satisfied
                if c_sat[ci] > 0 and all(c_sat[i] > 0 for i in range(m)):
                    break
                # Actually we should check if truly all sat
                # The while loop above found an unsat clause, so we're fine
                pass

            # Get the clause
            cl = clauses[ci]

            # Compute scores for each literal in the clause
            # Score for flipping var v in clause ci:
            # Gain = number of other unsat clauses that become sat
            # Loss = number of other sat clauses that become unsat
            # We want to maximize (Gain - Loss) with some randomness

            # For each literal in the clause, compute the score
            best_score = -999
            best_var = -1
            best_sign = 0  # +1 means set to True, -1 means set to False
            scores = []

            for lit in cl:
                v = abs(lit) - 1
                sign = 1 if lit > 0 else -1
                # If we flip v, what happens?
                # Current value of v: assign[v]
                # New value: 1 - assign[v]

                # For each clause containing v, check if flipping v changes sat status
                gain = 0
                loss = 0
                for (cci, sgn) in var_occ[v]:
                    if cci == ci:
                        continue
                    # Check if clause cci is currently sat
                    currently_sat = c_sat[cci] > 0
                    # After flipping v, does this clause's sat count change?
                    # The literal (v with sign sgn) in clause cci:
                    # Current contribution: 1 if assign[v] == (1 if sgn==1 else 0) else 0
                    # New contribution: 1 if (1-assign[v]) == (1 if sgn==1 else 0) else 0
                    cur_val = 1 if (assign[v] == (1 if sgn == 1 else 0)) else 0
                    new_val = 1 if ((1 - assign[v]) == (1 if sgn == 1 else 0)) else 0

                    if cur_val == 0 and new_val == 1:
                        # This literal becomes satisfied
                        if not currently_sat:
                            # Clause was unsat, now might become sat if this was the only unsat lit
                            # Actually, if clause was unsat (all lits false), and now one becomes true, it becomes sat
                            gain += 1
                    elif cur_val == 1 and new_val == 0:
                        # This literal becomes unsatisfied
                        # Clause was sat, now might become unsat if this was the only sat lit
                        if c_sat[cci] == 1:
                            loss += 1

                score = gain - loss
                scores.append((score, v, sign))

            # Noisy min: pick best with probability proportional to exp(score/alpha)
            # Or simply: sort by score, pick from top with random tie-breaking
            # Let's use: score + random noise
            # Better: softmax-like selection
            # For simplicity, let's pick the best score, with some probability pick a random one from the clause
            # This is the standard WalkSAT approach: with probability alpha, pick random lit; else pick best

            r = random.random()
            if r < alpha:
                idx = random.randrange(len(cl))
                _, v, sign = scores[idx]
            else:
                # Pick the one with max score
                idx = max(range(len(cl)), key=lambda i: scores[i][0])
                _, v, sign = scores[idx]

            # Flip variable v
            old_val = assign[v]
            new_val = 1 - old_val
            assign[v] = new_val

            # Update clause_sat for all clauses containing v
            for (cci, sgn) in var_occ[v]:
                # The literal (v with sign sgn) changed from old_val to new_val
                # Old contribution: 1 if old_val == (1 if sgn==1 else 0) else 0
                # New contribution: 1 if new_val == (1 if sgn==1 else 0) else 0
                cur_val = 1 if (old_val == (1 if sgn == 1 else 0)) else 0
                new_val_c = 1 if (new_val == (1 if sgn == 1 else 0)) else 0
                c_sat[cci] += new_val_c - cur_val

            steps += 1

            # Check if we solved it
            # We can periodically check all clauses
            if steps % 10 == 0:
                unsat_count = sum(1 for s in c_sat if s == 0)
                if unsat_count == 0:
                    break
                if unsat_count < best_unsat_count:
                    best_unsat_count = unsat_count
                    best_assignment = list(assign)

        # End of restart loop
        # Check if solved
        unsat_count = sum(1 for s in c_sat if s == 0)
        if unsat_count == 0:
            return assignment

        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = list(assign)

        # Decay alpha for next restart (less random)
        alpha = max(0.1, alpha * 0.95)

    # Return best assignment found
    return best_assignment
