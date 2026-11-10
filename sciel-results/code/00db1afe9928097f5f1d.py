"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 290  # 290s safety margin under 330s limit

    # Precompute clause structure
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

    def belief_propagation_init(rng, max_iters=200):
        """Run Belief Propagation on the factor graph to estimate variable beliefs.
        Returns a list of (probability_true, confidence) for each variable."""
        # Messages: var_to_clause[vi][ci] = log-odds that vi is true, given all clauses except ci
        #           clause_to_var[ci][vi] = log-odds contribution from clause ci to variable vi
        # We use log-odds (logit) representation for numerical stability.
        
        # Initialize messages with small random values
        var_to_clause = [[0.0] * m for _ in range(n)]
        clause_to_var = [[0.0] * 3 for _ in range(m)]  # Each clause has 3 vars
        
        # Map clause index to variable indices within the clause
        clause_var_map = []
        for ci in range(m):
            cvars = [vi for vi, s in clause_vars[ci]]
            clause_var_map.append(cvars)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            # Update clause-to-variable messages
            for ci in range(m):
                for pos, vi in enumerate(clause_var_map[ci]):
                    sign = clause_vars[ci][pos][1]
                    # Sum of incoming messages from other variables in this clause
                    log_sum = 0.0  # We'll compute log(1+exp(x)) carefully
                    incoming = []
                    for pos2, vi2 in enumerate(clause_var_map[ci]):
                        if pos2 == pos:
                            continue
                        msg = var_to_clause[vi2][ci]
                        if sign == 1:
                            # Literal is vi2 positive: satisfied if vi2=true
                            # Contribution: log(1 + exp(msg)) for the "unsatisfied" part
                            incoming.append(msg)
                        else:
                            # Literal is vi2 negative: satisfied if vi2=false
                            # Contribution: log(1 + exp(-msg))
                            incoming.append(-msg)
                    
                    # For a 3-clause, the message from clause to var is:
                    # P(clause unsatisfied | vi's value) depends on other vars
                    # If sign==1 (vi positive): clause unsat means all literals false
                    #   P(unsat|vi=true) = P(other two literals false)
                    #   In log-odds: the "evidence" that clause is satisfied given vi
                    # Simplified: use sum of incoming as proxy for belief
                    # Better: compute properly
                    # For vi with sign s in clause ci:
                    #   If s=1: clause is satisfied if vi=true OR any other literal true
                    #   The "force" pushing vi to true is proportional to P(all others false)
                    #   In log-odds, this is sum of log(1+exp(-msg_j)) for others... 
                    # Let's use a simpler approximation:
                    # message = -sum of other incoming messages (with sign adjustment)
                    
                    # Proper BP message: for clause C with variables v1,v2,v3
                    # message from C to vi: 
                    #   if vi appears positively: log(1 + exp(sum of log-odds of "other literals false"))
                    #   Actually: P(C unsat | vi=1) = P(v2_lit false) * P(v3_lit false)
                    #   = (1/(1+exp(msg_v2))) * (1/(1+exp(msg_v3)))  [if positive literals]
                    #   log-odds of "C is satisfied given vi" = -log(P(C unsat|vi)) + log(P(C sat|vi))
                    #   For planted SAT, we want to find the satisfying assignment.
                    #   Use: message = -[log(1+exp(-a)) + log(1+exp(-b))] where a,b are adjusted incoming
                    #   This pushes vi toward the value that makes the clause more likely satisfied.
                    
                    # Simplified stable version:
                    total = 0.0
                    for pos2, vi2 in enumerate(clause_var_map[ci]):
                        if pos2 == pos:
                            continue
                        msg2 = var_to_clause[vi2][ci]
                        sign2 = clause_vars[ci][pos2][1]
                        # Probability that literal (vi2, sign2) is false:
                        # if sign2=1: P(vi2=false) = 1/(1+exp(msg2))
                        # if sign2=-1: P(vi2=true) = 1/(1+exp(-msg2))
                        if sign2 == 1:
                            # log(1/(1+exp(msg2))) = -log(1+exp(msg2))
                            # Use log-sum-exp trick: log(1+exp(x)) ≈ max(0,x) + log1p(exp(-|x|))
                            x = msg2
                            total += -max(0, x) - math.log1p(math.exp(-abs(x))) if x > -700 else -x - math.log1p(math.exp(x))
                        else:
                            x = -msg2
                            total += -max(0, x) - math.log1p(math.exp(-abs(x))) if x > -700 else -x - math.log1p(math.exp(x))
                    
                    # The message from clause to var: higher means "clause more likely satisfied if vi takes its literal's preferred value"
                    # For sign=1: preferred is vi=true, message should be positive when clause wants vi=true
                    # message = -total (negative log-prob of all others being false = log-prob of clause being satisfied by others)
                    # Actually: if total is very negative (others unlikely to be false), clause is likely satisfied regardless
                    # We want the message to push vi toward making the clause satisfied
                    # message = -total gives: if others are likely false (total≈0), message≈0 (no push)
                    #          if others are likely true (total very negative), message very positive (strong push)
                    # Hmm, let me think again.
                    # P(clause sat | vi=val) = 1 - P(all other literals false | vi=val)
                    # For the "vi=true" branch (sign=1): P(sat) = 1 - P(others false)
                    # For the "vi=false" branch (sign=1): P(sat) = P(others true) = 1 - P(others false) ... no
                    # Actually if vi=false and sign=1, the literal is false, so clause sat only if some other literal is true
                    # P(sat | vi=false, sign=1) = 1 - P(all others false)
                    # P(sat | vi=true, sign=1) = 1
                    # So the clause strongly prefers vi=true when sign=1.
                    # The message should reflect: "how much does this clause want vi to be true?"
                    # = log(P(sat|vi=true) / P(sat|vi=false)) = log(1 / (1 - P(others false)))
                    # ≈ -log(1 - P(others false)) ≈ P(others false) when small
                    # In log-odds form: message = -total (where total = log P(others all false))
                    # If P(others all false) is small, -total is large positive → strong preference for vi=true
                    # This seems right for sign=1.
                    # For sign=-1: preferred is vi=false
                    # P(sat|vi=false) = 1, P(sat|vi=true) = 1 - P(others false)
                    # message = log(1/(1-P(others false))) = -total, but this should push toward vi=false
                    # So for sign=-1, the message should be negative of what we computed
                    # i.e., message = sign * (-total) ... no.
                    # Let me define: message = log-odds that vi should be TRUE
                    # For sign=1: clause wants vi=true → positive contribution
                    # For sign=-1: clause wants vi=false → negative contribution
                    # Strength = -total (log of inverse of P(others false))
                    clause_to_var[ci][pos] = sign * (-total)
            
            # Update variable-to-clause messages
            for vi in range(n):
                for ci, sign in var_occurrences[vi]:
                    # Sum of incoming messages from all clauses containing vi
                    total_in = 0.0
                    for oci, _ in var_occurrences[vi]:
                        pos = clause_var_map[oci].index(vi)
                        total_in += clause_to_var[oci][pos]
                    
                    # Message to clause ci: total_in minus the contribution from ci itself
                    # (to avoid self-feedback in standard BP)
                    pos = clause_var_map[ci].index(vi)
                    var_to_clause[vi][ci] = total_in - clause_to_var[ci][pos]
        
        # Compute final beliefs: for each variable, sum all incoming clause messages
        beliefs = []
        for vi in range(n):
            total_in = 0.0
            for ci, sign in var_occurrences[vi]:
                pos = clause_var_map[ci].index(vi)
                total_in += clause_to_var[ci][pos]
            # total_in is log-odds that vi is true
            # Convert to probability
            if total_in > 500:
                prob = 1.0
            elif total_in < -500:
                prob = 0.0
            else:
                prob = 1.0 / (1.0 + math.exp(-total_in))
            confidence = abs(total_in)
            beliefs.append((prob, confidence))
        
        return beliefs

    def bp_assignment(beliefs, rng):
        """Convert BP beliefs to an assignment, with optional randomization for low-confidence vars."""
        assignment = []
        for prob, conf in beliefs:
            if conf > 2.0:
                assignment.append(prob > 0.5)
            else:
                assignment.append(rng.random() < prob)
        return assignment

    def random_assignment():
        return [random.random() < 0.5 for _ in range(n)]

    def compute_clause_counts(assignment):
        """Compute for each clause how many literals are satisfied."""
        counts = [0] * m
        for ci in range(m):
            cnt = 0
            for vi, sign in clause_vars[ci]:
                if assignment[vi] == (sign == 1):
                    cnt += 1
            counts[ci] = cnt
        return counts

    def flip_variable(assignment, vi, counts):
        """Flip variable vi and update clause counts incrementally."""
        old_val = assignment[vi]
        new_val = not old_val
        assignment[vi] = new_val
        for ci, sign in var_occurrences[vi]:
            was_sat = old_val == (sign == 1)
            now_sat = new_val == (sign == 1)
            if was_sat and not now_sat:
                counts[ci] -= 1
            elif not was_sat and now_sat:
                counts[ci] += 1

    def is_satisfied(counts):
        return all(c >= 1 for c in counts)

    def count_unsat(counts):
        return sum(1 for c in counts if c == 0)

    def walk_sat(assignment, counts, max_steps, rng, noise=0.3):
        """WalkSAT with incremental tracking and noise parameter.
        Returns (success, assignment, counts)."""
        for step in range(max_steps):
            if time.time() > deadline:
                return False, assignment, counts
            unsat = [ci for ci in range(m) if counts[ci] == 0]
            if not unsat:
                return True, assignment, counts

            ci = rng.choice(unsat)
            lits_in_clause = clause_vars[ci]

            best_vi = None
            best_score = None
            for vi, sign in lits_in_clause:
                break_count = 0
                for oci, osign in var_occurrences[vi]:
                    if oci == ci:
                        continue
                    cur_sat = assignment[vi] == (osign == 1)
                    if cur_sat and counts[oci] == 1:
                        break_count += 1
                # Smooth surrogate: use soft penalty with noise
                score = break_count + rng.random() * noise
                if best_score is None or score < best_score:
                    best_score = score
                    best_vi = vi

            if best_vi is not None:
                flip_variable(assignment, best_vi, counts)

        return False, assignment, counts

    def local_search(assignment, counts, max_steps, rng):
        """Local search: pick variable that satisfies most unsat clauses."""
        for step in range(max_steps):
            if time.time() > deadline:
                return False, assignment, counts
            unsat = [ci for ci in range(m) if counts[ci] == 0]
            if not unsat:
                return True, assignment, counts

            var_score = [0] * n
            for ci in unsat:
                for vi, sign in clause_vars[ci]:
                    var_score[vi] += 1

            max_score = max(var_score) if var_score else 0
            if max_score == 0:
                vi = rng.randrange(n)
            else:
                candidates = [vi for vi in range(n) if var_score[vi] == max_score]
                vi = rng.choice(candidates)

            flip_variable(assignment, vi, counts)

        return False, assignment, counts

    def polish(assignment, counts, max_steps, rng):
        """High-precision polish: targeted flips to fix remaining unsat clauses.
        Uses a more aggressive strategy with smaller noise and more steps."""
        for step in range(max_steps):
            if time.time() > deadline:
                return False, assignment, counts
            unsat = [ci for ci in range(m) if counts[ci] == 0]
            if not unsat:
                return True, assignment, counts

            # Try to fix each unsat clause by finding a flip that reduces total unsat
            improved = False
            for ci in unsat:
                for vi, sign in clause_vars[ci]:
                    # Simulate flip
                    old_val = assignment[vi]
                    delta_unsat = 0
                    for oci, osign in var_occurrences[vi]:
                        cur_sat = old_val == (osign == 1)
                        new_sat = (not old_val) == (osign == 1)
                        if cur_sat and not new_sat:
                            if counts[oci] == 1:
                                delta_unsat += 1
                        elif not cur_sat and new_sat:
                            delta_unsat -= 1
                    # Also account for the clause ci itself: flipping vi satisfies one literal
                    # Since ci is unsat (counts[ci]==0), flipping vi will satisfy it
                    # But we already counted oci==ci in the loop above if ci contains vi
                    # Actually, in the loop, for oci==ci: cur_sat is False (unsat), new_sat is True
                    # So delta_unsat -= 1 for that clause. That's correct.
                    
                    if delta_unsat < 0:
                        # This flip reduces unsat count
                        flip_variable(assignment, vi, counts)
                        improved = True
                        break
                if improved:
                    break

            if not improved:
                # No single flip improves; try random flip with small probability
                # or pick the flip that minimizes delta_unsat
                best_delta = 0
                best_vi = None
                for ci in unsat:
                    for vi, sign in clause_vars[ci]:
                        old_val = assignment[vi]
                        delta_unsat = 0
                        for oci, osign in var_occurrences[vi]:
                            cur_sat = old_val == (osign == 1)
                            new_sat = (not old_val) == (osign == 1)
                            if cur_sat and not new_sat:
                                if counts[oci] == 1:
                                    delta_unsat += 1
                            elif not cur_sat and new_sat:
                                delta_unsat -= 1
                        if delta_unsat < best_delta:
                            best_delta = delta_unsat
                            best_vi = vi
                if best_vi is not None:
                    flip_variable(assignment, best_vi, counts)
                else:
                    # Truly stuck; random flip
                    vi = rng.randrange(n)
                    flip_variable(assignment, vi, counts)

        return False, assignment, counts

    # Main loop with restarts
    best_assignment = None
    best_unsat_count = m + 1

    # Run BP once to get initial beliefs
    bp_rng = random.Random(42)
    bp_beliefs = belief_propagation_init(bp_rng, max_iters=150)
    
    # Check if BP already gives a solution
    bp_assign = bp_assignment(bp_beliefs, bp_rng)
    bp_counts = compute_clause_counts(bp_assign)
    if is_satisfied(bp_counts):
        return bp_assign
    bp_unsat = count_unsat(bp_counts)
    if bp_unsat < best_unsat_count:
        best_unsat_count = bp_unsat
        best_assignment = bp_assign[:]

    for attempt in range(200):
        if time.time() > deadline:
            break

        rng = random.Random(attempt * 7919 + 13)
        
        # Use BP-guided initialization for first 60% of attempts, random for rest
        if attempt < 120:
            # Perturb BP assignment: flip low-confidence variables randomly
            assignment = []
            for vi in range(n):
                prob, conf = bp_beliefs[vi]
                if conf > 3.0:
                    # High confidence: keep BP value
                    assignment.append(prob > 0.5)
                elif conf > 1.0:
                    # Medium confidence: mostly keep, occasionally flip
                    if rng.random() < 0.15:
                        assignment.append(rng.random() < 0.5)
                    else:
                        assignment.append(prob > 0.5)
                else:
                    # Low confidence: randomize more
                    if rng.random() < 0.5:
                        assignment.append(rng.random() < prob)
                    else:
                        assignment.append(rng.random() < 0.5)
        else:
            assignment = random_assignment()
        
        counts = compute_clause_counts(assignment)

        if is_satisfied(counts):
            return assignment

        cur_unsat = count_unsat(counts)
        if cur_unsat < best_unsat_count:
            best_unsat_count = cur_unsat
            best_assignment = assignment[:]

        # Alternate between WalkSAT (with varying noise) and local search
        if attempt % 3 == 0:
            success, assignment, counts = walk_sat(assignment, counts, 5000, rng, noise=0.2)
        elif attempt % 3 == 1:
            success, assignment, counts = walk_sat(assignment, counts, 5000, rng, noise=0.5)
        else:
            success, assignment, counts = local_search(assignment, counts, 5000, rng)

        if success:
            return assignment

        cur_unsat = count_unsat(counts)
        if cur_unsat < best_unsat_count:
            best_unsat_count = cur_unsat
            best_assignment = assignment[:]

    # Final polish phase: use the best assignment found and try to solve it
    if best_assignment is not None and best_unsat_count < m:
        polish_rng = random.Random(12345)
        polish_assignment = best_assignment[:]
        polish_counts = compute_clause_counts(polish_assignment)
        if not is_satisfied(polish_counts):
            # Run polish with increasing effort
            for polish_attempt in range(5):
                if time.time() > deadline:
                    break
                p_rng = random.Random(polish_attempt * 31 + 7)
                success, polish_assignment, polish_counts = polish(
                    polish_assignment, polish_counts, 20000, p_rng
                )
                if success:
                    return polish_assignment
                cur_unsat = count_unsat(polish_counts)
                if cur_unsat < best_unsat_count:
                    best_unsat_count = cur_unsat
                    best_assignment = polish_assignment[:]

            # Additional WalkSAT polish on the best result
            remaining_time = deadline - time.time()
            if remaining_time > 5:
                for extra_attempt in range(3):
                    if time.time() > deadline:
                        break
                    e_rng = random.Random(999 + extra_attempt)
                    e_assign = best_assignment[:]
                    e_counts = compute_clause_counts(e_assign)
                    # Use lower noise for more deterministic behavior
                    success, e_assign, e_counts = walk_sat(e_assign, e_counts, 10000, e_rng, noise=0.1)
                    if success:
                        return e_assign
                    cur_unsat = count_unsat(e_counts)
                    if cur_unsat < best_unsat_count:
                        best_unsat_count = cur_unsat
                        best_assignment = e_assign[:]

    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
