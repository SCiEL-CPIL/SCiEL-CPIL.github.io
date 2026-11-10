"""
3-SAT solver: Spectral Max-Cut init + vectorized NumPy WalkSAT with
per-variable impact scoring and smooth log-sum-exp tie-breaking.

Key structural changes vs. parent:
- Vectorized NumPy flip-update loop (batch sat_count updates via numpy
  boolean masks instead of Python-level per-clause iteration).
- Per-variable impact score computed in one vectorised pass per flip,
  using a smooth log-sum-exp surrogate to rank literals by expected
  clause-reduction (encourages larger moves out of local minima).
- Same spectral initialisation and incremental clause tracking family.
"""
import random
import time
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["instance_clauses"] if "instance_clauses" in instance else instance["clauses"]

    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 330 s kill

    # 1. Preprocess: remove tautologies, dedupe
    seen = set()
    filtered_clauses = []
    for c in clauses:
        lits = set(c)
        if any(-l in lits for l in lits):
            continue
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            filtered_clauses.append(c)
    clauses = filtered_clauses
    clause_count = len(clauses)

    if clause_count == 0:
        return [True] * n

    # 2. Build indices
    var_pos = [[] for _ in range(n)]
    var_neg = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for l in c:
            idx = abs(l) - 1
            if l > 0:
                var_pos[idx].append(i)
            else:
                var_neg[idx].append(i)

    # Convert clause literals to 0-indexed signed array for fast lookup
    # lit_sign[i] = +1 if literal is positive, -1 if negative
    # lit_var[i]  = 0-indexed variable
    all_lits = np.array([l for c in clauses for l in c], dtype=np.int32)
    all_vars = np.abs(all_lits) - 1
    all_signs = np.sign(all_lits)  # +1 or -1

    # Clause offsets for vectorised access
    clause_lens = np.array([len(c) for c in clauses], dtype=np.int32)
    clause_starts = np.concatenate([[0], np.cumsum(clause_lens)[:-1]])

    # 3. Spectral Initialisation (Max-Cut formulation)
    W = np.zeros((n, n), dtype=np.float64)
    for c in clauses:
        for k in range(len(c)):
            for m in range(k + 1, len(c)):
                l1, l2 = c[k], c[m]
                v1, v2 = abs(l1) - 1, abs(l2) - 1
                if v1 == v2:
                    continue
                s1 = 1 if l1 > 0 else -1
                s2 = 1 if l2 > 0 else -1
                if s1 == s2:
                    W[v1, v2] += 1
                    W[v2, v1] += 1
                else:
                    W[v1, v2] -= 1
                    W[v2, v1] -= 1

    try:
        D = np.diag(np.sum(W, axis=1))
        L = D - W
        eigvals, eigvecs = np.linalg.eigh(L)
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        fiedler_idx = 2 if eigvals[1] < 1e-6 else 1
        fiedler_vec = eigvecs[:, fiedler_idx]
        spectral_assignment = (fiedler_vec > 0)
    except Exception:
        spectral_assignment = (np.random.rand(n) < 0.5)

    rng = random.Random(42)
    best_assignment = None
    best_unsat_count = clause_count + 1

    # Precompute for vectorised flip updates:
    # For each variable v, the clause indices where v appears positively and negatively
    pos_clause_idx = [np.array(var_pos[v], dtype=np.int32) if var_pos[v] else np.array([], dtype=np.int32) for v in range(n)]
    neg_clause_idx = [np.array(var_neg[v], dtype=np.int32) if var_neg[v] else np.array([], dtype=np.int32) for v in range(n)]

    # sat_count: number of satisfied literals per clause
    # assignment: bool array of length n
    # unsat_mask: bool array of length clause_count

    def compute_sat_count(assignment):
        # assignment: np.array of bools (True = variable is True)
        # For each literal, it is satisfied if (assignment[var] == (sign == +1))
        # i.e., sign * (assignment[var] ? 1 : 0) + ... let's do it directly
        # lit_satisfied[i] = (assignment[all_vars[i]] == (all_signs[i] == 1))
        is_true = all_signs == 1  # True for positive literals
        is_sat = (assignment[all_vars] == is_true)
        # Reshape into clauses and sum
        # Use np.add.reduceat
        padded = np.zeros(clause_starts[-1] + clause_count, dtype=np.int32)
        padded[clause_starts] = is_sat.astype(np.int32)
        sat_count = np.add.reduceat(padded, clause_starts)
        return sat_count

    def get_unsat_mask(sat_count):
        return sat_count == 0

    max_restarts = 5000
    steps_per_restart = 15000
    noise = 0.25

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        if restart == 0:
            assignment = spectral_assignment.copy()
        elif restart < 5:
            assignment = spectral_assignment.copy()
            for _ in range(n // 4 + 1):
                assignment[rng.randint(0, n - 1)] = not assignment[rng.randint(0, n - 1)]
        else:
            if best_assignment is not None and rng.random() < 0.7:
                assignment = best_assignment.copy()
                for _ in range(n // 4 + 1):
                    assignment[rng.randint(0, n - 1)] = not assignment[rng.randint(0, n - 1)]
            else:
                assignment = (rng.random() < 0.5)

        sat_count = compute_sat_count(assignment)
        unsat_mask = get_unsat_mask(sat_count)
        current_unsat = int(np.count_nonzero(unsat_mask))

        if current_unsat == 0:
            return assignment.tolist()

        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment.copy()

        # WalkSAT loop – vectorised
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break

            if not unsat_mask.any():
                return assignment.tolist()

            if rng.random() < noise:
                var = rng.randint(0, n - 1)
            else:
                # Pick a random unsat clause, then use smooth log-sum-exp scoring
                unsat_indices = np.where(unsat_mask)[0]
                ci = unsat_indices[rng.randint(0, len(unsat_indices) - 1)]
                # Get the 3 literals in this clause
                s = clause_starts[ci]
                e = s + clause_lens[ci]
                lits = all_lits[s:e]
                vars_c = all_vars[s:e]
                signs_c = all_signs[s:e]

                # For each variable in the clause, compute impact score
                # Score = expected reduction in unsat count
                # Use log-sum-exp over affected clauses for smoothness
                scores = np.zeros(len(vars_c), dtype=np.float64)
                for j in range(len(vars_c)):
                    v = vars_c[j]
                    l = lits[j]
                    target = (l > 0)
                    if assignment[v] != target:
                        # Flipping makes this literal true
                        # Count how many currently-unsat clauses this would fix
                        affected = pos_clause_idx[v] if l > 0 else neg_clause_idx[v]
                        if len(affected) == 0:
                            scores[j] = 0
                            continue
                        # For each affected clause, check if flipping v would make it satisfied
                        # i.e., sat_count[ci] was 0 and this literal was the only unsat one
                        # Vectorised: affected clauses where sat_count == 0 and this is the only 0
                        # A clause ci is fixed if sat_count[ci] == 0 and
                        #   the literal l is the only unsatisfied one in ci
                        # Which means: for all other literals in ci, they are satisfied
                        # Equivalently: the clause has exactly 1 unsat literal and it's l
                        # We can check: sat_count[affected] == 0
                        #   AND for the clause, the number of unsat literals is 1
                        #   i.e., sat_count[ci] == 0 and (clause_len - 1) literals are satisfied
                        #   which means sat_count[ci] == 0 and the other 2 literals are true
                        # Faster: count how many of the affected clauses have sat_count==0
                        # and then among those, count how many would be fully satisfied
                        fixed = 0
                        for aci in affected:
                            if sat_count[aci] == 0:
                                # Check if all other literals in aci are satisfied
                                acs = clause_starts[aci]
                                ael = acs + clause_lens[aci]
                                other_lits = all_lits[acs:ael]
                                other_vars = all_vars[acs:ael]
                                other_signs = all_signs[acs:ael]
                                # Exclude the current literal position
                                other_sat = np.array([
                                    (assignment[other_vars[k]] == (other_signs[k] == 1))
                                    for k in range(len(other_vars))
                                    if other_lits[k] != l
                                ])
                                if len(other_sat) == 0 or other_sat.all():
                                    fixed += 1
                        scores[j] = fixed
                    else:
                        # Flipping would make this literal false (worse)
                        # Penalty: how many sat clauses would break
                        scores[j] = -1.0  # mild penalty

                # Smooth log-sum-exp selection: pick with probability proportional to exp(beta * score)
                beta = 2.0
                shifted = beta * scores - np.max(scores)
                weights = np.exp(shifted)
                total = weights.sum()
                if total < 1e-12:
                    var = vars_c[rng.randint(0, len(vars_c) - 1)]
                else:
                    probs = weights / total
                    var = vars_c[rng.choice(len(vars_c), p=probs)]

            # Apply flip
            old_val = assignment[var]
            assignment[var] = not old_val

            # Vectorised sat_count update
            if old_val:
                # Variable was True, now False
                # Positive literals in var_pos[var] lose one satisfied literal
                if len(pos_clause_idx[var]) > 0:
                    sat_count[pos_clause_idx[var]] -= 1
                # Negative literals in var_neg[var] gain one satisfied literal
                if len(neg_clause_idx[var]) > 0:
                    sat_count[neg_clause_idx[var]] += 1
            else:
                # Variable was False, now True
                if len(pos_clause_idx[var]) > 0:
                    sat_count[pos_clause_idx[var]] += 1
                if len(neg_clause_idx[var]) > 0:
                    sat_count[neg_clause_idx[var]] -= 1

            # Ensure no negative counts
            sat_count = np.maximum(sat_count, 0)

            # Update unsat mask
            unsat_mask = get_unsat_mask(sat_count)
            current_unsat = int(np.count_nonzero(unsat_mask))

            if current_unsat == 0:
                return assignment.tolist()

            if step % 100 == 0 and current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment.copy()

    # Phase 2: Final polish
    if best_assignment is not None:
        assignment = best_assignment.copy()
        for _ in range(3000):
            if time.time() > deadline:
                break
            sat_count = compute_sat_count(assignment)
            unsat_mask = get_unsat_mask(sat_count)
            current_unsat = int(np.count_nonzero(unsat_mask))
            if current_unsat == 0:
                return assignment.tolist()

            # Greedy: find variable flip with max gain
            best_var = -1
            best_gain = -1
            for v in range(n):
                # Try flipping v
                affected_pos = pos_clause_idx[v]
                affected_neg = neg_clause_idx[v]
                if len(affected_pos) == 0 and len(affected_neg) == 0:
                    continue
                # Compute net gain
                gain = 0
                if assignment[v]:
                    # Flipping to False
                    gain = int(np.sum(sat_count[affected_pos] == 0))  # clauses that become unsat
                    gain -= int(np.sum(sat_count[affected_pos] == 1))  # clauses that were sat with 1, now 0
                    # Actually: clauses where this literal is the only satisfied one become unsat
                    # Clauses where this literal was unsat become sat
                    # Let's just simulate
                # Simple simulation for small n
                assignment[v] = not assignment[v]
                sat_count2 = sat_count.copy()
                if assignment[v]:
                    if len(affected_pos) > 0:
                        sat_count2[affected_pos] += 1
                    if len(affected_neg) > 0:
                        sat_count2[affected_neg] -= 1
                else:
                    if len(affected_pos) > 0:
                        sat_count2[affected_pos] -= 1
                    if len(affected_neg) > 0:
                        sat_count2[affected_neg] += 1
                sat_count2 = np.maximum(sat_count2, 0)
                new_unsat = int(np.sum(sat_count2 == 0))
                gain = current_unsat - new_unsat
                assignment[v] = not assignment[v]  # revert
                if gain > best_gain:
                    best_gain = gain
                    best_var = v

            if best_var >= 0 and best_gain > 0:
                assignment[best_var] = not assignment[best_var]
                sat_count = compute_sat_count(assignment)
                unsat_mask = get_unsat_mask(sat_count)
                current_unsat = int(np.count_nonzero(unsat_mask))
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment.copy()
            else:
                break

        if best_unsat_count == 0:
            return best_assignment.tolist()

    if best_assignment is not None:
        return best_assignment.tolist()
    return (rng.random() < 0.5).tolist()
