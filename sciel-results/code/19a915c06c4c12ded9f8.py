import numpy as np
import time
import random

def solve(instance):
    n = instance["n"]
    orig_clauses = instance["clauses"]
    m = len(orig_clauses)

    # Convert to 0-indexed, separated into positive/negative variable indices per clause
    # For each clause: list of (var_idx, sign) where sign=1 means positive literal, -1 means negative
    # Store as two arrays for vectorized operations
    clause_vars = []
    clause_signs = []
    for c in orig_clauses:
        vars_in_c = [abs(l) - 1 for l in c]
        signs_in_c = [1 if l > 0 else -1 for l in c]
        clause_vars.append(vars_in_c)
        clause_signs.append(signs_in_c)

    # Variable to clauses mapping
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clause_vars):
        for v in c:
            if i not in var_to_clauses[v]:
                var_to_clauses[v].append(i)

    # Also need: for each (var, clause) pair, the sign
    # var_sign_in_clause[var][clause_idx] = sign
    var_sign_in_clause = [{} for _ in range(n)]
    for i, c in enumerate(clause_vars):
        for j, v in enumerate(c):
            var_sign_in_clause[v][i] = clause_signs[i][j]

    # 1. Unit Propagation
    assignment = [None] * n
    active_clauses = [c[:] for c in orig_clauses]

    changed = True
    while changed:
        changed = False
        for i in range(len(active_clauses)):
            if len(active_clauses[i]) == 0:
                return [random.random() < 0.5 for _ in range(n)]
            if len(active_clauses[i]) == 1:
                lit = active_clauses[i][0]
                var = abs(lit) - 1
                val = lit > 0
                if assignment[var] is not None:
                    if assignment[var] != val:
                        return [random.random() < 0.5 for _ in range(n)]
                    else:
                        active_clauses[i] = []
                        changed = True
                        continue
                assignment[var] = val
                changed = True
                for j in range(len(active_clauses)):
                    if len(active_clauses[j]) == 0:
                        continue
                    new_clause = []
                    clause_sat = False
                    for l in active_clauses[j]:
                        v = abs(l) - 1
                        if v == var:
                            if (l > 0 and val) or (l < 0 and not val):
                                clause_sat = True
                                break
                            else:
                                continue
                        else:
                            new_clause.append(l)
                    active_clauses[j] = [] if clause_sat else new_clause
                break

    if all(a is not None for a in assignment):
        for c in orig_clauses:
            sat = False
            for l in c:
                v = abs(l) - 1
                if assignment[v] == (l > 0):
                    sat = True
                    break
            if not sat:
                return [random.random() < 0.5 for _ in range(n)]
        return assignment

    # 2. Vectorized WalkSAT
    # Start with random assignment
    rng = random.Random(42)
    for i in range(n):
        if assignment[i] is None:
            assignment[i] = rng.random() < 0.5

    # Build numpy arrays for vectorized operations
    # For each clause, we need to track how many literals are satisfied
    # Represent assignment as int array: 1 for True, 0 for False
    # A literal (var, sign) is satisfied iff assignment[var] == (1 if sign==1 else 0)
    # i.e., assignment[var] * sign == 1 when sign==1, or (1-assignment[var]) * (-sign) ... 
    # Simpler: sat = (assignment[var] == 1) if sign==1 else (assignment[var] == 0)
    # = assignment[var] if sign==1 else (1 - assignment[var])
    # = (assignment[var] + sign) // 2 ... let me think
    # sign=1: want assignment[var] (1 if True)
    # sign=-1: want 1 - assignment[var]
    # = (assignment[var] * sign + 1) // 2
    # sign=1: (a*1+1)//2 = (a+1)//2 = a (when a in {0,1})
    # sign=-1: (a*(-1)+1)//2 = (1-a)//2 = 1-a (when a in {0,1})
    # Yes!

    # Build flat arrays
    # For all clauses, flatten the variable indices and signs
    all_vars_flat = []
    all_signs_flat = []
    clause_starts = [0]
    for i in range(m):
        all_vars_flat.extend(clause_vars[i])
        all_signs_flat.extend(clause_signs[i])
        clause_starts.append(len(all_vars_flat))

    all_vars_arr = np.array(all_vars_flat, dtype=np.int32)
    all_signs_arr = np.array(all_signs_flat, dtype=np.int8)

    # Precompute for each clause the number of literals
    clause_lens = np.array([clause_starts[i+1] - clause_starts[i] for i in range(m)], dtype=np.int32)

    # sat_count per clause
    # For a given assignment (0/1 array), literal sat = (assignment[var] * sign + 1) // 2
    def compute_sat_counts(assign):
        # assign: numpy array of 0/1, length n
        vals = assign[all_vars_arr].astype(np.int8) * all_signs_arr + 1
        sat_bits = vals // 2  # 0 or 1
        # Sum per clause
        counts = np.zeros(m, dtype=np.int32)
        for i in range(m):
            s = clause_starts[i]
            e = clause_starts[i+1]
            counts[i] = np.sum(sat_bits[s:e])
        return counts

    # Better: use np.add.reduceat
    def compute_sat_counts_fast(assign):
        vals = assign[all_vars_arr].astype(np.int8) * all_signs_arr + 1
        sat_bits = vals // 2
        # Use reduceat
        if m == 0:
            return np.array([], dtype=np.int32)
        counts = np.add.reduceat(sat_bits, clause_starts[:-1])
        # Fix: reduceat gives sum from starts[i] to starts[i+1]-1, which is what we want
        # But the last element might not be correct if the last clause doesn't end at the array end
        # Actually reduceat: out[i] = sum(arr[starts[i]:starts[i+1]]) for i < len(starts)-1
        # and out[-1] = sum(arr[starts[-1]:]) 
        # Since clause_starts has m+1 entries, we pass clause_starts[:-1] which has m entries
        # out[i] = sum(sat_bits[clause_starts[i]:clause_starts[i+1]]) for i in 0..m-2
        # out[m-1] = sum(sat_bits[clause_starts[m-1]:]) which goes to the end
        # That's correct since clause_starts[m] = len(sat_bits)
        return counts

    # Variable to clause indices for incremental updates
    var_clause_idx = np.array([var_to_clauses[v] for v in range(n)])  # ragged, can't do directly
    # Instead, for each variable, store list of (clause_idx, sign)
    var_clauses_list = []
    for v in range(n):
        pairs = []
        for ci in var_to_clauses[v]:
            pairs.append((ci, var_sign_in_clause[v][ci]))
        var_clauses_list.append(pairs)

    # Also for the flat array: for each variable, the positions in all_vars_arr
    var_flat_positions = [[] for _ in range(n)]
    for i in range(m):
        for j, v in enumerate(clause_vars[i]):
            var_flat_positions[v].append(clause_starts[i] + j)

    assign = np.array([1 if a else 0 for a in assignment], dtype=np.int8)
    sat_count = compute_sat_counts_fast(assign)
    num_unsat = int(np.sum(sat_count == 0))

    if num_unsat == 0:
        return [bool(a) for a in assign]

    start_time = time.time()
    deadline = start_time + 290.0

    best_num_unsat = num_unsat
    best_assign = assign.copy()

    steps = 0
    current_restart_limit = 10000
    rng_np = np.random.default_rng(42)

    # Precompute clause variable ranges as numpy arrays for the heuristic
    # For each clause, the flat index range
    # And for each variable in each clause, we need to know the clause idx and sign

    # For the heuristic, when we consider flipping variable v in clause c:
    # gain = number of clauses containing v where v is currently unsatisfied and would become satisfied
    # loss = number of clauses containing v where v is currently satisfied with count==1 and would become unsatisfied
    # We can precompute for each variable: list of (clause_idx, sign)

    while time.time() < deadline:
        if num_unsat == 0:
            return [bool(a) for a in assign]

        if num_unsat < best_num_unsat:
            best_num_unsat = num_unsat
            best_assign = assign.copy()

        if steps > current_restart_limit:
            # Restart
            assign = rng_np.random(n) < 0.5
            assign = assign.astype(np.int8)
            sat_count = compute_sat_counts_fast(assign)
            num_unsat = int(np.sum(sat_count == 0))
            steps = 0
            current_restart_limit = int(current_restart_limit * 1.3)
            if current_restart_limit > 500000:
                current_restart_limit = 500000
            continue

        # Pick random unsatisfied clause
        unsat_indices = np.where(sat_count == 0)[0]
        if len(unsat_indices) == 0:
            return [bool(a) for a in assign]
        c_idx = unsat_indices[rng_np.integers(len(unsat_indices))]

        clause = clause_vars[c_idx]

        # Weighted heuristic
        if rng_np.random() < 0.25:
            # Random flip
            var_idx = clause[rng_np.integers(len(clause))]
        else:
            # Compute score for each variable in the clause
            best_score = -float('inf')
            best_var = None
            for lit_pos in range(len(clause)):
                v = clause[lit_pos]
                sign = clause_signs[c_idx][lit_pos]
                # For variable v, iterate over all clauses it appears in
                satisfied_gain = 0
                unsatisfied_loss = 0
                for (ci, si) in var_clauses_list[v]:
                    old_count = sat_count[ci]
                    # Current satisfaction of literal (v, si) in clause ci:
                    # sat = (assign[v] * si + 1) // 2
                    current_sat = (int(assign[v]) * si + 1) // 2
                    if current_sat == 1:
                        # Currently satisfied, flipping would make it unsatisfied
                        if old_count == 1:
                            unsatisfied_loss += 1
                    else:
                        # Currently unsatisfied, flipping would make it satisfied
                        satisfied_gain += 1

                score = satisfied_gain * 1.0 - unsatisfied_loss * 1.5
                if score > best_score:
                    best_score = score
                    best_var = v

            if best_var is None:
                best_var = clause[0]
            var_idx = best_var

        # Flip
        old_val = int(assign[var_idx])
        new_val = 1 - old_val
        assign[var_idx] = new_val

        # Update sat_count incrementally
        for (ci, si) in var_clauses_list[var_idx]:
            old_count = sat_count[ci]
            if old_val * si + 1 >= 2:
                # Was satisfied, now unsatisfied
                new_count = old_count - 1
            else:
                # Was unsatisfied, now satisfied
                new_count = old_count + 1
            was_sat = old_count > 0
            is_sat = new_count > 0
            if was_sat and not is_sat:
                num_unsat += 1
            elif not was_sat and is_sat:
                num_unsat -= 1
            sat_count[ci] = new_count

        steps += 1

    # Final: return best
    return [bool(a) for a in best_assign]
