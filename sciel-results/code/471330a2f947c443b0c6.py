"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    start_time = time.time()
    timeout = 325.0  # Leave margin for overhead (evaluator kills at 330s)

    if m == 0:
        return [True] * n

    # Precompute clause structure
    clause_vars = np.empty((m, 3), dtype=np.int32)
    clause_signs = np.empty((m, 3), dtype=np.int8)  # 1 = positive, 0 = negative

    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_vars[i, j] = abs(lit) - 1
            clause_signs[i, j] = 1 if lit > 0 else 0

    # Build variable-to-clauses mapping for incremental updates
    var_clause_pos = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_clause_pos[v].append((i, j))

    # Convert to numpy arrays for fast access
    var_clauses_np = [np.array([p[0] for p in var_clause_pos[v]], dtype=np.int32) if var_clause_pos[v] else np.empty(0, dtype=np.int32) for v in range(n)]
    var_pos_np = [np.array([p[1] for p in var_clause_pos[v]], dtype=np.int32) if var_clause_pos[v] else np.empty(0, dtype=np.int32) for v in range(n)]

    def compute_num_true(assignment):
        """Compute number of true literals per clause (0, 1, 2, or 3)."""
        vals = assignment[clause_vars]  # (m, 3) bools
        lit_true = (vals == clause_signs)  # (m, 3) bools
        return lit_true.sum(axis=1).astype(np.int32)  # (m,) ints

    def is_satisfied(num_true):
        return np.all(num_true > 0)

    def unit_propagate(assignment, num_true):
        """Iteratively set variables to satisfy unit clauses."""
        changed = False
        for _ in range(3 * n + 10):
            unit = np.where(num_true == 1)[0]
            if unit.size == 0:
                break
            progress = False
            for c in unit:
                # Find the unique true literal position
                sat_pos = np.where((assignment[clause_vars[c]] == clause_signs[c]))[0]
                if sat_pos.size == 0:
                    continue
                p = int(sat_pos[0])
                v = int(clause_vars[c, p])
                target = int(clause_signs[c, p])
                if int(assignment[v]) == target:
                    continue
                # Flip v to target
                assignment[v] = target
                changed = True
                progress = True
                c_list = var_clauses_np[v]
                p_list = var_pos_np[v]
                if c_list.size == 0:
                    continue
                s = clause_signs[c_list, p_list]
                now_sat = (assignment[v] == s)
                delta = (2 * now_sat.astype(np.int32)) - 1
                num_true[c_list] += delta
            if not progress:
                break
        return assignment, num_true, changed

    def polish(assignment, num_true, max_steps=50000):
        """High-precision polish: greedy flips targeting minimum unsat, then random-walk escape."""
        rng = random.Random(99999)
        best_unsat = int(np.sum(num_true == 0))
        best_assign = assignment.copy()

        # Phase 1: Greedy descent - always flip variable that reduces unsat most
        for step in range(max_steps // 2):
            if time.time() - start_time > timeout - 3:
                break
            if best_unsat == 0:
                return assignment.tolist()

            # Find best single flip
            best_delta = 0
            best_var = -1
            for v in range(n):
                if var_clauses_np[v].size == 0:
                    continue
                c_list = var_clauses_np[v]
                p_list = var_pos_np[v]
                s = clause_signs[c_list, p_list]
                now_sat = (assignment[v] == s)
                unsat_mask = (num_true[c_list] == 0)
                # Flipping v: clauses where v's literal becomes true gain +1, where it becomes false lose -1
                # Literal is true when assignment[v] == sign, so flipping changes truth value
                becomes_sat = int(np.sum(unsat_mask & ~now_sat))  # currently false, will be true
                becomes_unsat = int(np.sum((num_true[c_list] == 1) & now_sat))  # currently true, sole support, will be false
                delta = becomes_sat - becomes_unsat
                if delta > best_delta:
                    best_delta = delta
                    best_var = v

            if best_var < 0 or best_delta <= 0:
                break

            assignment[best_var] = not assignment[best_var]
            c_list = var_clauses_np[best_var]
            p_list = var_pos_np[best_var]
            if c_list.size > 0:
                s = clause_signs[c_list, p_list]
                now_sat = (assignment[best_var] == s)
                delta = (2 * now_sat.astype(np.int32)) - 1
                num_true[c_list] += delta

            current_unsat = int(np.sum(num_true == 0))
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assign = assignment.copy()

        # Phase 2: Random walk with bias toward improving moves
        for step in range(max_steps // 2):
            if time.time() - start_time > timeout - 3:
                break
            current_unsat = int(np.sum(num_true == 0))
            if current_unsat == 0:
                return assignment.tolist()

            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assign = assignment.copy()

            # Pick random unsatisfied clause
            unsat_idx = np.where(num_true == 0)[0]
            if len(unsat_idx) == 0:
                return assignment.tolist()
            ci = int(unsat_idx[rng.randint(0, len(unsat_idx) - 1)])

            # Score each variable in this clause
            clause_v = clause_vars[ci]
            deltas = []
            for v in clause_v:
                c_list = var_clauses_np[v]
                p_list = var_pos_np[v]
                s = clause_signs[c_list, p_list]
                now_sat = (assignment[v] == s)
                unsat_mask = (num_true[c_list] == 0)
                becomes_sat = int(np.sum(unsat_mask & ~now_sat))
                becomes_unsat = int(np.sum((num_true[c_list] == 1) & now_sat))
                deltas.append(becomes_sat - becomes_unsat)

            # 60% greedy, 40% random among top-2
            max_d = max(deltas)
            if rng.random() < 0.6:
                var_to_flip = clause_v[int(np.argmax(deltas))]
            else:
                sorted_idx = np.argsort(deltas)[::-1]
                if len(sorted_idx) >= 2 and rng.random() < 0.5:
                    var_to_flip = clause_v[int(sorted_idx[1])]
                else:
                    var_to_flip = clause_v[int(sorted_idx[0])]

            assignment[var_to_flip] = not assignment[var_to_flip]
            c_list = var_clauses_np[var_to_flip]
            p_list = var_pos_np[var_to_flip]
            if c_list.size > 0:
                s = clause_signs[c_list, p_list]
                now_sat = (assignment[var_to_flip] == s)
                delta = (2 * now_sat.astype(np.int32)) - 1
                num_true[c_list] += delta

        current_unsat = int(np.sum(num_true == 0))
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assign = assignment.copy()

        if best_unsat == 0:
            return best_assign.tolist()
        return None

    def walk_sat(restart_assignment, max_flips, seed):
        """Run WalkSAT with clause-weight tracking and efficient num_true tracking."""
        rng = random.Random(seed)
        assignment = restart_assignment.copy()
        num_true = compute_num_true(assignment)
        unsat_count = int(np.sum(num_true == 0))

        if unsat_count == 0:
            return assignment.tolist()

        best_unsat = unsat_count
        best_assignment = assignment.copy()

        for step in range(max_flips):
            if time.time() - start_time > timeout - 5:
                break

            if unsat_count == 0:
                return assignment.tolist()

            # Track best state
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment.copy()

            # Pick a random unsatisfied clause
            unsat_idx = np.where(num_true == 0)[0]
            if len(unsat_idx) == 0:
                return assignment.tolist()
            clause_idx = int(unsat_idx[rng.randint(0, len(unsat_idx) - 1)])

            # Choose variable to flip using score-based strategy
            clause_v = clause_vars[clause_idx]
            deltas = [0, 0, 0]

            for pos, v in enumerate(clause_v):
                c_list = var_clauses_np[v]
                p_list = var_pos_np[v]
                if c_list.size == 0:
                    deltas[pos] = 0
                    continue
                s = clause_signs[c_list, p_list]
                now_sat = (assignment[v] == s)
                unsat_mask = (num_true[c_list] == 0)
                becomes_sat = int(np.sum(unsat_mask & ~now_sat))
                becomes_unsat = int(np.sum((num_true[c_list] == 1) & now_sat))
                deltas[pos] = becomes_sat - becomes_unsat

            # Adaptive temperature: decrease over time for more greedy behavior
            temp = max(0.5, 3.0 - 0.05 * (step / max(1, max_flips / 10)))
            exp_deltas = np.exp(np.array(deltas, dtype=np.float64) / temp)
            probs = exp_deltas / exp_deltas.sum()

            # Greedy probability increases as we get closer to solution
            greedy_prob = min(0.9, 0.5 + 0.1 * (1 - unsat_count / max(1, m)))
            if rng.random() < greedy_prob:
                var_to_flip = clause_v[int(np.argmax(deltas))]
            else:
                var_to_flip = clause_v[rng.choices([0, 1, 2], weights=probs)[0]]

            # Flip the variable
            assignment[var_to_flip] = not assignment[var_to_flip]

            # Incrementally update num_true for affected clauses
            c_list = var_clauses_np[var_to_flip]
            p_list = var_pos_np[var_to_flip]
            if c_list.size > 0:
                s = clause_signs[c_list, p_list]
                now_sat = (assignment[var_to_flip] == s)
                delta = (2 * now_sat.astype(np.int32)) - 1
                num_true[c_list] += delta
                unsat_count = int(np.sum(num_true == 0))

        # Final check
        if unsat_count == 0:
            return assignment.tolist()

        # If close to solution, try polish
        if unsat_count <= 5:
            result = polish(assignment, num_true)
            if result is not None:
                return result

        return None

    # Polarity-based initialization: try to satisfy as many clauses as possible
    def polarity_init():
        """Initialize by setting each variable to satisfy more clauses."""
        assignment = np.zeros(n, dtype=np.int8)
        for v in range(n):
            if var_clauses_np[v].size == 0:
                assignment[v] = 1  # arbitrary
                continue
            c_list = var_clauses_np[v]
            p_list = var_pos_np[v]
            s = clause_signs[c_list, p_list]
            # Count how many clauses each polarity would satisfy
            pos_count = int(np.sum(s == 1))
            neg_count = int(np.sum(s == 0))
            if pos_count >= neg_count:
                assignment[v] = 1
            else:
                assignment[v] = 0
        return assignment

    # Main search loop with restarts
    best_result = None
    rng = random.Random(0)
    restart_count = 0
    best_unsat_ever = m + 1
    best_phase_assignment = None
    best_phase_num_true = None

    # Try polarity-based initialization first
    pol_assign = polarity_init()
    pol_num_true = compute_num_true(pol_assign)
    pol_unsat = int(np.sum(pol_num_true == 0))
    if pol_unsat < best_unsat_ever:
        best_unsat_ever = pol_unsat
        best_phase_assignment = pol_assign.copy()
        best_phase_num_true = pol_num_true.copy()

    # Try a few random initializations
    for _ in range(5):
        initial = (rng.random() < 0.5).astype(np.int8)
        num_true = compute_num_true(initial)
        unsat = int(np.sum(num_true == 0))
        if unsat < best_unsat_ever:
            best_unsat_ever = unsat
            best_phase_assignment = initial.copy()
            best_phase_num_true = num_true.copy()

    # Try quick WalkSAT from best initial
    if best_phase_assignment is not None:
        quick_result = walk_sat(best_phase_assignment, 20000, seed=42)
        if quick_result is not None:
            return quick_result

    while time.time() - start_time < timeout - 15:
        restart_count += 1

        # Use phase assignment with adaptive perturbation
        if best_phase_assignment is not None and restart_count > 1:
            initial = best_phase_assignment.copy()
            # Perturbation scales with restart count but caps out
            num_flips = max(1, int(n * (0.05 + 0.02 * min(restart_count, 25))))
            num_flips = min(num_flips, n)
            flip_indices = rng.sample(range(n), num_flips)
            for idx in flip_indices:
                initial[idx] = 1 - initial[idx]
        else:
            initial = (rng.random() < 0.5).astype(np.int8)

        # Apply unit propagation
        initial, init_num_true, _ = unit_propagate(initial, compute_num_true(initial))
        init_unsat = int(np.sum(init_num_true == 0))
        if init_unsat < best_unsat_ever:
            best_unsat_ever = init_unsat
            best_phase_assignment = initial.copy()
            best_phase_num_true = init_num_true.copy()

        if init_unsat == 0:
            return initial.tolist()

        # Adaptive flip budget: more flips for harder instances
        base_flips = 15000
        adaptive_factor = 1.0 + 0.15 * min(restart_count, 20)
        max_flips = int(base_flips * adaptive_factor)

        result = walk_sat(initial, max_flips, seed=restart_count * 12345 + 67890)

        if result is not None:
            best_result = result
            break

    if best_result is not None:
        return best_result

    # If we didn't find a solution, try one final polish from the best known state
    if best_phase_assignment is not None:
        final_num_true = compute_num_true(best_phase_assignment)
        final_result = polish(best_phase_assignment, final_num_true, max_steps=100000)
        if final_result is not None:
            return final_result
        return best_phase_assignment.astype(bool).tolist()

    # Fallback: return a random assignment
    return [(rng.random() < 0.5) for _ in range(n)]
# EVOLVE-BLOCK-END
