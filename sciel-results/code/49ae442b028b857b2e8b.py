"""Synthetic planted 3-SAT solver (50 vars) — hybrid SA + WalkSAT + FlipSAT with incremental state."""
import random
import time
import math


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    deadline = time.time() + 290  # safety margin before 300 s kill

    # ---- Precompute clause structure and incidence ----
    # clause_info[i] = [(var_idx, polarity), ...]  polarity: +1 pos, -1 neg
    # var_to_clauses[j] = [(clause_idx, polarity), ...]
    clause_info = []
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        info = []
        for l in clause:
            var_idx = abs(l) - 1
            pol = 1 if l > 0 else -1
            info.append((var_idx, pol))
            var_to_clauses[var_idx].append((i, pol))
        clause_info.append(info)

    # ---- Incremental state ----
    sat_count = [0] * num_clauses
    clause_sat = [False] * num_clauses
    total_sat = 0
    assignment = [False] * n

    def _clause_is_sat(ci):
        for var_idx, pol in clause_info[ci]:
            if (pol == 1 and assignment[var_idx]) or (pol == -1 and not assignment[var_idx]):
                return True
        return False

    def initialize_state(asm):
        nonlocal sat_count, clause_sat, total_sat
        assignment[:] = asm
        sat_count = [0] * num_clauses
        clause_sat = [False] * num_clauses
        total_sat = 0
        for ci in range(num_clauses):
            cnt = 0
            for var_idx, pol in clause_info[ci]:
                if (pol == 1 and assignment[var_idx]) or (pol == -1 and not assignment[var_idx]):
                    cnt += 1
            sat_count[ci] = cnt
            is_sat = cnt > 0
            clause_sat[ci] = is_sat
            if is_sat:
                total_sat += 1
        return total_sat

    def flip_var(var_idx):
        """Flip variable, update state incrementally. Returns delta in total_sat."""
        nonlocal total_sat
        assignment[var_idx] = not assignment[var_idx]
        delta = 0
        for ci, pol in var_to_clauses[var_idx]:
            # Recompute sat_count for this clause
            cnt = 0
            for vi2, pol2 in clause_info[ci]:
                if (pol2 == 1 and assignment[vi2]) or (pol2 == -1 and not assignment[vi2]):
                    cnt += 1
            was_sat = clause_sat[ci]
            sat_count[ci] = cnt
            is_sat = cnt > 0
            clause_sat[ci] = is_sat
            if is_sat and not was_sat:
                delta += 1
            elif not is_sat and was_sat:
                delta -= 1
        total_sat += delta
        return delta

    def get_unsat_clauses():
        return [ci for ci in range(num_clauses) if not clause_sat[ci]]

    def unsat_count():
        return num_clauses - total_sat

    def count_satisfied():
        return total_sat

    # ---- WalkSAT step with O(degree) damage heuristic ----
    def walksat_step(rng, unsat_list):
        if not unsat_list:
            return
        clause_idx = rng.choice(unsat_list)
        info = clause_info[clause_idx]

        # For each literal in the unsat clause, compute damage:
        # how many other clauses would become unsatisfied if we flip this variable.
        best_literal = None
        best_damage = float('inf')
        best_gain = -1

        for var_idx, pol in info:
            # Damage: clauses containing var_idx (other than clause_idx) that are
            # currently satisfied ONLY by var_idx with the current value.
            damage = 0
            for ci, pol_c in var_to_clauses[var_idx]:
                if ci == clause_idx:
                    continue
                if not clause_sat[ci]:
                    continue  # already unsat, flipping can't make it worse
                # Check if this clause is satisfied only by var_idx
                # i.e., sat_count[ci] == 1 and var_idx currently satisfies it
                if sat_count[ci] == 1:
                    # var_idx must be the one satisfying it (since sat_count==1)
                    # and flipping would break it
                    # Check: does var_idx currently satisfy this clause?
                    if (pol_c == 1 and assignment[var_idx]) or (pol_c == -1 and not assignment[var_idx]):
                        damage += 1
            # Gain: flipping this literal satisfies the chosen unsat clause (always +1 for the clause)
            # but we also want to consider if it helps other unsat clauses (minor)
            if damage < best_damage or (damage == best_damage and rng.random() < 0.5):
                best_damage = damage
                best_literal = var_idx

        if best_literal is not None:
            flip_var(best_literal)

    def flipsat_step(rng, unsat_list):
        if not unsat_list:
            return
        clause_idx = rng.choice(unsat_list)
        info = clause_info[clause_idx]
        var_idx, pol = rng.choice(info)
        flip_var(var_idx)

    # ---- Simulated Annealing ----
    def simulated_annealing(rng, start_temp, end_temp, max_steps):
        nonlocal total_sat
        current_score = total_sat
        best_score = current_score
        best_assignment = assignment[:]

        if current_score == num_clauses:
            return best_assignment, True

        for step in range(max_steps):
            if time.time() > deadline - 3:
                break

            progress = step / max_steps
            temp = start_temp * (end_temp / start_temp) ** progress

            var_idx = rng.randint(0, n - 1)
            old_val = assignment[var_idx]
            flip_var(var_idx)
            new_score = total_sat
            delta = new_score - current_score

            if delta >= 0:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_assignment = assignment[:]
                    if best_score == num_clauses:
                        return best_assignment, True
            else:
                if temp > 0:
                    prob = math.exp(delta / temp)
                    if rng.random() < prob:
                        current_score = new_score
                    else:
                        # Reject: undo the flip
                        flip_var(var_idx)

        return best_assignment, best_score == num_clauses

    # ---- WalkSAT search ----
    def walksat_search(rng, max_steps):
        unsat_list = get_unsat_clauses()
        if not unsat_list:
            return assignment, True

        best_assignment = assignment[:]
        best_unsat = len(unsat_list)

        for step in range(max_steps):
            if time.time() > deadline - 3:
                break
            if not unsat_list:
                return assignment, True

            cur_unsat = len(unsat_list)
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]

            walksat_step(rng, unsat_list)
            unsat_list = get_unsat_clauses()

        return best_assignment, best_unsat == 0

    # ---- FlipSAT search ----
    def flipsat_search(rng, max_steps):
        unsat_list = get_unsat_clauses()
        if not unsat_list:
            return assignment, True

        best_assignment = assignment[:]
        best_unsat = len(unsat_list)

        for step in range(max_steps):
            if time.time() > deadline - 3:
                break
            if not unsat_list:
                return assignment, True

            cur_unsat = len(unsat_list)
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]

            flipsat_step(rng, unsat_list)
            unsat_list = get_unsat_clauses()

        return best_assignment, best_unsat == 0

    # ---- Main search loop ----
    best_assignment = None
    best_score = -1

    # Strategy 1: Simulated Annealing with multiple restarts
    sa_restarts = 20
    sa_steps = 4000
    sa_start_temp = 1.0
    sa_end_temp = 0.01

    for restart in range(sa_restarts):
        if time.time() > deadline - 10:
            break
        rng = random.Random(1000 + restart)
        asm = [rng.random() < 0.5 for _ in range(n)]
        initialize_state(asm)
        if total_sat == num_clauses:
            return assignment[:]
        best_sa, solved = simulated_annealing(rng, sa_start_temp, sa_end_temp, sa_steps)
        if solved:
            return best_sa
        sa_score = count_satisfied()
        # best_sa is a snapshot; recompute its score by re-initializing
        initialize_state(best_sa)
        sa_score = total_sat
        if sa_score > best_score:
            best_score = sa_score
            best_assignment = best_sa[:]

    # Strategy 2: WalkSAT with multiple restarts
    ws_restarts = 30
    ws_steps = 8000

    for restart in range(ws_restarts):
        if time.time() > deadline - 10:
            break
        rng = random.Random(5000 + restart)
        asm = [rng.random() < 0.5 for _ in range(n)]
        initialize_state(asm)
        if total_sat == num_clauses:
            return assignment[:]
        best_ws, solved = walksat_search(rng, ws_steps)
        if solved:
            return best_ws
        initialize_state(best_ws)
        ws_score = total_sat
        if ws_score > best_score:
            best_score = ws_score
            best_assignment = best_ws[:]

    # Strategy 3: FlipSAT with multiple restarts
    fs_restarts = 15
    fs_steps = 4000

    for restart in range(fs_restarts):
        if time.time() > deadline - 10:
            break
        rng = random.Random(9000 + restart)
        asm = [rng.random() < 0.5 for _ in range(n)]
        initialize_state(asm)
        if total_sat == num_clauses:
            return assignment[:]
        best_fs, solved = flipsat_search(rng, fs_steps)
        if solved:
            return best_fs
        initialize_state(best_fs)
        fs_score = total_sat
        if fs_score > best_score:
            best_score = fs_score
            best_assignment = best_fs[:]

    # Final polish: WalkSAT from the best assignment found
    if best_assignment is not None and time.time() < deadline - 15:
        rng = random.Random(99999)
        initialize_state(best_assignment)
        unsat_list = get_unsat_clauses()
        if not unsat_list:
            return assignment[:]
        for step in range(30000):
            if time.time() > deadline - 5:
                break
            if not unsat_list:
                return assignment[:]
            walksat_step(rng, unsat_list)
            unsat_list = get_unsat_clauses()
        if not unsat_list:
            return assignment[:]

    if best_assignment is not None:
        return best_assignment

    return [False] * n
