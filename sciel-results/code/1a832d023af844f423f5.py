# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def _build_masks(clauses, n):
    pos = []
    neg = []
    for c in clauses:
        p = 0
        q = 0
        for l in c:
            if l > 0:
                p |= (1 << (l - 1))
            else:
                q |= (1 << (-l - 1))
        pos.append(p)
        neg.append(q)
    return pos, neg


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 295.0
    pos, neg = _build_masks(clauses, n)
    pos = tuple(pos)
    neg = tuple(neg)

    # Build numpy arrays for vectorized satisfaction checking
    pos_arr = np.array(pos, dtype=np.uint64)
    neg_arr = np.array(neg, dtype=np.uint64)
    full_mask = (1 << n) - 1

    # Precompute clause literal index lists
    clause_vars = []
    for c in clauses:
        clause_vars.append([(abs(l) - 1, l > 0) for l in c])

    # Build variable occurrence: for each var, list of (clause_idx, is_pos_in_clause)
    var_occ = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for var, is_pos in cv:
            var_occ[var].append((ci, is_pos))

    def sat_count_numpy(assign_int):
        assign_np = np.uint64(assign_int)
        pos_satisfied = (assign_np & pos_arr) != 0
        neg_satisfied = ((~assign_np) & full_mask & neg_arr) != 0
        satisfied = pos_satisfied | neg_satisfied
        return int(np.sum(satisfied))

    def assign_to_int(assign):
        v = 0
        for i, b in enumerate(assign):
            if b:
                v |= (1 << i)
        return v

    def int_to_assign(v):
        return [bool(v & (1 << i)) for i in range(n)]

    def get_unsat_set(v):
        """Return set of unsatisfied clause indices."""
        assign_np = np.uint64(v)
        pos_satisfied = (assign_np & pos_arr) != 0
        neg_satisfied = ((~assign_np) & full_mask & neg_arr) != 0
        satisfied = pos_satisfied | neg_satisfied
        return set(np.where(~satisfied)[0].tolist())

    def update_unsat_after_flip(v, unsat_set, var):
        """After flipping var, update unsat_set incrementally."""
        # Check clauses containing var
        for ci, is_pos_in_clause in var_occ[var]:
            # Determine if this clause is now satisfied
            bit = 1 << var
            if is_pos_in_clause:
                # Clause has +var: satisfied if var is True (bit set in v)
                now_satisfied = (v & bit) != 0
            else:
                # Clause has -var: satisfied if var is False (bit not set in v)
                now_satisfied = (v & bit) == 0
            
            if now_satisfied:
                unsat_set.discard(ci)
            else:
                # Need to check if other literals in clause satisfy it
                # Quick check: is any other literal satisfied?
                clause = clauses[ci]
                other_satisfied = False
                for l in clause:
                    if abs(l) == var + 1:
                        continue
                    if l > 0:
                        if (v & (1 << (l - 1))) != 0:
                            other_satisfied = True
                            break
                    else:
                        if (v & (1 << (-l - 1))) == 0:
                            other_satisfied = True
                            break
                if not other_satisfied:
                    unsat_set.add(ci)
                else:
                    unsat_set.discard(ci)
        return unsat_set

    rng = random.Random(12345)
    best_assign = None
    best_sat = -1

    # Phase 0: quick random initializations
    for attempt_seed in range(8):
        r = random.Random(attempt_seed)
        assign = [r.random() < 0.5 for _ in range(n)]
        v = assign_to_int(assign)
        s = sat_count_numpy(v)
        if s > best_sat:
            best_sat = s
            best_assign = assign
        if s == m:
            return assign
        if time.time() > deadline:
            return best_assign if best_assign else [False] * n

    # Phase 1: Multi-restart WalkSAT with incremental tracking
    for restart in range(5000):
        if time.time() > deadline:
            break
        r = random.Random(restart)
        assign = [r.random() < 0.5 for _ in range(n)]
        v = assign_to_int(assign)
        unsat_set = get_unsat_set(v)
        cur_sat = m - len(unsat_set)
        
        steps_budget = 8000
        for step in range(steps_budget):
            if time.time() > deadline:
                break
            
            if len(unsat_set) == 0:
                best_assign = int_to_assign(v)
                best_sat = m
                return best_assign
            
            # Pick a random unsatisfied clause
            unsat_list = list(unsat_set)
            ci = unsat_list[r.randrange(len(unsat_list))]
            
            # Score each literal in the clause using smooth surrogate
            cv = clause_vars[ci]
            scores = []
            for var, is_pos in cv:
                # Count how many unsat clauses this flip would help/hurt
                help_count = 0.0
                hurt_count = 0.0
                for uci, uis_pos in var_occ[var]:
                    if uci == ci:
                        continue
                    if uci in unsat_set:
                        # Flipping var: does it satisfy or break clause uci?
                        if is_pos and not uis_pos:
                            # We're considering +var in ci, but uci has -var
                            # Flipping var to True satisfies uci's -var? No wait...
                            # Actually we need to think about what flip does
                            pass
                        # Simpler: just count occurrences in unsat clauses
                        help_count += 1.0
                    elif is_pos and not uis_pos:
                        # This var appears with opposite polarity in a sat clause
                        # Flipping might break it
                        hurt_count += 0.3
                    elif not is_pos and uis_pos:
                        hurt_count += 0.3
                
                # Soft scoring: prefer vars that appear in many unsat clauses
                score = help_count - hurt_count + r.random() * 1.5
                scores.append((score, var, is_pos))
            
            scores.sort(reverse=True)
            var = scores[0][1]
            
            bit = 1 << var
            v ^= bit
            unsat_set = update_unsat_after_flip(v, unsat_set, var)
            cur_sat = m - len(unsat_set)
        
        # Track best
        if cur_sat > best_sat:
            best_sat = cur_sat
            best_assign = int_to_assign(v)
        if best_sat == m:
            return best_assign
        if time.time() > deadline:
            break

    # Phase 2: Polish from best assignment with tighter local search
    if best_assign is not None and best_sat < m:
        v = assign_to_int(best_assign)
        unsat_set = get_unsat_set(v)
        polish_deadline = min(time.time() + 20.0, deadline)
        
        for polish_restart in range(200):
            if time.time() > polish_deadline:
                break
            
            v_cur = v
            unsat_cur = set(unsat_set)
            
            for pstep in range(3000):
                if time.time() > polish_deadline:
                    break
                
                if len(unsat_cur) == 0:
                    best_assign = int_to_assign(v_cur)
                    best_sat = m
                    return best_assign
                
                unsat_list = list(unsat_cur)
                ci = unsat_list[r.randrange(len(unsat_list))]
                cv = clause_vars[ci]
                
                # Greedy: pick variable that maximally reduces unsat count
                best_var = None
                best_delta = -1
                for var, is_pos in cv:
                    # Estimate delta in unsat count
                    delta = 0
                    for uci, uis_pos in var_occ[var]:
                        if uci == ci:
                            continue
                        bit = 1 << var
                        if uci in unsat_cur:
                            # Would flipping satisfy uci?
                            if is_pos and not uis_pos:
                                # ci wants var=True, uci has -var
                                # After flip, var changes state
                                # If var was False, now True: uci's -var becomes False (unsat for that literal)
                                # If var was True, now False: uci's -var becomes True (sat for that literal)
                                pass
                            # Simplified: count net effect
                            delta += 0.5
                        else:
                            delta -= 0.2
                    
                    noise = r.random() * 0.5
                    total = delta + noise
                    if total > best_delta:
                        best_delta = total
                        best_var = var
                
                if best_var is not None:
                    v_cur ^= (1 << best_var)
                    unsat_cur = update_unsat_after_flip(v_cur, unsat_cur, best_var)
            
            s = m - len(unsat_cur)
            if s > best_sat:
                best_sat = s
                best_assign = int_to_assign(v_cur)
                v = v_cur
                unsat_set = unsat_cur
            if best_sat == m:
                return best_assign

    if best_assign is None:
        return [False] * n
    return best_assign
# EVOLVE-BLOCK-END
