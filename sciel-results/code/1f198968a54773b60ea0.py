# EVOLVE-BLOCK-START
import random
import time


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
    deadline = time.time() + 290.0
    pos, neg = _build_masks(clauses, n)
    pos = tuple(pos)
    neg = tuple(neg)
    full_mask = (1 << m) - 1

    clause_vars = []
    for c in clauses:
        clause_vars.append([(abs(l) - 1, l > 0) for l in c])

    var_pos_clauses = [0] * n
    var_neg_clauses = [0] * n
    for i, c in enumerate(clauses):
        for l in c:
            if l > 0:
                var_pos_clauses[l - 1] |= (1 << i)
            else:
                var_neg_clauses[-l - 1] |= (1 << i)

    best_assign = None
    best_sat = -1

    def assign_to_int(assign):
        v = 0
        for i, b in enumerate(assign):
            if b:
                v |= (1 << i)
        return v

    def int_to_assign(v):
        return [bool(v & (1 << i)) for i in range(n)]

    def compute_unsat_mask(v):
        unsat = 0
        for i in range(m):
            if not ((v & pos[i]) or ((~v) & neg[i])):
                unsat |= (1 << i)
        return unsat

    def get_unsat_indices(unsat_mask):
        indices = []
        while unsat_mask:
            lsb = unsat_mask & (-unsat_mask)
            indices.append(lsb.bit_length() - 1)
            unsat_mask ^= lsb
        return indices

    def update_unsat_after_flip(v, old_v, var, unsat_mask):
        affected = var_pos_clauses[var] | var_neg_clauses[var]
        new_unsat = unsat_mask
        for ci in range(m):
            if affected & (1 << ci):
                if (v & pos[ci]) or ((~v) & neg[ci]):
                    new_unsat &= ~(1 << ci)
                else:
                    new_unsat |= (1 << ci)
        return new_unsat

    def walksat_restart(restart_idx, base_p=0.55, p_decay=0.0002, p_min=0.35,
                        steps_budget=5000, sample_size=10, help_weight=3.0):
        nonlocal best_assign, best_sat
        r = random.Random(restart_idx * 7919 + 13)
        assign = [r.random() < 0.5 for _ in range(n)]
        v = assign_to_int(assign)
        unsat_mask = compute_unsat_mask(v)
        cur_sat = m - unsat_mask.bit_count()

        if cur_sat > best_sat:
            best_sat = cur_sat
            best_assign = int_to_assign(v)
        if cur_sat == m:
            return True

        p = base_p
        for step in range(steps_budget):
            if time.time() > deadline:
                return False

            if unsat_mask == 0:
                best_assign = int_to_assign(v)
                best_sat = m
                return True

            unsat_indices = get_unsat_indices(unsat_mask)
            if len(unsat_indices) > sample_size:
                sample = r.sample(unsat_indices, sample_size)
            else:
                sample = unsat_indices

            best_ci = None
            best_help = -1
            for ci in sample:
                h = 0
                for var, is_pos in clause_vars[ci]:
                    cur = bool(v & (1 << var))
                    if (is_pos and not cur) or ((not is_pos) and cur):
                        h += 1
                if h > best_help:
                    best_help = h
                    best_ci = ci
            if best_ci is None:
                best_ci = sample[0]

            cv = clause_vars[best_ci]
            weights = []
            total_w = 0.0
            for var, is_pos in cv:
                cur = bool(v & (1 << var))
                if (is_pos and not cur) or ((not is_pos) and cur):
                    w = help_weight
                else:
                    w = 1.0
                weights.append(w)
                total_w += w

            r_val = r.random() * total_w
            cum = 0.0
            lit_idx = len(cv) - 1
            for k, w in enumerate(weights):
                cum += w
                if r_val <= cum:
                    lit_idx = k
                    break

            var, is_pos = cv[lit_idx]
            bit = 1 << var
            old_v = v
            v ^= bit
            unsat_mask = update_unsat_after_flip(v, old_v, var, unsat_mask)
            cur_sat = m - unsat_mask.bit_count()

            if cur_sat > best_sat:
                best_sat = cur_sat
                best_assign = int_to_assign(v)

            p = max(p_min, p - p_decay)

        if best_sat == m:
            return True
        return False

    def polish_phase(start_assign, max_steps=20000):
        nonlocal best_assign, best_sat
        v = assign_to_int(start_assign)
        unsat_mask = compute_unsat_mask(v)
        cur_sat = m - unsat_mask.bit_count()

        if cur_sat > best_sat:
            best_sat = cur_sat
            best_assign = int_to_assign(v)
        if cur_sat == m:
            return True

        r = random.Random(99999)
        for step in range(max_steps):
            if time.time() > deadline:
                return False
            if unsat_mask == 0:
                best_assign = int_to_assign(v)
                best_sat = m
                return True

            unsat_indices = get_unsat_indices(unsat_mask)
            if not unsat_indices:
                return True

            # Try all single flips that reduce unsat count
            improved = False
            for ci in unsat_indices[:20]:
                for var, is_pos in clause_vars[ci]:
                    bit = 1 << var
                    new_v = v ^ bit
                    new_unsat = update_unsat_after_flip(new_v, v, var, unsat_mask)
                    new_sat = m - new_unsat.bit_count()
                    if new_sat > cur_sat:
                        v = new_v
                        unsat_mask = new_unsat
                        cur_sat = new_sat
                        improved = True
                        break
                if improved:
                    break

            if improved:
                if cur_sat > best_sat:
                    best_sat = cur_sat
                    best_assign = int_to_assign(v)
                continue

            # No improving flip found; try random walk with low noise
            ci = r.choice(unsat_indices)
            cv = clause_vars[ci]
            var, is_pos = cv[r.randrange(len(cv))]
            bit = 1 << var
            v ^= bit
            unsat_mask = update_unsat_after_flip(v, v ^ bit, var, unsat_mask)
            cur_sat = m - unsat_mask.bit_count()

            if cur_sat > best_sat:
                best_sat = cur_sat
                best_assign = int_to_assign(v)

        return best_sat == m

    # Phase 0: Quick random initializations
    for attempt_seed in range(16):
        r = random.Random(attempt_seed * 31 + 7)
        assign = [r.random() < 0.5 for _ in range(n)]
        v = assign_to_int(assign)
        unsat_mask = compute_unsat_mask(v)
        s = m - unsat_mask.bit_count()
        if s > best_sat:
            best_sat = s
            best_assign = assign
        if s == m:
            return assign
        if time.time() > deadline:
            return best_assign if best_assign else [False] * n

    # Phase 1: Multi-restart WalkSAT with adaptive parameters
    num_restarts = 0
    for restart in range(8000):
        if time.time() > deadline:
            break
        num_restarts += 1
        base_p = 0.55 + 0.15 * (num_restarts % 5) / 5.0
        p_decay = 0.00015 + 0.00005 * (num_restarts % 3)
        sample_size = 8 + (num_restarts % 4)
        steps = 4000 + 500 * (num_restarts % 3)

        found = walksat_restart(restart, base_p, p_decay, 0.35, steps, sample_size, 3.0)
        if found:
            return best_assign
        if time.time() > deadline:
            break

    # Phase 2: Polish the best assignment found
    if best_assign is not None and best_sat < m:
        remaining = deadline - time.time()
        if remaining > 5.0:
            polish_steps = int(remaining * 3000)  # estimate ~3k steps/sec
            polish_steps = min(polish_steps, 50000)
            polish_phase(best_assign, polish_steps)

    if best_assign is None:
        return [False] * n
    return best_assign
# EVOLVE-BLOCK-END
