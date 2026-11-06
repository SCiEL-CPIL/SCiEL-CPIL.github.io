import time
import random
import bisect

def construct():
    """
    Find a Sidon set containing 0 with all differences 1..30, minimizing the largest element.
    Combines a maximum-coverage greedy (initial point + fallback) with a simulated-annealing
    search (structural change: different optimization family) and a final local-refine pass.
    """
    start_time = time.time()
    DEADLINE = time.time() + 145.0  # 5s margin before 150s kill

    TARGET = set(range(1, 31))

    def get_diffs(S):
        S = sorted(S)
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                diffs.add(S[j] - S[i])
        return diffs

    def is_sidon(S):
        S = sorted(S)
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d = S[j] - S[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def covers_all(S):
        return TARGET.issubset(get_diffs(S))

    def missing_count(S):
        return 30 - len(get_diffs(S) & TARGET)

    def score_set(S):
        S = sorted(S)
        missing = missing_count(S)
        return (missing, max(S))

    # --- Fast conflict machinery (bitset) for the annealer ---
    def element_diff_bits(S, x):
        """Bitset of differences between x and every element of S."""
        bits = 0
        for v in S:
            bits |= 1 << (abs(v - x))
        return bits

    def diff_bitset(S):
        bits = 0
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                bits |= 1 << (S[j] - S[i])
        return bits

    def try_add_fast(S, x, existing_bits):
        """Return (new_set, new_existing_bits) if adding x stays Sidon, else (None, None)."""
        if x in S:
            return None, None
        b = element_diff_bits(S, x)
        if b & existing_bits:  # conflicts with existing differences
            return None, None
        return sorted(S + [x]), (existing_bits | b)

    # --- Greedy maximum-coverage construction (initial point + fallback) ---
    def greedy_construct():
        S = [0]
        covered = set()
        existing_bits = 0
        for _ in range(31):
            if time.time() > DEADLINE - 5:
                break
            if covered == TARGET:
                break
            missing = TARGET - covered
            best_x = None
            best_cov = 0
            best_trial = None
            best_bits = None
            # candidates: v +/- d for v in S, d in missing
            cands = set()
            for v in S:
                for d in missing:
                    cands.add(v + d)
                    if v - d > 0:
                        cands.add(v - d)
            cands = sorted(c for c in cands if c > 0 and c not in S and c <= 10**6)
            if len(cands) > 800:
                cands = cands[:800]
            for x in cands:
                trial, nb = try_add_fast(S, x, existing_bits)
                if trial is None:
                    continue
                new_cov = len(element_diff_bits(S, x) & (1 << 31 - 1) and 0)  # placeholder
                # compute coverage of missing diffs created by x
                cov = 0
                for v in S:
                    if abs(v - x) in missing:
                        cov += 1
                if cov > best_cov or (cov == best_cov and best_x is not None and x < best_x):
                    best_cov = cov
                    best_x = x
                    best_trial = trial
                    best_bits = nb
                    if cov == len(missing):
                        break
            if best_trial is None:
                # fallback: scan upward
                found = False
                y = max(S) + 1
                while y < 10**6 and time.time() < DEADLINE - 5:
                    trial, nb = try_add_fast(S, y, existing_bits)
                    if trial is not None:
                        best_trial, best_bits = trial, nb
                        found = True
                        break
                    y += 1
                if not found:
                    break
            S = best_trial
            existing_bits = best_bits
            covered = get_diffs(S) & TARGET
        return S

    # --- Simulated annealing (structural change) ---
    def simulated_anneal(S0, deadline):
        S = list(S0)
        S.sort()
        existing_bits = diff_bitset(S)
        missing = missing_count(S)
        best_S = S[:]
        best_missing = missing
        best_max = max(S)
        best_score = (missing, best_max)

        T0 = 200.0
        Tend = 0.5
        steps = 0
        total_steps = 0

        def eval_state(state, bits):
            m = missing_count(state)
            mx = max(state)
            return m, mx

        m, mx = eval_state(S, existing_bits)
        if (m, mx) < best_score:
            best_S, best_missing, best_max, best_score = S[:], m, mx, (m, mx)

        while time.time() < deadline:
            steps += 1
            total_steps += 1
            # cooling schedule
            frac = min(1.0, total_steps / 200000.0)
            T = T0 * (Tend / T0) ** frac

            n = len(S)
            move = random.random()
            cur_missing, cur_max = m, mx

            if move < 0.45 and n > 1:
                # swap: replace one non-zero element with a random smaller/larger value
                i = random.randrange(n)
                if S[i] == 0:
                    i = random.randrange(1, n)
                old = S[i]
                # candidate range: bias toward smaller to reduce diameter
                hi = max(1, old + random.randint(-50, 50))
                lo = 1
                new_val = random.randint(lo, max(lo + 1, hi))
                if new_val == old or new_val in S:
                    continue
                # remove old, try add new
                without = S[:i] + S[i+1:]
                # recompute bits without old
                bits_wo = diff_bitset(without)
                trial, nb = try_add_fast(without, new_val, bits_wo)
                if trial is None:
                    continue
                nm, nmx = eval_state(trial, nb)
                # accept if better or with probability exp(-delta/T)
                delta = (nm - cur_missing) + (nmx - cur_max) * 0.01
                if delta < 0 or random.random() < (2.718281828 ** (-delta / max(T, 0.001))):
                    S, existing_bits = trial, nb
                    m, mx = nm, nmx
                    if (m, mx) < best_score:
                        best_S, best_missing, best_max, best_score = S[:], m, mx, (m, mx)
            elif move < 0.70 and n > 1:
                # remove an element (only if we can still cover or are improving)
                i = random.randrange(1, n)  # never remove 0
                trial = S[:i] + S[i+1:]
                nm, nmx = eval_state(trial, diff_bitset(trial))
                delta = (nm - cur_missing) + (nmx - cur_max) * 0.01
                if delta < 0 or random.random() < (2.718281828 ** (-delta / max(T, 0.001))):
                    S = trial
                    existing_bits = diff_bitset(S)
                    m, mx = nm, nmx
                    if (m, mx) < best_score:
                        best_S, best_missing, best_max, best_score = S[:], m, mx, (m, mx)
            else:
                # insert a new element
                hi = max(S) + random.randint(1, 40)
                new_val = random.randint(1, hi)
                if new_val in S:
                    continue
                trial, nb = try_add_fast(S, new_val, existing_bits)
                if trial is None:
                    continue
                if len(trial) > 32:
                    continue
                nm, nmx = eval_state(trial, nb)
                delta = (nm - cur_missing) + (nmx - cur_max) * 0.01
                if delta < 0 or random.random() < (2.718281828 ** (-delta / max(T, 0.001))):
                    S, existing_bits = trial, nb
                    m, mx = nm, nmx
                    if (m, mx) < best_score:
                        best_S, best_missing, best_max, best_score = S[:], m, mx, (m, mx)

            # keep size bounded
            if len(S) > 32:
                # remove the largest non-zero element
                idx = max(range(1, len(S)), key=lambda k: S[k])
                S = S[:idx] + S[idx+1:]
                existing_bits = diff_bitset(S)
                m, mx = eval_state(S, existing_bits)

        return best_S

    # --- Local refine: reduce diameter via single-element replacement ---
    def local_refine(S, time_limit):
        S = list(S)
        S.sort()
        best = S[:]
        best_score = score_set(best)
        t0 = time.time()
        while time.time() - t0 < time_limit:
            improved = False
            for i in range(len(S)):
                if S[i] == 0:
                    continue
                old_val = S[i]
                if time.time() - t0 > time_limit:
                    break
                for new_val in range(1, old_val):
                    if new_val in S:
                        continue
                    trial = S[:i] + S[i+1:]
                    t2, _ = try_add_fast(trial, new_val, diff_bitset(trial))
                    if t2 is None:
                        continue
                    t2.sort()
                    ns = score_set(t2)
                    if ns < best_score:
                        S = t2
                        best = S[:]
                        best_score = ns
                        improved = True
                        break
            if not improved:
                break
        return best

    # --- Main ---
    best_S = None
    best_score = (30, 10**9)

    S0 = greedy_construct()
    if S0:
        s0 = score_set(S0)
        if s0 < best_score:
            best_score = s0
            best_S = S0

    # Annealing from the greedy point (and a few random restarts)
    random.seed(12345)
    for restart in range(8):
        if time.time() > DEADLINE - 25:
            break
        if restart == 0:
            init = S0
        else:
            random.seed(restart * 999 + 7)
            init = greedy_construct()
            if not init:
                init = S0
        refined = simulated_anneal(init, DEADLINE - 25)
        if refined:
            rs = score_set(refined)
            if rs < best_score:
                best_score = rs
                best_S = refined

    # Final local refinement
    if best_S and time.time() < DEADLINE - 3:
        final = local_refine(best_S, max(1.0, DEADLINE - 3 - time.time()))
        if final:
            fs = score_set(final)
            if fs < best_score:
                best_score = fs
                best_S = final

    # Ensure validity
    if best_S is None or not covers_all(best_S) or not is_sidon(best_S):
        best_S = S0 if (S0 and covers_all(S0) and is_sidon(S0)) else [0]
        if not covers_all(best_S) or not is_sidon(best_S):
            # last-resort fallback
            best_S = [0]
            for d in range(1, 31):
                if d in get_diffs(best_S):
                    continue
                placed = False
                for x in range(1, 20000):
                    if x in best_S:
                        continue
                    trial, _ = try_add_fast(best_S, x, diff_bitset(best_S))
                    if trial is not None and d in get_diffs(trial):
                        best_S = trial
                        placed = True
                        break
                if not placed:
                    y = 2 * max(best_S) + d + 1
                    while True:
                        t1, _ = try_add_fast(best_S, y, diff_bitset(best_S))
                        if t1 is not None:
                            t2, _ = try_add_fast(t1, y + d, diff_bitset(t1))
                            if t2 is not None:
                                best_S = t2
                                break
                        y += 1

    return best_S
