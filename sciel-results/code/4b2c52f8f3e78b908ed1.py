"""Program for kissing number in dimension 11.

Constructs nonzero integer vectors in R^11 whose minimum pairwise distance is
at least the maximum vector norm.  Returns an integer array of shape (k, 11).
The score is k.  The kissing number in dimension 11 is 60.
"""
import numpy as np
import itertools
import time

D = 11


def construct():
    start = time.time()
    deadline = start + 112

    # Candidate pool: weight-4 vectors (entries +/-1, norm^2=4) plus +/-2 e_i.
    v4 = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4.append(v)
    v1 = []
    for pos in range(D):
        for sign in (1, -1):
            v = [0] * D
            v[pos] = 2 * sign
            v1.append(v)

    cand = np.array(v4 + v1, dtype=np.int64)
    N = len(cand)

    dots = cand @ cand.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    neighbors = [np.nonzero(conflicts[i])[0] for i in range(N)]
    deg = np.array([len(n) for n in neighbors], dtype=np.int32)

    def is_valid(in_set):
        idx = np.nonzero(in_set)[0]
        if len(idx) < 2:
            return True
        dd = cand[idx] @ cand[idx].T
        np.fill_diagonal(dd, 0)
        return bool(np.all(dd <= 2))

    def greedy_init(seed):
        rng = np.random.default_rng(seed)
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        d = deg.copy()
        while alive.any() and time.time() < deadline - 8:
            ai = np.nonzero(alive)[0]
            if ai.size == 0:
                break
            dv = d[ai]
            m = dv.min()
            cands_ai = ai[dv == m]
            v = int(cands_ai[rng.integers(len(cands_ai))])
            in_set[v] = True
            to_rm = np.unique(np.concatenate(([v], neighbors[v])))
            alive[to_rm] = False
            # decrement degrees of alive neighbors of removed vertices
            for u in to_rm:
                for w in neighbors[u]:
                    if alive[w]:
                        d[w] -= 1
        return in_set

    def energy_local_search(in_set0, max_time, T0):
        rng = np.random.default_rng(int(time.time() * 1e6) % (2**31) + 1)
        in_set = in_set0.copy()
        cnt = np.zeros(N, dtype=np.int32)
        ii = np.nonzero(in_set)[0]
        for i in ii:
            cnt[neighbors[i]] += 1
        cur_size = int(in_set.sum())

        def recompute_energy():
            idx = np.nonzero(in_set)[0]
            if len(idx) < 2:
                return 0.0
            dd = cand[idx] @ cand[idx].T
            np.fill_diagonal(dd, 0)
            viol = np.maximum(dd - 2, 0)
            return float(np.sum(viol * viol)) / 2.0

        best_in_set = in_set.copy()
        best_size = cur_size
        T = T0
        cooling = 0.9995
        Tmin = 0.03
        sa_deadline = time.time() + max_time
        stuck = 0

        while time.time() < sa_deadline:
            ii = np.nonzero(in_set)[0]
            ni = np.nonzero(~in_set)[0]
            move = rng.random()
            accepted = False

            if move < 0.60 and ni.size:
                cc = cnt[ni]
                m = cc.min()
                ccands = ni[cc == m]
                dd = deg[ccands]
                md = dd.min()
                final = ccands[dd == md]
                v = int(final[rng.integers(len(final))])
                k = int(cc[v])
                if k == 0:
                    acc = True
                elif k == 1:
                    acc = rng.random() < 0.5
                else:
                    acc = rng.random() < np.exp(-k / T)
                if acc:
                    to_rm = np.nonzero(conflicts[v] & in_set)[0]
                    for u in to_rm:
                        in_set[u] = False
                        cnt[neighbors[u]] -= 1
                        cur_size -= 1
                    in_set[v] = True
                    cnt[neighbors[v]] += 1
                    cur_size += 1
                    accepted = True
                    if cur_size > best_size and is_valid(in_set):
                        best_size = cur_size
                        best_in_set = in_set.copy()
                        stuck = 0
                else:
                    stuck += 1

            elif move < 0.75 and ii.size:
                v = int(ii[rng.integers(len(ii))])
                acc = rng.random() < np.exp(-1.0 / T)
                if acc:
                    in_set[v] = False
                    cnt[neighbors[v]] -= 1
                    cur_size -= 1
                    accepted = True
                else:
                    stuck += 1

            elif move < 0.90 and ii.size and ni.size:
                vr = int(ii[rng.integers(len(ii))])
                nsamp = min(20, ni.size)
                samp = ni[rng.choice(ni.size, size=nsamp, replace=False)]
                tc = cnt.copy()
                tc[neighbors[vr]] -= 1
                tc[vr] = 0
                cc = tc[samp]
                m = cc.min()
                gc = samp[cc == m]
                va = int(gc[rng.integers(len(gc))])
                k = int(tc[va])
                if k == 0 or rng.random() < np.exp(-k / T):
                    to_rm = np.nonzero(conflicts[va] & in_set)[0]
                    for u in to_rm:
                        in_set[u] = False
                        cnt[neighbors[u]] -= 1
                        cur_size -= 1
                    in_set[va] = True
                    cnt[neighbors[va]] += 1
                    cur_size += 1
                    accepted = True
                    if cur_size > best_size and is_valid(in_set):
                        best_size = cur_size
                        best_in_set = in_set.copy()
                        stuck = 0
                else:
                    stuck += 1
            else:
                if ii.size >= 2:
                    flips = ii[rng.choice(ii.size, size=min(3, ii.size), replace=False)]
                    for v in flips:
                        in_set[v] = False
                        cnt[neighbors[v]] -= 1
                        cur_size -= 1
                    if ni.size >= 2:
                        adds = ni[rng.choice(ni.size, size=min(3, ni.size), replace=False)]
                        for v in adds:
                            in_set[v] = True
                            cnt[neighbors[v]] += 1
                            cur_size += 1
                    accepted = True
                else:
                    stuck += 1

            if not accepted:
                stuck += 1
            if stuck > 2000:
                T = max(T, 1.5)
                stuck = 0
            T *= cooling
            if T < Tmin:
                T = Tmin

        # Repair to validity.
        rep = best_in_set.copy()
        guard = 0
        while not is_valid(rep) and time.time() < deadline - 4 and guard < 500:
            guard += 1
            idx = np.nonzero(rep)[0]
            if idx.size < 2:
                break
            dd = cand[idx] @ cand[idx].T
            np.fill_diagonal(dd, 0)
            viol = np.maximum(dd - 2, 0)
            tot = viol.sum(axis=1)
            worst = int(np.argmax(tot))
            rep[idx[worst]] = False
        return rep, int(rep.sum())

    def polish(in_set0):
        in_set = in_set0.copy()
        cnt = np.zeros(N, dtype=np.int32)
        ii = np.nonzero(in_set)[0]
        for i in ii:
            cnt[neighbors[i]] += 1
        changed = True
        while changed and time.time() < deadline - 3:
            changed = False
            ni = np.nonzero(~in_set)[0]
            if ni.size == 0:
                break
            addable = ni[cnt[ni] == 0]
            if addable.size:
                v = int(addable[0])
                in_set[v] = True
                cnt[neighbors[v]] += 1
                changed = True
        return in_set, int(in_set.sum())

    best_in_set = np.zeros(N, dtype=bool)
    best_size = 0

    n_restarts = 0
    max_restarts = 200
    luby = [1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, 1, 1, 2, 1, 1, 2, 4, 8, 16]
    li = 0

    while time.time() < deadline - 6 and n_restarts < max_restarts:
        n_restarts += 1
        luby_len = luby[li % len(luby)]
        li += 1

        seed = n_restarts * 7919 + 13
        in_set = greedy_init(seed)
        cur = int(in_set.sum())
        if cur > best_size and is_valid(in_set):
            best_size = cur
            best_in_set = in_set.copy()

        remaining = deadline - 6 - time.time()
        if remaining < 2:
            break
        base = min(remaining * 0.5, 12.0)
        sa_time = base * (luby_len / 4.0)
        sa_time = max(sa_time, 1.5)
        sa_time = min(sa_time, remaining * 0.7)

        sa_set, sa_size = energy_local_search(in_set, sa_time, T0=2.0)
        if sa_size > best_size and is_valid(sa_set):
            best_size = sa_size
            best_in_set = sa_set.copy()

        remaining = deadline - 4 - time.time()
        if remaining > 2:
            pol_set, pol_size = polish(sa_set if sa_size >= cur else in_set)
            if pol_size > best_size and is_valid(pol_set):
                best_size = pol_size
                best_in_set = pol_set.copy()

        remaining = deadline - 3 - time.time()
        if remaining > 5 and n_restarts % 4 == 0 and best_size > 0:
            pt = min(remaining * 0.6, 10.0)
            if pt > 2:
                atk_set, atk_size = energy_local_search(best_in_set, pt, T0=1.0)
                if atk_size > best_size and is_valid(atk_set):
                    best_size = atk_size
                    best_in_set = atk_set.copy()

        if best_size >= 60:
            break

    if time.time() < deadline - 2:
        fp_set, fp_size = polish(best_in_set)
        if fp_size > best_size and is_valid(fp_set):
            best_size = fp_size
            best_in_set = fp_set

    if not is_valid(best_in_set):
        in_set = greedy_init(0)
        if is_valid(in_set):
            best_in_set = in_set
            best_size = int(in_set.sum())

    idx = np.nonzero(best_in_set)[0]
    return cand[idx]
