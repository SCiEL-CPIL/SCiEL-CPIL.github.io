import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin under 90s wall-clock

    # 1. Diagonal check: 2*i must not be a perfect square
    sq = np.zeros(513, dtype=bool)
    for k in range(1, 23):
        sq[k * k] = True
    vals = np.arange(1, 257, dtype=np.int64)
    candidates = vals[~sq[2 * vals]]
    N = len(candidates)
    cand = candidates.astype(np.int64)

    # 2. Conflict matrix via vectorization
    s = cand[:, None] + cand[None, :]
    conflict = sq[s]
    # diagonal already guaranteed false (2*i not square)
    masks = np.packbits(conflict, axis=1)  # shape (N, ceil(N/8))

    # 3. Greedy init (min-degree)
    def greedy_init():
        remaining = np.ones(N, dtype=bool)
        selected = np.zeros(N, dtype=bool)
        while remaining.any():
            deg = (conflict & remaining[None, :]).sum(axis=1)
            deg[~remaining] = N + 1
            node = int(np.argmin(deg))
            if not remaining[node]:
                break
            selected[node] = True
            remaining &= ~conflict[node]
        return selected

    sel = greedy_init()
    best_sel = sel.copy()
    best_size = int(sel.sum())

    def try_add(sel_arr):
        # add any node with no conflict, min-degree
        nonlocal best_sel, best_size
        conflict_mask = np.zeros(N, dtype=bool)
        if sel_arr.any():
            conflict_mask = conflict[sel_arr].any(axis=0)
        addable = (~sel_arr) & (~conflict_mask)
        if not addable.any():
            return
        deg = (conflict & addable[None, :]).sum(axis=1)
        deg[~addable] = N + 1
        node = int(np.argmin(deg))
        sel_arr[node] = True

    def local_search(sel_arr):
        nonlocal best_sel, best_size
        cur = sel_arr.copy()
        cur_sz = int(cur.sum())
        best_local = cur.copy()
        best_local_sz = cur_sz
        # Phase 1: greedy add
        while True:
            before = cur.sum()
            try_add(cur)
            if cur.sum() == before:
                break
            if cur.sum() > best_local_sz:
                best_local_sz = int(cur.sum()); best_local = cur.copy()
            if cur.sum() > best_size:
                best_size = int(cur.sum()); best_sel = cur.copy()
        # Phase 2: swaps (1-for-1 and 2-for-2)
        for _ in range(1200):
            if time.time() > deadline:
                break
            found = False
            # 1-for-1
            for _ in range(40):
                u = random.randrange(N)
                if cur[u]:
                    continue
                # find v in cur such that removing v lets u in
                candidates_v = np.where(cur)[0]
                ok = []
                for v in candidates_v:
                    rest = cur.copy(); rest[v] = False
                    if not (conflict[u] & rest).any():
                        ok.append(v)
                if ok:
                    v = random.choice(ok)
                    cur[v] = False; cur[u] = True
                    found = True
                    break
            if not found:
                # 2-for-2
                for _ in range(20):
                    u1 = random.randrange(N)
                    if cur[u1]:
                        continue
                    u2 = random.randrange(N)
                    if cur[u2] or u2 == u1 or conflict[u1, u2]:
                        continue
                    candidates_v = np.where(cur)[0]
                    ok2 = []
                    for i in range(len(candidates_v)):
                        for j in range(i + 1, len(candidates_v)):
                            v1 = candidates_v[i]; v2 = candidates_v[j]
                            rest = cur.copy(); rest[v1] = False; rest[v2] = False
                            if not (conflict[u1] & rest).any() and not (conflict[u2] & rest).any():
                                ok2.append((v1, v2)); break
                        if ok2:
                            break
                    if ok2:
                        v1, v2 = random.choice(ok2)
                        cur[v1] = False; cur[v2] = False; cur[u1] = True; cur[u2] = True
                        found = True
            if not found:
                break
            # greedy add after swap
            while True:
                if time.time() > deadline:
                    break
                before = cur.sum()
                try_add(cur)
                if cur.sum() == before:
                    break
                if cur.sum() > best_local_sz:
                    best_local_sz = int(cur.sum()); best_local = cur.copy()
                if cur.sum() > best_size:
                    best_size = int(cur.sum()); best_sel = cur.copy()
        # Phase 3: final greedy add
        while True:
            if time.time() > deadline:
                break
            before = cur.sum()
            try_add(cur)
            if cur.sum() == before:
                break
            if cur.sum() > best_local_sz:
                best_local_sz = int(cur.sum()); best_local = cur.copy()
            if cur.sum() > best_size:
                best_size = int(cur.sum()); best_sel = cur.copy()
        return best_local, best_local_sz

    best_sel, best_size = local_search(best_sel)

    # 4. Simulated Annealing (escape local optima; accept shrinking moves)
    if time.time() < deadline - 15.0:
        cur = best_sel.copy()
        cur_sz = int(cur.sum())
        best_sa = best_sel.copy()
        best_sa_sz = best_size
        T = 1.0
        cooling = 0.985
        steps = 0
        max_steps = 200000
        while time.time() < deadline - 12.0 and steps < max_steps and T > 0.01:
            steps += 1
            # choose a move
            if random.random() < 0.5:
                # try add
                conflict_mask = conflict[cur].any(axis=0) if cur.any() else np.zeros(N, dtype=bool)
                addable = np.where((~cur) & (~conflict_mask))[0]
                if len(addable) == 0:
                    continue
                node = int(random.choice(addable))
                cur[node] = True
                cur_sz += 1
                # accept (improvement)
                if cur_sz > best_sa_sz:
                    best_sa_sz = cur_sz; best_sa = cur.copy()
                    if cur_sz > best_size:
                        best_size = cur_sz; best_sel = cur.copy()
            else:
                # try remove (shrinking, SA)
                in_cur = np.where(cur)[0]
                if len(in_cur) == 0:
                    continue
                node = int(random.choice(in_cur))
                # accept with probability
                if random.random() < np.exp(-1.0 / max(T, 1e-6)):
                    cur[node] = False
                    cur_sz -= 1
                    if cur_sz > best_sa_sz:
                        best_sa_sz = cur_sz; best_sa = cur.copy()
            T *= cooling
        if best_sa_sz > best_size:
            best_size = best_sa_sz; best_sel = best_sa.copy()
        # polish
        while True:
            if time.time() > deadline:
                break
            before = best_sel.sum()
            try_add(best_sel)
            if best_sel.sum() == before:
                break
            best_size = int(best_sel.sum())

    # 5. Multiple restarts
    if time.time() < deadline - 6.0:
        for restart in range(8):
            if time.time() > deadline - 3.0:
                break
            pert = best_sel.copy()
            n_remove = max(1, best_size // 6)
            for _ in range(n_remove):
                if not pert.any():
                    break
                pos = random.randrange(N)
                if pert[pos]:
                    pert[pos] = False
            pert, psz = local_search(pert)
            if psz > best_size:
                best_size = psz; best_sel = pert.copy()

    # 6. Final polish
    if time.time() < deadline - 1.0:
        for i in range(N):
            if not best_sel[i] and not (conflict[i] & best_sel).any():
                best_sel[i] = True
                best_size += 1

    result = [int(candidates[i]) for i in range(N) if best_sel[i]]
    return result

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
