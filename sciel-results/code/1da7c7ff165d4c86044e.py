"""Difference triangle set (4,7): simulated-annealing packing of row difference-sets."""
import time
import random
import numpy as np


def _greedy(n_rows, k, seed=0):
    """Deterministic greedy: each row takes the smallest new mark whose diffs are unused."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _positions(ds, k):
    """Recover the increasing row from a difference-set bitset (ds is a Python int)."""
    pos = [0]
    x = 0
    for _ in range(k):
        x += 1
        while (ds >> x) & 1:
            x += 1
        pos.append(x)
    return pos


def _row_diffs(row):
    return [row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))]


def _scope(rows):
    return max(max(r) for r in rows)


def construct(n_rows, k):
    start = time.time()
    DEADLINE = start + 140.0
    rng = random.Random(20250601)

    # Fast valid baseline (guaranteed return).
    base = _greedy(n_rows, k, 0)
    base_scope = _scope(base)

    C = 1000000
    n = n_rows * k
    max_iter = 600000

    # seed rows with greedy rows (valid, distinct diffs)
    rows = [set(_row_diffs(r)) for r in base]
    bitsets = [0] * n_rows
    for i, s in enumerate(rows):
        b = 0
        for d in s:
            b |= (1 << d)
        bitsets[i] = b

    # difference-frequency arrays (index d holds count of rows containing d)
    freq = [0] * (base_scope + 2)
    for s in rows:
        for d in s:
            if d >= len(freq):
                freq.extend([0] * (d - len(freq) + 1))
            freq[d] += 1
    dup = sum(c - 1 for c in freq[1:] if c >= 2)
    over = 0

    def last_pos(ds):
        pos = [0]
        x = 0
        for _ in range(k):
            x += 1
            while (ds >> x) & 1:
                x += 1
            pos.append(x)
        return pos[-1]

    cur_scope = sum(last_pos(b) for b in bitsets)
    cur_cost = cur_scope + C * (dup + over)
    best_scope = base_scope
    best_rows = [set(s) for s in rows]

    T = 1000.0
    T_min = 1.0
    decay = 0.99997
    inv_sqrt = 1.0 / np.sqrt(np.arange(1, base_scope + 2, dtype=float))
    inv_sqrt /= inv_sqrt.sum()
    idx = np.arange(1, base_scope + 2)

    it = 0
    while it < max_iter:
        if time.time() > DEADLINE:
            break
        it += 1
        r = rng.randrange(n_rows)
        ds = bitsets[r]
        lp = last_pos(ds)
        ds_set = rows[r]

        new_ds = ds
        new_lp = lp
        dup_delta = 0
        changed = False

        op = rng.random()
        if op < 0.45:  # add a low difference
            d = int(idx[rng.choice(idx, p=inv_sqrt)])
            if not (ds >> d) & 1:
                new_ds = ds | (1 << d)
                ds_set.add(d)
                if d >= len(freq):
                    freq.extend([0] * (d - len(freq) + 1))
                freq[d] += 1
                if freq[d] == 2:
                    dup_delta += 1
                if d > lp:
                    new_lp = d
                changed = True
        elif op < 0.75:  # remove a difference
            if ds_set:
                d = rng.choice(list(ds_set))
                new_ds = ds & ~(1 << d)
                ds_set.remove(d)
                freq[d] -= 1
                if freq[d] == 1:
                    dup_delta -= 1
                if d == lp:
                    x = d - 1
                    while (new_ds >> x) & 1:
                        x -= 1
                    new_lp = x
                changed = True
        else:  # swap
            if ds_set:
                old = rng.choice(list(ds_set))
                d = int(idx[rng.choice(idx, p=inv_sqrt)])
                if not (ds >> d) & 1:
                    new_ds = (ds & ~(1 << old)) | (1 << d)
                    ds_set.remove(old)
                    ds_set.add(d)
                    freq[old] -= 1
                    if freq[old] == 1:
                        dup_delta -= 1
                    if d >= len(freq):
                        freq.extend([0] * (d - len(freq) + 1))
                    freq[d] += 1
                    if freq[d] == 2:
                        dup_delta += 1
                    if d > lp:
                        new_lp = d
                    elif old == lp:
                        x = d - 1
                        while (new_ds >> x) & 1:
                            x -= 1
                        new_lp = x
                    changed = True

        if changed:
            new_scope = cur_scope - lp + new_lp
            new_cost = new_scope + C * ((dup + dup_delta) + over)
            if new_cost <= cur_cost or rng.random() < np.exp(-(new_cost - cur_cost) / max(T, 1e-6)):
                cur_scope = new_scope
                cur_cost = new_cost
                bitsets[r] = new_ds
                dup += dup_delta
                if cur_scope < best_scope or (cur_scope == best_scope and dup == 0):
                    best_scope = cur_scope
                    best_rows = [set(s) for s in rows]
                    if dup == 0:
                        base_scope = cur_scope
                        base = [_positions(bitsets[i], k) for i in range(n_rows)]
        # (if not changed, state untouched)

        T *= decay
        if T < T_min:
            T = T_min

    # Final: prefer a valid candidate with the smallest scope.
    final_rows = base
    final_scope = _scope(base)
    try:
        trial = []
        used = set()
        ok = True
        for i, s in enumerate(best_rows):
            b = 0
            for d in s:
                b |= (1 << d)
            row = _positions(b, k)
            trial.append(row)
        for row in trial:
            for d in _row_diffs(row):
                if d in used:
                    ok = False
                    break
                used.add(d)
            if not ok:
                break
        if ok and _scope(trial) < final_scope:
            final_rows = trial
    except Exception:
        final_rows = base

    return final_rows
