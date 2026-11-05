import time
import random
import heapq

def construct(n_rows, k):
    t0 = time.time()
    deadline = t0 + 105.0  # safety margin under 120s

    # ---- helpers ----
    def row_diffs(row):
        s = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ri)
        return s

    def scope(rows):
        return max(r[-1] for r in rows)

    def is_valid(rows):
        used = set()
        for r in rows:
            if r[0] != 0:
                return False
            for i in range(len(r)):
                if i > 0 and r[i] <= r[i-1]:
                    return False
            for d in row_diffs(r):
                if d in used:
                    return False
                used.add(d)
        return True

    # ---- randomized greedy with best-fit fragmentation guidance ----
    def greedy(rng, max_scope):
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _pos in range(k):
                last = row[-1]
                chosen = None
                # scan candidates from smallest upward; stop at first that doesn't
                # collide and has best fragmentation score among a small window
                best_frag = None
                window = 6
                cand_list = []
                for x in range(last + 1, max_scope + 1):
                    dmask = 0
                    for a in row:
                        dmask |= 1 << (x - a)
                    if dmask & used:
                        continue
                    # fragmentation: number of "blocks" of used diffs after adding
                    # approximate by counting set bits in (used | dmask) vs (used)
                    # simpler: count gaps. Use popcount of union.
                    union = used | dmask
                    frag = bin(union).count('1')
                    cand_list.append((frag, x))
                    if len(cand_list) >= window:
                        break
                if not cand_list:
                    return None
                # pick the one with smallest fragmentation, tie-break smaller x
                cand_list.sort()
                frag, x = cand_list[0]
                # small randomization: sometimes pick 2nd to diversify
                if len(cand_list) > 1 and rng.random() < 0.2:
                    frag, x = cand_list[1]
                dmask = 0
                for a in row:
                    dmask |= 1 << (x - a)
                used |= dmask
                row.append(x)
            rows.append(row)
        return rows

    # ---- targeted max-reduction polish on the worst row ----
    def polish(rows, max_scope):
        # find row with largest max
        worst = max(range(n_rows), key=lambda i: rows[i][-1])
        row = rows[worst]
        others = [rows[i] for i in range(n_rows) if i != worst]
        used = 0
        for r in others:
            for d in row_diffs(r):
                used |= 1 << d
        # rebuild worst row greedily with tighter max
        new_row = [0]
        for _pos in range(k):
            last = new_row[-1]
            chosen = None
            for x in range(last + 1, max_scope + 1):
                dmask = 0
                for a in new_row:
                    dmask |= 1 << (x - a)
                if dmask & used:
                    continue
                chosen = x
                break
            if chosen is None:
                return None  # can't fit within max_scope
            dmask = 0
            for a in new_row:
                dmask |= 1 << (chosen - a)
            used |= dmask
            new_row.append(chosen)
        new_rows = [rows[i] for i in range(n_rows)]
        new_rows[worst] = new_row
        return new_rows

    # ---- main loop ----
    best_rows = None
    best_scope = float('inf')
    rng = random.Random(12345)

    # initial max_scope estimate
    max_scope = max(120, best_scope if best_scope < float('inf') else 120)

    attempts = 0
    while time.time() < deadline:
        rows = greedy(rng, max_scope)
        if rows is not None:
            sc = scope(rows)
            if sc < best_scope:
                best_scope = sc
                best_rows = rows
                # try to polish to reduce further
                max_scope = sc
                # a few polish attempts
                for _ in range(3):
                    if time.time() >= deadline:
                        break
                    pol = polish(best_rows, max_scope)
                    if pol is not None:
                        psc = scope(pol)
                        if psc < best_scope:
                            best_scope = psc
                            best_rows = pol
                            max_scope = psc
                if best_scope <= 105:
                    break
        attempts += 1
        # occasionally widen scope to escape local minima
        if attempts % 50 == 0:
            max_scope = best_scope + 5

    if best_rows is None:
        # deterministic fallback
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _pos in range(k):
                last = row[-1]
                x = last + 1
                while True:
                    dmask = 0
                    for a in row:
                        dmask |= 1 << (x - a)
                    if not (dmask & used):
                        break
                    x += 1
                used |= dmask
                row.append(x)
            rows.append(row)
        best_rows = rows

    return best_rows
