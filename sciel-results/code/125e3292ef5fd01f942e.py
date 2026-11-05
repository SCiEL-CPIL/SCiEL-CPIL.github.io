import time
import random


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before the 90s kill

    # ---------- helpers ----------

    def get_all_products(seq):
        """All non-empty consecutive block products for a strictly increasing seq."""
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                products.add(prod)
        return products

    def is_valid(seq):
        if not seq:
            return True
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i + 1]:
                return False
        if any(x < 2 or x > 128 for x in seq):
            return False
        n = len(seq)
        return len(get_all_products(seq)) == n * (n + 1) // 2

    def find_insert_pos(seq, m):
        """Index where m should be inserted to keep strictly increasing order."""
        lo, hi = 0, len(seq)
        while lo < hi:
            mid = (lo + hi) // 2
            if seq[mid] < m:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def fast_valid_with_insert(seq, prods, m):
        """
        Return True iff inserting m into seq (sorted position) keeps every
        consecutive block product distinct, using only the incremental tails.
        """
        n = len(seq)
        pos = find_insert_pos(seq, m)
        if pos < n and seq[pos] == m:
            return False  # already present

        # New block products = all blocks that contain m.
        # A block [i..j] contains m (at index pos) iff i <= pos <= j.
        # Compute them by growing left and right from m.
        new_products = set()

        # blocks ending at m: [i..pos] for i in pos..0
        prod = m
        new_products.add(prod)
        for i in range(pos - 1, -1, -1):
            prod *= seq[i]
            new_products.add(prod)

        # blocks starting at m (length>=2): [pos..j] for j in pos+1..n-1
        prod = m
        for j in range(pos + 1, n):
            prod *= seq[j]
            new_products.add(prod)

        # blocks spanning m: [i..j] with i<pos<j  =>  (left suffix ending at i) * (m) * (right prefix to j)
        # left suffix products L[i] = prod(seq[i..pos-1]) for i in pos-1..0
        left = {}
        lp = 1
        for i in range(pos - 1, -1, -1):
            lp *= seq[i]
            left[i] = lp
        # right prefix products R[j] = prod(seq[pos+1..j]) for j in pos+1..n-1
        right = {}
        rp = 1
        for j in range(pos + 1, n):
            rp *= seq[j]
            right[j] = rp
        for i, lv in left.items():
            for j, rv in right.items():
                new_products.add(lv * m * rv)

        # collision check
        return not (new_products & prods)

    # ---------- beam search (fixed-point insertion) ----------

    def run_beam(width, time_limit):
        """
        Beam search over subsets of [2,128]. State = (tuple, product_set).
        At each phase we try to insert every missing element into every beam
        state; we repeat until no new state is produced (fixed point) or the
        time budget is exhausted. Beam is pruned by (length desc, sum asc).
        """
        beam = {(): set()}
        best = []
        best_len = 0

        def prune(states):
            # states: dict tuple->set
            if len(states) <= width:
                return states
            keys = sorted(states.keys(), key=lambda s: (-len(s), sum(s)))
            return {k: states[k] for k in keys[:width]}

        while time.time() < time_limit:
            changed = False
            new_states = dict(beam)  # carry over all current states (skip option)
            for seq, prods in beam.items():
                seq_set = set(seq)
                for m in range(2, 129):
                    if m in seq_set:
                        continue
                    if not fast_valid_with_insert(seq, prods, m):
                        continue
                    # build new seq
                    pos = find_insert_pos(seq, m)
                    new_seq = seq[:pos] + (m,) + seq[pos:]
                    # update product set incrementally
                    new_prods = set(prods)
                    # recompute only the new products (blocks containing m)
                    n = len(seq)
                    prod = m
                    new_prods.add(prod)
                    for i in range(pos - 1, -1, -1):
                        prod *= seq[i]
                        new_prods.add(prod)
                    prod = m
                    for j in range(pos + 1, n):
                        prod *= seq[j]
                        new_prods.add(prod)
                    left = {}
                    lp = 1
                    for i in range(pos - 1, -1, -1):
                        lp *= seq[i]
                        left[i] = lp
                    right = {}
                    rp = 1
                    for j in range(pos + 1, n):
                        rp *= seq[j]
                        right[j] = rp
                    for i, lv in left.items():
                        for j, rv in right.items():
                            new_prods.add(lv * m * rv)

                    if new_seq not in new_states:
                        new_states[new_seq] = new_prods
                        changed = True

            beam = prune(new_states)

            # update best
            for s in beam:
                if len(s) > best_len:
                    best_len = len(s)
                    best = list(s)

            if not changed:
                break

        return best, best_len

    # ---------- local polish (insert any missing, then remove-reinsert) ----------

    def polish(seq, time_limit):
        seq = list(seq)
        prods = get_all_products(seq)
        best = seq
        best_len = len(seq)

        # Phase 1: greedily insert any missing element
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            seq_set = set(seq)
            for m in range(2, 129):
                if m in seq_set:
                    continue
                if not fast_valid_with_insert(seq, prods, m):
                    continue
                pos = find_insert_pos(seq, m)
                new_seq = seq[:pos] + [m] + seq[pos:]
                new_prods = get_all_products(new_seq)
                seq = new_seq
                prods = new_prods
                seq_set.add(m)
                improved = True
                if len(seq) > best_len:
                    best_len = len(seq)
                    best = seq[:]
                break  # restart scan with updated state

        # Phase 2: remove one element, then try to insert two missing (net +1)
        if time.time() < time_limit:
            for i in range(len(seq)):
                if time.time() > time_limit:
                    break
                base = seq[:i] + seq[i + 1:]
                base_prods = get_all_products(base)
                base_set = set(base)
                for m in range(2, 129):
                    if m in base_set:
                        continue
                    if not fast_valid_with_insert(base, base_prods, m):
                        continue
                    pos = find_insert_pos(base, m)
                    cand = base[:pos] + [m] + base[pos:]
                    # need net gain: cand length == len(seq) (same) -> not helpful for length
                    # but it may enable further inserts; accept to explore
                    if len(cand) >= best_len:
                        # try to extend cand
                        c2 = cand[:]
                        p2 = get_all_products(c2)
                        s2 = set(c2)
                        for m2 in range(2, 129):
                            if m2 in s2:
                                continue
                            if not fast_valid_with_insert(c2, p2, m2):
                                continue
                            pos2 = find_insert_pos(c2, m2)
                            c3 = c2[:pos2] + [m2] + c2[pos2:]
                            if len(c3) > best_len:
                                best_len = len(c3)
                                best = c3
                                seq = c3
                                prods = get_all_products(c3)
                                break
        return best

    # ---------- driver ----------

    # 1. Quick greedy baseline (fast, gives a floor)
    greedy = []
    gprods = set()
    for x in range(2, 129):
        tails = [x]
        prod = x
        for a in reversed(greedy):
            prod *= a
            tails.append(prod)
        if not gprods.intersection(tails):
            greedy.append(x)
            gprods.update(tails)
    best = greedy
    best_len = len(greedy)

    # 2. Beam search with increasing widths, time-budgeted
    #    Start narrow, widen as time allows. Fixed-point insertion.
    widths = [30, 80, 150, 300, 600]
    for w in widths:
        remaining = deadline - time.time()
        if remaining < 8:
            break
        # allocate a chunk of time to this width
        chunk = min(remaining * 0.5, 30.0)
        t_limit = time.time() + chunk
        seq, ln = run_beam(w, t_limit)
        if len(seq) > best_len:
            best_len = len(seq)
            best = seq

    # 3. Polish the best found
    if time.time() < deadline - 4:
        polished = polish(best, deadline - 4)
        if len(polished) > best_len:
            best_len = len(polished)
            best = polished

    # 4. Final safety: ensure validity; fallback to greedy
    if not is_valid(best):
        best = greedy

    return best
