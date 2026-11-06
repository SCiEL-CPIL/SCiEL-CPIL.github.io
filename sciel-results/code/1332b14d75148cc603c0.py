"""Combined: greedy+local search + branch-and-bound for Maximum Independent Set."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    t0 = time.time()
    DEADLINE = t0 + 78.0

    N = 256

    # Precompute conflict masks: cm[i] = bitmask of j where (i+1)*(j+1) % ((i+1)+(j+1)) == 0
    cm = [0] * N
    for i in range(N):
        a = i + 1
        m = 0
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                m |= (1 << j)
        cm[i] = m

    cc = [0] * N
    for i in range(N):
        c = cm[i]
        cnt = 0
        while c:
            c &= c - 1
            cnt += 1
        cc[i] = cnt

    def popcount(x):
        return x.bit_count()

    def is_valid(mask):
        for i in range(N):
            if (mask >> i) & 1:
                if cm[i] & mask:
                    return False
        return True

    def greedy_add(order):
        m = 0
        for e in order:
            if (m >> e) & 1:
                continue
            if cm[e] & m:
                continue
            m |= (1 << e)
        return m

    def local_search(m, iters=5000):
        best = m
        bestsz = popcount(m)
        for _ in range(iters):
            if time.time() > DEADLINE:
                break
            e = random.randrange(N)
            if (m >> e) & 1:
                continue
            if not (cm[e] & m):
                m |= (1 << e)
                s = popcount(m)
                if s > bestsz:
                    best, bestsz = m, s
            else:
                conf = cm[e] & m
                if conf:
                    c = (conf & -conf).bit_length() - 1
                    newm = (m & ~(1 << c)) | (1 << e)
                    if is_valid(newm):
                        s = popcount(newm)
                        if s > bestsz:
                            best, bestsz = newm, s
                            m = newm
        return best

    # Branch and bound tracking best mask
    best_size = 0
    best_mask = 0

    def bb_solve(cand_mask, cur_mask, cur_size):
        """cand_mask: bitmask of available candidates. cur_mask: current selected."""
        nonlocal best_size, best_mask
        if time.time() > DEADLINE:
            return
        if cand_mask == 0:
            if cur_size > best_size:
                best_size = cur_size
                best_mask = cur_mask
            return

        # Get list of candidates
        cand_list = []
        c = cand_mask
        while c:
            lsb = c & (-c)
            idx = lsb.bit_length() - 1
            cand_list.append(idx)
            c ^= lsb

        if not cand_list:
            return

        # Color bound
        color_of = {}
        max_color = 0
        for v in cand_list:
            used = set()
            for u in cand_list:
                if u == v:
                    continue
                if (cm[v] >> u) & 1 and u in color_of:
                    used.add(color_of[u])
            c_col = 0
            while c_col in used:
                c_col += 1
            color_of[v] = c_col
            if c_col > max_color:
                max_color = c_col

        if cur_size + max_color + 1 <= best_size:
            return

        # Sort by color descending
        sorted_cands = sorted(cand_list, key=lambda v: color_of[v], reverse=True)

        for v in sorted_cands:
            if time.time() > DEADLINE:
                return
            # Include v
            new_cand = cand_mask & ~(1 << v) & ~cm[v]
            bb_solve(new_cand, cur_mask | (1 << v), cur_size + 1)

            # Exclude v
            cand_mask &= ~(1 << v)
            if cur_size + popcount(cand_mask) <= best_size:
                break

    # Main
    odds = [i for i in range(N) if (i + 1) % 2 == 1]
    evens = [i for i in range(N) if (i + 1) % 2 == 0]

    # Baseline
    base = 0
    for i in odds:
        base |= (1 << i)
    if is_valid(base):
        best_size = popcount(base)
        best_mask = base

    # Greedy evens into odds
    m = base
    for e in evens:
        if not (cm[e] & m):
            m |= (1 << e)
    if is_valid(m):
        s = popcount(m)
        if s > best_size:
            best_size = s
            best_mask = m

    # Random greedy + local search
    for _ in range(20):
        if time.time() > DEADLINE:
            break
        ro = list(range(N))
        random.shuffle(ro)
        m = greedy_add(ro)
        m = local_search(m, 1500)
        if is_valid(m):
            s = popcount(m)
            if s > best_size:
                best_size = s
                best_mask = m

    # B&B
    if time.time() < DEADLINE - 5:
        all_mask = (1 << N) - 1
        bb_solve(all_mask, 0, 0)

    # Final refinement
    if time.time() < DEADLINE - 2:
        refined = local_search(best_mask, 8000)
        if is_valid(refined):
            s = popcount(refined)
            if s > best_size:
                best_size = s
                best_mask = refined

    # More restarts
    if time.time() < DEADLINE - 1:
        for _ in range(10):
            if time.time() > DEADLINE:
                break
            ro = list(range(N))
            random.shuffle(ro)
            m = greedy_add(ro)
            m = local_search(m, 2000)
            if is_valid(m):
                s = popcount(m)
                if s > best_size:
                    best_size = s
                    best_mask = m

    result = sorted([i + 1 for i in range(N) if (best_mask >> i) & 1])
    if not result:
        result = [1]

    # Verify
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if (a * b) % (a + b) == 0:
                result = [x for x in result if x != b]
                break

    return result
# EVOLVE-BLOCK-END
