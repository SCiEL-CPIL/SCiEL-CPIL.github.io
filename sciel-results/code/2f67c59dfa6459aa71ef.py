import time
import random

def construct():
    t0 = time.time()
    DEADLINE = t0 + 80.0

    N = 256
    FULL = (1 << N) - 1

    # conflict_mask[i]: bitmask of j where (i+1)*(j+1) % ((i+1)+(j+1)) == 0
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

    odds = [i for i in range(N) if (i + 1) % 2 == 1]
    evens = [i for i in range(N) if (i + 1) % 2 == 0]

    # Precompute 2-adic valuation for each number
    def v2(n):
        c = 0
        while n % 2 == 0:
            n //= 2
            c += 1
        return c

    v2_class = [v2(i + 1) for i in range(N)]
    # Group indices by 2-adic valuation
    v2_groups = {}
    for i in range(N):
        v2_groups.setdefault(v2_class[i], []).append(i)

    def popcount(x):
        return x.bit_count()

    # Precompute conflict counts for each element with current mask
    # conflict_count[e] = popcount(cm[e] & m)
    def get_conflict_count(e, m):
        return popcount(cm[e] & m)

    # ---- candidate builders (each returns a valid mask) ----

    def greedy_add(start, order):
        m = start
        for e in order:
            if (m >> e) & 1:
                continue
            if cm[e] & m:
                continue
            m |= (1 << e)
        return m

    def local_search(m, iters):
        best = m
        bestsz = popcount(m)
        # Track conflict counts for faster swap evaluation
        cc = [0] * N  # conflict count for each element with current mask
        for i in range(N):
            if (m >> i) & 1:
                cc[i] = popcount(cm[i] & m)

        for _ in range(iters):
            if time.time() > DEADLINE:
                break
            if random.random() < 0.7:
                e = random.randrange(N)
                if (m >> e) & 1:
                    continue
                conf_e = cm[e] & m
                if not conf_e:
                    m |= (1 << e)
                    cc[e] = 0
                    for i in range(N):
                        if (m >> i) & 1 and i != e:
                            if cm[i] & (1 << e):
                                cc[i] += 1
                    s = popcount(m)
                    if s > bestsz:
                        best, bestsz = m, s
                else:
                    # Try swap: remove one conflicting element c, add e
                    # Use 2-adic valuation to prioritize swaps
                    conf_list = []
                    tmp = conf_e
                    while tmp:
                        c = (tmp & -tmp).bit_length() - 1
                        conf_list.append(c)
                        tmp &= tmp - 1

                    # Sort by 2-adic valuation to try same-class swaps first
                    conf_list.sort(key=lambda x: v2_class[x])

                    for c in conf_list[:8]:  # Try up to 8 swap candidates
                        if c == e:
                            continue
                        # Check if removing c and adding e is valid
                        # After removing c, e must not conflict with remaining elements
                        # After adding e, c's removal must not break others (but c was already in m)
                        # Key check: for all i in m \ {c}, cm[i] & (1 << e) == 0 is NOT required
                        # Actually we need: for all i in (m \ {c}) ∪ {e}, no conflicts
                        # Since m was valid, removing c only helps. Adding e: need cm[e] & (m \ {c}) == 0
                        # cm[e] & (m \ {c}) = (cm[e] & m) & ~(1 << c) = conf_e & ~(1 << c)
                        if conf_e & ~(1 << c):
                            continue
                        # Also need: for all i in m \ {c}, cm[i] & (1 << e) == 0
                        # This is equivalent to: no element in m\{c} has e in its conflict mask
                        # i.e., (OR of cm[i] for i in m\{c}) & (1 << e) == 0
                        # But we can check: for each i in m\{c}, if cm[i] has bit e set, conflict
                        # Simpler: check if any i in m\{c} has cm[i] & (1<<e)
                        # We can precompute: which elements conflict with e? That's cm_inv[e]
                        # But we don't have that. Alternative: just verify after swap.
                        # For speed, do a quick check: count how many i in m\{c} conflict with e
                        # Actually, cm[e] & m gives elements that conflict with e (symmetric? No, not necessarily)
                        # Wait, (a*b)%(a+b)==0 is symmetric in a,b, so conflict is symmetric!
                        # So cm[i] has bit e iff cm[e] has bit i.
                        # So conf_e = cm[e] & m already tells us which elements in m conflict with e.
                        # After removing c, the remaining conflicting elements are conf_e & ~(1<<c)
                        # We already checked that's 0. So the swap is valid!
                        newm = (m & ~(1 << c)) | (1 << e)
                        # Update conflict counts
                        for i in range(N):
                            if (m >> i) & 1 and i != c:
                                if cm[i] & (1 << e):
                                    cc[i] += 1
                            if (newm >> i) & 1 and i != e:
                                if cm[i] & (1 << c):
                                    cc[i] -= 1
                        cc[e] = 0
                        cc[c] = 0
                        m = newm
                        s = popcount(m)
                        if s > bestsz:
                            best, bestsz = m, s
                        break  # Only do one swap per iteration for stability
            else:
                if m == 0:
                    continue
                c = (m & -m).bit_length() - 1
                trial = m & ~(1 << c)
                added = 0
                for _ in range(60):
                    if added >= 2:
                        break
                    e = random.randrange(N)
                    if (trial >> e) & 1:
                        continue
                    if not (cm[e] & trial):
                        trial |= (1 << e)
                        added += 1
                s = popcount(trial)
                if s > popcount(m):
                    m = trial
                    # Recompute cc
                    for i in range(N):
                        if (m >> i) & 1:
                            cc[i] = popcount(cm[i] & m)
                        else:
                            cc[i] = 0
                    if s > bestsz:
                        best, bestsz = m, s
        return best

    def is_valid(m):
        for i in range(N):
            if (m >> i) & 1:
                if cm[i] & m:
                    return False
        return True

    def polish(m, iters):
        """Final high-precision polish: try all single swaps systematically"""
        best = m
        bestsz = popcount(m)
        for _ in range(iters):
            if time.time() > DEADLINE:
                break
            improved = False
            for e in range(N):
                if (best >> e) & 1:
                    continue
                conf_e = cm[e] & best
                if not conf_e:
                    best |= (1 << e)
                    improved = True
                    continue
                # Try swapping out each conflicting element
                tmp = conf_e
                while tmp:
                    c = (tmp & -tmp).bit_length() - 1
                    tmp &= tmp - 1
                    if c == e:
                        continue
                    if conf_e & ~(1 << c):
                        continue
                    # Valid swap
                    newm = (best & ~(1 << c)) | (1 << e)
                    s = popcount(newm)
                    if s > popcount(best):
                        best = newm
                        bestsz = s
                        improved = True
                        break
            if not improved:
                break
        return best

    best = 0
    bestsz = 0

    def consider(m):
        nonlocal best, bestsz
        if not is_valid(m):
            return
        s = popcount(m)
        if s > bestsz:
            best, bestsz = m, s

    # 1) all odds baseline
    base = 0
    for i in odds:
        base |= (1 << i)
    consider(base)

    # 2) greedy evens on odds, several orders
    consider(greedy_add(base, evens))
    consider(greedy_add(base, list(reversed(evens))))
    for _ in range(30):
        if time.time() > DEADLINE:
            break
        ro = evens[:]
        random.shuffle(ro)
        consider(greedy_add(base, ro))

    # 3) random small cores + greedy
    for _ in range(40):
        if time.time() > DEADLINE:
            break
        core = 0
        for i in random.sample(range(N), random.randint(1, 6)):
            if not (cm[i] & core):
                core |= (1 << i)
        ro = evens[:]
        random.shuffle(ro)
        consider(greedy_add(core, ro))

    # 4) local search from best
    if time.time() < DEADLINE - 10:
        consider(local_search(best, 150000))

    # 5) polish pass
    if time.time() < DEADLINE - 3:
        polished = polish(best, 50)
        consider(polished)

    # 6) if time remains, more local search from polished
    if time.time() < DEADLINE - 3:
        consider(local_search(best, 30000))

    if best == 0:
        best = 1  # number 1 alone is valid
    result = sorted([i + 1 for i in range(N) if (best >> i) & 1])
    return result
