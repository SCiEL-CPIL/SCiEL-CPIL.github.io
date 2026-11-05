import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 82.0

    n = 128
    numbers = list(range(1, n + 1))

    # ---- prime factorization (square-free mask + full mask) ----
    spf = list(range(n + 1))
    for i in range(2, int(n ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, n + 1, i):
                if spf[j] == j:
                    spf[j] = i
    prime_list = [i for i in range(2, n + 1) if spf[i] == i]
    pidx = {p: i for i, p in enumerate(prime_list)}

    def fac(x):
        full = [0] * len(prime_list)
        sq = 0
        while x > 1:
            p = spf[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            full[pidx[p]] = e
            if e >= 2:
                sq |= (1 << pidx[p])
        return full, sq

    full_exp = [None] * (n + 1)
    sq_mask = [0] * (n + 1)
    for x in range(1, n + 1):
        full_exp[x], sq_mask[x] = fac(x)

    # ---- LCM table (full exponent vectors) ----
    lcm_exp = [[None] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        fi = full_exp[i]
        for j in range(i, n + 1):
            fj = full_exp[j]
            vec = [a if a >= b else b for a, b in zip(fi, fj)]
            lcm_exp[i][j] = vec
            lcm_exp[j][i] = vec

    def div_mask_of(vec):
        m = 0
        for x in range(1, n + 1):
            fx = full_exp[x]
            ok = True
            for a, b in zip(fx, vec):
                if a > b:
                    ok = False
                    break
            if ok:
                m |= (1 << x)
        return m

    L = n + 1
    div_mask = [0] * L
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            vec = lcm_exp[i][j]
            if vec is not None:
                div_mask[i * L + j] = div_mask[j * L + i] = div_mask_of(vec)

    def is_maximal_divisor_of(x, vec, sel_mask):
        fx = full_exp[x]
        for a, b in zip(fx, vec):
            if a > b:
                return False
        for y in range(1, n + 1):
            if x == y or not (sel_mask >> y) & 1:
                continue
            fy = full_exp[y]
            if all(fx[k] <= fy[k] for k in range(len(fx))) and any(fx[k] < fy[k] for k in range(len(fx))):
                return False
        return True

    # ---- core set: all prime powers + 1 (independently safe) ----
    core = set()
    for p in prime_list:
        q = p
        while q <= n:
            core.add(q)
            q *= p
    core.add(1)
    core_mask = 0
    for x in core:
        core_mask |= (1 << x)

    # ---- simulated annealing over the complement ----
    comp = [x for x in range(1, n + 1) if x not in core]
    random.seed(20240517)
    random.shuffle(comp)

    # initial greedy fill
    sel = core_mask
    for x in comp:
        ok = True
        for a in range(1, n + 1):
            if (sel >> a) & 1:
                vec = lcm_exp[x][a]
                if is_maximal_divisor_of(x, vec, sel) and is_maximal_divisor_of(a, vec, sel):
                    ok = False
                    break
        if ok:
            sel |= (1 << x)

    best_sel = sel
    best_size = bin(sel).count("1")

    def size(m):
        return bin(m).count("1")

    def delta_add(x, sel):
        c = 0
        for a in range(1, n + 1):
            if (sel >> a) & 1:
                vec = lcm_exp[x][a]
                if is_maximal_divisor_of(x, vec, sel) and is_maximal_divisor_of(a, vec, sel):
                    c += 1
        return c

    def delta_remove(x, sel):
        c = 0
        for a in range(1, n + 1):
            if (sel >> a) & 1 and a != x:
                vec = lcm_exp[x][a]
                if is_maximal_divisor_of(x, vec, sel) and is_maximal_divisor_of(a, vec, sel):
                    c += 1
        return c

    T0 = 1.0
    Tend = 0.0008
    it = 0
    total = 200000
    while time.time() < deadline - 3.0:
        it += 1
        T = max(Tend, T0 * (Tend / T0) ** (it / total))
        x = random.choice(comp)
        if (sel >> x) & 1:
            # remove x, then try to add a random excluded element
            sel ^= (1 << x)
            gain = delta_remove(x, sel | (1 << x))
            r = random.choice(comp)
            while (sel >> r) & 1:
                r = random.choice(comp)
            loss = delta_add(r, sel)
            d = loss - gain
            if d < 0 or random.random() < pow(2.718281828, -d / T):
                sel |= (1 << r)
            else:
                sel ^= (1 << x)
        else:
            # swap: remove x, add a random selected non-core element
            gain = delta_add(x, sel)
            pool = [a for a in range(1, n + 1) if (sel >> a) & 1 and a not in core]
            if not pool:
                sel |= (1 << x)
                continue
            r = random.choice(pool)
            sel ^= (1 << r)
            loss = delta_remove(r, sel | (1 << r))
            d = gain - loss
            if d < 0 or random.random() < pow(2.718281828, -d / T):
                sel |= (1 << x)
            else:
                sel |= (1 << r)

        s = size(sel)
        if s > best_size:
            best_size = s
            best_sel = sel

    # ---- final validation ----
    S = [x for x in range(1, n + 1) if (best_sel >> x) & 1]
    valid = True
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            lab = lcm_exp[S[i]][S[j]]
            for k in range(j + 1, len(S)):
                if lcm_exp[S[i]][S[k]] == lab and lcm_exp[S[j]][S[k]] == lab:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        return sorted(core)

    return sorted(S)
