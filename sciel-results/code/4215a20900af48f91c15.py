"""Combined program: bitmask + precomputed masks + weighted objective + simulated annealing."""
import time
import random

N = 256

# ---- Precompute per-element 3-AP partner lists and 4-AP conflict masks ----
f3 = [[] for _ in range(N + 1)]   # x as FIRST: (x+d, x+2d)
m3 = [[] for _ in range(N + 1)]   # x as MIDDLE: (x-d, x+d)
l3 = [[] for _ in range(N + 1)]   # x as LAST: (x-2d, x-d)
f4_masks = [[] for _ in range(N + 1)]  # bitmask of the OTHER three elements in each 4-AP containing x

def to_mask(vals):
    m = 0
    for v in vals:
        m |= (1 << (v - 1))
    return m

for x in range(1, N + 1):
    for d in range(1, (N - x) // 2 + 1):
        f3[x].append((x + d, x + 2 * d))
    for d in range(1, min(x - 1, N - x) + 1):
        m3[x].append((x - d, x + d))
    for d in range(1, (x - 1) // 2 + 1):
        l3[x].append((x - 2 * d, x - d))
    # 4-AP positions
    for d in range(1, (N - x) // 3 + 1):
        f4_masks[x].append(to_mask([x + d, x + 2 * d, x + 3 * d]))
    for d in range(1, min(x - 1, (N - x) // 2) + 1):
        f4_masks[x].append(to_mask([x - d, x + d, x + 2 * d]))
    for d in range(1, min((x - 1) // 2, N - x) + 1):
        f4_masks[x].append(to_mask([x - 2 * d, x - d, x + d]))
    for d in range(1, (x - 1) // 3 + 1):
        f4_masks[x].append(to_mask([x - 3 * d, x - 2 * d, x - d]))

def bit(A, x):
    return (A >> (x - 1)) & 1

def creates_4AP(A, x):
    for mask in f4_masks[x]:
        if (A & mask) == mask:
            return True
    return False

def gain_add(A, x):
    g = 0
    for a, b in f3[x]:
        if bit(A, a) and bit(A, b):
            g += 1
    for a, b in m3[x]:
        if bit(A, a) and bit(A, b):
            g += 1
    for a, b in l3[x]:
        if bit(A, a) and bit(A, b):
            g += 1
    return g

def count_3AP(A):
    c = 0
    for x in range(1, N + 1):
        if bit(A, x):
            for a, b in m3[x]:
                if bit(A, a) and bit(A, b):
                    c += 1
    return c

def count_near_4AP(A):
    """Number of 4-APs that would complete by adding exactly one element (i.e. 3 of the 4 are present)."""
    cnt = 0
    for x in range(1, N + 1):
        if not bit(A, x):
            for mask in f4_masks[x]:
                # how many of the other three are present
                if ((A & mask).bit_count() == 3):
                    cnt += 1
                    break  # count each potential completion once per missing x
    return cnt

def objective(A, lam):
    return count_3AP(A) - lam * count_near_4AP(A)

def greedy_seed(rng, lam):
    A = 0
    order = list(range(1, N + 1))
    rng.shuffle(order)
    for x in order:
        if creates_4AP(A, x):
            continue
        # weighted: prefer high 3-AP gain, penalize creating near-4-APs
        g = gain_add(A, x)
        # cheap near-4-AP increment: adding x can complete a 4-AP only if 2 of the other three are present
        near = 0
        for mask in f4_masks[x]:
            if (A & mask).bit_count() == 2:
                near += 1
        score = g - lam * near
        if score >= 0:
            A |= (1 << (x - 1))
    return A

def local_search(A, deadline, lam):
    """Best-improvement add + swap using weighted objective."""
    def valid_add(x):
        return not creates_4AP(A, x)

    def weighted_gain(x):
        g = gain_add(A, x)
        near = 0
        for mask in f4_masks[x]:
            if (A & mask).bit_count() == 2:
                near += 1
        return g - lam * near

    while time.time() < deadline:
        improved = False
        # best positive-gain addition
        bx, bs = None, 0
        for x in range(1, N + 1):
            if bit(A, x):
                continue
            if not valid_add(x):
                continue
            s = weighted_gain(x)
            if s > bs:
                bs, bx = s, x
        if bx is not None and bs > 0:
            A |= (1 << (bx - 1))
            improved = True
        # best net-positive swap
        if not improved:
            bnet, br, bsx = 0, None, None
            for r in range(1, N + 1):
                if not bit(A, r):
                    continue
                Ar = A & ~(1 << (r - 1))
                # loss of removing r (approx: 3-APs containing r)
                loss = 0
                for a, b in m3[r]:
                    if bit(Ar, a) and bit(Ar, b):
                        loss += 1
                for x in range(1, N + 1):
                    if bit(A, x):
                        continue
                    if creates_4AP(Ar, x):
                        continue
                    g = gain_add(Ar, x)
                    near = 0
                    for mask in f4_masks[x]:
                        if (Ar & mask).bit_count() == 2:
                            near += 1
                    net = (g - lam * near) - loss
                    if net > bnet:
                        bnet, br, bsx = net, r, x
            if br is not None and bnet > 0:
                A &= ~(1 << (br - 1))
                A |= (1 << (bsx - 1))
                improved = True
        if not improved:
            break
    return A

def simulated_annealing(A, deadline, lam):
    """SA polish: accept downhill moves with decaying temperature."""
    rng = random.Random()
    temp = 4.0
    cooling = 0.995
    cur_obj = objective(A, lam)
    while time.time() < deadline and temp > 0.05:
        # propose a random add or remove that keeps validity
        r = rng.random()
        if r < 0.5:
            # random add
            x = rng.randint(1, N)
            if bit(A, x) or creates_4AP(A, x):
                temp *= cooling
                continue
            A2 = A | (1 << (x - 1))
        else:
            # random remove
            present = [x for x in range(1, N + 1) if bit(A, x)]
            if not present:
                temp *= cooling
                continue
            x = rng.choice(present)
            A2 = A & ~(1 << (x - 1))
        new_obj = objective(A2, lam)
        delta = new_obj - cur_obj
        if delta >= 0 or rng.random() < (2.718281828 ** (delta / temp)):
            A = A2
            cur_obj = new_obj
        temp *= cooling
    return A

def sanitize(A):
    """Hard validity: remove any element in a completed 4-AP."""
    changed = True
    while changed:
        changed = False
        for x in range(1, N + 1):
            if not bit(A, x):
                continue
            # check if removing x is needed: x is in a 4-AP with three others present
            # 4-APs where x is one of the four; the other three must all be present
            # We use f4_masks but those are defined for x as a member: mask = other three
            for mask in f4_masks[x]:
                if (A & mask) == mask:
                    A &= ~(1 << (x - 1))
                    changed = True
                    break
    return A

def construct():
    start = time.time()
    deadline = start + 52.0  # safety margin under 90s kill
    lam = 0.15

    best_A = 0
    best_score = -10**9
    rng = random.Random(12345)

    # multi-restart
    while time.time() < deadline - 8.0:
        A = greedy_seed(rng, lam)
        A = local_search(A, min(deadline - 6.0, time.time() + 2.0), lam)
        s = count_3AP(A)
        if s > best_score:
            best_score = s
            best_A = A

    # SA polish on the incumbent with remaining time
    if time.time() < deadline - 1.0 and best_A:
        A = simulated_annealing(best_A, deadline - 0.5, lam)
        s = count_3AP(A)
        if s > best_score:
            best_score = s
            best_A = A

    # final local search
    if time.time() < deadline - 0.3 and best_A:
        A = local_search(best_A, deadline - 0.2, lam)
        s = count_3AP(A)
        if s > best_score:
            best_score = s
            best_A = A

    # hard validity guarantee
    best_A = sanitize(best_A)

    A_list = [x for x in range(1, N + 1) if bit(best_A, x)]
    if not A_list:
        A_list = [1]
    return A_list
