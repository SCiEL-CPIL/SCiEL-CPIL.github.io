import time
import random

def construct():
    deadline = time.time() + 80.0
    N = 128

    # all nonconstant 4-term APs, 0-indexed
    aps = []
    for a in range(1, N+1):
        for d in range(1, N+1):
            if a + 3*d <= N:
                aps.append((a-1, a+d-1, a+2*d-1, a+3*d-1))
    n_aps = len(aps)

    def full_score(arr):
        v = 0
        for ap in aps:
            if len({arr[ap[0]], arr[ap[1]], arr[ap[2]], arr[ap[3]]}) < 3:
                v += 1
        return v, len(set(arr))

    # start all-distinct: valid, 128 colors
    cur = list(range(N))
    cur_v, cur_c = full_score(cur)

    best = cur[:]
    best_v, best_c = cur_v, cur_c

    BIG = 10**6
    def obj(v, c):
        return v*BIG + c

    best_obj = obj(best_v, best_c)

    random.seed(12345)
    t0 = time.time()
    T = 1.0
    cool = 0.9999

    while time.time() < deadline:
        p = random.randrange(N)
        oldc = cur[p]
        # candidate: reuse a color from current palette (shrinks palette) or a random label
        palette = set(cur)
        if random.random() < 0.7:
            newc = random.choice(list(palette))
        else:
            newc = random.randrange(128)
        if newc == oldc:
            continue
        cur[p] = newc
        nv, nc = full_score(cur)
        no = obj(nv, nc)
        d = no - obj(cur_v, cur_c)
        if d <= 0 or random.random() < pow(2.718281828, -d / max(T, 1e-9)):
            cur_v, cur_c = nv, nc
            if no < best_obj:
                best_obj = no
                best_v, best_c = nv, nc
                best = cur[:]
        else:
            cur[p] = oldc
        T *= cool

    # final polish: try to drop colors by remapping last occurrences
    # ensure best valid
    if best_v != 0:
        return list(range(N))
    return best
