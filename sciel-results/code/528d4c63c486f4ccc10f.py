import time
import random

def sidon4_count(a_set):
    s = list(a_set)
    n = len(s)
    count = 0
    for a in range(n):
        sa = s[a]
        for b in range(a+1, n):
            sb = s[b]
            for c in range(b+1, n):
                sc = s[c]
                for d in range(c+1, n):
                    sd = s[d]
                    # 10 pair sums
                    if len({sa+sa, sa+sb, sa+sc, sa+sd,
                           sb+sb, sb+sc, sb+sd,
                           sc+sc, sc+sd,
                           sd+sd}) == 10:
                        count += 1
    return count

def construct():
    start = time.time()
    deadline = start + 82.0

    best = list(range(12))
    best_count = sidon4_count(best)

    rng = random.Random(7)

    def now_ok():
        return time.time() < deadline

    while now_ok():
        # Start a fresh local search from a dense random set
        r = rng.randint(12, 22)
        cur = rng.sample(range(r+1), 12)
        cur_count = sidon4_count(cur)
        if cur_count < best_count:
            best, best_count = cur[:], cur_count
            if best_count == 0:
                return best

        # Local search
        stagnant = 0
        while now_ok():
            i = rng.randrange(12)
            new_val = rng.randrange(0, r+1)  # stay in dense range
            if new_val in cur and cur[i] != new_val:
                stagnant += 1
                continue
            old = cur[i]
            cur[i] = new_val
            new_count = sidon4_count(cur)
            if new_count < cur_count or (new_count == cur_count and rng.random() < 0.25):
                cur_count = new_count
                stagnant = 0
                if new_count < best_count:
                    best, best_count = cur[:], new_count
                    if best_count == 0:
                        return best
            else:
                cur[i] = old
                stagnant += 1
            if stagnant > 400:
                break

    return best
