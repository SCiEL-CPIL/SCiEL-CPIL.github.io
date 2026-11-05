"""B_3[1] set construction: maximize cardinality with unique unordered triple sums."""
import time
import random

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    N = 4096

    def check_and_build(order):
        """Greedy build following candidate order; returns selected list."""
        selected = []
        n = 0
        # used bitset over possible sums: min sum 3, max sum 3*4096=12288
        # use a Python int as bitset (bit i = sum i used)
        used = 0
        for x in order:
            if x <= 0 or x > N:
                continue
            # compute new sums involving x
            # 3x
            s3 = 3 * x
            if (used >> s3) & 1:
                continue
            new_sums = [s3]
            ok = True
            # 2x + s
            for s in selected:
                v = 2 * x + s
                if (used >> v) & 1:
                    ok = False
                    break
                new_sums.append(v)
            if not ok:
                continue
            # x + 2s
            for s in selected:
                v = x + 2 * s
                if (used >> v) & 1:
                    ok = False
                    break
                new_sums.append(v)
            if not ok:
                continue
            # x + si + sj for i<=j
            for i in range(n):
                si = selected[i]
                for j in range(i, n):
                    sj = selected[j]
                    v = x + si + sj
                    if (used >> v) & 1:
                        ok = False
                        break
                    new_sums.append(v)
                if not ok:
                    break
            if not ok:
                continue
            # check internal collisions among new_sums
            # (they could collide with each other)
            if len(new_sums) != len(set(new_sums)):
                continue
            # accept
            for v in new_sums:
                used |= (1 << v)
            selected.append(x)
            n += 1
            if len(selected) >= 80:
                break
        return selected

    # Phase 1: deterministic greedy in increasing order
    best = check_and_build(range(1, N + 1))
    best_size = len(best)

    if time.time() > deadline:
        return best

    # Phase 2: randomized restarts with shuffled orders, keep best
    rng = random.Random(12345)
    attempts = 0
    while time.time() < deadline - 2.0:
        # try a few strategies:
        # (a) fully random order
        # (b) increasing with random swaps
        strategy = rng.random()
        if strategy < 0.5:
            order = list(range(1, N + 1))
            rng.shuffle(order)
        else:
            order = list(range(1, N + 1))
            # random perturbation
            for _ in range(rng.randint(0, 50)):
                i = rng.randint(0, len(order) - 2)
                order[i], order[i+1] = order[i+1], order[i]
        cand = check_and_build(order)
        if len(cand) > best_size:
            best = cand
            best_size = len(cand)
        attempts += 1
        if best_size >= 80:
            break

    # Phase 3: local search - try to add more elements by trying all x not in best
    if time.time() < deadline - 1.0 and best_size < 80:
        best_set = set(best)
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for x in range(1, N + 1):
                if x in best_set:
                    continue
                trial = best + [x]
                # verify
                trial.sort()
                # quick check via rebuild sums
                sums = set()
                n = len(trial)
                valid = True
                for i in range(n):
                    for j in range(i, n):
                        for k in range(j, n):
                            s = trial[i] + trial[j] + trial[k]
                            if s in sums:
                                valid = False
                                break
                            sums.add(s)
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    best = trial
                    best_set.add(x)
                    best_size += 1
                    improved = True
                    if best_size >= 80:
                        break

    return best

if __name__ == "__main__":
    result = construct()
    print(len(result))
