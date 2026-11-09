import random
import time

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115  # safety margin before 120s/150s evaluator limit

    N = 284
    PRESCRIBED = {1, 2, 5, 11}
    PRESCRIBED_LIST = sorted(PRESCRIBED)
    candidates = [x for x in range(1, N + 1) if x not in PRESCRIBED]
    MAX_SIZE = 20

    # ---------- fast validity / cost helpers ----------
    def is_valid(S):
        if len(S) != len(set(S)):
            return False
        if not PRESCRIBED.issubset(S):
            return False
        if any(x < 1 or x > N for x in S):
            return False
        sums = set()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def calc_cost(S):
        # number of "excess" duplicate pair-sums (0 == valid Sidon set)
        sums = {}
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                v = S[i] + S[j]
                sums[v] = sums.get(v, 0) + 1
        cost = 0
        for c in sums.values():
            if c > 1:
                cost += c - 1
        return cost

    def compute_sums(S):
        sums = set()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                sums.add(S[i] + S[j])
        return sums

    def can_add(S, sums, x):
        if x in S:
            return False
        for y in S:
            if x + y in sums:
                return False
        if 2 * x in sums:
            return False
        return True

    # ---------- fast local search (add + swap) ----------
    def local_search(S, rng, max_rounds=40):
        S = list(S)
        S.sort()
        S_set = set(S)
        sums = compute_sums(S)
        best_S = list(S)
        best_size = len(S)
        no_improve = 0

        for _ in range(max_rounds):
            if time.time() > DEADLINE:
                break
            improved = False

            # try to add one element
            cand = [x for x in range(1, N + 1) if x not in S_set]
            rng.shuffle(cand)
            for x in cand:
                if time.time() > DEADLINE:
                    break
                if can_add(S, sums, x):
                    S.append(x)
                    S.sort()
                    S_set.add(x)
                    for y in S[:-1]:
                        sums.add(x + y)
                    sums.add(2 * x)
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = list(S)
                    break

            # try a swap: drop a non-prescribed element, add a different one
            if not improved:
                removable = [i for i, x in enumerate(S) if x not in PRESCRIBED]
                rng.shuffle(removable)
                for i in removable:
                    if time.time() > DEADLINE:
                        break
                    removed = S[i]
                    S_without = S[:i] + S[i + 1:]
                    S_without.sort()
                    S_without_set = set(S_without)
                    sums_without = compute_sums(S_without)
                    cand = [x for x in range(1, N + 1) if x not in S_without_set]
                    rng.shuffle(cand)
                    found = False
                    for x in cand:
                        if time.time() > DEADLINE:
                            break
                        if can_add(S_without, sums_without, x):
                            S = S_without + [x]
                            S.sort()
                            S_set = set(S)
                            sums = compute_sums(S)
                            improved = True
                            found = True
                            if len(S) > best_size:
                                best_size = len(S)
                                best_S = list(S)
                            break
                    if found:
                        break

            if improved:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 4:
                    break

        return best_S

    # ---------- size-targeted simulated annealing (from current program) ----------
    def run_sa(target_size, time_limit):
        num_vars = target_size - len(PRESCRIBED)
        if num_vars < 0:
            return None
        end_time = time.time() + time_limit

        while time.time() < end_time:
            random_vars = random.sample(candidates, num_vars)
            s = PRESCRIBED_LIST + random_vars
            current_cost = calc_cost(s)
            if current_cost == 0:
                return sorted(set(s))

            T = 5.0
            T_min = 0.01
            cooling = 0.9995
            steps = 5000

            for _ in range(steps):
                if time.time() > end_time or time.time() > DEADLINE:
                    return None

                idx = random.randint(0, target_size - 1)
                while s[idx] in PRESCRIBED:
                    idx = random.randint(0, target_size - 1)

                old_val = s[idx]
                new_val = random.choice(candidates)
                if new_val == old_val or new_val in s:
                    continue

                s[idx] = new_val
                new_cost = calc_cost(s)
                delta = new_cost - current_cost
                if delta < 0 or random.random() < pow(2.71828, -delta / T):
                    current_cost = new_cost
                    if current_cost == 0:
                        return sorted(set(s))
                else:
                    s[idx] = old_val

                T *= cooling
                if T < T_min:
                    break

        return None

    # ---------- main strategy ----------
    best_S = list(PRESCRIBED_LIST)
    best_size = len(PRESCRIBED_LIST)

    # Phase 1: many randomized greedy restarts + fast local-search polish
    for seed in range(400):
        if time.time() > DEADLINE:
            break
        rng = random.Random(seed)
        order = list(candidates)
        rng.shuffle(order)

        S = list(PRESCRIBED_LIST)
        S.sort()
        S_set = set(S)
        sums = compute_sums(S)
        for x in order:
            if x in S_set:
                continue
            if can_add(S, sums, x):
                S.append(x)
                S_set.add(x)
                for y in S[:-1]:
                    sums.add(x + y)
                sums.add(2 * x)
                if len(S) == MAX_SIZE:
                    break
        S.sort()

        S = local_search(S, rng, max_rounds=30)
        if is_valid(S) and len(S) > best_size:
            best_size = len(S)
            best_S = list(S)

        if best_size == MAX_SIZE:
            return best_S

    # Phase 2: targeted SA on the largest sizes, using remaining time
    remaining = DEADLINE - time.time()
    if remaining > 20:
        s20 = run_sa(MAX_SIZE, min(70, remaining - 5))
        if s20 and is_valid(s20):
            return s20

    remaining = DEADLINE - time.time()
    if remaining > 10 and best_size < 19:
        s19 = run_sa(19, min(remaining - 3, 40))
        if s19 and is_valid(s19):
            return s19

    remaining = DEADLINE - time.time()
    if remaining > 5 and best_size < 18:
        s18 = run_sa(18, min(remaining - 2, 25))
        if s18 and is_valid(s18):
            return s18

    return best_S
