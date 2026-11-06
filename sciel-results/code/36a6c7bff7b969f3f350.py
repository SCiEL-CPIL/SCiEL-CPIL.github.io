import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 52.0  # safety margin before 60s (evaluator kills at 90s)
    N = 256
    P = 100.0  # heavy penalty for 4-APs

    # ---- Precompute 3-APs and 4-APs (0-based) ----
    threes = []
    for d in range(1, N // 2 + 1):
        for a in range(0, N - 2 * d):
            threes.append((a, a + d, a + 2 * d))

    fours = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            fours.append((a, a + d, a + 2 * d, a + 3 * d))

    # neighbor lists: for each index, the "other" members of each AP it belongs to
    index_3aps = [[] for _ in range(N)]
    for a, b, c in threes:
        index_3aps[a].append((b, c))
        index_3aps[b].append((a, c))
        index_3aps[c].append((a, b))

    index_4aps = [[] for _ in range(N)]
    for a, b, c, d in fours:
        index_4aps[a].append((b, c, d))
        index_4aps[b].append((a, c, d))
        index_4aps[c].append((a, b, d))
        index_4aps[d].append((a, b, c))

    def calc_scores(state):
        s3 = 0
        s4 = 0
        for a, b, c in threes:
            if state[a] and state[b] and state[c]:
                s3 += 1
        for a, b, c, d in fours:
            if state[a] and state[b] and state[c] and state[d]:
                s4 += 1
        return s3, s4

    def init_greedy():
        in_set = [False] * N
        for x in range(N):
            creates_4ap = False
            max_d = (N - 1 - x) // 3
            for d in range(1, max_d + 1):
                if in_set[x + d] and in_set[x + 2 * d] and in_set[x + 3 * d]:
                    creates_4ap = True
                    break
            if not creates_4ap:
                max_d = min(x, (N - 1 - x) // 2)
                for d in range(1, max_d + 1):
                    if in_set[x - d] and in_set[x + d] and in_set[x + 2 * d]:
                        creates_4ap = True
                        break
            if not creates_4ap:
                max_d = min(x // 2, N - 1 - x)
                for d in range(1, max_d + 1):
                    if in_set[x - 2 * d] and in_set[x - d] and in_set[x + d]:
                        creates_4ap = True
                        break
            if not creates_4ap:
                max_d = x // 3
                for d in range(1, max_d + 1):
                    if in_set[x - 3 * d] and in_set[x - 2 * d] and in_set[x - d]:
                        creates_4ap = True
                        break
            if not creates_4ap:
                in_set[x] = True
        return in_set

    # ---- Delta helpers ----
    def delta_flip(state, x):
        """Change in (s3, s4) if we flip state[x] (toggle)."""
        new_val = not state[x]
        sign = 1 if new_val else -1
        d3 = 0
        d4 = 0
        for o1, o2 in index_3aps[x]:
            if state[o1] and state[o2]:
                d3 += sign
        for o1, o2, o3 in index_4aps[x]:
            if state[o1] and state[o2] and state[o3]:
                d4 += sign
        return d3, d4

    def delta_two_flip(state, x, y):
        """Change in (s3, s4) if we flip both state[x] and state[y]."""
        if x == y:
            return 0, 0
        # Combined effect. We compute by checking each AP's membership change.
        # For an AP, its contribution changes only if the number of "in-set" members changes.
        # We'll compute net delta by considering each AP that contains x or y.
        d3 = 0
        d4 = 0
        # 3-APs
        seen3 = set()
        for o1, o2 in index_3aps[x]:
            ap = tuple(sorted((x, o1, o2)))
            if ap in seen3:
                continue
            seen3.add(ap)
            # members: x, o1, o2
            cnt = (1 if state[x] else 0) + (1 if state[o1] else 0) + (1 if state[o2] else 0)
            new_cnt = (0 if state[x] else 1) + (1 if state[o1] else 0) + (1 if state[o2] else 0)
            before = 1 if cnt == 3 else 0
            after = 1 if new_cnt == 3 else 0
            d3 += after - before
        for o1, o2 in index_3aps[y]:
            ap = tuple(sorted((y, o1, o2)))
            if ap in seen3:
                continue
            seen3.add(ap)
            cnt = (1 if state[y] else 0) + (1 if state[o1] else 0) + (1 if state[o2] else 0)
            new_cnt = (0 if state[y] else 1) + (1 if state[o1] else 0) + (1 if state[o2] else 0)
            before = 1 if cnt == 3 else 0
            after = 1 if new_cnt == 3 else 0
            d3 += after - before
        # 4-APs
        seen4 = set()
        for o1, o2, o3 in index_4aps[x]:
            ap = tuple(sorted((x, o1, o2, o3)))
            if ap in seen4:
                continue
            seen4.add(ap)
            cnt = (1 if state[x] else 0) + (1 if state[o1] else 0) + (1 if state[o2] else 0) + (1 if state[o3] else 0)
            new_cnt = (0 if state[x] else 1) + (1 if state[o1] else 0) + (1 if state[o2] else 0) + (1 if state[o3] else 0)
            before = 1 if cnt == 4 else 0
            after = 1 if new_cnt == 4 else 0
            d4 += after - before
        for o1, o2, o3 in index_4aps[y]:
            ap = tuple(sorted((y, o1, o2, o3)))
            if ap in seen4:
                continue
            seen4.add(ap)
            cnt = (1 if state[y] else 0) + (1 if state[o1] else 0) + (1 if state[o2] else 0) + (1 if state[o3] else 0)
            new_cnt = (0 if state[y] else 1) + (1 if state[o1] else 0) + (1 if state[o2] else 0) + (1 if state[o3] else 0)
            before = 1 if cnt == 4 else 0
            after = 1 if new_cnt == 4 else 0
            d4 += after - before
        return d3, d4

    # ---- Parallel tempering setup ----
    num_replicas = 12
    T_min = 0.3
    T_max = 8.0
    temperatures = [T_min * (T_max / T_min) ** (i / (num_replicas - 1)) for i in range(num_replicas)]

    replicas = []
    for i in range(num_replicas):
        if i < num_replicas // 2:
            r = init_greedy()
            for _ in range(random.randint(5, 20)):
                idx = random.randint(0, N - 1)
                r[idx] = not r[idx]
        else:
            r = init_greedy()
            # stronger perturbation for high-T replicas
            for _ in range(random.randint(30, 80)):
                idx = random.randint(0, N - 1)
                r[idx] = not r[idx]
        s3, s4 = calc_scores(r)
        replicas.append({
            'state': r,
            's3': s3,
            's4': s4,
            'score': s3 - P * s4,
            'T': temperatures[i],
        })

    best_valid_set = None
    best_valid_score = -1
    for r in replicas:
        if r['s4'] == 0 and r['s3'] > best_valid_score:
            best_valid_score = r['s3']
            best_valid_set = r['state'].copy()

    # ---- Local search polishing (single-flip, greedy, on a valid set) ----
    def polish(state, s3, s4, time_limit):
        """Greedy single-flip hill climb. Returns (state, s3, s4). Only keeps valid (s4==0) moves that gain 3-APs, plus accepts 3-AP-neutral moves that keep validity to escape plateaus."""
        if s4 != 0:
            return state, s3, s4
        local_deadline = time.time() + time_limit
        improved = True
        while improved and time.time() < local_deadline:
            improved = False
            # try all flips that keep validity and gain 3-APs
            best_gain = 0
            best_x = -1
            for x in range(N):
                d3, d4 = delta_flip(state, x)
                new_s4 = s4 + d4
                if new_s4 == 0 and s3 + d3 > s3 + best_gain:
                    best_gain = d3
                    best_x = x
            if best_x >= 0 and best_gain > 0:
                state[best_x] = not state[best_x]
                s3 += best_gain
                s4 += 0
                improved = True
            else:
                # try neutral moves (d3==0, s4 stays 0) to escape plateau, limited
                neutral = [x for x in range(N) if (lambda d: d[1] == 0 and d[0] == 0)(delta_flip(state, x))]
                if neutral and random.random() < 0.3:
                    x = random.choice(neutral)
                    state[x] = not state[x]
                    improved = True
        return state, s3, s4

    # ---- Main SA / PT loop with swap moves ----
    swap_prob = 0.35
    while time.time() < deadline:
        # pick a replica (biased toward low-T for exploitation)
        i = random.randint(0, num_replicas - 1)
        r = replicas[i]
        state = r['state']
        s3, s4 = r['s3'], r['s4']
        T = r['T']

        if random.random() < swap_prob:
            # swap move: pick x to remove and y to add (or both flips)
            x = random.randint(0, N - 1)
            y = random.randint(0, N - 1)
            if x == y:
                continue
            d3, d4 = delta_two_flip(state, x, y)
            new_s3 = s3 + d3
            new_s4 = s4 + d4
            new_score = new_s3 - P * new_s4
            cur_score = r['score']
            if new_score > cur_score or random.random() < math.exp((new_score - cur_score) / T):
                state[x] = not state[x]
                state[y] = not state[y]
                r['s3'], r['s4'], r['score'] = new_s3, new_s4, new_score
                if new_s4 == 0 and new_s3 > best_valid_score:
                    best_valid_score = new_s3
                    best_valid_set = state.copy()
        else:
            # single flip
            x = random.randint(0, N - 1)
            d3, d4 = delta_flip(state, x)
            new_s3 = s3 + d3
            new_s4 = s4 + d4
            new_score = new_s3 - P * new_s4
            cur_score = r['score']
            if new_score > cur_score or random.random() < math.exp((new_score - cur_score) / T):
                state[x] = not state[x]
                r['s3'], r['s4'], r['score'] = new_s3, new_s4, new_score
                if new_s4 == 0 and new_s3 > best_valid_score:
                    best_valid_score = new_s3
                    best_valid_set = state.copy()

        # periodic tempering swap between adjacent replicas
        if random.random() < 0.1:
            a = random.randint(0, num_replicas - 2)
            b = a + 1
            ra, rb = replicas[a], replicas[b]
            # Metropolis swap: accept with prob 1 if it moves state from high-T to low-T favorably
            # Standard PT: accept swap with prob min(1, exp((1/ra.T - 1/rb.T)*(rb.score - ra.score)))
            try:
                prob = math.exp((1.0 / ra['T'] - 1.0 / rb['T']) * (rb['score'] - ra['score']))
            except OverflowError:
                prob = 1.0
            if random.random() < prob:
                # swap states
                ra['state'], rb['state'] = rb['state'], ra['state']
                ra['s3'], rb['s3'] = rb['s3'], ra['s3']
                ra['s4'], rb['s4'] = rb['s4'], ra['s4']
                ra['score'], rb['score'] = rb['score'], ra['score']
                # re-check best
                for rr in (ra, rb):
                    if rr['s4'] == 0 and rr['s3'] > best_valid_score:
                        best_valid_score = rr['s3']
                        best_valid_set = rr['state'].copy()

        # periodic local search polishing on best valid replica
        if time.time() < deadline - 3.0 and best_valid_set is not None and random.random() < 0.02:
            ps = best_valid_set.copy()
            ps3, ps4 = best_valid_score, 0
            ps, ps3, ps4 = polish(ps, ps3, ps4, time_limit=1.5)
            if ps4 == 0 and ps3 > best_valid_score:
                best_valid_score = ps3
                best_valid_set = ps
                # seed a low-T replica with this improved state
                replicas[0]['state'] = ps.copy()
                replicas[0]['s3'], replicas[0]['s4'], replicas[0]['score'] = ps3, ps4, ps3 - P * ps4

    if best_valid_set is None:
        # fallback: greedy
        best_valid_set = init_greedy()

    return [i + 1 for i in range(N) if best_valid_set[i]]
