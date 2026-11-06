import time
import random
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit (evaluator kills at 90s)

    # Precompute all 3-APs and 4-APs
    ap3s = []
    for d in range(1, n // 2 + 1):
        for a in range(0, n - 2 * d):
            ap3s.append((a, a + d, a + 2 * d))

    ap4s = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            ap4s.append((a, a + d, a + 2 * d, a + 3 * d))

    M3 = len(ap3s)
    M4 = len(ap4s)

    # Build membership lists for fast delta computation
    # For each element v, store the "other" elements for each AP it's in
    ap3_members = [[] for _ in range(n)]
    ap4_members = [[] for _ in range(n)]

    for a, b, c in ap3s:
        ap3_members[a].append((b, c))
        ap3_members[b].append((a, c))
        ap3_members[c].append((a, b))

    for a, b, c, d in ap4s:
        ap4_members[a].append((b, c, d))
        ap4_members[b].append((a, c, d))
        ap4_members[c].append((a, b, d))
        ap4_members[d].append((a, b, c))

    def count_3aps(state):
        """Count 3-APs in state (list of bools)."""
        cnt = 0
        for a, b, c in ap3s:
            if state[a] and state[b] and state[c]:
                cnt += 1
        return cnt

    def count_4aps(state):
        """Count 4-APs in state (list of bools)."""
        cnt = 0
        for a, b, c, d in ap4s:
            if state[a] and state[b] and state[c] and state[d]:
                cnt += 1
        return cnt

    def delta_3ap(state, x, new_val):
        """Compute change in 3-AP count when flipping state[x] to new_val."""
        sign = 1 if new_val else -1
        d = 0
        for o1, o2 in ap3_members[x]:
            if state[o1] and state[o2]:
                d += sign
        return d

    def delta_4ap(state, x, new_val):
        """Compute change in 4-AP count when flipping state[x] to new_val."""
        sign = 1 if new_val else -1
        d = 0
        for o1, o2, o3 in ap4_members[x]:
            if state[o1] and state[o2] and state[o3]:
                d += sign
        return d

    def greedy_init():
        """Greedy: add elements in order, skip if creates 4-AP."""
        state = [False] * n
        for x in range(n):
            creates_4ap = False
            for o1, o2, o3 in ap4_members[x]:
                if state[o1] and state[o2] and state[o3]:
                    creates_4ap = True
                    break
            if not creates_4ap:
                state[x] = True
        return state

    def greedy_init_random():
        """Greedy with random order."""
        state = [False] * n
        order = list(range(n))
        random.shuffle(order)
        for x in order:
            creates_4ap = False
            for o1, o2, o3 in ap4_members[x]:
                if state[o1] and state[o2] and state[o3]:
                    creates_4ap = True
                    break
            if not creates_4ap:
                state[x] = True
        return state

    def hill_climb(state, s3, s4, max_iters=5000):
        """Strict hill climbing: only accept moves that improve (s3 - P*s4) or maintain validity."""
        P = 100.0
        score = s3 - P * s4
        improved = True
        iters = 0

        while improved and iters < max_iters and (time.time() - start_time < time_limit):
            improved = False
            iters += 1
            # Try all single flips
            for x in range(n):
                new_val = not state[x]
                d3 = delta_3ap(state, x, new_val)
                d4 = delta_4ap(state, x, new_val)
                new_s3 = s3 + d3
                new_s4 = s4 + d4
                new_score = new_s3 - P * new_s4

                # Accept if: improves score AND (new_s4 == 0 or new_score > score)
                # Prefer moves that keep s4=0
                if new_s4 == 0 and new_score > score:
                    state[x] = new_val
                    s3 = new_s3
                    s4 = new_s4
                    score = new_score
                    improved = True
                    break
                elif new_s4 > 0 and new_score > score:
                    # Only allow if it's a big improvement (unlikely in practice)
                    pass

        return state, s3, s4, score

    def simulated_annealing(state, s3, s4, steps=20000):
        """SA with penalty for 4-APs."""
        P = 100.0
        score = s3 - P * s4
        T_start = 5.0
        T_end = 0.1
        best_state = state[:]
        best_score = score

        for step in range(steps):
            if time.time() - start_time > time_limit:
                break

            T = T_start * (T_end / T_start) ** (step / steps)

            # Random flip
            x = random.randint(0, n - 1)
            new_val = not state[x]
            d3 = delta_3ap(state, x, new_val)
            d4 = delta_4ap(state, x, new_val)
            new_s3 = s3 + d3
            new_s4 = s4 + d4
            new_score = new_s3 - P * new_s4
            delta = new_score - score

            if delta > 0 or random.random() < np.exp(delta / T):
                state[x] = new_val
                s3 = new_s3
                s4 = new_s4
                score = new_score

                if new_s4 == 0 and new_s3 > best_score:
                    best_score = new_s3
                    best_state = state[:]

        return best_state, best_score

    best_valid_count = 0
    best_valid_state = None

    # Phase 1: Multiple greedy starts + hill climbing
    num_greedy = 8
    for i in range(num_greedy):
        if time.time() - start_time > time_limit:
            break

        if i == 0:
            state = greedy_init()
        else:
            state = greedy_init_random()

        s3 = count_3aps(state)
        s4 = count_4aps(state)

        if s4 == 0:
            state, s3, s4, score = hill_climb(state, s3, s4)
            if s4 == 0 and s3 > best_valid_count:
                best_valid_count = s3
                best_valid_state = state[:]

    # Phase 2: SA from multiple starts
    num_sa = 12
    for i in range(num_sa):
        if time.time() - start_time > time_limit:
            break

        if i == 0 and best_valid_state is not None:
            state = best_valid_state[:]
            s3 = best_valid_count
            s4 = 0
        elif i == 1:
            state = greedy_init()
            s3 = count_3aps(state)
            s4 = count_4aps(state)
        else:
            state = greedy_init_random()
            s3 = count_3aps(state)
            s4 = count_4aps(state)

        # Perturb
        for _ in range(random.randint(5, 15)):
            x = random.randint(0, n - 1)
            state[x] = not state[x]

        s3 = count_3aps(state)
        s4 = count_4aps(state)

        state2, s3_best = simulated_annealing(state, s3, s4, steps=15000)
        if s3_best > best_valid_count:
            best_valid_count = s3_best
            best_valid_state = state2[:]

        # Also try hill climb on the SA result
        if s3_best > best_valid_count:
            s4_check = count_4aps(state2)
            if s4_check == 0:
                state2, s3_hc, s4_hc, _ = hill_climb(state2, s3_best, 0)
                if s4_hc == 0 and s3_hc > best_valid_count:
                    best_valid_count = s3_hc
                    best_valid_state = state2[:]

    # Phase 3: More hill climbing from best found
    if best_valid_state is not None and time.time() - start_time < time_limit:
        for _ in range(3):
            state = best_valid_state[:]
            # Small perturbation
            for _ in range(random.randint(3, 8)):
                x = random.randint(0, n - 1)
                state[x] = not state[x]
            s3 = count_3aps(state)
            s4 = count_4aps(state)
            if s4 == 0:
                state, s3, s4, _ = hill_climb(state, s3, s4)
                if s4 == 0 and s3 > best_valid_count:
                    best_valid_count = s3
                    best_valid_state = state[:]
            elif s4 > 0:
                # Try to fix 4-APs by removing elements
                for _ in range(20):
                    if count_4aps(state) == 0:
                        break
                    # Remove a random element that's in a 4-AP
                    found = False
                    for a, b, c, d in ap4s:
                        if state[a] and state[b] and state[c] and state[d]:
                            rem = random.choice([a, b, c, d])
                            state[rem] = False
                            found = True
                            break
                    if not found:
                        break
                s3 = count_3aps(state)
                s4 = count_4aps(state)
                if s4 == 0:
                    state, s3, s4, _ = hill_climb(state, s3, s4)
                    if s4 == 0 and s3 > best_valid_count:
                        best_valid_count = s3
                        best_valid_state = state[:]

    # Fallback
    if best_valid_state is None or not any(best_valid_state):
        return [1]

    result = [i + 1 for i in range(n) if best_valid_state[i]]
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res), res[:10])
