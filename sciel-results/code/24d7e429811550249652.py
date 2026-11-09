import time
import random

def construct():
    N = 1000
    max_val = 1000
    deadline = time.time() + 145

    best_size = 0
    best_S = []

    def solve_one(seed_val):
        rng = random.Random(seed_val)

        S = []
        in_S = [False] * (max_val + 1)
        counts = [0] * (2 * max_val + 1)

        def try_add(x):
            if counts[2 * x] >= 2:
                return False
            for y in S:
                if y != x and counts[x + y] >= 2:
                    return False
            S.append(x)
            in_S[x] = True
            counts[2 * x] += 1
            for y in S[:-1]:
                counts[x + y] += 1
            return True

        def try_remove(x):
            idx = S.index(x)
            S.pop(idx)
            in_S[x] = False
            counts[2 * x] -= 1
            for y in S:
                if y != x:
                    counts[x + y] -= 1

        def can_add(x):
            if counts[2 * x] >= 2:
                return False
            for y in S:
                if y != x and counts[x + y] >= 2:
                    return False
            return True

        def conflict_score(x):
            score = counts[2 * x]
            for y in S:
                if y != x:
                    score += counts[x + y]
            return score

        def is_valid_swap(a, c):
            delta = {}
            delta[2 * a] = delta.get(2 * a, 0) - 1
            for y in S:
                if y != a:
                    s = a + y
                    delta[s] = delta.get(s, 0) - 1
            delta[2 * c] = delta.get(2 * c, 0) + 1
            for y in S:
                if y != a:
                    s = c + y
                    delta[s] = delta.get(s, 0) + 1
            for s, d in delta.items():
                if counts[s] + d > 2 or counts[s] + d < 0:
                    return False
            return True

        # Smart greedy with conflict scoring
        unused = set(range(1, max_val + 1))
        while unused and time.time() < deadline:
            best_x = None
            best_score = float('inf')
            for x in unused:
                if not can_add(x):
                    continue
                score = conflict_score(x)
                if score < best_score:
                    best_score = score
                    best_x = x
            if best_x is None:
                break
            try_add(best_x)
            unused.remove(best_x)

        # Local search
        no_improve = 0
        while time.time() < deadline and no_improve < 100:
            improved = False
            unused_list = [x for x in range(1, max_val + 1) if not in_S[x]]
            rng.shuffle(unused_list)

            for c in unused_list:
                if try_add(c):
                    improved = True
                    break

            if improved:
                no_improve = 0
                continue

            if S and unused_list:
                for _ in range(500):
                    a = rng.choice(S)
                    c = rng.choice(unused_list)
                    if is_valid_swap(a, c):
                        try_remove(a)
                        try_add(c)
                        improved = True
                        break

            if improved:
                no_improve = 0
                continue

            if len(S) > 10:
                k = rng.randint(2, 5)
                to_remove = rng.sample(S, k)
                for a in to_remove:
                    try_remove(a)

            no_improve += 1

        return S

    restart = 0
    while time.time() < deadline:
        S = solve_one(restart)
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)
        restart += 1

    return best_S
