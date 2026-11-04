import time
import random
from itertools import permutations as itperms

def construct(n):
    start_time = time.time()
    deadline = start_time + 140

    if n <= 4:
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    all_perms = list(itperms(range(1, n + 1)))
    num_perms = len(all_perms)

    ov = [[0] * num_perms for _ in range(num_perms)]
    for i, a in enumerate(all_perms):
        for j, b in enumerate(all_perms):
            if i == j:
                ov[i][j] = n
            else:
                k = 0
                for kk in range(1, n + 1):
                    if a[n-kk:] == b[:kk]:
                        k = kk
                    else:
                        break
                ov[i][j] = k

    order = list(range(num_perms))
    random.shuffle(order)

    def full_cost(order):
        c = n
        for i in range(num_perms - 1):
            c += n - ov[order[i]][order[i+1]]
        return c

    current_cost = full_cost(order)
    best_cost = current_cost
    best_order = list(order)

    T = 3.0
    T_min = 0.05
    cooling = 0.99995

    while time.time() < deadline:
        T = max(T_min, T * cooling)

        i = random.randint(0, num_perms - 2)
        B = order[i]
        C = order[i+1]

        delta = 0
        if i > 0:
            A = order[i-1]
            delta += (n - ov[A][C]) - (n - ov[A][B])
        delta += (n - ov[C][B]) - (n - ov[B][C])
        if i + 2 < num_perms:
            D = order[i+2]
            delta += (n - ov[B][D]) - (n - ov[C][D])

        if delta < 0 or random.random() < pow(2.718281828, -delta / max(T, 0.001)):
            order[i], order[i+1] = C, B
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_order = list(order)

    word = list(all_perms[best_order[0]])
    for i in range(1, num_perms):
        ov_len = ov[best_order[i-1]][best_order[i]]
        word.extend(all_perms[best_order[i]][ov_len:])

    return "".join(map(str, word))
