import time
import random
import numpy as np

def _score(s):
    """Count distinct d > 0 such that a, a+d, a+2d all in s for some a."""
    s_set = set(s)
    min_s = min(s)
    max_s = max(s)
    max_d = (max_s - min_s) // 2
    count = 0
    for d in range(1, max_d + 1):
        found = False
        for a in s:
            if a + d in s_set and a + 2 * d in s_set:
                found = True
                break
        if found:
            count += 1
    return count

def _random_init():
    return sorted(random.sample(range(1, 513), 32))

def _structured_init():
    strategies = []
    base = random.randint(1, 20)
    step = random.randint(15, 18)
    s = [base + i * step for i in range(32)]
    if max(s) <= 512:
        strategies.append(s)
    base1 = random.randint(1, 10)
    base2 = base1 + random.randint(1, 10)
    step1 = random.randint(16, 20)
    s = []
    for i in range(16):
        s.append(base1 + i * step1)
        s.append(base2 + i * step1)
    s = [x for x in s if 1 <= x <= 512]
    if len(set(s)) >= 32:
        strategies.append(sorted(list(set(s)))[:32])
    clusters = []
    for _ in range(4):
        center = random.randint(30, 480)
        for _ in range(8):
            v = center + random.randint(-15, 15)
            if 1 <= v <= 512:
                clusters.append(v)
    clusters = list(set(clusters))
    while len(clusters) < 32:
        clusters.append(random.randint(1, 512))
    clusters = list(set(clusters))[:32]
    if len(clusters) == 32:
        strategies.append(sorted(clusters))
    if strategies:
        return random.choice(strategies)
    return _random_init()

def _anneal(initial, deadline):
    s = sorted(initial)
    current_score = _score(s)
    best_s = list(s)
    best_score = current_score
    T = 10.0
    T_min = 0.01
    alpha = 0.9995
    max_iter = 500000
    for it in range(max_iter):
        if time.time() > deadline:
            break
        move_type = random.random()
        if move_type < 0.5:
            i, j = random.sample(range(32), 2)
            new_s = list(s)
            new_s[i], new_s[j] = new_s[j], new_s[i]
            new_s = sorted(new_s)
        elif move_type < 0.8:
            i = random.randint(0, 31)
            old_val = s[i]
            new_val = random.randint(1, 512)
            if new_val == old_val:
                continue
            new_s = list(s)
            new_s.pop(i)
            new_s.append(new_val)
            new_s = sorted(new_s)
            if len(set(new_s)) != 32:
                continue
        else:
            i, j = random.sample(range(32), 2)
            new_s = list(s)
            new_s[i] += random.choice([-1, 1]) * random.randint(1, 5)
            new_s[j] += random.choice([-1, 1]) * random.randint(1, 5)
            if not all(1 <= x <= 512 for x in new_s):
                continue
            new_s = sorted(new_s)
            if len(set(new_s)) != 32:
                continue
        new_score = _score(new_s)
        delta = new_score - current_score
        if delta > 0 or random.random() < np.exp(delta / max(T, 1e-10)):
            s = new_s
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_s = list(s)
        T *= alpha
        if T < T_min:
            T = T_min
    return best_s, best_score

def construct():
    deadline = time.time() + 140.0
    best_overall = None
    best_overall_score = 0
    n_restarts = 20
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        if r == 0:
            init = _structured_init()
        elif r == 1:
            init = _random_init()
        elif r == 2:
            init = sorted(random.sample(range(1, 256), 16) + random.sample(range(257, 513), 16))
        else:
            init = _random_init() if random.random() < 0.5 else _structured_init()
        remaining = deadline - time.time()
        sub_deadline = time.time() + max(remaining / (n_restarts - r), 2.0)
        result, score = _anneal(init, sub_deadline)
        if score > best_overall_score:
            best_overall_score = score
            best_overall = result
    if best_overall is not None and time.time() < deadline - 5.0:
        remaining = deadline - time.time() - 5.0
        if remaining > 2.0:
            final_result, final_score = _anneal(best_overall, time.time() + remaining)
            if final_score > best_overall_score:
                best_overall = final_result
                best_overall_score = final_score
    if best_overall is None:
        return list(range(1, 33))
    return best_overall

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
