import time
import random
import numpy as np

def _square_nonzero_count(coeffs):
    n = len(coeffs)
    size = 2 * n - 1
    sq = np.zeros(size, dtype=np.int64)
    for i in range(n):
        if coeffs[i] == 0:
            continue
        ci = coeffs[i]
        for j in range(i, n):
            if coeffs[j] == 0:
                continue
            cj = coeffs[j]
            val = ci * cj
            if i == j:
                sq[2 * i] += val
            else:
                sq[i + j] += 2 * val
    return int(np.count_nonzero(sq))

def construct():
    start_time = time.time()
    deadline = start_time + 70.0

    best_count = 23
    best_poly = None

    def random_poly():
        coeffs = [0] * 41
        positions = sorted(random.sample(range(41), 12))
        for pos in positions:
            coeffs[pos] = random.choice([-2, -1, 1, 2])
        return coeffs

    # Structured candidates
    structured = []
    for start in range(0, 10):
        for diff in [2, 3, 4, 5, 6, 7, 8, 10]:
            coeffs = [0] * 41
            count = 0
            for k in range(12):
                pos = start + k * diff
                if pos < 41:
                    coeffs[pos] = 1
                    count += 1
            if count == 12:
                structured.append(coeffs)

    for start in range(0, 10):
        for diff in [3, 5, 7]:
            coeffs = [0] * 41
            count = 0
            for k in range(12):
                pos = start + k * diff
                if pos < 41:
                    coeffs[pos] = 1 if k % 2 == 0 else -1
                    count += 1
            if count == 12:
                structured.append(coeffs)

    all_candidates = structured + [random_poly() for _ in range(30)]

    for cand in all_candidates:
        cnt = _square_nonzero_count(cand)
        if cnt < best_count:
            best_count = cnt
            best_poly = cand[:]

    def local_search(poly, max_iter=15000):
        nonlocal best_count, best_poly
        current = poly[:]
        current_count = _square_nonzero_count(current)

        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if current_count < best_count:
                best_count = current_count
                best_poly = current[:]

            pos = random.randint(0, 40)
            old_val = current[pos]
            new_val = random.choice([-2, -1, 0, 1, 2])
            if new_val == old_val:
                continue

            current[pos] = new_val
            nonzero = sum(1 for c in current if c != 0)
            if nonzero != 12:
                current[pos] = old_val
                continue

            new_count = _square_nonzero_count(current)
            if new_count <= current_count:
                current_count = new_count
            else:
                current[pos] = old_val

        return current, current_count

    for _ in range(80):
        if time.time() > deadline:
            break

        if best_poly is not None and random.random() < 0.6:
            start = best_poly[:]
            for _ in range(random.randint(1, 2)):
                pos = random.randint(0, 40)
                start[pos] = random.choice([-2, -1, 0, 1, 2])
        else:
            start = random_poly()

        nonzero = sum(1 for c in start if c != 0)
        if nonzero != 12:
            start = random_poly()

        local_search(start, max_iter=12000)

    if best_poly is None:
        best_poly = [1] * 12

    nonzero = sum(1 for c in best_poly if c != 0)
    if nonzero != 12:
        best_poly = [1] * 12

    return best_poly
