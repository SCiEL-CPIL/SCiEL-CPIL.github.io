"""Rewritten program: simulated annealing on reciprocal Schur triple constraint."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 75.0

    N = 180
    # Precompute all forbidden triples (a, b, c) with distinct elements where 1/a = 1/b + 1/c
    # 1/a = 1/b + 1/c  <=>  bc = a(b+c)
    forbidden_idx = []
    for a in range(1, N + 1):
        for b in range(1, N + 1):
            if b == a:
                continue
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if b * c == a * (b + c):
                    forbidden_idx.append((a - 1, b - 1, c - 1))

    # Build list of triples per element
    elem_triples = [[] for _ in range(N)]
    for idx, (a, b, c) in enumerate(forbidden_idx):
        elem_triples[a].append(idx)
        elem_triples[b].append(idx)
        elem_triples[c].append(idx)

    def count_violations(present):
        count = 0
        for (a, b, c) in forbidden_idx:
            if present[a] and present[b] and present[c]:
                count += 1
        return count

    # Initial solution: [91, 180]
    present = [False] * N
    for i in range(90, N):
        present[i] = True
    assert count_violations(present) == 0

    best_present = present[:]
    best_size = 90

    # Simulated annealing
    current = best_present[:]
    current_violations = 0
    current_size = 90

    T = 10.0
    cooling = 0.9998
    min_T = 0.01

    random.seed(42)

    while T > min_T:
        if time.time() > deadline:
            break

        i = random.randint(0, N - 1)
        old_state = current[i]
        current[i] = not current[i]

        # Compute delta in violations
        delta_v = 0
        for t_idx in elem_triples[i]:
            a, b, c = forbidden_idx[t_idx]
            count_now = current[a] + current[b] + current[c]
            if old_state:
                # i was present, now absent
                count_before = count_now + 1
                if count_before == 3:
                    delta_v -= 1
            else:
                # i was absent, now present
                if count_now == 3:
                    delta_v += 1

        new_violations = current_violations + delta_v
        new_size = current_size + (1 if not old_state else -1)

        # Objective: minimize violations, maximize size
        # Use a combined score: score = size - violations * 1000
        old_score = current_size - current_violations * 1000
        new_score = new_size - new_violations * 1000

        if new_score >= old_score:
            accept = True
        else:
            delta = old_score - new_score  # positive means worse
            # Accept with probability exp(-delta / T)
            if delta < T * 50:  # avoid overflow
                import math
                accept = random.random() < math.exp(-delta / T)
            else:
                accept = False

        if accept:
            current_violations = new_violations
            current_size = new_size
            if current_violations == 0 and current_size > best_size:
                best_size = current_size
                best_present = current[:]
        else:
            current[i] = old_state

        T *= cooling

    # Verify best solution
    if count_violations(best_present) != 0:
        best_present = [False] * N
        for i in range(90, N):
            best_present[i] = True
        best_size = 90

    return [i + 1 for i in range(N) if best_present[i]]
# EVOLVE-BLOCK-END
