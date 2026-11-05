import time
import random
import numpy as np
from itertools import product

N = 12
MAX_THEORETICAL = (N - 1) ** 2  # 121

def _bfs_threshold(a_trans, b_trans):
    """
    Compute the exact shortest reset-word length for a binary deterministic automaton
    with N states. Returns -1 if not synchronizing.
    Uses BFS on the subset automaton with bitmasks.
    """
    size = 1 << N
    all_states = size - 1

    # Precompute transition for each subset under 'a' and 'b'
    # For each state i, a_trans[i] is the destination under 'a'
    # For a subset mask, the image under 'a' is the OR of (1 << a_trans[i]) for all i in mask
    a_bit = [1 << a_trans[i] for i in range(N)]
    b_bit = [1 << b_trans[i] for i in range(N)]

    trans_a = np.zeros(size, dtype=np.int32)
    trans_b = np.zeros(size, dtype=np.int32)

    for mask in range(1, size):
        ra = 0
        rb = 0
        m = mask
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            ra |= a_bit[i]
            rb |= b_bit[i]
            m ^= lsb
        trans_a[mask] = ra
        trans_b[mask] = rb

    # BFS from all_states
    visited = np.full(size, -1, dtype=np.int32)
    visited[all_states] = 0
    queue = [all_states]
    head = 0

    while head < len(queue):
        curr = queue[head]
        head += 1
        dist = visited[curr]

        # Check if singleton (power of 2)
        if curr and (curr & (curr - 1)) == 0:
            return int(dist)

        na = trans_a[curr]
        if visited[na] == -1:
            visited[na] = dist + 1
            queue.append(na)

        nb = trans_b[curr]
        if visited[nb] == -1:
            visited[nb] = dist + 1
            queue.append(nb)

    return -1

def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # Leave 10s safety margin for the 90s limit

    best_score = -1
    best_a = None
    best_b = None

    def try_candidate(a, b):
        nonlocal best_score, best_a, best_b
        if best_score == MAX_THEORETICAL:
            return True
        s = _bfs_threshold(a, b)
        if s > best_score:
            best_score = s
            best_a = list(a)
            best_b = list(b)
        return best_score == MAX_THEORETICAL

    # ---- Phase 1: Structured families (fast) ----

    # 1a. a = full cycle (shift by 1), b = identity with exactly 1 defect
    a_cycle = [(i + 1) % N for i in range(N)]
    for k in range(N):
        for j in range(N):
            if j == k:
                continue
            b = list(range(N))
            b[k] = j
            if try_candidate(a_cycle, b):
                return [best_a, best_b]
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    if time.time() > deadline:
        if best_a is None:
            best_a = a_cycle
            best_b = [0] * N
        return [best_a, best_b]

    # 1b. a = full cycle, b = identity with exactly 2 defects
    for k1 in range(N):
        for k2 in range(k1 + 1, N):
            for j1 in range(N):
                for j2 in range(N):
                    b = list(range(N))
                    b[k1] = j1
                    b[k2] = j2
                    if try_candidate(a_cycle, b):
                        return [best_a, best_b]
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    if time.time() > deadline:
        if best_a is None:
            best_a = a_cycle
            best_b = [0] * N
        return [best_a, best_b]

    # 1c. a = shift by d (d=2..6), b = identity with 1 defect
    for d in range(2, 7):
        a_shift = [(i + d) % N for i in range(N)]
        for k in range(N):
            for j in range(N):
                if j == k:
                    continue
                b = list(range(N))
                b[k] = j
                if try_candidate(a_shift, b):
                    return [best_a, best_b]
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    if time.time() > deadline:
        if best_a is None:
            best_a = a_cycle
            best_b = [0] * N
        return [best_a, best_b]

    # ---- Phase 2: Targeted random search with local refinement ----
    # Use a structured random approach: start from known good structures and perturb

    rng = np.random.default_rng(42)

    def random_structured():
        """Generate a random automaton biased toward synchronizing structures."""
        # Strategy: a is a random permutation or near-permutation,
        # b has at least one collision to enable synchronization
        a = rng.permutation(N).tolist()
        b = rng.integers(0, N, N).tolist()
        # Ensure at least one collision in b
        if len(set(b)) == N:
            b[rng.integers(N)] = b[rng.integers(N)]
        return a, b

    def local_refine(a, b, budget_sec=2.0):
        """
        Hill-climbing: try single-position changes to a and b,
        keep if score improves. Stop when budget exhausted or no improvement.
        """
        nonlocal best_score, best_a, best_b
        t_start = time.time()
        current_score = _bfs_threshold(a, b)
        cur_a = list(a)
        cur_b = list(b)

        if current_score > best_score:
            best_score = current_score
            best_a = list(cur_a)
            best_b = list(cur_b)

        no_improve_count = 0
        max_no_improve = 50

        while time.time() - t_start < budget_sec and no_improve_count < max_no_improve:
            if best_score == MAX_THEORETICAL:
                break

            improved = False
            # Try changing one position in a
            for i in range(N):
                old = cur_a[i]
                for new_val in range(N):
                    if new_val == old:
                        continue
                    cur_a[i] = new_val
                    s = _bfs_threshold(cur_a, cur_b)
                    if s > current_score:
                        current_score = s
                        if s > best_score:
                            best_score = s
                            best_a = list(cur_a)
                            best_b = list(cur_b)
                        improved = True
                        no_improve_count = 0
                        break
                    else:
                        cur_a[i] = old

            if improved:
                continue

            # Try changing one position in b
            for i in range(N):
                old = cur_b[i]
                for new_val in range(N):
                    if new_val == old:
                        continue
                    cur_b[i] = new_val
                    s = _bfs_threshold(cur_a, cur_b)
                    if s > current_score:
                        current_score = s
                        if s > best_score:
                            best_score = s
                            best_a = list(cur_a)
                            best_b = list(cur_b)
                        improved = True
                        no_improve_count = 0
                        break
                    else:
                        cur_b[i] = old

            if not improved:
                no_improve_count += 1

        return best_score == MAX_THEORETICAL

    # Phase 2a: Try random structured candidates with local refinement
    attempts = 0
    max_attempts = 200
    while attempts < max_attempts:
        if time.time() > deadline:
            break
        if best_score == MAX_THEORETICAL:
            break

        a, b = random_structured()
        # Quick check: is it even synchronizing?
        s = _bfs_threshold(a, b)
        if s >= 0:
            if s > best_score:
                best_score = s
                best_a = list(a)
                best_b = list(b)
            if best_score == MAX_THEORETICAL:
                break
            # Do local refinement if score is promising
            if s >= best_score - 5:
                if local_refine(a, b):
                    break
        attempts += 1

    if time.time() > deadline:
        if best_a is None:
            best_a = a_cycle
            best_b = [0] * N
        return [best_a, best_b]

    # ---- Phase 3: Simulated annealing-style search ----
    # Start from best found, do random walks with occasional restarts

    if best_a is not None and best_score > 0:
        cur_a = list(best_a)
        cur_b = list(best_b)
        cur_score = best_score
        t_start = time.time()
        sa_budget = deadline - time.time()
        if sa_budget > 5.0:
            rng2 = np.random.default_rng(123)
            while time.time() - t_start < sa_budget - 2.0:
                if best_score == MAX_THEORETICAL:
                    break

                # Random perturbation
                if rng2.random() < 0.7:
                    # Change one position in a
                    i = int(rng2.integers(N))
                    cur_a[i] = int(rng2.integers(N))
                else:
                    # Change one position in b
                    i = int(rng2.integers(N))
                    cur_b[i] = int(rng2.integers(N))

                s = _bfs_threshold(cur_a, cur_b)
                if s > cur_score:
                    cur_score = s
                    if s > best_score:
                        best_score = s
                        best_a = list(cur_a)
                        best_b = list(cur_b)
                else:
                    # Revert with some probability (accept worse to escape local opt)
                    if s < 0 or (s < cur_score and rng2.random() > 0.1):
                        # Revert
                        if rng2.random() < 0.7:
                            cur_a = list(best_a) if best_a is not None else a_cycle
                            cur_b = list(best_b) if best_b is not None else [0] * N
                            cur_score = best_score
                        else:
                            # Keep the worse state temporarily
                            pass

    # ---- Phase 4: Final fallback ----
    if best_a is None:
        best_a = a_cycle
        best_b = [0] * N
    # Ensure it's at least synchronizing
    final_score = _bfs_threshold(best_a, best_b)
    if final_score < 0:
        # Use a known synchronizing automaton
        best_a = a_cycle
        best_b = list(range(N))
        best_b[0] = 1  # one defect to ensure sync

    return [best_a, best_b]
