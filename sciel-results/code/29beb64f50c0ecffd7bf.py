import time
import random
import math
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1
DEADLINE_SECONDS = 85.0


def _build_subset_trans(trans):
    size = 1 << N
    next_subset = [0] * size
    for mask in range(1, size):
        lsb = mask & (-mask)
        i = lsb.bit_length() - 1
        rest = mask ^ lsb
        next_subset[mask] = next_subset[rest] | (1 << trans[i])
    return next_subset


def compute_threshold(trans_a, trans_b):
    """BFS on subset automaton; returns exact shortest reset word length, 0 if not synchronizing."""
    next_a = _build_subset_trans(trans_a)
    next_b = _build_subset_trans(trans_b)
    size = 1 << N
    dist = [-1] * size
    dist[FULL_MASK] = 0
    queue = deque([FULL_MASK])
    while queue:
        cur = queue.popleft()
        if cur & (cur - 1) == 0:
            return dist[cur]
        nxt_a = next_a[cur]
        if dist[nxt_a] == -1:
            dist[nxt_a] = dist[cur] + 1
            queue.append(nxt_a)
        nxt_b = next_b[cur]
        if dist[nxt_b] == -1:
            dist[nxt_b] = dist[cur] + 1
            queue.append(nxt_b)
    return 0


def _construct_cycle_compress(n, e, t, w):
    a = [(i + 1) % n for i in range(n)]
    b = [0] * n
    for i in range(w):
        b[i] = i
    b[e] = t
    return a, b


def _sa_search(start_a, start_b, rng, deadline, best_holder):
    """Simulated Annealing over the full transition space."""
    n = N
    cur_a = start_a[:]
    cur_b = start_b[:]
    cur_score = compute_threshold(cur_a, cur_b)
    if cur_score == 0:
        return

    best_a = cur_a[:]
    best_b = cur_b[:]
    best_score = cur_score
    if best_score > best_holder[0]:
        best_holder[0] = best_score
        best_holder[1] = best_a[:]
        best_holder[2] = best_b[:]

    # SA parameters
    T_start = 30.0
    T_end = 0.5
    total_iterations = 0
    iterations_per_check = 200
    
    # Estimate how many iterations we can do
    # Each iteration: 1-2 random changes + 1-2 threshold computations
    # Threshold computation is ~fast for N=12
    # Let's just run until deadline
    
    while time.time() < deadline:
        # Geometric cooling
        total_iterations += 1
        # Use a cooling schedule that reaches T_end after some fraction of budget
        # We'll use a dynamic approach: cool based on elapsed time
        elapsed = time.time() - (deadline - (deadline - time.time()))  # not quite right
        # Simpler: just use a fixed number of iterations with geometric cooling
        # Let's do: T = T_start * (T_end/T_start)^(iteration / total_expected)
        # But we don't know total_expected. Use time-based cooling.
        
        # Time-based cooling
        time_elapsed = time.time() - (deadline - 85.0)  # approximate
        # Actually let's just track iterations and use a reasonable schedule
        # For simplicity, use a fixed schedule assuming ~5000 iterations max
        T = T_start * (T_end / T_start) ** (total_iterations / 5000.0)
        if T < T_end:
            T = T_end
        
        # Generate a move: change 1-2 random entries
        num_changes = rng.random() < 0.3 and 2 or 1
        changes = []
        for _ in range(num_changes):
            if rng.random() < 0.5:
                idx = rng.randint(0, n - 1)
                old_val = cur_a[idx]
                new_val = rng.randint(0, n - 1)
                if new_val != old_val:
                    changes.append(('a', idx, old_val, new_val))
            else:
                idx = rng.randint(0, n - 1)
                old_val = cur_b[idx]
                new_val = rng.randint(0, n - 1)
                if new_val != old_val:
                    changes.append(('b', idx, old_val, new_val))
        
        if not changes:
            continue
        
        # Apply changes
        for (letter, idx, old_val, new_val) in changes:
            if letter == 'a':
                cur_a[idx] = new_val
            else:
                cur_b[idx] = new_val
        
        new_score = compute_threshold(cur_a, cur_b)
        
        # Accept or reject
        if new_score > cur_score:
            # Always accept improvement
            cur_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_a = cur_a[:]
                best_b = cur_b[:]
                if best_score > best_holder[0]:
                    best_holder[0] = best_score
                    best_holder[1] = best_a[:]
                    best_holder[2] = best_b[:]
        else:
            # Metropolis acceptance
            delta = new_score - cur_score  # negative
            if delta == 0:
                # Accept with some probability to explore
                if rng.random() < 0.3:
                    cur_score = new_score
                else:
                    # Revert
                    for (letter, idx, old_val, new_val) in changes:
                        if letter == 'a':
                            cur_a[idx] = old_val
                        else:
                            cur_b[idx] = old_val
            else:
                # delta < 0
                accept_prob = math.exp(delta / T)
                if rng.random() < accept_prob:
                    cur_score = new_score
                else:
                    # Revert
                    for (letter, idx, old_val, new_val) in changes:
                        if letter == 'a':
                            cur_a[idx] = old_val
                        else:
                            cur_b[idx] = old_val


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    rng = random.Random(42)

    best_holder = [0, None, None]  # [score, a, b]

    # Phase 1: Find best in cycle+compress family
    n = N
    best_family = None
    best_family_score = 0
    for w in range(1, n // 2 + 1):
        for e in range(w, n):
            for t in range(w, n):
                if t == e:
                    continue
                # Analytic estimate
                if e >= w and t >= w:
                    score = (n - 1) - w - ((e - t - 1) % n)
                    if score > best_family_score:
                        best_family_score = score
                        best_family = (e, t, w)

    # Verify top candidates with real BFS
    candidates = []
    for w in range(1, n // 2 + 1):
        for e in range(w, n):
            for t in range(w, n):
                if t == e:
                    continue
                if e >= w and t >= w:
                    score = (n - 1) - w - ((e - t - 1) % n)
                    candidates.append((score, e, t, w))
    candidates.sort(reverse=True)
    
    for score_est, e, t, w in candidates[:20]:  # Check top 20
        a, b = _construct_cycle_compress(n, e, t, w)
        real = compute_threshold(a, b)
        if real > best_holder[0]:
            best_holder[0] = real
            best_holder[1] = a[:]
            best_holder[2] = b[:]
        if time.time() > deadline - 10.0:
            break

    # Phase 2: Simulated Annealing from best found
    if best_holder[1] is not None and time.time() < deadline - 5.0:
        _sa_search(best_holder[1], best_holder[2], rng, deadline - 5.0, best_holder)

    # Phase 3: Try SA from a few other good starting points
    if time.time() < deadline - 5.0:
        for score_est, e, t, w in candidates[20:40]:
            if time.time() > deadline - 5.0:
                break
            a, b = _construct_cycle_compress(n, e, t, w)
            real = compute_threshold(a, b)
            if real > 0:
                _sa_search(a, b, rng, deadline - 5.0, best_holder)

    # Fallback: guaranteed synchronizing automaton
    if best_holder[1] is None or best_holder[0] == 0:
        a = [(i + 1) % n for i in range(n)]
        b = [0] * n
        return [a, b]

    return [best_holder[1], best_holder[2]]


if __name__ == "__main__":
    a, b = construct()
    print("Best score:", compute_threshold(a, b))
    print("A:", a)
    print("B:", b)
