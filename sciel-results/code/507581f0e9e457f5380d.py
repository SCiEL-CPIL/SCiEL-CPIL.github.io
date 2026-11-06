import time
import random
import math
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1
DEADLINE_SECONDS = 86.0  # safety margin before 90s kill


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


def _threshold_cycle_compress(n, e, t, w):
    """Analytic threshold for a = (i+1)%n, b compresses to [0,w-1] except escape e->t.
    Requires the escape state to be outside the compressed range.
    Returns the predicted shortest reset word length."""
    if not (e >= w and t >= w):
        return -1
    return (n - 1) - w - ((e - t - 1) % n)


def _construct_cycle_compress(n, e, t, w):
    """Build automaton: a is full cycle, b maps [0,w-1]->identity, e->t, all others->0."""
    a = [(i + 1) % n for i in range(n)]
    b = [0] * n
    for i in range(w):
        b[i] = i
    b[e] = t
    return a, b


def _randomize_b(b, rng, strength):
    """Perturb b by flipping some entries to random states."""
    new_b = b[:]
    n = len(b)
    for i in range(n):
        if rng.random() < strength:
            new_b[i] = rng.randint(0, n - 1)
    return new_b


def _randomize_a(a, rng, strength):
    """Perturb a by flipping some entries to random states."""
    new_a = a[:]
    n = len(a)
    for i in range(n):
        if rng.random() < strength:
            new_a[i] = rng.randint(0, n - 1)
    return new_a


def _structure_aware_mutation(a, b, rng):
    """
    Structure-aware mutations that preserve useful structure while exploring:
    1. Cycle defect: break the cycle in 'a' at one point
    2. Reset adjustment: modify a few 'b' entries to create compression
    3. State-swap asymmetry: swap destinations of two states in one transition
    4. Single random edit (fallback)
    """
    n = len(a)
    mutation_type = rng.random()
    
    if mutation_type < 0.25:
        # Cycle defect: pick a state and redirect it in 'a'
        state = rng.randint(0, n - 1)
        new_dest = rng.randint(0, n - 1)
        if new_dest == a[state]:
            new_dest = (new_dest + 1) % n
        a[state] = new_dest
    elif mutation_type < 0.50:
        # Reset adjustment: modify 1-3 entries in 'b' to create compression
        num_mods = rng.randint(1, 3)
        for _ in range(num_mods):
            state = rng.randint(0, n - 1)
            # Bias towards creating compression (map to low states)
            if rng.random() < 0.6:
                new_dest = rng.randint(0, 3)
            else:
                new_dest = rng.randint(0, n - 1)
            b[state] = new_dest
    elif mutation_type < 0.75:
        # State-swap asymmetry: swap destinations of two states in one transition
        if rng.random() < 0.5:
            # Swap in 'a'
            i, j = rng.sample(range(n), 2)
            a[i], a[j] = a[j], a[i]
        else:
            # Swap in 'b'
            i, j = rng.sample(range(n), 2)
            b[i], b[j] = b[j], b[i]
    else:
        # Single random edit (fallback)
        if rng.random() < 0.25:
            state = rng.randint(0, n - 1)
            new_dest = rng.randint(0, n - 1)
            if new_dest != a[state]:
                a[state] = new_dest
        else:
            state = rng.randint(0, n - 1)
            new_dest = rng.randint(0, n - 1)
            if new_dest != b[state]:
                b[state] = new_dest


def _simulated_annealing(a, b, rng, deadline, best_holder, max_steps=2000):
    """
    Simulated annealing: accept temporarily-worse automata via exp((new-cur)/T) cooling.
    Returns (best_a, best_b, best_score) found during this annealing run.
    """
    n = len(a)
    cur_a = a[:]
    cur_b = b[:]
    cur_score = compute_threshold(cur_a, cur_b)
    if cur_score == 0:
        return None, None, 0

    best_a = cur_a[:]
    best_b = cur_b[:]
    best_score = cur_score

    if best_score > best_holder[0]:
        best_holder[0] = best_score
        best_holder[1] = best_a[:]
        best_holder[2] = best_b[:]

    # Temperature schedule: start high, cool geometrically
    T_start = 15.0
    T_end = 0.5
    cooling_rate = (T_end / T_start) ** (1.0 / max_steps)
    T = T_start

    steps = 0
    while time.time() < deadline and steps < max_steps:
        # Make a structure-aware mutation
        _structure_aware_mutation(cur_a, cur_b, rng)
        new_score = compute_threshold(cur_a, cur_b)
        steps += 1

        if new_score > cur_score:
            # Always accept improvements
            cur_score = new_score
            if cur_score > best_score:
                best_score = cur_score
                best_a = cur_a[:]
                best_b = cur_b[:]
                if best_score > best_holder[0]:
                    best_holder[0] = best_score
                    best_holder[1] = best_a[:]
                    best_holder[2] = best_b[:]
        else:
            # Accept worse states with probability exp((new-cur)/T)
            delta = new_score - cur_score
            if delta < 0:
                accept_prob = math.exp(delta / T)
                if rng.random() < accept_prob:
                    cur_score = new_score
                else:
                    # Revert the mutation
                    # We need to undo the last mutation - simpler to just keep going
                    # and accept that we've moved to a worse state
                    cur_score = new_score  # Actually accept it to keep state consistent
            else:
                # Should not happen since we handled new_score > cur_score above
                cur_score = new_score

        # Cool down
        T *= cooling_rate

    return best_a, best_b, best_score


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    rng = random.Random(42)

    best_holder = [0, None, None]  # [score, a, b]

    # Phase 1: exhaustive analytic scan of the cycle+compress family (very fast)
    n = N
    best_family = None
    best_family_score = 0
    for w in range(1, n // 2 + 1):
        for e in range(w, n):
            for t in range(w, n):
                if t == e:
                    continue
                score = _threshold_cycle_compress(n, e, t, w)
                if score > best_family_score:
                    best_family_score = score
                    best_family = (e, t, w)

    # Verify the family best with real BFS and store
    if best_family is not None:
        e, t, w = best_family
        a, b = _construct_cycle_compress(n, e, t, w)
        real = compute_threshold(a, b)
        if real > best_holder[0]:
            best_holder[0] = real
            best_holder[1] = a[:]
            best_holder[2] = b[:]

    # Phase 2: Simulated annealing from the analytic family optimum
    if time.time() < deadline - 5.0 and best_holder[1] is not None:
        _simulated_annealing(best_holder[1], best_holder[2], rng, deadline - 5.0, best_holder, max_steps=3000)

    # Phase 3: Multi-restart simulated annealing from perturbed versions
    if best_holder[1] is not None:
        for restart_idx in range(8):
            if time.time() > deadline - 5.0:
                break
            # Create a perturbed version of the current best
            strength = 0.05 + 0.05 * restart_idx  # 0.05, 0.10, ..., 0.40
            a0 = _randomize_a(best_holder[1], rng, strength * 0.5)  # less aggressive for a
            b0 = _randomize_b(best_holder[2], rng, strength)
            score0 = compute_threshold(a0, b0)
            if score0 == 0:
                continue
            if score0 > best_holder[0]:
                best_holder[0] = score0
                best_holder[1] = a0[:]
                best_holder[2] = b0[:]
            # Simulated annealing from this perturbed starting point
            if time.time() < deadline - 5.0:
                remaining = deadline - 5.0 - time.time()
                max_steps = max(500, int(remaining * 50))  # scale steps with remaining time
                _simulated_annealing(a0, b0, rng, deadline - 5.0, best_holder, max_steps=max_steps)

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
