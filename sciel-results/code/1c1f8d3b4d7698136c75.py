import time
import random

_DEADLINE = 50.0
_N = 256

def _build_conflict_bitmasks():
    """Build conflict as bitmasks for O(1) set intersection checks."""
    conflict = [0] * (_N + 1)
    for a in range(1, _N + 1):
        for b in range(a + 1, _N + 1):
            if (a * b) % (a + b) == 0:
                conflict[a] |= (1 << b)
                conflict[b] |= (1 << a)
    return conflict

def _validate(S):
    s = set(S)
    if len(s) != len(S):
        return False
    if any(x < 1 or x > _N for x in s):
        return False
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            a, b = S[i], S[j]
            if (a * b) % (a + b) == 0:
                return False
    return True

def _greedy_build(conflict, order, S_mask=0):
    """Greedy: add x if it doesn't conflict with any already-chosen element.
    Returns (S_set, S_mask)."""
    S = set()
    for x in order:
        if not (S_mask & conflict[x]):
            S.add(x)
            S_mask |= (1 << x)
    return S, S_mask

def _local_search(conflict, S_init, S_mask_init, time_limit, seed):
    """
    Local search on an independent set using bitmask conflict checks.
    Moves: add a vertex (if no conflict), swap (remove one, add one), multi-swap.
    Objective: maximize |S|.
    """
    rng = random.Random(seed)
    S = set(S_init)
    S_mask = S_mask_init
    best_S = set(S)
    best_size = len(S)
    start = time.time()

    all_vertices = list(range(1, _N + 1))
    # Precompute list of vertices not in S for faster sampling
    not_in_S = [v for v in all_vertices if v not in S]

    while time.time() - start < time_limit:
        improved = False

        # Try to add any vertex that has no conflict with S
        candidates_to_add = [v for v in not_in_S if not (S_mask & conflict[v])]
        if candidates_to_add:
            v = rng.choice(candidates_to_add)
            S.add(v)
            S_mask |= (1 << v)
            not_in_S.remove(v)
            improved = True
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)
            continue

        # Try swaps: remove u from S, add v not in S
        swap_found = False
        for _ in range(300):
            u = rng.choice(list(S))
            v = rng.choice(not_in_S)
            # Check if v conflicts with S \ {u}
            S_without_u_mask = S_mask & ~(1 << u)
            if not (S_without_u_mask & conflict[v]):
                S.discard(u)
                S_mask &= ~(1 << u)
                S.add(v)
                S_mask |= (1 << v)
                not_in_S.remove(v)
                not_in_S.append(u)
                swap_found = True
                break

        if swap_found:
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)
            continue

        # Try 2-swap: remove u1, u2 from S, add v1, v2 not in S
        two_swap_found = False
        if len(S) >= 2:
            for _ in range(100):
                u1, u2 = rng.sample(list(S), 2)
                v1, v2 = rng.sample(not_in_S, 2)
                S_without_mask = S_mask & ~(1 << u1) & ~(1 << u2)
                if not (S_without_mask & conflict[v1]) and not (S_without_mask & conflict[v2]):
                    # Also check v1 and v2 don't conflict with each other
                    if not (conflict[v1] & (1 << v2)):
                        S.discard(u1)
                        S.discard(u2)
                        S_mask &= ~(1 << u1) & ~(1 << u2)
                        S.add(v1)
                        S.add(v2)
                        S_mask |= (1 << v1) | (1 << v2)
                        not_in_S.remove(v1)
                        not_in_S.remove(v2)
                        not_in_S.extend([u1, u2])
                        two_swap_found = True
                        break

        if two_swap_found:
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)
            continue

        # If no improvement, do a small random perturbation
        if S:
            u = rng.choice(list(S))
            S.discard(u)
            S_mask &= ~(1 << u)
            not_in_S.append(u)
            # Add a random element if possible
            candidates = [v for v in not_in_S if not (S_mask & conflict[v])]
            if candidates:
                v = rng.choice(candidates)
                S.add(v)
                S_mask |= (1 << v)
                not_in_S.remove(v)

        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)

    return best_S

def _seeded_greedy_with_local_search(conflict, total_time):
    """Run multiple rounds: randomized greedy init + local search."""
    start = time.time()
    best_S = set()
    best_size = 0
    seed = 42

    while time.time() - start < total_time:
        # Phase 1: Randomized greedy initialization
        order = list(range(1, _N + 1))
        random.shuffle(order)
        S_init, S_mask_init = _greedy_build(conflict, order)

        # Phase 2: Local search on this initialization
        elapsed = time.time() - start
        remaining = total_time - elapsed
        if remaining < 0.5:
            break
        ls_time = min(remaining * 0.4, 3.0)
        S_ls = _local_search(conflict, S_init, S_mask_init, ls_time, seed)
        seed += 1

        if len(S_ls) > best_size:
            best_size = len(S_ls)
            best_S = set(S_ls)

        # Also try a "complement-focused" greedy: prioritize vertices with fewer conflicts
        elapsed = time.time() - start
        remaining = total_time - elapsed
        if remaining < 0.5:
            break

        deg_order = sorted(range(1, _N + 1), key=lambda v: bin(conflict[v]).count('1'))
        random.shuffle(deg_order)
        S_deg, S_mask_deg = _greedy_build(conflict, deg_order)
        if len(S_deg) > best_size:
            best_size = len(S_deg)
            best_S = set(S_deg)

        remaining = total_time - (time.time() - start)
        if remaining > 1.0:
            ls_time = min(remaining * 0.3, 2.0)
            S_deg_ls = _local_search(conflict, S_deg, S_mask_deg, ls_time, seed + 1000)
            seed += 1
            if len(S_deg_ls) > best_size:
                best_size = len(S_deg_ls)
                best_S = set(S_deg_ls)

    return best_S

def _try_add_to_set(conflict, S):
    """Try to add more elements to S by finding vertices with no conflict."""
    S_mask = 0
    for x in S:
        S_mask |= (1 << x)
    added = True
    while added:
        added = False
        for v in range(1, _N + 1):
            if v not in S and not (S_mask & conflict[v]):
                S.add(v)
                S_mask |= (1 << v)
                added = True
    return S

def construct():
    start = time.time()
    conflict = _build_conflict_bitmasks()

    # Quick baseline: all odds
    S_odd = set(range(1, _N + 1, 2))
    S_odd_mask = 0
    for x in S_odd:
        S_odd_mask |= (1 << x)
    for e in range(2, _N + 1, 2):
        if not (S_odd_mask & conflict[e]):
            S_odd.add(e)
            S_odd_mask |= (1 << e)
    baseline_size = len(S_odd)

    # Allocate time
    elapsed = time.time() - start
    remaining = _DEADLINE - elapsed - 2.0
    if remaining < 5:
        remaining = 5

    # Run the main search
    best_S = _seeded_greedy_with_local_search(conflict, remaining)

    # Final greedy fill
    best_S = _try_add_to_set(conflict, best_S)

    # Ensure baseline is at least as good
    if len(best_S) < baseline_size:
        best_S = S_odd
        best_S = _try_add_to_set(conflict, best_S)

    result = sorted(best_S)
    if not result:
        return [1]
    return result
