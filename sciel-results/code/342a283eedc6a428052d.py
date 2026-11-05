import time
import random

_DEADLINE = 50.0
_N = 256

def _build_conflict_graph():
    """Build adjacency sets: conflict[a] = set of b != a such that (a*b)%(a+b)==0."""
    conflict = [set() for _ in range(_N + 1)]
    for a in range(1, _N + 1):
        for b in range(a + 1, _N + 1):
            if (a * b) % (a + b) == 0:
                conflict[a].add(b)
                conflict[b].add(a)
    return conflict

def _validate(S):
    s = set(S)
    if len(s) != len(S):
        return False
    if any(x < 1 or x > _N for x in s):
        return False
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            a, b = S[i], S[j]
            if (a * b) % (a + b) == 0:
                return False
    return True

def _greedy_build(conflict, order):
    """Greedy: add x if it doesn't conflict with any already-chosen element."""
    S = set()
    for x in order:
        if not (S & conflict[x]):
            S.add(x)
    return S

def _local_search(conflict, S, time_limit, seed):
    """
    Local search on an independent set.
    Moves: add a vertex (if no conflict), remove a vertex, or swap (remove one, add one).
    Objective: maximize |S|.
    """
    rng = random.Random(seed)
    S = set(S)
    best_S = set(S)
    best_size = len(S)
    start = time.time()

    # Precompute: for each vertex, its conflict count within S
    # We'll do simple neighborhood exploration
    all_vertices = list(range(1, _N + 1))

    while time.time() - start < time_limit:
        improved = False

        # Try to add any vertex that has no conflict with S
        candidates_to_add = [v for v in all_vertices if v not in S and not (S & conflict[v])]
        if candidates_to_add:
            # Add the one with fewest conflicts (to leave room)
            # Actually just add one at random for diversity
            v = rng.choice(candidates_to_add)
            S.add(v)
            improved = True
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)

        # Try swaps: remove u from S, add v not in S
        if not improved:
            # Sample a few random swaps
            for _ in range(200):
                u = rng.choice(list(S))
                v = rng.choice(all_vertices)
                if v in S:
                    continue
                # Check if v conflicts with S \ {u}
                new_S = S - {u}
                if not (new_S & conflict[v]):
                    S = new_S | {v}
                    improved = True
                    break

        # Try: remove a "high-conflict" vertex to free up space, then try to add two
        if not improved and len(S) > 2:
            # Find vertex in S with most conflicts within S
            max_conf = -1
            max_v = None
            for u in S:
                c = len(S & conflict[u])
                if c > max_conf:
                    max_conf = c
                    max_v = u
            if max_v is not None and max_conf >= 2:
                S.discard(max_v)
                # Now try to add two vertices
                remaining = [v for v in all_vertices if v not in S and not (S & conflict[v])]
                if len(remaining) >= 2:
                    # Greedily add up to 2
                    added = 0
                    rng.shuffle(remaining)
                    for v in remaining:
                        if not (S & conflict[v]):
                            S.add(v)
                            added += 1
                            if added >= 2:
                                break
                    if added >= 1:
                        improved = True

        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)

        # If no improvement, do a small random perturbation
        if not improved:
            # Remove a random element
            if S:
                S.discard(rng.choice(list(S)))
            # Add a random element if possible
            candidates = [v for v in all_vertices if v not in S and not (S & conflict[v])]
            if candidates:
                S.add(rng.choice(candidates))

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
        S_init = _greedy_build(conflict, order)

        # Phase 2: Local search on this initialization
        # Allocate a fraction of remaining time to local search
        elapsed = time.time() - start
        remaining = total_time - elapsed
        if remaining < 0.5:
            break
        ls_time = min(remaining * 0.4, 3.0)  # Up to 3s per local search, 40% of remaining
        S_ls = _local_search(conflict, S_init, ls_time, seed)
        seed += 1

        if len(S_ls) > best_size:
            best_size = len(S_ls)
            best_S = set(S_ls)

        # Also try a "complement-focused" greedy: prioritize vertices with fewer conflicts
        elapsed = time.time() - start
        remaining = total_time - elapsed
        if remaining < 0.5:
            break

        # Sort by degree (number of conflicts) ascending - prefer low-degree vertices
        deg_order = sorted(range(1, _N + 1), key=lambda v: len(conflict[v]))
        random.shuffle(deg_order)  # slight randomness
        S_deg = _greedy_build(conflict, deg_order)
        if len(S_deg) > best_size:
            best_size = len(S_deg)
            best_S = set(S_deg)

        # Local search on degree-ordered greedy
        remaining = total_time - (time.time() - start)
        if remaining > 1.0:
            ls_time = min(remaining * 0.3, 2.0)
            S_deg_ls = _local_search(conflict, S_deg, ls_time, seed + 1000)
            seed += 1
            if len(S_deg_ls) > best_size:
                best_size = len(S_deg_ls)
                best_S = set(S_deg_ls)

    return best_S

def _try_add_to_set(conflict, S):
    """Try to add more elements to S by finding vertices with no conflict."""
    added = True
    while added:
        added = False
        for v in range(1, _N + 1):
            if v not in S and not (S & conflict[v]):
                S.add(v)
                added = True
    return S

def construct():
    start = time.time()
    conflict = _build_conflict_graph()

    # Quick baseline: all odds
    S_odd = set(range(1, _N + 1, 2))
    # Try to add evens that don't conflict with odds
    for e in range(2, _N + 1, 2):
        if not (S_odd & conflict[e]):
            S_odd.add(e)
    baseline_size = len(S_odd)

    # Allocate time
    elapsed = time.time() - start
    remaining = _DEADLINE - elapsed - 2.0  # 2s safety margin
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
