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

def _local_search(conflict, S, time_limit, rng):
    """
    Local search on an independent set.
    Moves: add a vertex (if no conflict), 1-for-2 swap (remove one, add two),
    or 2-for-3 swap (remove two, add three).
    Objective: maximize |S|.
    """
    S = set(S)
    best_S = set(S)
    best_size = len(S)
    start = time.time()
    all_vertices = list(range(1, _N + 1))

    while time.time() - start < time_limit:
        improved = False

        # Try to add any vertex that has no conflict with S
        candidates_to_add = [v for v in all_vertices if v not in S and not (S & conflict[v])]
        if candidates_to_add:
            v = rng.choice(candidates_to_add)
            S.add(v)
            improved = True
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)

        # Try 1-for-2: remove u from S, add two vertices v1, v2 not in S
        if not improved and len(S) > 0:
            for _ in range(50):
                u = rng.choice(list(S))
                new_S = S - {u}
                # Find vertices that can be added to new_S
                addable = [v for v in all_vertices if v not in new_S and not (new_S & conflict[v])]
                if len(addable) >= 2:
                    # Try to find two that don't conflict with each other
                    rng.shuffle(addable)
                    found = False
                    for i in range(len(addable)):
                        for j in range(i+1, len(addable)):
                            v1, v2 = addable[i], addable[j]
                            if v1 not in conflict[v2]:
                                S = new_S | {v1, v2}
                                improved = True
                                if len(S) > best_size:
                                    best_size = len(S)
                                    best_S = set(S)
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break

        # Try 2-for-3: remove two vertices u1, u2 from S, add three vertices
        if not improved and len(S) > 2:
            for _ in range(30):
                u1, u2 = rng.sample(list(S), 2)
                new_S = S - {u1, u2}
                addable = [v for v in all_vertices if v not in new_S and not (new_S & conflict[v])]
                if len(addable) >= 3:
                    rng.shuffle(addable)
                    found = False
                    for i in range(len(addable)):
                        for j in range(i+1, len(addable)):
                            for k in range(j+1, len(addable)):
                                v1, v2, v3 = addable[i], addable[j], addable[k]
                                if v1 not in conflict[v2] and v1 not in conflict[v3] and v2 not in conflict[v3]:
                                    S = new_S | {v1, v2, v3}
                                    improved = True
                                    if len(S) > best_size:
                                        best_size = len(S)
                                        best_S = set(S)
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break

        # If no improvement, do a small random perturbation
        if not improved:
            if S:
                S.discard(rng.choice(list(S)))
            candidates = [v for v in all_vertices if v not in S and not (S & conflict[v])]
            if candidates:
                S.add(rng.choice(candidates))
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)

    return best_S

def _simulated_annealing(conflict, S, time_limit, rng):
    """
    Simulated annealing local search on an independent set.
    Allows temporary removal of elements to escape local optima.
    Uses 1-for-2 and 2-for-3 moves for size increases.
    """
    S = set(S)
    best_S = set(S)
    best_size = len(S)
    start = time.time()
    all_vertices = list(range(1, _N + 1))
    
    # Temperature schedule
    T = 10.0
    cooling = 0.999
    
    while time.time() - start < time_limit:
        T *= cooling
        if T < 0.01:
            break
        
        improved = False
        
        # Try to add any vertex that has no conflict with S
        candidates_to_add = [v for v in all_vertices if v not in S and not (S & conflict[v])]
        if candidates_to_add:
            v = rng.choice(candidates_to_add)
            S.add(v)
            improved = True
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)
        
        # Try 1-for-2 swap
        if not improved and len(S) > 0:
            for _ in range(30):
                u = rng.choice(list(S))
                new_S = S - {u}
                addable = [v for v in all_vertices if v not in new_S and not (new_S & conflict[v])]
                if len(addable) >= 2:
                    rng.shuffle(addable)
                    found = False
                    for i in range(min(len(addable), 20)):
                        for j in range(i+1, min(len(addable), 20)):
                            v1, v2 = addable[i], addable[j]
                            if v1 not in conflict[v2]:
                                S = new_S | {v1, v2}
                                improved = True
                                if len(S) > best_size:
                                    best_size = len(S)
                                    best_S = set(S)
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
        
        # Simulated annealing move: remove a random element and try to add back more
        if not improved and len(S) > 3 and rng.random() < T / 10.0:
            u = rng.choice(list(S))
            new_S = S - {u}
            candidates = [v for v in all_vertices if v not in new_S and not (new_S & conflict[v])]
            if len(candidates) >= 2:
                rng.shuffle(candidates)
                added = 0
                for v in candidates:
                    if not (new_S & conflict[v]):
                        new_S.add(v)
                        added += 1
                        if added >= 2:
                            break
                if added >= 2:
                    S = new_S
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
        
        # If no improvement and temperature is low, do a more aggressive perturbation
        if not improved and T < 1.0:
            # Remove 1-2 random elements
            for _ in range(rng.randint(1, 2)):
                if S:
                    S.discard(rng.choice(list(S)))
            # Greedily add back
            _try_add_to_set(conflict, S)
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)

    return best_S

def _search(conflict, total_time):
    """Run multiple rounds of greedy init + local search / simulated annealing."""
    start = time.time()
    best_S = set()
    best_size = 0
    rng = random.Random(42)

    while time.time() - start < total_time:
        # Phase 1: Randomized greedy initialization
        order = list(range(1, _N + 1))
        rng.shuffle(order)
        S_init = _greedy_build(conflict, order)
        
        # Also try degree-ordered greedy
        deg_order = sorted(range(1, _N + 1), key=lambda v: len(conflict[v]))
        rng.shuffle(deg_order)
        S_deg = _greedy_build(conflict, deg_order)
        
        # Pick the better initialization
        S_curr = S_init if len(S_init) >= len(S_deg) else S_deg
        
        if len(S_curr) > best_size:
            best_size = len(S_curr)
            best_S = set(S_curr)
        
        # Phase 2: Local search or simulated annealing
        elapsed = time.time() - start
        remaining = total_time - elapsed
        if remaining < 0.5:
            break
        
        ls_time = min(remaining * 0.3, 2.0)
        
        # Alternate between local search and simulated annealing
        if rng.random() < 0.5:
            S_ls = _local_search(conflict, S_curr, ls_time, rng)
        else:
            S_ls = _simulated_annealing(conflict, S_curr, ls_time, rng)
        
        if len(S_ls) > best_size:
            best_size = len(S_ls)
            best_S = set(S_ls)

    return best_S

def construct():
    start = time.time()
    conflict = _build_conflict_graph()
    
    # Quick baseline: all odds
    S_odd = set(range(1, _N + 1, 2))
    for e in range(2, _N + 1, 2):
        if not (S_odd & conflict[e]):
            S_odd.add(e)
    baseline_size = len(S_odd)
    
    # Allocate time
    elapsed = time.time() - start
    remaining = _DEADLINE - elapsed - 2.0
    if remaining < 5:
        remaining = 5
    
    # Run the main search
    best_S = _search(conflict, remaining)
    
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
