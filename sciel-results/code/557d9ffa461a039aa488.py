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

def _maximal_independent_set(conflict, S, all_vertices):
    """
    Structural change: Instead of local search / simulated annealing,
    use a maximal independent set extension. Given a current set S,
    find all vertices that can be added (no conflict with S), and
    greedily add them in an order that minimizes future conflicts.
    This is a different search paradigm: we don't remove and re-add,
    we just extend to a maximal set.
    """
    S = set(S)
    # Find all candidates that can be added
    candidates = [v for v in all_vertices if v not in S and not (S & conflict[v])]
    if not candidates:
        return S
    
    # Sort candidates by their degree in the conflict graph (fewer conflicts first)
    # This tends to leave more room for other vertices
    candidates.sort(key=lambda v: len(conflict[v]))
    
    for v in candidates:
        if not (S & conflict[v]):
            S.add(v)
    
    return S

def _conflict_aware_greedy(conflict, all_vertices, rng):
    """
    A different greedy strategy: at each step, pick the vertex with the 
    minimum number of conflicts with already-chosen vertices, breaking ties randomly.
    This is a structural change from random-order greedy to conflict-aware greedy.
    """
    S = set()
    remaining = set(all_vertices)
    
    while remaining:
        # Find the vertex in remaining with minimum conflicts with S
        best_v = None
        best_conf = float('inf')
        best_candidates = []
        
        for v in remaining:
            c = len(S & conflict[v])
            if c < best_conf:
                best_conf = c
                best_candidates = [v]
            elif c == best_conf:
                best_candidates.append(v)
        
        if best_conf > 0:
            # All remaining vertices conflict with S, stop
            break
        
        # Pick randomly from best candidates
        v = rng.choice(best_candidates)
        S.add(v)
        remaining.discard(v)
    
    return S

def _search(conflict, total_time):
    """
    Run multiple rounds with a mix of strategies:
    1. Randomized greedy
    2. Degree-ordered greedy
    3. Conflict-aware greedy (new structural approach)
    4. Maximal independent set extension (new structural approach)
    
    This combines the refinement machinery (multiple initializations, 
    local improvement) with the structural idea of conflict-aware selection.
    """
    start = time.time()
    best_S = set()
    best_size = 0
    rng = random.Random(42)
    all_vertices = list(range(1, _N + 1))

    while time.time() - start < total_time:
        elapsed = time.time() - start
        remaining = total_time - elapsed
        if remaining < 0.1:
            break
        
        # Choose a strategy randomly
        strategy = rng.random()
        
        if strategy < 0.3:
            # Randomized greedy
            order = list(all_vertices)
            rng.shuffle(order)
            S_init = _greedy_build(conflict, order)
        elif strategy < 0.5:
            # Degree-ordered greedy
            deg_order = sorted(all_vertices, key=lambda v: len(conflict[v]))
            rng.shuffle(deg_order)
            S_init = _greedy_build(conflict, deg_order)
        elif strategy < 0.7:
            # Conflict-aware greedy (new)
            S_init = _conflict_aware_greedy(conflict, all_vertices, rng)
        else:
            # Start with a small random set, then extend to maximal
            S_init = set(rng.sample(all_vertices, rng.randint(5, 20)))
        
        # Extend to a maximal independent set
        S_curr = _maximal_independent_set(conflict, S_init, all_vertices)
        
        if len(S_curr) > best_size:
            best_size = len(S_curr)
            best_S = set(S_curr)
        
        # Try a few more extensions with different orderings
        for _ in range(3):
            if time.time() - start >= total_time:
                break
            
            # Try adding vertices in different orders
            candidates = [v for v in all_vertices if v not in S_curr and not (S_curr & conflict[v])]
            if not candidates:
                break
            
            # Shuffle candidates and add greedily
            rng.shuffle(candidates)
            S_try = set(S_curr)
            for v in candidates:
                if not (S_try & conflict[v]):
                    S_try.add(v)
            
            if len(S_try) > best_size:
                best_size = len(S_try)
                best_S = set(S_try)

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
