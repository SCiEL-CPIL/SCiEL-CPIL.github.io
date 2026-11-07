import time
import random
import itertools

N = 128
DEADLINE = 85.0  # safety margin under 90s

# Precompute all forbidden triples (a, b, c) with a < b < c
# such that a+b, a+c, b+c are all in 1..128
_FORBIDDEN_TRIPLES = []
def _build_forbidden_triples():
    """Precompute all (a,b,c) with a<b<c where a+b, a+c, b+c are all in 1..128."""
    triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            ab = a + b
            if ab > N:
                break
            for c in range(b + 1, N + 1):
                ac = a + c
                bc = b + c
                if ac > N or bc > N:
                    break
                triples.append((a, b, c))
    return triples

_FORBIDDEN_TRIPLES = _build_forbidden_triples()

# For each element x, precompute which triples it participates in
_ELEMENT_TRIPLES = {x: [] for x in range(1, N + 1)}
for _ti, (a, b, c) in enumerate(_FORBIDDEN_TRIPLES):
    _ELEMENT_TRIPLES[a].append(_ti)
    _ELEMENT_TRIPLES[b].append(_ti)
    _ELEMENT_TRIPLES[c].append(_ti)

def _check_valid(Sset):
    """Full validity check using precomputed triples. Returns True if valid."""
    for (a, b, c) in _FORBIDDEN_TRIPLES:
        if a in Sset and b in Sset and c in Sset:
            if (a + b) in Sset and (a + c) in Sset and (b + c) in Sset:
                return False
    return True

def _count_violations(Sset):
    """Count the number of violated forbidden triples."""
    total = 0
    for (a, b, c) in _FORBIDDEN_TRIPLES:
        if a in Sset and b in Sset and c in Sset:
            if (a + b) in Sset and (a + c) in Sset and (b + c) in Sset:
                total += 1
    return total

def _can_add(Sset, x):
    """Check if adding x to Sset creates any violation. Fast using precomputed triples."""
    if x in Sset:
        return False
    # Check all triples containing x
    for ti in _ELEMENT_TRIPLES[x]:
        a, b, c = _FORBIDDEN_TRIPLES[ti]
        # x is one of a, b, c; check if the other two are in Sset
        others = [v for v in (a, b, c) if v != x]
        if others[0] in Sset and others[1] in Sset:
            # All three in S (after adding x), check pair sums
            if (a + b) in Sset and (a + c) in Sset and (b + c) in Sset:
                return False
    return True

def _greedy_construct():
    """Greedy construction: add elements in order of minimal conflict."""
    Sset = set()
    candidates = list(range(1, N+1))
    random.shuffle(candidates)
    for x in candidates:
        if _can_add(Sset, x):
            Sset.add(x)
    return Sset

def _lagrangian_primal(dual_vars, Sset_init=None):
    """
    Lagrangian primal heuristic: given dual variables (penalties for violated triples),
    greedily select elements maximizing (1 - penalty_contribution).
    
    For each element x, its "cost" is the sum of dual_vars[ti] for triples ti containing x
    where the other two elements are already selected. We select elements in order of
    decreasing benefit (1 - cost), skipping those that would violate constraints.
    """
    Sset = set(Sset_init) if Sset_init else set()
    remaining = [x for x in range(1, N + 1) if x not in Sset]
    
    # Sort by number of triples (fewer constraints first tends to be easier to add)
    remaining.sort(key=lambda x: len(_ELEMENT_TRIPLES[x]))
    
    for x in remaining:
        if _can_add(Sset, x):
            Sset.add(x)
    
    return Sset

def _lagrangian_subgradient(Sset_init, max_iters=200, step_size=0.05):
    """
    Lagrangian relaxation with subgradient method.
    
    Objective: maximize |S| subject to no violated triples.
    Lagrangian: L = |S| - sum_t lambda_t * violation_t(S)
    where violation_t(S) = 1 if triple t is fully selected AND all pair sums are in S.
    
    Primal: greedy selection guided by dual variables.
    Dual: subgradient update of lambda_t.
    """
    rng = random.Random(12345)
    best_set = set(Sset_init)
    best_size = len(best_set)
    
    # Initialize dual variables
    n_triples = len(_FORBIDDEN_TRIPLES)
    lambdas = [0.0] * n_triples
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 8:
            break
        
        # Primal: greedy selection with dual-guided ordering
        # Elements with high penalty on their triples are less attractive
        Sset = set(Sset_init)
        remaining = [x for x in range(1, N + 1) if x not in Sset]
        
        # Compute "attractiveness" for each remaining element
        # Attractiveness = 1 - sum of lambda_t for triples t containing x where other two are in Sset
        scored = []
        for x in remaining:
            penalty = 0.0
            for ti in _ELEMENT_TRIPLES[x]:
                a, b, c = _FORBIDDEN_TRIPLES[ti]
                others = [v for v in (a, b, c) if v != x]
                if others[0] in Sset and others[1] in Sset:
                    penalty += lambdas[ti]
            scored.append((1.0 - penalty, x))
        
        # Sort by decreasing attractiveness
        scored.sort(key=lambda t: -t[0])
        
        for _, x in scored:
            if _can_add(Sset, x):
                Sset.add(x)
        
        if len(Sset) > best_size:
            best_size = len(Sset)
            best_set = set(Sset)
        
        # Dual update: subgradient
        # For each triple t, if it's violated in current Sset, increase lambda_t
        # Otherwise, decrease slightly (or keep)
        for ti in range(n_triples):
            a, b, c = _FORBIDDEN_TRIPLES[ti]
            if a in Sset and b in Sset and c in Sset:
                if (a + b) in Sset and (a + c) in Sset and (b + c) in Sset:
                    lambdas[ti] += step_size
                else:
                    lambdas[ti] = max(0.0, lambdas[ti] * 0.95)
            else:
                lambdas[ti] = max(0.0, lambdas[ti] * 0.95)
        
        # Occasionally reset some lambdas to avoid getting stuck
        if it % 20 == 19:
            for ti in range(n_triples):
                lambdas[ti] *= 0.5
    
    return best_set

def _local_search(Sset, max_iters=2000):
    """Local search: try removing one element and adding another to improve size."""
    Sset = set(Sset)
    best_size = len(Sset)
    rng = random.Random(42)
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 5:
            break
        
        # Try a random swap: remove y, add x
        notin = [x for x in range(1, N+1) if x not in Sset]
        if not notin:
            break
        
        x = rng.choice(notin)
        y = rng.choice(list(Sset))
        
        # Remove y
        S2 = Sset - {y}
        
        # Try to add x
        if _can_add(S2, x):
            S2.add(x)
            # Try to add back more elements
            improved = True
            while improved:
                improved = False
                for z in range(1, N+1):
                    if z not in S2 and _can_add(S2, z):
                        S2.add(z)
                        improved = True
            
            if len(S2) > best_size:
                best_size = len(S2)
                Sset = S2
    
    return Sset

def _polish(Sset, max_iters=5000):
    """High-precision polish: iterative add-remove with aggressive fill."""
    Sset = set(Sset)
    best_size = len(Sset)
    rng = random.Random(999)
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 3:
            break
        
        # Phase 1: Try to add any element that fits
        added_any = True
        while added_any:
            added_any = False
            for x in range(1, N + 1):
                if x not in Sset and _can_add(Sset, x):
                    Sset.add(x)
                    added_any = True
        
        if len(Sset) > best_size:
            best_size = len(Sset)
        
        # Phase 2: Try random swaps
        notin = [x for x in range(1, N+1) if x not in Sset]
        if not notin:
            break
        
        x = rng.choice(notin)
        y = rng.choice(list(Sset))
        
        S2 = Sset - {y}
        if _can_add(S2, x):
            S2.add(x)
            # Aggressive fill
            improved = True
            while improved:
                improved = False
                for z in range(1, N+1):
                    if z not in S2 and _can_add(S2, z):
                        S2.add(z)
                        improved = True
            
            if len(S2) >= len(Sset):
                Sset = S2
                if len(Sset) > best_size:
                    best_size = len(Sset)
    
    return Sset

def _two_part_construct():
    """Two-part construction: L (weakly sum-free) union U (sum-free).
    Choose gap so that 2*max(L) < min(U) to prevent mixed-sum violations."""
    L = set(x for x in range(1, 33) if x % 2 == 1)
    U = set(range(65, 129))
    Sset = L | U
    
    if not _check_valid(Sset):
        Sset = set(range(65, 129))
    
    return Sset

def _random_restart_construct():
    """Random restart with greedy + local search."""
    best_set = set()
    best_size = 0
    rng = random.Random(7)
    
    for trial in range(20):
        if time.time() > DEADLINE - 10:
            break
        
        Sset = _greedy_construct()
        Sset = _local_search(Sset, max_iters=500)
        
        if len(Sset) > best_size:
            best_size = len(Sset)
            best_set = set(Sset)
    
    return best_set

def _lagrangian_search(Sset_init, max_iters=100):
    """Lagrangian relaxation search to improve the set."""
    result = _lagrangian_subgradient(Sset_init, max_iters=max_iters)
    return result

def construct():
    start = time.time()
    
    # Start with the two-part construction (known to give ~80)
    base_set = _two_part_construct()
    best_set = set(base_set)
    best_size = len(best_set)
    
    # Lagrangian relaxation search from the base set
    if time.time() < DEADLINE - 20:
        lag_set = _lagrangian_search(base_set, max_iters=150)
        if len(lag_set) > best_size:
            best_set = lag_set
            best_size = len(best_set)
    
    # Try random restarts to see if we can do better
    if time.time() < DEADLINE - 20:
        rs_set = _random_restart_construct()
        if len(rs_set) > best_size:
            best_set = rs_set
            best_size = len(best_set)
    
    # Lagrangian on the best set found so far
    if time.time() < DEADLINE - 15:
        lag_set2 = _lagrangian_search(best_set, max_iters=100)
        if len(lag_set2) > best_size:
            best_set = lag_set2
            best_size = len(best_set)
    
    # High-precision polish pass
    if time.time() < DEADLINE - 5:
        best_set = _polish(best_set, max_iters=5000)
        best_size = len(best_set)
    
    # Final local search
    if time.time() < DEADLINE - 3:
        best_set = _local_search(best_set, max_iters=2000)
        best_size = len(best_set)
    
    # Verify validity
    if not _check_valid(best_set):
        # Fallback to safe set
        best_set = set(range(65, 129))
    
    res = sorted(best_set)
    return [int(x) for x in res]
