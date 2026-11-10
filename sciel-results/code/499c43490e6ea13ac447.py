"""Optimized program for small sums and products along selected graph edges."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

P = {'n': 14, 'm': 35, 'maximum': 10000}

def _score(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        vi = values[i]
        vj = values[j]
        sums.add(vi + vj)
        prods.add(vi * vj)
    return max(len(sums), len(prods))

def _score_parts(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        vi = values[i]
        vj = values[j]
        sums.add(vi + vj)
        prods.add(vi * vj)
    return len(sums), len(prods)

def _all_edges(n):
    return list(combinations(range(n), 2))

def _sidon_base(n, max_val=10000, seed=0):
    rng = random.Random(seed)
    vals = []
    for i in range(n):
        while True:
            v = rng.randint(1, max_val)
            ok = True
            for j, w in enumerate(vals):
                if v == w:
                    ok = False
                    break
                if v * w in [vals[k] * vals[l] for k in range(len(vals)) for l in range(k+1, len(vals)) if k != j and l != j]:
                    pass  # we just check distinctness for now; product uniqueness handled later
            if ok and v not in vals:
                vals.append(v)
                break
    return vals

def _distinct_products(values):
    prods = set()
    for i in range(len(values)):
        for j in range(i+1, len(values)):
            prods.add(values[i] * values[j])
    return len(prods)

def _build_sidon_values(n, max_val=10000, seed=0):
    """Build n distinct values such that all C(n,2) products are distinct."""
    rng = random.Random(seed)
    # Start with small values and grow to ensure product uniqueness
    vals = []
    for i in range(n):
        # Try to find a value that keeps all products unique
        attempts = 0
        while attempts < 20000:
            attempts += 1
            if i == 0:
                v = rng.randint(1, 3)
            else:
                # Prefer values that don't create product collisions
                # Use values that are "large enough" to avoid collisions
                lo = max(1, vals[-1] + 1)
                hi = min(max_val, 10000)
                if lo >= hi:
                    v = lo
                else:
                    v = rng.randint(lo, hi)
            # Check distinctness
            if v in vals:
                continue
            # Check product uniqueness
            new_prods = set()
            for w in vals:
                new_prods.add(v * w)
            existing_prods = set()
            for a in range(len(vals)):
                for b in range(a+1, len(vals)):
                    existing_prods.add(vals[a] * vals[b])
            if new_prods & existing_prods:
                continue
            vals.append(v)
            break
        else:
            # Fallback: just pick a large unique value
            while True:
                v = rng.randint(1, max_val)
                if v not in vals:
                    vals.append(v)
                    break
    return vals

def _fix_products(values, n, max_val=10000):
    """Ensure all C(n,2) products are distinct by adjusting values."""
    vals = list(values)
    for _ in range(10000):
        prods = {}
        dup_found = False
        for i in range(n):
            for j in range(i+1, n):
                p = vals[i] * vals[j]
                if p in prods:
                    dup_found = True
                    # Adjust the larger value
                    k = j
                    while True:
                        vals[k] += 1
                        if vals[k] > max_val:
                            vals[k] = 1
                        if vals[k] in vals[:k] or vals[k] in vals[k+1:]:
                            continue
                        # Recheck
                        new_prods = set()
                        for a in range(n):
                            for b in range(a+1, n):
                                if a == k or b == k:
                                    new_prods.add(vals[a] * vals[b])
                        rest_prods = set()
                        for a in range(n):
                            for b in range(a+1, n):
                                if a != k and b != k:
                                    rest_prods.add(vals[a] * vals[b])
                        if not (new_prods & rest_prods):
                            break
                    break
                else:
                    prods[p] = (i, j)
        if not dup_found:
            break
    return vals

def _find_best_subgraph(values, n, m, all_edges_list, seed=0, max_iter=50000, timeout=10.0):
    """Find the m-edge subgraph that minimizes max(#distinct sums, #distinct products)."""
    rng = random.Random(seed)
    total_edges = len(all_edges_list)
    m = min(m, total_edges)
    
    # Start with random m edges
    idx = list(range(total_edges))
    rng.shuffle(idx)
    current = set(idx[:m])
    
    def eval_set(edge_set):
        sums = set()
        prods = set()
        for ei in edge_set:
            i, j = all_edges_list[ei]
            vi = values[i]
            vj = values[j]
            sums.add(vi + vj)
            prods.add(vi * vj)
        return max(len(sums), len(prods))
    
    best_score = eval_set(current)
    best_set = set(current)
    
    start = time.time()
    no_improve = 0
    max_no_improve = 5000
    
    it = 0
    while it < max_iter:
        if time.time() - start > timeout:
            break
        it += 1
        
        # Try swap: remove one edge, add another
        if len(current) == 0:
            break
        remove_idx = rng.choice(list(current))
        # Find a candidate to add
        candidates = [e for e in idx if e not in current]
        if not candidates:
            break
        add_idx = rng.choice(candidates)
        
        trial = set(current)
        trial.remove(remove_idx)
        trial.add(add_idx)
        
        sc = eval_set(trial)
        if sc < best_score:
            best_score = sc
            best_set = set(trial)
            current = trial
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > max_no_improve:
            break
    
    return [all_edges_list[i] for i in best_set], best_score

def _local_search_values(values, edges, n, max_val=10000, max_iter=30000, timeout=10.0, seed=0):
    """Local search on values to minimize score."""
    rng = random.Random(seed)
    current = list(values)
    best_score = _score(current, edges)
    best_values = list(current)
    
    start = time.time()
    no_improve = 0
    max_no_improve = 3000
    
    it = 0
    while it < max_iter:
        if time.time() - start > timeout:
            break
        it += 1
        
        # Pick a vertex and perturb its value
        i = rng.randrange(n)
        old_val = current[i]
        
        # Try a random new value
        new_val = rng.randint(1, max_val)
        if new_val == old_val:
            continue
        if new_val in [current[j] for j in range(n) if j != i]:
            continue
        
        current[i] = new_val
        sc = _score(current, edges)
        
        if sc < best_score:
            best_score = sc
            best_values = list(current)
            no_improve = 0
        else:
            current[i] = old_val
            no_improve += 1
        
        if no_improve > max_no_improve:
            # Reset
            current = list(best_values)
            no_improve = 0
    
    return best_values, best_score

def _simulated_annealing(values, edges, n, max_val=10000, max_iter=50000, timeout=15.0, seed=0):
    """Simulated annealing on values."""
    rng = random.Random(seed)
    current = list(values)
    current_score = _score(current, edges)
    best_values = list(current)
    best_score = current_score
    
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    start = time.time()
    it = 0
    while it < max_iter:
        if time.time() - start > timeout:
            break
        it += 1
        
        i = rng.randrange(n)
        old_val = current[i]
        new_val = rng.randint(1, max_val)
        if new_val == old_val:
            continue
        if new_val in [current[j] for j in range(n) if j != i]:
            continue
        
        current[i] = new_val
        new_score = _score(current, edges)
        delta = new_score - current_score
        
        if delta <= 0 or rng.random() < math.exp(-delta / max(T, T_min)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_values = list(current)
        else:
            current[i] = old_val
        
        T *= cooling
    
    return best_values, best_score

def _try_configuration(values, edges, n, seed=0, deadline=None):
    """Try to improve a (values, edges) configuration."""
    if deadline is not None and time.time() > deadline:
        return values, edges, _score(values, edges)
    
    # First, optimize the subgraph for these values
    all_edges_list = _all_edges(n)
    if deadline is not None:
        t_left = max(1.0, deadline - time.time())
    else:
        t_left = 10.0
    best_edges, sc = _find_best_subgraph(values, n, P['m'], all_edges_list, seed=seed, max_iter=20000, timeout=t_left)
    if best_edges is None or len(best_edges) != P['m']:
        # Fallback
        best_edges = [list(e) for e in edges[:P['m']]]
        sc = _score(values, best_edges)
    
    # Now optimize values for these edges
    if deadline is not None:
        t_left = max(1.0, deadline - time.time())
    else:
        t_left = 10.0
    vals, sc2 = _local_search_values(values, best_edges, n, max_iter=20000, timeout=t_left, seed=seed)
    if sc2 < sc:
        sc = sc2
        values = vals
    
    # Simulated annealing pass
    if deadline is not None:
        t_left = max(1.0, deadline - time.time())
    else:
        t_left = 10.0
    vals, sc3 = _simulated_annealing(values, best_edges, n, max_iter=30000, timeout=t_left, seed=seed+1)
    if sc3 < sc:
        sc = sc3
        values = vals
    
    return values, best_edges, sc

def construct():
    start = time.time()
    deadline = start + 75.0  # safety margin (90s limit, 15s buffer)
    
    n = P['n']
    m = P['m']
    
    best_values = None
    best_edges = None
    best_score = float('inf')
    
    # Strategy 1: Sidon-like values + subgraph optimization
    for seed in [0, 1, 42, 123, 999]:
        if time.time() > deadline - 15:
            break
        vals = _build_sidon_values(n, seed=seed)
        vals = _fix_products(vals, n)
        # Initial edges: random
        all_e = _all_edges(n)
        rng = random.Random(seed)
        rng.shuffle(all_e)
        init_edges = [list(e) for e in all_e[:m]]
        
        vals, edges, sc = _try_configuration(vals, init_edges, n, seed=seed, deadline=deadline)
        if sc < best_score:
            best_score = sc
            best_values = vals
            best_edges = edges
    
    # Strategy 2: Small consecutive values
    if time.time() < deadline - 15:
        vals = list(range(1, n+1))
        all_e = _all_edges(n)
        rng = random.Random(7)
        rng.shuffle(all_e)
        init_edges = [list(e) for e in all_e[:m]]
        vals, edges, sc = _try_configuration(vals, init_edges, n, seed=7, deadline=deadline)
        if sc < best_score:
            best_score = sc
            best_values = vals
            best_edges = edges
    
    # Strategy 3: Geometric progression (powers of 2, capped)
    if time.time() < deadline - 15:
        vals = []
        for i in range(n):
            v = 2 ** i
            if v > 10000:
                v = 10000 - (n - 1 - i)
            vals.append(v)
        # Ensure distinct
        seen = set()
        result = []
        for v in vals:
            while v in seen or v < 1 or v > 10000:
                v += 1
            seen.add(v)
            result.append(v)
        vals = result
        all_e = _all_edges(n)
        rng = random.Random(11)
        rng.shuffle(all_e)
        init_edges = [list(e) for e in all_e[:m]]
        vals, edges, sc = _try_configuration(vals, init_edges, n, seed=11, deadline=deadline)
        if sc < best_score:
            best_score = sc
            best_values = vals
            best_edges = edges
    
    # Strategy 4: Random values
    for seed in [100, 200, 300]:
        if time.time() > deadline - 10:
            break
        rng = random.Random(seed)
        vals = sorted(rng.sample(range(1, 10001), n))
        all_e = _all_edges(n)
        rng2 = random.Random(seed+1)
        rng2.shuffle(all_e)
        init_edges = [list(e) for e in all_e[:m]]
        vals, edges, sc = _try_configuration(vals, init_edges, n, seed=seed, deadline=deadline)
        if sc < best_score:
            best_score = sc
            best_values = vals
            best_edges = edges
    
    # Final refinement with more time
    if best_values is not None and time.time() < deadline - 5:
        t_left = deadline - time.time() - 1.0
        vals, edges, sc = _try_configuration(best_values, best_edges, n, seed=9999, deadline=deadline)
        if sc < best_score:
            best_score = sc
            best_values = vals
            best_edges = edges
    
    # Fallback
    if best_values is None:
        best_values = list(range(1, n+1))
        all_e = _all_edges(n)
        best_edges = [list(e) for e in all_e[:m]]
        best_score = _score(best_values, best_edges)
    
    # Final validation
    assert len(best_values) == n
    assert len(set(best_values)) == n
    assert all(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 10000 for v in best_values)
    assert len(best_edges) == m
    edge_set = set()
    for e in best_edges:
        assert isinstance(e, list) and len(e) == 2
        a, b = e
        assert isinstance(a, int) and not isinstance(a, bool)
        assert isinstance(b, int) and not isinstance(b, bool)
        assert a != b
        assert 0 <= a < n and 0 <= b < n
        key = (min(a, b), max(a, b))
        assert key not in edge_set
        edge_set.add(key)
    
    return {'values': [int(v) for v in best_values], 'edges': [[int(a), int(b)] for a, b in best_edges]}
# EVOLVE-BLOCK-END
