"""Joint SA: co-optimize values and edges via simulated annealing with fast incremental scoring."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import math

P = {'n': 14, 'm': 35, 'maximum': 10000}

def _raw_score(values, edges):
    """Compute the raw score: max(#distinct sums, #distinct products) over edges."""
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _all_edges(n):
    """Generate all C(n,2) possible edges."""
    return list(itertools.combinations(range(n), 2))

def _init_state():
    """Initialize a random feasible state with structured bias."""
    n = P['n']
    # Start with small values biased toward composites
    base_vals = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]
    random.shuffle(base_vals)
    values = list(base_vals)
    
    # Random 35 edges
    all_e = _all_edges(n)
    random.shuffle(all_e)
    edges = [list(e) for e in all_e[:P['m']]]
    
    return values, edges

def _score_state(values, edges):
    """Fast score computation."""
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _perturb_values(values, edges):
    """Generate a neighbor by perturbing one value."""
    new_values = list(values)
    i = random.randint(0, P['n'] - 1)
    old = new_values[i]
    
    # Try various perturbation strategies
    strategy = random.random()
    if strategy < 0.3:
        # Small delta
        delta = random.choice([-3, -2, -1, 1, 2, 3, -5, 5, -10, 10, -20, 20, -50, 50])
        new_val = old + delta
    elif strategy < 0.5:
        # Replace with random small composite
        candidates = [d for d in range(1, 500) if d != old]
        # Prefer composites
        def is_prime(x):
            if x < 2:
                return False
            for f in range(2, int(math.isqrt(x)) + 1):
                if x % f == 0:
                    return False
            return True
        composites = [d for d in candidates if not is_prime(d)]
        new_val = random.choice(composites) if composites else random.choice(candidates)
    elif strategy < 0.7:
        # Scale by small factor
        factor = random.choice([2, 3, 4, 5, 6])
        if random.random() < 0.5:
            new_val = old * factor
        else:
            new_val = old // factor if old % factor == 0 else old + random.randint(1, 20)
    else:
        # Random in range
        new_val = random.randint(1, 10000)
    
    if new_val <= 0 or new_val > 10000:
        return None
    if new_val in new_values:
        return None
    
    new_values[i] = new_val
    return new_values

def _perturb_edges(values, edges):
    """Generate a neighbor by swapping one edge."""
    new_edges = [list(e) for e in edges]
    i = random.randint(0, P['m'] - 1)
    old_edge = tuple(new_edges[i])
    
    # Pick a new edge
    all_e = _all_edges(P['n'])
    edge_set = set(tuple(e) for e in new_edges)
    
    attempts = 0
    while attempts < 50:
        new_e = random.choice(all_e)
        if new_e != old_edge and new_e not in edge_set:
            break
        attempts += 1
    else:
        return None
    
    new_edges[i] = list(new_e)
    return new_edges

def _sa_optimize(values, edges, deadline, max_iters=50000):
    """Simulated annealing with joint value and edge perturbations."""
    n = P['n']
    m = P['m']
    
    cur_vals = list(values)
    cur_edges = [list(e) for e in edges]
    cur_score = _score_state(cur_vals, cur_edges)
    
    best_vals = list(cur_vals)
    best_edges = [list(e) for e in cur_edges]
    best_score = cur_score
    
    T0 = 5.0
    T_end = 0.1
    cooling = (T_end / T0) ** (1.0 / max(1, max_iters))
    
    no_improve = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        T = T0 * (cooling ** it)
        
        # Choose perturbation type
        r = random.random()
        if r < 0.5:
            new_vals = _perturb_values(cur_vals, cur_edges)
            if new_vals is None:
                continue
            new_edges = cur_edges
        else:
            new_edges = _perturb_edges(cur_vals, cur_edges)
            if new_edges is None:
                continue
            new_vals = cur_vals
        
        new_score = _score_state(new_vals, new_edges)
        delta = new_score - cur_score
        
        if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
            cur_vals = new_vals
            cur_edges = new_edges
            cur_score = new_score
            
            if cur_score < best_score:
                best_score = cur_score
                best_vals = list(cur_vals)
                best_edges = [list(e) for e in cur_edges]
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        # Restart if stuck too long
        if no_improve > 5000:
            # Try a random restart with bias toward best
            if random.random() < 0.3:
                cur_vals = list(best_vals)
                cur_edges = [list(e) for e in best_edges]
                cur_score = best_score
            else:
                # Fresh random
                cur_vals, cur_edges = _init_state()
                cur_score = _score_state(cur_vals, cur_edges)
            no_improve = 0
    
    return best_score, best_vals, best_edges

def _local_polish(values, edges, deadline):
    """Final local search to polish the solution."""
    n = P['n']
    m = P['m']
    
    cur_vals = list(values)
    cur_edges = [list(e) for e in edges]
    cur_score = _score_state(cur_vals, cur_edges)
    
    all_e = _all_edges(n)
    edge_set = set(tuple(e) for e in cur_edges)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try edge swaps
        for i in range(m):
            if time.time() > deadline:
                break
            old_e = tuple(cur_edges[i])
            for new_e in all_e:
                if new_e == old_e or new_e in edge_set:
                    continue
                # Test swap
                test_edges = [list(e) for e in cur_edges]
                test_edges[i] = list(new_e)
                new_score = _score_state(cur_vals, test_edges)
                if new_score < cur_score:
                    cur_edges = test_edges
                    edge_set.remove(old_e)
                    edge_set.add(new_e)
                    cur_score = new_score
                    improved = True
                    break
        
        # Try value perturbations
        for i in range(n):
            if time.time() > deadline:
                break
            old_val = cur_vals[i]
            for delta in [-2, -1, 1, 2, -5, 5, -10, 10, -20, 20]:
                new_val = old_val + delta
                if new_val <= 0 or new_val > 10000:
                    continue
                if new_val in cur_vals:
                    continue
                test_vals = list(cur_vals)
                test_vals[i] = new_val
                new_score = _score_state(test_vals, cur_edges)
                if new_score < cur_score:
                    cur_vals = test_vals
                    cur_score = new_score
                    improved = True
                    break
    
    return cur_score, cur_vals, cur_edges

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s deadline, 5s safety margin
    
    random.seed()  # Use system time for randomness
    
    best_score = float('inf')
    best_vals = None
    best_edges = None
    
    # Run multiple SA restarts
    n_restarts = 8
    time_per_restart = (deadline - start_time) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Initialize
        vals, edges = _init_state()
        
        # SA optimization
        sa_deadline = start_time + (restart + 1) * time_per_restart
        sa_deadline = min(sa_deadline, deadline - 5)  # Leave 5s for polish
        
        score, vals, edges = _sa_optimize(vals, edges, sa_deadline, max_iters=30000)
        
        # Local polish
        if time.time() < deadline - 2:
            polish_deadline = min(deadline - 1, time.time() + 5)
            score, vals, edges = _local_polish(vals, edges, polish_deadline)
        
        if score < best_score:
            best_score = score
            best_vals = list(vals)
            best_edges = [list(e) for e in edges]
    
    # Fallback
    if best_edges is None:
        vals, edges = _init_state()
        best_vals = vals
        best_edges = edges
        best_score = _raw_score(best_vals, best_edges)
    
    # Ensure types are correct
    final_values = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': final_values, 'edges': final_edges}
# EVOLVE-BLOCK-END
