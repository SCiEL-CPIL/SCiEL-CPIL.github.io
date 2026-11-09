"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _instance(size, seed, limit=10000):
    rng = random.Random(seed)
    values = set()
    while len(values) < size:
        x = rng.randint(-limit, limit)
        if x:
            values.add(x)
    return sorted(values)

def _build_conflict_graph(values):
    """Build adjacency as bitmasks for fast conflict counting."""
    val_set = set(values)
    n = len(values)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            s = values[i] + values[j]
            if s in val_set:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _greedy_mis(adj, n, rng, order=None):
    """Greedy MIS using bitmask adjacency."""
    if order is None:
        order = list(range(n))
        rng.shuffle(order)
    
    in_set = [False] * n
    remaining_mask = (1 << n) - 1
    
    for v in order:
        if not (remaining_mask & (1 << v)):
            continue
        in_set[v] = True
        remaining_mask &= ~(1 << v)
        remaining_mask &= ~adj[v]
    
    return in_set

def _count_conflicts(in_set, adj, n):
    """Count edges where both endpoints are in_set."""
    cnt = 0
    for i in range(n):
        if in_set[i]:
            common = adj[i] & ((1 << n) - 1)
            # Count bits j > i
            upper_mask = ((1 << n) - 1) ^ ((1 << (i + 1)) - 1)
            cnt += bin(common & upper_mask).count('1')
    return cnt

def _simulate_annealing(adj, n, rng, in_set_init, max_iters, t_start, t_end, deadline):
    """Simulated annealing with 1-flip moves on conflict graph."""
    in_set = in_set_init[:]
    cur_size = sum(in_set)
    cur_conf = _count_conflicts(in_set, adj, n)
    
    best_in_set = in_set[:]
    best_size = cur_size if cur_conf == 0 else 0
    best_conf = cur_conf
    
    full_mask = (1 << n) - 1
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        t = t_start * (t_end / t_start) ** (it / max_iters)
        
        # Random flip
        v = rng.randint(0, n - 1)
        old_state = in_set[v]
        in_set[v] = not old_state
        
        # Compute delta in conflicts
        if old_state:
            # Removed v: conflicts decrease by number of neighbors in set
            delta_conf = -bin(adj[v] & full_mask).count('1')
            # But we need to count only neighbors that are in_set
            nb_in_set = 0
            for j in range(n):
                if in_set[j] and (adj[v] & (1 << j)):
                    nb_in_set += 1
            delta_conf = -nb_in_set
            delta_size = -1
        else:
            # Added v: conflicts increase by number of neighbors in set
            nb_in_set = 0
            for j in range(n):
                if in_set[j] and (adj[v] & (1 << j)):
                    nb_in_set += 1
            delta_conf = nb_in_set
            delta_size = 1
        
        new_conf = cur_conf + delta_conf
        new_size = cur_size + delta_size
        
        # Objective: maximize size, penalize conflicts heavily
        if new_conf == 0:
            new_obj = new_size
        else:
            new_obj = new_size - 10000 * new_conf
        
        if old_state:
            cur_obj = cur_size - (10000 * cur_conf if cur_conf > 0 else 0)
        else:
            cur_obj = cur_size - (10000 * cur_conf if cur_conf > 0 else 0)
        
        delta_obj = new_obj - cur_obj
        
        if delta_obj >= 0 or rng.random() < math.exp(delta_obj / max(t, 1e-10)):
            cur_conf = new_conf
            cur_size = new_size
            if cur_conf == 0 and cur_size > best_size:
                best_size = cur_size
                best_in_set = in_set[:]
        else:
            in_set[v] = old_state  # revert
    
    return best_in_set

def _solve_instance(values, time_budget, seed, global_deadline):
    """Solve one instance: find max sum-free subset."""
    n = len(values)
    if n == 0:
        return []
    
    rng = random.Random(seed)
    
    # Build conflict graph as bitmasks
    adj = _build_conflict_graph(values)
    
    best_subset = []
    best_size = 0
    
    # Parity baseline
    odds = [values[i] for i in range(n) if values[i] % 2 != 0]
    evens = [values[i] for i in range(n) if values[i] % 2 == 0]
    if len(odds) > best_size:
        best_size = len(odds)
        best_subset = odds[:]
    if len(evens) > best_size:
        best_size = len(evens)
        best_subset = evens[:]
    
    # Multiple greedy restarts
    n_restarts = min(30, max(5, int(time_budget * 2)))
    greedy_results = []
    for restart in range(n_restarts):
        if time.time() > global_deadline:
            break
        greedy = _greedy_mis(adj, n, rng)
        size = sum(greedy)
        greedy_results.append((size, greedy))
        if size > best_size:
            best_size = size
            best_subset = [values[i] for i in range(n) if greedy[i]]
    
    # Sort greedy results by size descending, take top few for SA
    greedy_results.sort(key=lambda x: -x[0])
    
    # Simulated annealing from top greedy solutions
    sa_deadline = min(global_deadline - 0.5, time.time() + time_budget * 0.7)
    n_sa = min(5, len(greedy_results))
    
    for k in range(n_sa):
        if time.time() > sa_deadline:
            break
        size, greedy = greedy_results[k]
        remaining_time = sa_deadline - time.time()
        if remaining_time < 0.5:
            break
        
        # Scale iterations with time
        max_iters = int(remaining_time * 2000)
        max_iters = max(max_iters, 500)
        max_iters = min(max_iters, 100000)
        
        sa_result = _simulate_annealing(adj, n, rng, greedy, max_iters, 5.0, 0.01, sa_deadline)
        sa_size = sum(sa_result)
        if sa_size > best_size:
            best_size = sa_size
            best_subset = [values[i] for i in range(n) if sa_result[i]]
    
    return best_subset

_start_time = None

def construct():
    global _start_time
    _start_time = time.time()
    global_deadline = _start_time + 140  # 10s safety margin before 150s kill
    
    instances = [
        _instance(60, 801),
        _instance(120, 802),
        _instance(240, 803)
    ]
    
    total_size = sum(len(v) for v in instances)
    results = []
    for i, values in enumerate(instances):
        elapsed = time.time() - _start_time
        remaining = global_deadline - elapsed
        if remaining < 0.1:
            remaining = 0.1
        
        # Allocate more time to larger instances
        weight = len(values) / total_size
        time_budget = max(remaining * weight * 0.85, 0.5)
        
        subset = _solve_instance(values, time_budget, 801 + i, global_deadline)
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
