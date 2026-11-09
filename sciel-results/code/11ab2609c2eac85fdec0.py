"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _instance(size, seed, limit=10000):
    rng = random.Random(seed)
    values = set()
    while len(values) < size:
        x = rng.randint(-limit, limit)
        if x:
            values.add(x)
    return sorted(values)

def _build_conflict_graph(values):
    """Build adjacency list for the conflict graph.
    Edge (i,j) exists if values[i]+values[j] is also in values.
    Returns adj as list of lists (sorted) for cache efficiency.
    """
    val_set = set(values)
    n = len(values)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = values[i] + values[j]
            if s in val_set:
                adj[i].append(j)
                adj[j].append(i)
    # Convert to sorted lists for deterministic iteration
    for i in range(n):
        adj[i].sort()
    return adj

def _greedy_mis(adj, n, rng, deadline):
    """Greedy Maximum Independent Set: repeatedly pick lowest-degree remaining vertex."""
    in_set = [False] * n
    remaining = [True] * n
    rem_count = n
    cur_deg = [len(adj[i]) for i in range(n)]
    
    while rem_count > 0:
        if time.time() > deadline:
            break
        # Find vertex with minimum current degree among remaining
        best_v = -1
        best_deg = float('inf')
        for v in range(n):
            if not remaining[v]:
                continue
            deg = sum(1 for nb in adj[v] if remaining[nb])
            if deg < best_deg:
                best_deg = deg
                best_v = v
                if deg == 0:
                    break
        if best_v < 0:
            break
        in_set[best_v] = True
        remaining[best_v] = False
        rem_count -= 1
        for nb in adj[best_v]:
            if remaining[nb]:
                remaining[nb] = False
                rem_count -= 1
    
    return in_set

def _conflict_count(in_set, adj, n):
    """Count number of conflicting edges within in_set."""
    cnt = 0
    for i in range(n):
        if in_set[i]:
            for j in adj[i]:
                if j > i and in_set[j]:
                    cnt += 1
    return cnt

def _local_search_swap(in_set, adj, n, rng, time_limit):
    """Local search: try many swap pairs (remove one in-set, add one out-of-set).
    Also does add-only moves when possible.
    """
    cur = in_set[:]
    cur_size = sum(cur)
    best = cur[:]
    best_size = cur_size
    
    # Precompute in/out lists
    in_list = [i for i in range(n) if cur[i]]
    out_list = [i for i in range(n) if not cur[i]]
    
    start = time.time()
    no_improve = 0
    
    while time.time() - start < time_limit:
        if not in_list or not out_list:
            break
        
        improved_this_round = False
        
        # Try swap: remove v_in, add v_out
        # Sample up to 50 in-set and 30 out-set candidates
        n_in = len(in_list)
        n_out = len(out_list)
        in_sample = rng.sample(in_list, min(50, n_in))
        out_sample = rng.sample(out_list, min(30, n_out))
        
        for v_in in in_sample:
            if not cur[v_in]:
                continue
            for v_out in out_sample:
                if cur[v_out]:
                    continue
                # Check if v_out conflicts with any current in-set member (excluding v_in)
                ok = True
                for nb in adj[v_out]:
                    if cur[nb] and nb != v_in:
                        ok = False
                        break
                if ok:
                    # Perform swap
                    cur[v_in] = False
                    cur[v_out] = True
                    in_list.remove(v_in)
                    in_list.append(v_out)
                    out_list.remove(v_out)
                    out_list.append(v_in)
                    improved_this_round = True
                    break
            if improved_this_round:
                break
        
        if improved_this_round:
            no_improve = 0
        else:
            # Try add-only: add a random out vertex with no conflicts
            added = False
            for _ in range(min(20, n_out)):
                v_try = rng.choice(out_list)
                if cur[v_try]:
                    continue
                ok = True
                for nb in adj[v_try]:
                    if cur[nb]:
                        ok = False
                        break
                if ok:
                    cur[v_try] = True
                    cur_size += 1
                    in_list.append(v_try)
                    out_list.remove(v_try)
                    added = True
                    if cur_size > best_size:
                        best_size = cur_size
                        best = cur[:]
                    break
            if added:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 200:
                    break
    
    # Final polish: add-only pass
    changed = True
    while changed and time.time() - start < time_limit:
        changed = False
        for v in out_list[:]:
            if cur[v]:
                continue
            ok = True
            for nb in adj[v]:
                if cur[nb]:
                    ok = False
                    break
            if ok:
                cur[v] = True
                cur_size += 1
                in_list.append(v)
                out_list.remove(v)
                changed = True
                if cur_size > best_size:
                    best_size = cur_size
                    best = cur[:]
    
    return best

def _simulated_annealing(adj, n, rng, t_start, t_end, time_limit, initial=None):
    """Simulated annealing with tracked conflict count for O(degree) objective updates."""
    if initial is not None:
        cur = initial[:]
    else:
        cur = [False] * n
        # Quick greedy init
        for v in range(n):
            if not any(cur[nb] for nb in adj[v]):
                cur[v] = True
    
    cur_size = sum(cur)
    cur_conf = _conflict_count(cur, adj, n)
    
    best = cur[:]
    best_size = cur_size - 10000 * cur_conf
    
    start = time.time()
    it = 0
    
    while time.time() - start < time_limit:
        it += 1
        t = t_start * (t_end / t_start) ** (min(it, 100000) / 100000.0)
        t = max(t, 1e-10)
        
        v = rng.randint(0, n - 1)
        old_state = cur[v]
        cur[v] = not old_state
        
        # Compute delta in objective
        if old_state:
            # Removing v: size decreases by 1, conflicts involving v are removed
            removed_conf = sum(1 for nb in adj[v] if cur[nb])  # cur[v] is now False
            new_size = cur_size - 1
            new_conf = max(0, cur_conf - removed_conf)
            delta = (new_size - 10000 * new_conf) - (cur_size - 10000 * cur_conf)
        else:
            # Adding v: size increases by 1, conflicts involving v are added
            added_conf = sum(1 for nb in adj[v] if cur[nb])
            new_size = cur_size + 1
            new_conf = cur_conf + added_conf
            delta = (new_size - 10000 * new_conf) - (cur_size - 10000 * cur_conf)
        
        if delta >= 0 or rng.random() < np.exp(delta / t):
            cur_size += (1 if not old_state else -1)
            if old_state:
                removed_conf = sum(1 for nb in adj[v] if cur[nb])
                cur_conf = max(0, cur_conf - removed_conf)
            else:
                added_conf = sum(1 for nb in adj[v] if cur[nb])
                cur_conf += added_conf
            
            obj = cur_size - 10000 * cur_conf
            if obj > best_size:
                best_size = obj
                best = cur[:]
        else:
            cur[v] = old_state  # revert
    
    return best

def _solve_instance(values, seed, deadline):
    """Solve one instance: find max sum-free subset."""
    n = len(values)
    if n == 0:
        return []
    
    rng = random.Random(seed)
    adj = _build_conflict_graph(values)
    
    best_in_set = None
    best_size = 0
    
    # Baseline: odds and evens
    odds_mask = [values[i] % 2 != 0 for i in range(n)]
    evens_mask = [values[i] % 2 == 0 for i in range(n)]
    if sum(odds_mask) > best_size:
        best_size = sum(odds_mask)
        best_in_set = odds_mask
    if sum(evens_mask) > best_size:
        best_size = sum(evens_mask)
        best_in_set = evens_mask
    
    # Greedy MIS
    if time.time() < deadline - 5:
        greedy = _greedy_mis(adj, n, rng, deadline - 2)
        gsize = sum(greedy)
        if gsize > best_size:
            best_size = gsize
            best_in_set = greedy
    
    # Local search with swaps
    if time.time() < deadline - 3 and best_in_set is not None:
        ls_time = min(deadline - time.time() - 2, 20.0)
        ls_result = _local_search_swap(best_in_set, adj, n, rng, ls_time)
        ls_size = sum(ls_result)
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result
    
    # Simulated annealing
    if time.time() < deadline - 2 and best_in_set is not None:
        sa_time = min(deadline - time.time() - 1, 15.0)
        sa_result = _simulated_annealing(adj, n, rng, 10.0, 0.01, sa_time, initial=best_in_set)
        sa_size = sum(sa_result)
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result
    
    # Final polish
    if time.time() < deadline - 1 and best_in_set is not None:
        polish_time = min(deadline - time.time() - 0.5, 5.0)
        polish_result = _local_search_swap(best_in_set, adj, n, rng, polish_time)
        p_size = sum(polish_result)
        if p_size > best_size:
            best_size = p_size
            best_in_set = polish_result
    
    if best_in_set is None:
        return []
    
    return [values[i] for i in range(n) if best_in_set[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s deadline with 5s safety margin
    
    instances = [
        _instance(60, 801),
        _instance(120, 802),
        _instance(240, 803)
    ]
    
    results = []
    for i, values in enumerate(instances):
        # Allocate time: give more to larger instances
        # Total budget ~140s, split: 20% for n=60, 30% for n=120, 50% for n=240
        weights = [0.2, 0.3, 0.5]
        total_budget = 140.0
        time_budget = total_budget * weights[i]
        
        inst_deadline = min(start_time + time_budget + 10, deadline - 2)
        subset = _solve_instance(values, 801 + i, inst_deadline)
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
