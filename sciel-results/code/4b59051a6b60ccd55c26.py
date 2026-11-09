"""Combined program: set-based O(1) can_add + validity-preserving SA with numpy conflict counting."""
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
    """Build adjacency: edge (i,j) if values[i]+values[j] is in the set (a+b=c constraint)."""
    val_set = set(values)
    n = len(values)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = values[i] + values[j]
            if s in val_set:
                adj[i].add(j)
                adj[j].add(i)
    # Convert to lists for iteration
    adj = [sorted(lst) for lst in adj]
    return adj

def _greedy_mis(adj, n, rng, deadline):
    """Greedy Maximum Independent Set: pick lowest-degree remaining vertex."""
    in_set = [False] * n
    remaining = [True] * n
    rem_count = n
    
    while rem_count > 0:
        if time.time() > deadline:
            break
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

def _local_search_swap(in_set, adj, n, rng, time_limit):
    """Local search with O(1) can_add via selected set. Swap + add-only moves."""
    cur = in_set[:]
    selected = set(i for i in range(n) if cur[i])
    cur_size = len(selected)
    best = cur[:]
    best_size = cur_size
    
    start = time.time()
    no_improve = 0
    
    while time.time() - start < time_limit:
        if not selected:
            break
        
        improved = False
        out_list = [i for i in range(n) if not cur[i]]
        
        if not out_list:
            break
        
        # Try swaps: remove one in-set, add one out-set
        in_list = list(selected)
        n_in = len(in_list)
        n_out = len(out_list)
        
        in_sample = rng.sample(in_list, min(80, n_in))
        out_sample = rng.sample(out_list, min(50, n_out))
        
        for v_out in out_sample:
            if cur[v_out]:
                continue
            # Check if v_out can be added (no conflicts with current selected)
            ok = all(nb not in selected for nb in adj[v_out])
            if ok:
                # Add-only: just add v_out
                cur[v_out] = True
                selected.add(v_out)
                cur_size += 1
                improved = True
                if cur_size > best_size:
                    best_size = cur_size
                    best = cur[:]
                break
        
        if not improved:
            # Try swap: remove v_in, add v_out where v_out conflicts only with v_in
            for v_in in in_sample:
                if not cur[v_in]:
                    continue
                for v_out in out_sample:
                    if cur[v_out]:
                        continue
                    # v_out must not conflict with any selected except possibly v_in
                    conflicts_with = [nb for nb in adj[v_out] if nb in selected and nb != v_in]
                    if not conflicts_with:
                        cur[v_in] = False
                        selected.discard(v_in)
                        cur[v_out] = True
                        selected.add(v_out)
                        improved = True
                        if cur_size > best_size:
                            best_size = cur_size
                            best = cur[:]
                        break
                if improved:
                    break
        
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 300:
                break
    
    # Final add-only polish
    changed = True
    while changed and time.time() - start < time_limit:
        changed = False
        for v in range(n):
            if cur[v]:
                continue
            if all(nb not in selected for nb in adj[v]):
                cur[v] = True
                selected.add(v)
                cur_size += 1
                changed = True
                if cur_size > best_size:
                    best_size = cur_size
                    best = cur[:]
    
    return best

def _sa_validity_preserving(adj, n, rng, t_start, t_end, time_limit, initial):
    """SA that only accepts moves maintaining sum-freeness.
    Moves: add a free vertex (always accept if size increases),
           remove a vertex (accept with prob exp(-1/T)),
           swap (remove one, add one if valid).
    Uses O(degree) delta checks via selected set.
    """
    cur = initial[:]
    selected = set(i for i in range(n) if cur[i])
    cur_size = len(selected)
    
    best = cur[:]
    best_size = cur_size
    
    start = time.time()
    it = 0
    max_it = 500000
    
    while time.time() - start < time_limit and it < max_it:
        it += 1
        t = t_start * (t_end / t_start) ** (min(it, 100000) / 100000.0)
        t = max(t, 1e-10)
        
        move_type = rng.random()
        
        if move_type < 0.5:
            # Try ADD: pick a random vertex not in selected
            candidates = [v for v in range(n) if not cur[v]]
            if not candidates:
                continue
            v = rng.choice(candidates)
            # Check if v is free (no conflicts with selected)
            if all(nb not in selected for nb in adj[v]):
                cur[v] = True
                selected.add(v)
                cur_size += 1
                if cur_size > best_size:
                    best_size = cur_size
                    best = cur[:]
        
        elif move_type < 0.8:
            # Try REMOVE: pick a random vertex in selected
            if not selected:
                continue
            v = rng.choice(list(selected))
            # Remove: size decreases by 1, accept with prob exp(-1/T)
            if rng.random() < np.exp(-1.0 / t):
                cur[v] = False
                selected.discard(v)
                cur_size -= 1
        
        else:
            # Try SWAP: remove v_in, add v_out where v_out only conflicts with v_in
            if not selected:
                continue
            v_in = rng.choice(list(selected))
            # Find vertices that conflict with v_in but not with rest of selected
            # v_out must be: not in selected, and its only conflict with selected is v_in
            candidates = []
            for nb in adj[v_in]:
                if not cur[nb]:
                    # Check nb doesn't conflict with any other selected
                    other_conflicts = any(x in selected and x != v_in for x in adj[nb])
                    if not other_conflicts:
                        candidates.append(nb)
            if candidates:
                v_out = rng.choice(candidates)
                cur[v_in] = False
                selected.discard(v_in)
                cur[v_out] = True
                selected.add(v_out)
                # Size unchanged, but structure changed - always accept for exploration
    
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
        best_in_set = odds_mask[:]
    if sum(evens_mask) > best_size:
        best_size = sum(evens_mask)
        best_in_set = evens_mask[:]
    
    # Greedy MIS
    if time.time() < deadline - 3:
        greedy = _greedy_mis(adj, n, rng, deadline - 1)
        gsize = sum(greedy)
        if gsize > best_size:
            best_size = gsize
            best_in_set = greedy
    
    # Local search with fast O(1) can_add
    if time.time() < deadline - 2 and best_in_set is not None:
        ls_time = min(deadline - time.time() - 1, 15.0)
        ls_result = _local_search_swap(best_in_set, adj, n, rng, ls_time)
        ls_size = sum(ls_result)
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result
    
    # Validity-preserving SA
    if time.time() < deadline - 1 and best_in_set is not None:
        sa_time = min(deadline - time.time() - 0.5, 15.0)
        sa_result = _sa_validity_preserving(adj, n, rng, 5.0, 0.01, sa_time, initial=best_in_set)
        sa_size = sum(sa_result)
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result
    
    # Final polish
    if time.time() < deadline - 0.5 and best_in_set is not None:
        polish_time = min(deadline - time.time() - 0.3, 5.0)
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
    deadline = start_time + 145
    
    instances = [
        _instance(60, 801),
        _instance(120, 802),
        _instance(240, 803)
    ]
    
    results = []
    for i, values in enumerate(instances):
        weights = [0.2, 0.3, 0.5]
        total_budget = 140.0
        time_budget = total_budget * weights[i]
        
        inst_deadline = min(start_time + time_budget + 10, deadline - 2)
        subset = _solve_instance(values, 801 + i, inst_deadline)
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
