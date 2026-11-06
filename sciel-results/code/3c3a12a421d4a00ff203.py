import time
import random
import numpy as np

# EVOLVE-BLOCK-START

# Precompute: for each element x and difference d, what are the other two elements
# of the 3-AP (a, a+d, a+2d) that x participates in.
# For each x, store list of (d, other1, other2) where {x, other1, other2} is a 3-AP with diff d.

def _build_ap_data():
    """Build AP adjacency data using numpy for speed."""
    # ap_data[x] = list of (d, y, z) where {x,y,z} forms a 3-AP with difference d
    ap_data = [[] for _ in range(513)]
    
    for d in range(1, 256):
        for a in range(1, 512 - 2*d + 1):
            b = a + d
            c = a + 2*d
            # x=a: partners are b,c
            ap_data[a].append((d, b, c))
            # x=b: partners are a,c
            ap_data[b].append((d, a, c))
            # x=c: partners are a,b
            ap_data[c].append((d, a, b))
    
    return ap_data

AP_DATA = _build_ap_data()

def _compute_score(S):
    """Compute score: number of distinct d with a 3-AP in S."""
    S_set = set(S)
    covered = np.zeros(256, dtype=np.int8)
    for x in S:
        for d, y, z in AP_DATA[x]:
            if y in S_set and z in S_set:
                covered[d] = 1
    return int(np.sum(covered))

def _init_state(S):
    """Initialize state: S_set, d_count array, score."""
    S_set = set(S)
    d_count = np.zeros(256, dtype=np.int32)
    for x in S:
        for d, y, z in AP_DATA[x]:
            if y in S_set and z in S_set:
                d_count[d] += 1
    d_count //= 3  # each AP counted 3 times (once per element)
    score = int(np.sum(d_count > 0))
    return S_set, d_count, score

def _apply_remove(S_set, d_count, x):
    """Remove x from state, return new score."""
    score_change = 0
    for d, y, z in AP_DATA[x]:
        if y in S_set and z in S_set:
            d_count[d] -= 1
            if d_count[d] == 0:
                score_change -= 1
    S_set.discard(x)
    return score_change

def _apply_add(S_set, d_count, x):
    """Add x to state, return new score."""
    score_change = 0
    for d, y, z in AP_DATA[x]:
        if y in S_set and z in S_set:
            if d_count[d] == 0:
                score_change += 1
            d_count[d] += 1
    S_set.add(x)
    return score_change

def _swap_delta(S_set, d_count, old, new):
    """Compute score change for swapping old->new. Returns delta."""
    # Remove old
    delta = 0
    for d, y, z in AP_DATA[old]:
        if y in S_set and z in S_set:
            d_count[d] -= 1
            if d_count[d] == 0:
                delta -= 1
    S_set.discard(old)
    
    # Add new
    for d, y, z in AP_DATA[new]:
        if y in S_set and z in S_set:
            if d_count[d] == 0:
                delta += 1
            d_count[d] += 1
    S_set.add(new)
    
    return delta

def _swap_apply(S_set, d_count, old, new):
    """Apply swap old->new. Returns delta."""
    delta = 0
    for d, y, z in AP_DATA[old]:
        if y in S_set and z in S_set:
            d_count[d] -= 1
            if d_count[d] == 0:
                delta -= 1
    S_set.discard(old)
    
    for d, y, z in AP_DATA[new]:
        if y in S_set and z in S_set:
            if d_count[d] == 0:
                delta += 1
            d_count[d] += 1
    S_set.add(new)
    
    return delta

def _swap_revert(S_set, d_count, old, new):
    """Revert swap: remove new, add old. Returns delta (should be -swap_delta)."""
    delta = 0
    for d, y, z in AP_DATA[new]:
        if y in S_set and z in S_set:
            d_count[d] -= 1
            if d_count[d] == 0:
                delta -= 1
    S_set.discard(new)
    
    for d, y, z in AP_DATA[old]:
        if y in S_set and z in S_set:
            if d_count[d] == 0:
                delta += 1
            d_count[d] += 1
    S_set.add(old)
    
    return delta

def _greedy_init(rng, n=32):
    """Greedy set-cover style initialization."""
    S = []
    S_set = set()
    # Start with a random element
    S.append(rng.randint(1, 512))
    S_set.add(S[0])
    
    while len(S) < n:
        best_elem = None
        best_gain = -1
        # Try all candidates
        for cand in range(1, 513):
            if cand in S_set:
                continue
            gain = 0
            for d, y, z in AP_DATA[cand]:
                if y in S_set and z in S_set:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_elem = cand
        if best_elem is None:
            available = [x for x in range(1, 513) if x not in S_set]
            if available:
                best_elem = rng.choice(available)
            else:
                break
        S.append(best_elem)
        S_set.add(best_elem)
    
    return S

def _local_search(S_init, time_budget, rng):
    """Aggressive local search with swaps."""
    S = list(S_init)
    S_set, d_count, score = _init_state(S)
    best_S = list(S)
    best_score = score
    
    start = time.time()
    T = 3.0
    cooling = 0.9995
    min_T = 0.05
    
    while time.time() - start < time_budget:
        T = max(T * cooling, min_T)
        
        i = rng.randint(0, 31)
        old = S[i]
        new = rng.randint(1, 512)
        if new == old or new in S_set:
            continue
        
        delta = _swap_apply(S_set, d_count, old, new)
        new_score = score + delta
        
        if delta >= 0 or rng.random() < np.exp(delta / T):
            score = new_score
            S[i] = new
            if score > best_score:
                best_score = score
                best_S = list(S)
        else:
            _swap_revert(S_set, d_count, old, new)
            score -= delta
    
    return best_S, best_score

def _greedy_polish(S_init, time_budget):
    """Greedy polish: try all swaps for each position, keep best."""
    S = list(S_init)
    S_set, d_count, score = _init_state(S)
    
    start = time.time()
    improved = True
    while improved and time.time() - start < time_budget:
        improved = False
        for i in range(32):
            old = S[i]
            best_new = None
            best_delta = 0
            
            for cand in range(1, 513):
                if cand in S_set:
                    continue
                # Try swap
                d = _swap_apply(S_set, d_count, old, cand)
                if d > best_delta:
                    best_delta = d
                    best_new = cand
                _swap_revert(S_set, d_count, old, cand)
            
            if best_new is not None and best_delta > 0:
                _swap_apply(S_set, d_count, old, best_new)
                score += best_delta
                S[i] = best_new
                improved = True
    
    return S, score

def _pairwise_swap_search(S_init, time_budget):
    """Try all pairwise swaps to escape local optima."""
    S = list(S_init)
    S_set, d_count, score = _init_state(S)
    best_S = list(S)
    best_score = score
    
    start = time.time()
    # Try random pairs of positions
    while time.time() - start < time_budget:
        i, j = rng.sample(range(32), 2)
        old_i, old_j = S[i], S[j]
        
        # Try swapping S[i] with random new, S[j] with random new
        new_i = rng.randint(1, 512)
        new_j = rng.randint(1, 512)
        
        if new_i in S_set or new_j in S_set or new_i == new_j:
            continue
        if new_i == old_i or new_j == old_j:
            continue
        
        # Apply both changes
        d1 = _swap_apply(S_set, d_count, old_i, new_i)
        d2 = _swap_apply(S_set, d_count, old_j, new_j)
        total_d = d1 + d2
        
        if total_d > 0 or rng.random() < np.exp(total_d / 2.0):
            score += total_d
            S[i] = new_i
            S[j] = new_j
            if score > best_score:
                best_score = score
                best_S = list(S)
        else:
            _swap_revert(S_set, d_count, old_j, new_j)
            _swap_revert(S_set, d_count, old_i, new_i)
    
    return best_S, best_score

def _structured_init(rng, strategy):
    """Generate diverse structured initializations."""
    if strategy == 0:
        return sorted(rng.sample(range(1, 513), 32))
    elif strategy == 1:
        # Clustered
        S = set()
        for _ in range(3):
            center = rng.randint(20, 490)
            for off in range(-8, 9):
                v = center + off
                if 1 <= v <= 512:
                    S.add(v)
        while len(S) < 32:
            S.add(rng.randint(1, 512))
        return sorted(list(S)[:32])
    elif strategy == 2:
        # Spread with perturbation
        base = [int(1 + i * 511.0 / 31) for i in range(32)]
        S = set(base)
        for _ in range(10):
            S.add(rng.randint(1, 512))
        return sorted(list(S)[:32])
    elif strategy == 3:
        # Residue class based
        S = set()
        mod = rng.choice([3, 4, 5, 6, 7, 8])
        rem = rng.randint(0, mod - 1)
        candidates = [x for x in range(1, 513) if x % mod == rem]
        S = set(rng.sample(candidates, min(32, len(candidates))))
        while len(S) < 32:
            S.add(rng.randint(1, 512))
        return sorted(list(S)[:32])
    else:
        return sorted(rng.sample(range(1, 513), 32))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    rng = random.Random(12345)
    
    best_S = list(range(1, 33))
    best_score = _compute_score(best_S)
    
    # Phase 1: Multiple restarts with local search
    n_restarts = 15
    per_restart = 7.0
    
    for r in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        
        if r == 0:
            S_init = _greedy_init(rng)
        elif r == 1:
            S_init = _greedy_init(rng)
        else:
            strategy = (r - 2) % 4
            S_init = _structured_init(rng, strategy)
        
        remaining = min(per_restart, deadline - time.time() - 15.0)
        if remaining < 2.0:
            break
        
        S, score = _local_search(S_init, remaining, rng)
        
        if score > best_score:
            best_score = score
            best_S = list(S)
    
    # Phase 2: Greedy polish on best
    if time.time() < deadline - 10.0:
        best_S, best_score = _greedy_polish(best_S, time_budget=8.0)
    
    # Phase 3: More local search from polished result
    if time.time() < deadline - 5.0:
        S, score = _local_search(best_S, min(5.0, deadline - time.time() - 3.0), rng)
        if score > best_score:
            best_score = score
            best_S = list(S)
    
    # Phase 4: Final greedy polish
    if time.time() < deadline - 2.0:
        best_S, best_score = _greedy_polish(best_S, time_budget=min(3.0, deadline - time.time() - 1.0))
    
    # Final verification
    final_score = _compute_score(best_S)
    if final_score > best_score:
        best_score = final_score
    
    return sorted(best_S)

# EVOLVE-BLOCK-END
