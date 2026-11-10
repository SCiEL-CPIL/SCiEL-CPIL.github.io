"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
N = 256

def _precompute():
    """Precompute 3-AP and 4-AP partner tables for fast lookups."""
    ap3_partners = [[] for _ in range(N + 1)]
    ap4_partners = [[] for _ in range(N + 1)]
    
    # 3-APs: (a, a+d, a+2d)
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            ap3_partners[a].append((b, c))
            ap3_partners[b].append((a, c))
            ap3_partners[c].append((a, b))
            
    # 4-APs: (a, a+d, a+2d, a+3d)
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap4_partners[a].append((b, c, e))
            ap4_partners[b].append((a, c, e))
            ap4_partners[c].append((a, b, e))
            ap4_partners[e].append((a, b, c))
            
    return ap3_partners, ap4_partners

_ap3_partners, _ap4_partners = _precompute()

def _is_valid_addition(x, in_set):
    """Check if adding x to in_set creates a 4-AP."""
    for j, k, l in _ap4_partners[x]:
        if in_set[j] and in_set[k] and in_set[l]:
            return False
    return True

def _count_3aps(in_set):
    """Count total 3-APs in the set."""
    total = 0
    for i in range(1, N + 1):
        if in_set[i]:
            for j, k in _ap3_partners[i]:
                if in_set[j] and in_set[k]:
                    total += 1
    return total // 3

def _gain(x, in_set, exclude=None):
    """Number of 3-APs formed by adding x to in_set (optionally excluding one element)."""
    count = 0
    for j, k in _ap3_partners[x]:
        j_ok = in_set[j] and (j != exclude)
        k_ok = in_set[k] and (k != exclude)
        if j_ok and k_ok:
            count += 1
    return count

def _loss(x, in_set):
    """Number of 3-APs lost by removing x from in_set."""
    count = 0
    for j, k in _ap3_partners[x]:
        if in_set[j] and in_set[k]:
            count += 1
    return count

def _greedy_weighted_construct(rng, deadline):
    """GRASP with gain-based selection from top-K valid candidates."""
    in_set = np.zeros(N + 1, dtype=bool)
    candidates = list(range(1, N + 1))
    
    while candidates and time.time() < deadline:
        valid = []
        gains = {}
        for y in candidates:
            if _is_valid_addition(y, in_set):
                g = _gain(y, in_set)
                valid.append(y)
                gains[y] = g
        
        if not valid:
            break
        
        # Select from top K by gain
        K = min(5, len(valid))
        valid.sort(key=lambda x: gains[x], reverse=True)
        top_k = valid[:K]
        y = rng.choice(top_k)
        
        in_set[y] = True
        candidates.remove(y)
    
    return in_set

def _local_search_add(in_set):
    """Greedy add: add any element that doesn't create a 4-AP, prioritizing high gain."""
    changed = True
    while changed:
        changed = False
        best_x = -1
        best_gain = -1
        for x in range(1, N + 1):
            if not in_set[x]:
                if _is_valid_addition(x, in_set):
                    g = _gain(x, in_set)
                    if g > best_gain:
                        best_gain = g
                        best_x = x
        if best_x > 0:
            in_set[best_x] = True
            changed = True
    return in_set

def _tabu_search(in_set, current_score, deadline, rng):
    """Tabu search with add/remove/swap moves using precomputed partners."""
    tabu_add = {}   # x -> iteration when non-tabu for adding
    tabu_remove = {} # x -> iteration when non-tabu for removing
    tabu_swap = {}   # (x,y) -> iteration when non-tabu for swapping
    tabu_tenure = 8
    best_in_set = in_set.copy()
    best_score = current_score
    iter_count = 0
    no_improve = 0
    max_no_improve = 60
    
    while time.time() < deadline and no_improve < max_no_improve:
        iter_count += 1
        
        best_delta = -10**9
        best_move = None
        
        # Add moves
        for x in range(1, N + 1):
            if not in_set[x]:
                if x in tabu_add and tabu_add[x] > iter_count:
                    continue
                if _is_valid_addition(x, in_set):
                    delta = _gain(x, in_set)
                    if delta > best_delta:
                        best_delta = delta
                        best_move = ('add', x, delta)
        
        # Remove moves
        for x in range(1, N + 1):
            if in_set[x]:
                if x in tabu_remove and tabu_remove[x] > iter_count:
                    continue
                delta = -_loss(x, in_set)
                if delta > best_delta:
                    best_delta = delta
                    best_move = ('remove', x, delta)
        
        # Swap moves (limited sampling for speed)
        present = [x for x in range(1, N + 1) if in_set[x]]
        absent = [x for x in range(1, N + 1) if not in_set[x]]
        
        if present and absent:
            n_sample = min(15, len(present), len(absent))
            if len(present) > n_sample:
                present_s = rng.sample(present, n_sample)
            else:
                present_s = present
            if len(absent) > n_sample:
                absent_s = rng.sample(absent, n_sample)
            else:
                absent_s = absent
            
            for x_out in present_s:
                if x_out in tabu_remove and tabu_remove[x_out] > iter_count:
                    continue
                for x_in in absent_s:
                    if x_in in tabu_add and tabu_add[x_in] > iter_count:
                        continue
                    # Check swap validity
                    valid = True
                    for j, k, l in _ap4_partners[x_in]:
                        if (in_set[j] and j != x_out) and (in_set[k] and k != x_out) and (in_set[l] and l != x_out):
                            valid = False
                            break
                    if not valid:
                        continue
                    # Check tabu for swap
                    swap_key = (min(x_out, x_in), max(x_out, x_in))
                    if swap_key in tabu_swap and tabu_swap[swap_key] > iter_count:
                        continue
                    gain_in = _gain(x_in, in_set, exclude=x_out)
                    loss_out = _loss(x_out, in_set)
                    delta = gain_in - loss_out
                    if delta > best_delta:
                        best_delta = delta
                        best_move = ('swap', (x_out, x_in), delta)
        
        if best_move is None:
            break
        
        mtype, mdata, mdelta = best_move
        
        if mtype == 'add':
            x = mdata
            in_set[x] = True
            current_score += mdelta
            tabu_add[x] = iter_count + tabu_tenure
        elif mtype == 'remove':
            x = mdata
            in_set[x] = False
            current_score += mdelta
            tabu_remove[x] = iter_count + tabu_tenure
        elif mtype == 'swap':
            x_out, x_in = mdata
            in_set[x_out] = False
            in_set[x_in] = True
            current_score += mdelta
            tabu_remove[x_out] = iter_count + tabu_tenure
            tabu_add[x_in] = iter_count + tabu_tenure
            tabu_swap[(min(x_out, x_in), max(x_out, x_in))] = iter_count + tabu_tenure
        
        if current_score > best_score:
            best_score = current_score
            best_in_set = in_set.copy()
            no_improve = 0
        else:
            no_improve += 1
    
    return best_in_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 82s deadline with 8s safety margin
    
    rng = random.Random(42)
    
    best_in_set = None
    best_score = -1
    
    n_restarts = 25
    time_per_restart = (deadline - start_time) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        # Phase 1: Greedy weighted construction
        construct_deadline = min(deadline - 5, time.time() + time_per_restart * 0.3)
        in_set = _greedy_weighted_construct(rng, construct_deadline)
        
        # Phase 2: Local search - add all possible high-gain elements
        in_set = _local_search_add(in_set)
        current_score = _count_3aps(in_set)
        
        # Phase 3: Tabu search
        tabu_deadline = min(deadline - 5, time.time() + time_per_restart * 0.6)
        in_set, current_score = _tabu_search(in_set, current_score, tabu_deadline, rng)
        
        # Phase 4: Final local search polish
        in_set = _local_search_add(in_set)
        current_score = _count_3aps(in_set)
        
        if current_score > best_score:
            best_score = current_score
            best_in_set = in_set.copy()
    
    # Convert to sorted list
    if best_in_set is None:
        # Fallback: just return [1]
        return [1]
    
    result = [x for x in range(1, N + 1) if best_in_set[x]]
    return sorted(result)
# EVOLVE-BLOCK-END
