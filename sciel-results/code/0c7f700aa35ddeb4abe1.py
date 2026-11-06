"""Two Sidon sets with disjoint difference sets – Pair-feasibility + SA swap search."""
# EVOLVE-BLOCK-START
import time
import random
import math

N = 256
MAX_EACH = 64
DEADLINE = 52.0  # safety margin before 60s kill


def _build_pair_feasibility():
    """Precompute for each pair (i,j) whether they can coexist in a Sidon set.
    Two elements a,b can coexist iff a+b hasn't been used and |a-b| hasn't been used.
    We precompute a bitset for each element showing which other elements it conflicts with
    via sum or difference (within the Sidon constraint alone, ignoring the cross-set diff constraint).
    """
    # For a Sidon set, pair (a,b) with a!=b is valid iff no other pair has same sum or same diff.
    # We precompute: for each element a, which elements b conflict with a via sum (a+b) or diff (|a-b|).
    # sum_conflict[a] = bitmask of b where a+b equals some other pair's sum
    # diff_conflict[a] = bitmask of b where |a-b| equals some other pair's diff
    # This is complex; instead we'll use a simpler approach:
    # Precompute all pair sums and pair diffs, and for each element, which other elements
    # share a sum or diff with it.
    
    # sum_map[s] = list of pairs (a,b) with a+b=s
    # diff_map[d] = list of pairs (a,b) with |a-b|=d
    sum_map = {}
    diff_map = {}
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            s = a + b
            sum_map.setdefault(s, []).append((a, b))
            if a != b:
                d = b - a
                diff_map.setdefault(d, []).append((a, b))
    
    # For each element a, compute which elements b are "sum-incompatible" with a
    # i.e., there exists c != a, c != b such that a+b = c+d for some d, or |a-b| = |c-d|
    # Actually, the Sidon condition is: all pair sums (including a+a) are unique.
    # So (a,b) is valid in a Sidon set iff no other pair (c,d) with c<=d has c+d = a+b.
    # And for differences: |a-b| must not equal |c-d| for any other pair.
    
    # Precompute: for each a, a bitmask sum_incompat[a] where bit b is set if
    # a+b conflicts with some other pair sum.
    # This is O(N^3) which is too slow. Let's use a different approach.
    
    # Alternative: just precompute pair_sums and pair_diffs as sets for quick lookup.
    return sum_map, diff_map


def _can_add_fast(elem, elems, sum_used, diff_used, other_diff_used):
    """Fast check using precomputed structures."""
    s2 = elem + elem
    if s2 in sum_used:
        return False
    for a in elems:
        s = elem + a
        if s in sum_used:
            return False
        d = abs(elem - a)
        if d in diff_used or d in other_diff_used:
            return False
    return True


def _add_fast(state, side, elem):
    """Add elem to side using set-based tracking."""
    elems = state['elems'][side]
    sum_used = state['sum_used'][side]
    diff_used = state['diff_used'][side]
    
    sum_used.add(elem + elem)
    for a in elems:
        sum_used.add(elem + a)
        diff_used.add(abs(elem - a))
    
    elems.append(elem)


def _remove_fast(state, side, elem):
    """Remove elem from side, rebuild sets."""
    elems = state['elems'][side]
    idx = elems.index(elem)
    elems[idx] = elems[-1]
    elems.pop()
    
    # Rebuild
    sum_used = set()
    diff_used = set()
    for i, a in enumerate(elems):
        sum_used.add(a + a)
        for j in range(i + 1, len(elems)):
            b = elems[j]
            sum_used.add(a + b)
            diff_used.add(abs(a - b))
    
    state['sum_used'][side] = sum_used
    state['diff_used'][side] = diff_used


def _init_state():
    return {
        'elems': [[], []],
        'sum_used': [set(), set()],
        'diff_used': [set(), set()]
    }


def _total_size(state):
    return len(state['elems'][0]) + len(state['elems'][1])


def _greedy_build(order, deadline):
    """Greedy build in given order."""
    state = _init_state()
    
    for elem in order:
        if time.time() > deadline:
            break
        if _can_add_fast(elem, state['elems'][0], state['sum_used'][0],
                         state['diff_used'][0], state['diff_used'][1]):
            _add_fast(state, 0, elem)
        elif _can_add_fast(elem, state['elems'][1], state['sum_used'][1],
                           state['diff_used'][1], state['diff_used'][0]):
            _add_fast(state, 1, elem)
    
    return state, _total_size(state)


def _sa_search(state, deadline):
    """Simulated annealing with swap moves to escape local minima."""
    best_state = {
        'elems': [list(state['elems'][0]), list(state['elems'][1])],
        'sum_used': [set(state['sum_used'][0]), set(state['sum_used'][1])],
        'diff_used': [set(state['diff_used'][0]), set(state['diff_used'][1])]
    }
    best_total = _total_size(best_state)
    cur_total = best_total
    
    # Temperature schedule
    t_start = 1.0
    t_end = 0.01
    n_steps = 2000
    
    all_elems_set = set(best_state['elems'][0]) | set(best_state['elems'][1])
    
    step = 0
    while time.time() < deadline and step < n_steps:
        step += 1
        T = t_start * (t_end / t_start) ** (step / n_steps)
        
        # Choose a random move: swap x in side i with y not in either set
        side = random.randint(0, 1)
        elems = best_state['elems'][side]
        if not elems:
            continue
        
        x = random.choice(elems)
        
        # Remove x
        _remove_fast(best_state, side, x)
        all_elems_set.discard(x)
        
        # Try to find a y to add
        candidates = [y for y in range(1, N + 1) if y not in all_elems_set]
        if not candidates:
            _add_fast(best_state, side, x)
            all_elems_set.add(x)
            continue
        
        # Try a random candidate first, then best if time permits
        found = False
        y = random.choice(candidates)
        if _can_add_fast(y, best_state['elems'][side], best_state['sum_used'][side],
                         best_state['diff_used'][side], best_state['diff_used'][1 - side]):
            _add_fast(best_state, side, y)
            all_elems_set.add(y)
            found = True
            new_total = _total_size(best_state)
        else:
            # Try a few more random candidates
            for _ in range(10):
                y = random.choice(candidates)
                if _can_add_fast(y, best_state['elems'][side], best_state['sum_used'][side],
                                 best_state['diff_used'][side], best_state['diff_used'][1 - side]):
                    _add_fast(best_state, side, y)
                    all_elems_set.add(y)
                    found = True
                    new_total = _total_size(best_state)
                    break
        
        if not found:
            # Revert: put x back
            _add_fast(best_state, side, x)
            all_elems_set.add(x)
            continue
        
        new_total = _total_size(best_state)
        delta = new_total - cur_total
        
        # Accept if better or with probability exp(delta/T) for worse
        if delta >= 0 or random.random() < math.exp(delta / max(T, 1e-9)):
            cur_total = new_total
            if new_total > best_total:
                best_total = new_total
                best_state['elems'] = [list(best_state['elems'][0]), list(best_state['elems'][1])]
                best_state['sum_used'] = [set(best_state['sum_used'][0]), set(best_state['sum_used'][1])]
                best_state['diff_used'] = [set(best_state['diff_used'][0]), set(best_state['diff_used'][1])]
        else:
            # Revert: remove y, add x back
            _remove_fast(best_state, side, y)
            all_elems_set.discard(y)
            _add_fast(best_state, side, x)
            all_elems_set.add(x)
            cur_total = best_total  # revert to previous total
    
    return best_state, best_total


def _hill_climb(state, deadline):
    """Fast hill-climbing: try all 1-for-1 swaps, keep first improvement."""
    best_state = {
        'elems': [list(state['elems'][0]), list(state['elems'][1])],
        'sum_used': [set(state['sum_used'][0]), set(state['sum_used'][1])],
        'diff_used': [set(state['diff_used'][0]), set(state['diff_used'][1])]
    }
    best_total = _total_size(best_state)
    
    all_elems_set = set(best_state['elems'][0]) | set(best_state['elems'][1])
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        for side in (0, 1):
            elems = list(best_state['elems'][side])
            for x in elems:
                if time.time() > deadline:
                    return best_state, best_total
                
                # Remove x
                _remove_fast(best_state, side, x)
                all_elems_set.discard(x)
                
                # Try adding each candidate y
                for y in range(1, N + 1):
                    if y in all_elems_set:
                        continue
                    if _can_add_fast(y, best_state['elems'][side], best_state['sum_used'][side],
                                     best_state['diff_used'][side], best_state['diff_used'][1 - side]):
                        _add_fast(best_state, side, y)
                        all_elems_set.add(y)
                        new_total = _total_size(best_state)
                        if new_total > best_total:
                            best_total = new_total
                            improved = True
                        break
                else:
                    # No improvement, revert
                    _add_fast(best_state, side, x)
                    all_elems_set.add(x)
    
    return best_state, best_total


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_total = 0
    best_result = ([1], [2])  # fallback
    
    random.seed(42)
    
    while time.time() < deadline - 3.0:
        # Generate random order with some structure: alternate low/high
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        # Greedy build
        state, total = _greedy_build(order, deadline)
        
        if total > best_total:
            best_total = total
            best_result = (list(state['elems'][0]), list(state['elems'][1]))
        
        # Hill climb first (fast, deterministic improvement)
        if time.time() < deadline - 2.0:
            hc_state, hc_total = _hill_climb(state, deadline)
            if hc_total > best_total:
                best_total = hc_total
                best_result = (list(hc_state['elems'][0]), list(hc_state['elems'][1]))
                state = hc_state
        
        # SA search to escape local minima
        if time.time() < deadline - 1.0:
            sa_state, sa_total = _sa_search(state, deadline)
            if sa_total > best_total:
                best_total = sa_total
                best_result = (list(sa_state['elems'][0]), list(sa_state['elems'][1]))
    
    # Ensure nonempty
    if not best_result[0]:
        best_result = ([1], list(best_result[1]) if best_result[1] else [2])
    if not best_result[1]:
        best_result = (list(best_result[0]), [2])
    
    return list(best_result)
# EVOLVE-BLOCK-END
