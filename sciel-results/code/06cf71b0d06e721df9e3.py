"""Two Sidon sets with disjoint difference sets – Bitset-based fast search."""
# EVOLVE-BLOCK-START
import time
import random
import math

N = 256
MAX_EACH = 64
DEADLINE = 55.0  # safety margin before 60s kill


def _can_add(elem, elems, sum_used, diff_used, other_diff_used):
    """Check if elem can be added to the set with given used sums/diffs."""
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


def _add(elem, elems, sum_used, diff_used):
    """Add elem to the set, updating used sums and diffs."""
    sum_used.add(elem + elem)
    for a in elems:
        sum_used.add(elem + a)
        diff_used.add(abs(elem - a))
    elems.append(elem)


def _remove(elem, elems, sum_used, diff_used):
    """Remove elem from the set, rebuilding used sums and diffs."""
    idx = elems.index(elem)
    elems[idx] = elems[-1]
    elems.pop()
    sum_used.clear()
    diff_used.clear()
    for i in range(len(elems)):
        a = elems[i]
        sum_used.add(a + a)
        for j in range(i + 1, len(elems)):
            b = elems[j]
            sum_used.add(a + b)
            diff_used.add(abs(a - b))


def _total_size(elems0, elems1):
    return len(elems0) + len(elems1)


def _greedy_build(order, deadline):
    """Greedy build in given order."""
    elems0, elems1 = [], []
    sum_used0, sum_used1 = set(), set()
    diff_used0, diff_used1 = set(), set()
    
    for elem in order:
        if time.time() > deadline:
            break
        if _can_add(elem, elems0, sum_used0, diff_used0, diff_used1):
            _add(elem, elems0, sum_used0, diff_used0)
        elif _can_add(elem, elems1, sum_used1, diff_used1, diff_used0):
            _add(elem, elems1, sum_used1, diff_used1)
    
    return (elems0, elems1), (sum_used0, sum_used1), (diff_used0, diff_used1), _total_size(elems0, elems1)


def _hill_climb(elems, sum_used, diff_used, deadline):
    """Fast hill-climbing: try all 1-for-1 swaps, keep first improvement."""
    elems0, elems1 = list(elems[0]), list(elems[1])
    sum_used0, sum_used1 = set(sum_used[0]), set(sum_used[1])
    diff_used0, diff_used1 = set(diff_used[0]), set(diff_used[1])
    best_total = _total_size(elems0, elems1)
    
    all_elems_set = set(elems0) | set(elems1)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        for side in (0, 1):
            elems_side = elems0 if side == 0 else elems1
            sum_side = sum_used0 if side == 0 else sum_used1
            diff_side = diff_used0 if side == 0 else diff_used1
            other_diff = diff_used1 if side == 0 else diff_used0
            elems_other = elems1 if side == 0 else elems0
            
            for x in list(elems_side):
                if time.time() > deadline:
                    return (elems0, elems1), (sum_used0, sum_used1), (diff_used0, diff_used1), best_total
                
                # Remove x
                _remove(x, elems_side, sum_side, diff_side)
                all_elems_set.discard(x)
                
                # Try adding each candidate y
                found = False
                for y in range(1, N + 1):
                    if y in all_elems_set:
                        continue
                    if _can_add(y, elems_side, sum_side, diff_side, other_diff):
                        _add(y, elems_side, sum_side, diff_side)
                        all_elems_set.add(y)
                        new_total = _total_size(elems0, elems1)
                        if new_total > best_total:
                            best_total = new_total
                            improved = True
                        found = True
                        break
                
                if not found:
                    # Revert: put x back
                    _add(x, elems_side, sum_side, diff_side)
                    all_elems_set.add(x)
    
    return (elems0, elems1), (sum_used0, sum_used1), (diff_used0, diff_used1), best_total


def _sa_search(elems, sum_used, diff_used, deadline):
    """Simulated annealing with swap moves to escape local minima."""
    elems0, elems1 = list(elems[0]), list(elems[1])
    sum_used0, sum_used1 = set(sum_used[0]), set(sum_used[1])
    diff_used0, diff_used1 = set(diff_used[0]), set(diff_used[1])
    best_total = _total_size(elems0, elems1)
    cur_total = best_total
    
    all_elems_set = set(elems0) | set(elems1)
    
    t_start = 1.0
    t_end = 0.01
    n_steps = 3000
    
    step = 0
    while time.time() < deadline and step < n_steps:
        step += 1
        T = t_start * (t_end / t_start) ** (step / n_steps)
        
        # Choose a random move: swap x in side i with y not in either set
        side = random.randint(0, 1)
        elems_side = elems0 if side == 0 else elems1
        sum_side = sum_used0 if side == 0 else sum_used1
        diff_side = diff_used0 if side == 0 else diff_used1
        other_diff = diff_used1 if side == 0 else diff_used0
        
        if not elems_side:
            continue
        
        x = random.choice(elems_side)
        
        # Remove x
        _remove(x, elems_side, sum_side, diff_side)
        all_elems_set.discard(x)
        
        # Try to find a y to add
        candidates = [y for y in range(1, N + 1) if y not in all_elems_set]
        if not candidates:
            _add(x, elems_side, sum_side, diff_side)
            all_elems_set.add(x)
            continue
        
        found = False
        y = random.choice(candidates)
        if _can_add(y, elems_side, sum_side, diff_side, other_diff):
            _add(y, elems_side, sum_side, diff_side)
            all_elems_set.add(y)
            found = True
            new_total = _total_size(elems0, elems1)
        else:
            # Try a few more random candidates
            for _ in range(10):
                y = random.choice(candidates)
                if _can_add(y, elems_side, sum_side, diff_side, other_diff):
                    _add(y, elems_side, sum_side, diff_side)
                    all_elems_set.add(y)
                    found = True
                    new_total = _total_size(elems0, elems1)
                    break
        
        if not found:
            # Revert: put x back
            _add(x, elems_side, sum_side, diff_side)
            all_elems_set.add(x)
            continue
        
        delta = new_total - cur_total
        
        # Accept if better or with probability exp(delta/T) for worse
        if delta >= 0 or random.random() < math.exp(delta / max(T, 1e-9)):
            cur_total = new_total
            if new_total > best_total:
                best_total = new_total
        else:
            # Revert: remove y, add x back
            _remove(y, elems_side, sum_side, diff_side)
            all_elems_set.discard(y)
            _add(x, elems_side, sum_side, diff_side)
            all_elems_set.add(x)
            cur_total = best_total  # revert to previous total
    
    return (elems0, elems1), (sum_used0, sum_used1), (diff_used0, diff_used1), best_total


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_total = 0
    best_result = ([1], [2])  # fallback
    
    random.seed(42)
    
    while time.time() < deadline - 3.0:
        # Generate random order with some structure
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        # Greedy build
        elems, sum_used, diff_used, total = _greedy_build(order, deadline)
        
        if total > best_total:
            best_total = total
            best_result = (list(elems[0]), list(elems[1]))
        
        # Hill climb first (fast, deterministic improvement)
        if time.time() < deadline - 2.0:
            hc_elems, hc_sum, hc_diff, hc_total = _hill_climb(elems, sum_used, diff_used, deadline)
            if hc_total > best_total:
                best_total = hc_total
                best_result = (list(hc_elems[0]), list(hc_elems[1]))
                elems, sum_used, diff_used = hc_elems, hc_sum, hc_diff
        
        # SA search to escape local minima
        if time.time() < deadline - 1.0:
            sa_elems, sa_sum, sa_diff, sa_total = _sa_search(elems, sum_used, diff_used, deadline)
            if sa_total > best_total:
                best_total = sa_total
                best_result = (list(sa_elems[0]), list(sa_elems[1]))
    
    # Ensure nonempty
    if not best_result[0]:
        best_result = ([1], list(best_result[1]) if best_result[1] else [2])
    if not best_result[1]:
        best_result = (list(best_result[0]), [2])
    
    return list(best_result)
# EVOLVE-BLOCK-END
