import time
import math
import random
from itertools import combinations

_N = 128
_DEADLINE = 85.0
_CAND = list(range(1, _N + 1))
_SEED = 42

_GCD = None
_BAD_MASK = None

def _ensure_pre():
    global _GCD, _BAD_MASK
    if _GCD is not None:
        return
    
    _GCD = [[0] * (_N + 1) for _ in range(_N + 1)]
    for i in range(1, _N + 1):
        for j in range(i, _N + 1):
            g = math.gcd(i, j)
            _GCD[i][j] = g
            _GCD[j][i] = g

    _BAD_MASK = [[0] * (_N + 1) for _ in range(_N + 1)]
    for u in range(1, _N + 1):
        for v in range(u + 1, _N + 1):
            g_uv = _GCD[u][v]
            mask = 0
            for w in range(1, _N + 1):
                if w == u or w == v:
                    continue
                if _GCD[u][w] == g_uv and _GCD[v][w] == g_uv:
                    mask |= (1 << (w - 1))
            _BAD_MASK[u][v] = mask
            _BAD_MASK[v][u] = mask

def _popcount(x):
    return x.bit_count()

def _violations_count(mask):
    s = [i for i in range(1, _N + 1) if (mask >> (i - 1)) & 1]
    n = len(s)
    if n < 3:
        return 0
    total = 0
    for i in range(n):
        u = s[i]
        for j in range(i + 1, n):
            v = s[j]
            total += _popcount(mask & _BAD_MASK[u][v])
    return total // 3

def _delta_violations_add(mask, x):
    if (mask >> (x - 1)) & 1:
        return 0
    delta = 0
    for u in range(1, _N + 1):
        if (mask >> (u - 1)) & 1:
            delta += _popcount(mask & _BAD_MASK[x][u])
    return delta

def _delta_violations_remove(mask, x):
    if not ((mask >> (x - 1)) & 1):
        return 0
    delta = 0
    mask_without_x = mask & ~(1 << (x - 1))
    for u in range(1, _N + 1):
        if (mask_without_x >> (u - 1)) & 1:
            delta += _popcount(mask_without_x & _BAD_MASK[x][u])
    return delta

def _delta_violations_swap(mask, x_out, x_in):
    """Delta when removing x_out and adding x_in."""
    if (mask >> (x_out - 1)) & 1 and not ((mask >> (x_in - 1)) & 1):
        # Remove x_out first
        mask_no_out = mask & ~(1 << (x_out - 1))
        # Violations removed by removing x_out
        removed = 0
        for u in range(1, _N + 1):
            if (mask_no_out >> (u - 1)) & 1:
                removed += _popcount(mask_no_out & _BAD_MASK[x_out][u])
        # Violations added by adding x_in to mask_no_out
        added = 0
        for u in range(1, _N + 1):
            if (mask_no_out >> (u - 1)) & 1:
                added += _popcount(mask_no_out & _BAD_MASK[x_in][u])
        return added - removed
    return 0

def _greedy_construct(order):
    mask = 0
    violations = 0
    for x in order:
        dv = _delta_violations_add(mask, x)
        if dv == 0:
            mask |= (1 << (x - 1))
    return mask, 0

def _local_search_polish(mask, deadline):
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, _N + 1):
            if not ((mask >> (x - 1)) & 1):
                dv = _delta_violations_add(mask, x)
                if dv == 0:
                    mask |= (1 << (x - 1))
                    changed = True
    return mask

def _remove_conflicts_greedy(mask, deadline):
    cur_mask = mask
    cur_viol = _violations_count(cur_mask)
    
    while cur_viol > 0 and time.time() < deadline:
        conflict_count = [0] * (_N + 1)
        elems = [i for i in range(1, _N + 1) if (cur_mask >> (i - 1)) & 1]
        n = len(elems)
        if n < 3:
            break
        
        for i in range(n):
            u = elems[i]
            for j in range(i + 1, n):
                v = elems[j]
                cnt = _popcount(cur_mask & _BAD_MASK[u][v])
                if cnt > 0:
                    conflict_count[u] += cnt
                    conflict_count[v] += cnt
        
        max_conflict = 0
        remove_elem = -1
        for i in range(1, _N + 1):
            if (cur_mask >> (i - 1)) & 1 and conflict_count[i] > max_conflict:
                max_conflict = conflict_count[i]
                remove_elem = i
        
        if remove_elem == -1:
            break
        
        cur_mask &= ~(1 << (remove_elem - 1))
        cur_viol = _violations_count(cur_mask)
    
    return cur_mask, cur_viol

def _anneal(start_mask, deadline):
    """Simulated annealing on the 128-bit mask."""
    rng = random.Random(_SEED + 1)
    
    cur_mask = start_mask
    cur_viol = _violations_count(cur_mask)
    cur_size = _popcount(cur_mask)
    
    best_mask = cur_mask
    best_viol = cur_viol
    best_size = cur_size
    
    # Temperature schedule
    T_start = 10.0
    T_end = 0.01
    num_steps = 50000
    
    for step in range(num_steps):
        if time.time() > deadline:
            break
        
        # Cooling
        progress = step / num_steps
        T = T_start * (T_end / T_start) ** progress
        
        # Choose move type
        r = rng.random()
        
        if r < 0.3:
            # Add a random element
            if cur_size < _N:
                x = rng.randint(1, _N)
                if not ((cur_mask >> (x - 1)) & 1):
                    new_mask = cur_mask | (1 << (x - 1))
                    new_viol = cur_viol + _delta_violations_add(cur_mask, x)
                    new_size = cur_size + 1
                else:
                    continue
            else:
                continue
        elif r < 0.6:
            # Remove a random element
            if cur_size > 1:
                x = rng.randint(1, _N)
                if (cur_mask >> (x - 1)) & 1:
                    new_mask = cur_mask & ~(1 << (x - 1))
                    new_viol = cur_viol - _delta_violations_remove(cur_mask, x)
                    new_size = cur_size - 1
                else:
                    continue
            else:
                continue
        else:
            # Swap: remove one, add another
            if cur_size > 1:
                x_out = rng.randint(1, _N)
                x_in = rng.randint(1, _N)
                if (cur_mask >> (x_out - 1)) & 1 and not ((cur_mask >> (x_in - 1)) & 1):
                    new_mask = (cur_mask & ~(1 << (x_out - 1))) | (1 << (x_in - 1))
                    new_viol = cur_viol + _delta_violations_swap(cur_mask, x_out, x_in)
                    new_size = cur_size
                else:
                    continue
            else:
                continue
        
        # Acceptance criterion
        delta = new_viol - cur_viol
        # We want to minimize violations, maximize size
        # Objective: minimize (violations * 100 - size)
        old_obj = cur_viol * 100 - cur_size
        new_obj = new_viol * 100 - new_size
        obj_delta = new_obj - old_obj
        
        if obj_delta <= 0 or rng.random() < math.exp(-obj_delta / max(T, 1e-10)):
            cur_mask = new_mask
            cur_viol = new_viol
            cur_size = new_size
            
            if new_viol == 0 and new_size > best_size:
                best_size = new_size
                best_mask = new_mask
                best_viol = 0
            elif new_viol < best_viol:
                best_viol = new_viol
                best_size = new_size
                best_mask = new_mask
            elif new_viol == best_viol and new_size > best_size:
                best_size = new_size
                best_mask = new_mask
    
    return best_mask, best_viol, best_size

def _structural_initial(deadline):
    """Create diverse initial solutions."""
    solutions = []
    
    # 1. Greedy with natural order
    mask1, _ = _greedy_construct(list(range(1, _N + 1)))
    solutions.append(mask1)
    
    # 2. Greedy with reverse order
    mask2, _ = _greedy_construct(list(range(_N, 0, -1)))
    solutions.append(mask2)
    
    # 3. Prime powers first
    def is_prime_power(n):
        if n < 2: return False
        v = n
        p = 2
        found = False
        while p * p <= v:
            if v % p == 0:
                if found: return False
                found = True
                while v % p == 0:
                    v //= p
            p += 1
        return True
    
    prime_powers = [x for x in range(1, _N + 1) if is_prime_power(x)]
    others = [x for x in range(1, _N + 1) if x not in prime_powers]
    mask3, _ = _greedy_construct(prime_powers + others)
    solutions.append(mask3)
    
    # 4. Random initial with conflict removal
    rng = random.Random(_SEED + 2)
    for _ in range(3):
        sample = rng.sample(range(1, _N + 1), 70)
        m = 0
        for x in sample:
            m |= (1 << (x - 1))
        cleaned, _ = _remove_conflicts_greedy(m, deadline - 30)
        solutions.append(cleaned)
    
    return solutions

def construct():
    _ensure_pre()
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Get diverse initial solutions
    initial_solutions = _structural_initial(deadline - 30)
    
    best_mask = 0
    best_viol = float('inf')
    best_size = 0
    
    # Polish all initial solutions
    for init_mask in initial_solutions:
        if time.time() > deadline - 40:
            break
        polished = _local_search_polish(init_mask, deadline - 40)
        viol = _violations_count(polished)
        size = _popcount(polished)
        
        if viol == 0 and size > best_size:
            best_size = size
            best_mask = polished
            best_viol = 0
        elif viol < best_viol:
            best_viol = viol
            best_size = size
            best_mask = polished
        elif viol == best_viol and size > best_size:
            best_size = size
            best_mask = polished
    
    # Run simulated annealing from the best initial solution
    if time.time() < deadline - 5:
        bm, bv, bs = _anneal(best_mask, deadline - 5)
        
        if bv == 0 and bs > best_size:
            best_size = bs
            best_mask = bm
            best_viol = 0
        elif bv < best_viol:
            best_viol = bv
            best_size = bs
            best_mask = bm
        elif bv == best_viol and bs > best_size:
            best_size = bs
            best_mask = bm
    
    # Final polish
    if time.time() < deadline - 1:
        best_mask = _local_search_polish(best_mask, deadline - 1)
        best_viol = _violations_count(best_mask)
        best_size = _popcount(best_mask)
    
    # If still have violations, remove conflicts
    if best_viol > 0 and time.time() < deadline - 0.5:
        best_mask, best_viol = _remove_conflicts_greedy(best_mask, deadline - 0.5)
        best_size = _popcount(best_mask)
    
    result = [i for i in range(1, _N + 1) if (best_mask >> (i - 1)) & 1]
    
    if not result:
        result = [2]
        
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    P = _GCD
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            for k in range(j+1, len(res)):
                a, b, c = res[i], res[j], res[k]
                if P[a][b] == P[a][c] == P[b][c]:
                    valid = False
    print(f"Valid: {valid}")
