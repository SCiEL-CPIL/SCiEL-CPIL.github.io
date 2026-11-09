"""New design: bitmask fast 4-AP checking + simulated annealing local search."""
import time
import random

N = 256
DEADLINE = 85  # seconds, safety margin before 90s kill


def _bitmask_from_set(s):
    """Convert a set of 1-based indices to a 256-bit integer."""
    m = 0
    for x in s:
        m |= (1 << (x - 1))
    return m


def _set_from_bitmask(m):
    """Convert a 256-bit integer back to a sorted list of 1-based indices."""
    return [i + 1 for i in range(N) if (m >> i) & 1]


def _has_4ap_bitmask(m, x, d):
    """Check if positions x, x+d, x+2d, x+3d are all set in bitmask m.
    x is 0-based index (0..255), d is positive difference."""
    # All four must be present
    return ((m >> x) & 1) and ((m >> (x + d)) & 1) and ((m >> (x + 2*d)) & 1) and ((m >> (x + 3*d)) & 1)


def _count_3aps_bitmask(m):
    """Count distinct 3-term APs (positive difference) in the bitmask."""
    count = 0
    for a in range(N):
        if not ((m >> a) & 1):
            continue
        for d in range(1, (N - a) // 2 + 1):
            if ((m >> (a + d)) & 1) and ((m >> (a + 2*d)) & 1):
                count += 1
    return count


def _has_any_4ap_bitmask(m):
    """Check if the bitmask contains any nonconstant 4-term AP."""
    for a in range(N):
        if not ((m >> a) & 1):
            continue
        for d in range(1, (N - a) // 3 + 1):
            if _has_4ap_bitmask(m, a, d):
                return True
    return False


def _violations_bitmask(m):
    """Return list of (a, d) for all 4-APs present. a is 0-based."""
    v = []
    for a in range(N):
        if not ((m >> a) & 1):
            continue
        for d in range(1, (N - a) // 3 + 1):
            if _has_4ap_bitmask(m, a, d):
                v.append((a, d))
    return v


def _greedy_initial():
    """Greedy construction: add elements in order if no 4-AP is created."""
    m = 0
    for x in range(N):  # 0-based
        # Tentatively add x
        m2 = m | (1 << x)
        # Check if this creates any 4-AP
        # Only need to check 4-APs that include position x
        ok = True
        # x as first element: x, x+d, x+2d, x+3d
        for d in range(1, (N - x) // 3 + 1):
            if ((m2 >> (x + d)) & 1) and ((m2 >> (x + 2*d)) & 1) and ((m2 >> (x + 3*d)) & 1):
                ok = False
                break
        if not ok:
            continue
        # x as second element: x-d, x, x+d, x+2d
        for d in range(1, x // 3 + 1):
            if ((m2 >> (x - d)) & 1) and ((m2 >> (x + d)) & 1) and ((m2 >> (x + 2*d)) & 1):
                ok = False
                break
        if not ok:
            continue
        # x as third element: x-2d, x-d, x, x+d
        for d in range(1, x // 2 + 1):
            if ((m2 >> (x - 2*d)) & 1) and ((m2 >> (x - d)) & 1) and ((m2 >> (x + d)) & 1):
                ok = False
                break
        if not ok:
            continue
        # x as fourth element: x-3d, x-2d, x-d, x
        for d in range(1, x // 3 + 1):
            if ((m2 >> (x - 3*d)) & 1) and ((m2 >> (x - 2*d)) & 1) and ((m2 >> (x - d)) & 1):
                ok = False
                break
        if ok:
            m = m2
    return m


def _local_search_sa(m, deadline):
    """Simulated annealing local search to maximize 3-APs while maintaining no 4-APs."""
    best_m = m
    best_score = _count_3aps_bitmask(m)
    current_m = m
    current_score = best_score
    T = 5.0  # initial temperature
    cooling = 0.9995
    random.seed(42)
    
    while time.time() < deadline:
        if T < 0.01:
            break
        
        # Pick a random move: add or remove a random element
        if random.random() < 0.5:
            # Try adding a random element
            x = random.randint(0, N - 1)
            if (current_m >> x) & 1:
                continue  # already present
            new_m = current_m | (1 << x)
            # Check for 4-AP violations
            valid = True
            for d in range(1, (N - x) // 3 + 1):
                if ((new_m >> (x + d)) & 1) and ((new_m >> (x + 2*d)) & 1) and ((new_m >> (x + 3*d)) & 1):
                    valid = False
                    break
            if valid:
                for d in range(1, x // 3 + 1):
                    if ((new_m >> (x - d)) & 1) and ((new_m >> (x + d)) & 1) and ((new_m >> (x + 2*d)) & 1):
                        valid = False
                        break
            if valid:
                for d in range(1, x // 2 + 1):
                    if ((new_m >> (x - 2*d)) & 1) and ((new_m >> (x - d)) & 1) and ((new_m >> (x + d)) & 1):
                        valid = False
                        break
            if valid:
                for d in range(1, x // 3 + 1):
                    if ((new_m >> (x - 3*d)) & 1) and ((new_m >> (x - 2*d)) & 1) and ((new_m >> (x - d)) & 1):
                        valid = False
                        break
            if not valid:
                continue
        else:
            # Try removing a random element
            x = random.randint(0, N - 1)
            if not ((current_m >> x) & 1):
                continue
            new_m = current_m & ~(1 << x)
        
        # Compute new score
        new_score = _count_3aps_bitmask(new_m)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < _exp_neg(delta / T):
            current_m = new_m
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_m = current_m
        
        T *= cooling
    
    return best_m, best_score


def _exp_neg(x):
    """exp(-x) with clamping."""
    if x < -50:
        return 0.0
    import math
    return math.exp(-x)


def construct():
    start = time.time()
    deadline = start + DEADLINE
    
    # Phase 1: Greedy initial solution
    m = _greedy_initial()
    score = _count_3aps_bitmask(m)
    best_m = m
    best_score = score
    
    # Phase 2: Simulated annealing local search
    if time.time() < deadline - 5:
        best_m, best_score = _local_search_sa(m, deadline - 5)
    
    # Phase 3: Try multiple random restarts with different seeds if time allows
    if time.time() < deadline - 10:
        for seed in [1, 2, 3, 7, 13, 42, 99, 123, 777, 2024]:
            if time.time() > deadline - 5:
                break
            random.seed(seed)
            # Random perturbation of best solution
            m2 = best_m
            # Remove a few random elements
            for _ in range(random.randint(1, 5)):
                x = random.randint(0, N - 1)
                if (m2 >> x) & 1:
                    m2 &= ~(1 << x)
            # Run short SA from perturbed state
            m2, s2 = _local_search_sa(m2, deadline - 3)
            if s2 > best_score:
                best_score = s2
                best_m = m2
    
    # Convert back to sorted list
    result = _set_from_bitmask(best_m)
    
    # Verify no 4-APs
    assert not _has_any_4ap_bitmask(best_m), "4-AP detected in result!"
    
    return result


# Quick self-test
if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Cardinality: {len(result)}")
    print(f"Time: {t1-t0:.2f}s")
    # Count 3-APs
    m = _bitmask_from_set(set(result))
    print(f"3-APs: {_count_3aps_bitmask(m)}")
