"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import random
import sys

DEADLINE = 300.0
TIME_LIMIT = 320.0  # safety margin


def _dist2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _check_no_isosceles(candidate, chosen):
    """Check if candidate can be added to chosen without forming an isosceles triangle."""
    for i in range(len(chosen)):
        d1 = _dist2(candidate, chosen[i])
        for j in range(i + 1, len(chosen)):
            d2 = _dist2(candidate, chosen[j])
            d3 = _dist2(chosen[i], chosen[j])
            if d1 == d2 or d1 == d3 or d2 == d3:
                return False
    return True


def _is_maximal(chosen, all_points):
    """Check if no point can be added to chosen."""
    chosen_set = set(chosen)
    for p in all_points:
        if p not in chosen_set:
            if _check_no_isosceles(p, chosen):
                return False
    return True


def _greedy_random_order():
    """Greedy construction with random point ordering."""
    points = [(x, y) for x in range(N) for y in range(N)]
    random.shuffle(points)
    chosen = []
    for p in points:
        if _check_no_isosceles(p, chosen):
            chosen.append(p)
    return chosen


def _greedy_reverse_order():
    """Greedy construction trying to remove points from a full set."""
    points = [(x, y) for x in range(N) for y in range(N)]
    chosen = list(points)
    # Try to remove points that don't affect maximality
    for p in points:
        test = [q for q in chosen if q != p]
        if all(_check_no_isosceles(r, test) for r in test):
            # Check if removing p still leaves a valid no-isosceles set
            # and whether the set remains maximal
            chosen = test
    return chosen


def _local_search(initial):
    """Local search: try to swap or remove points to reduce size while maintaining maximality."""
    chosen = list(initial)
    improved = True
    while improved:
        improved = False
        for i in range(len(chosen)):
            # Try removing point i
            test = [chosen[j] for j in range(len(chosen)) if j != i]
            if all(_check_no_isosceles(p, test) for p in test):
                chosen = test
                improved = True
                break
    return chosen


def construct():
    start_time = time.time()
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    best = None
    best_len = N * N + 1
    
    # Try multiple random greedy constructions
    attempts = 0
    while time.time() - start_time < DEADLINE:
        attempts += 1
        candidate = _greedy_random_order()
        
        # Verify no-isosceles property
        valid = True
        for i in range(len(candidate)):
            for j in range(i + 1, len(candidate)):
                for k in range(j + 1, len(candidate)):
                    d1 = _dist2(candidate[i], candidate[j])
                    d2 = _dist2(candidate[i], candidate[k])
                    d3 = _dist2(candidate[j], candidate[k])
                    if d1 == d2 or d1 == d3 or d2 == d3:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Check maximality
            if _is_maximal(candidate, all_points):
                if len(candidate) < best_len:
                    best = candidate
                    best_len = len(candidate)
                    print(f"Found maximal no-isosceles set of size {best_len} at attempt {attempts}", file=sys.stderr)
        
        # Time check
        if time.time() - start_time > DEADLINE - 30:
            break
    
    if best is None:
        # Fallback: use a simple greedy in row-major order
        chosen = []
        for p in all_points:
            if _check_no_isosceles(p, chosen):
                chosen.append(p)
        best = chosen
    
    return best
# EVOLVE-BLOCK-END
