"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import random
import time
import itertools


def _dist_sq(p, q):
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def _creates_isosceles(p, chosen):
    """Check if adding p to chosen creates any isosceles triangle."""
    n = len(chosen)
    for i in range(n):
        a = chosen[i]
        d_pa = _dist_sq(p, a)
        for j in range(i + 1, n):
            b = chosen[j]
            d_pb = _dist_sq(p, b)
            d_ab = _dist_sq(a, b)
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                return True
    return False


def _greedy(order):
    """Run greedy construction with given point ordering. Returns maximal no-isosceles set."""
    chosen = []
    for p in order:
        if not _creates_isosceles(p, chosen):
            chosen.append(p)
    return chosen


def _verify_maximal(chosen):
    """Verify that no grid point can be added to chosen."""
    chosen_set = set(chosen)
    for p in itertools.product(range(N), repeat=2):
        if p in chosen_set:
            continue
        if not _creates_isosceles(p, chosen):
            return False
    return True


def construct():
    deadline = time.time() + 280  # 280s safety margin before 330s kill
    all_points = list(itertools.product(range(N), repeat=2))
    
    best = None
    best_size = float('inf')
    
    # First: row-major greedy as baseline
    baseline = _greedy(all_points)
    best = baseline
    best_size = len(baseline)
    
    # Generate diverse orderings and run greedy
    rng = random.Random(42)
    
    # Structured orderings
    orderings = []
    
    # Reverse row-major
    orderings.append(list(reversed(all_points)))
    
    # Column-major
    orderings.append(list(itertools.product(range(N), repeat=2)))
    orderings.append([p for p in all_points if p[0] % 2 == 0] + [p for p in all_points if p[0] % 2 == 1])
    
    # Diagonal ordering
    diag = []
    for d in range(2 * N - 1):
        for x in range(N):
            y = d - x
            if 0 <= y < N:
                diag.append((x, y))
    orderings.append(diag)
    orderings.append(list(reversed(diag)))
    
    # Spiral-like ordering (from center outward)
    cx, cy = (N - 1) / 2, (N - 1) / 2
    spiral = sorted(all_points, key=lambda p: (abs(p[0] - cx) + abs(p[1] - cy), p[0] + p[1]))
    orderings.append(spiral)
    orderings.append(list(reversed(spiral)))
    
    # Random orderings
    num_random = 200
    for i in range(num_random):
        shuffled = list(all_points)
        rng.shuffle(shuffled)
        orderings.append(shuffled)
    
    # Also try some biased random orderings (weight toward corners/edges)
    for i in range(50):
        weights = []
        for p in all_points:
            # Weight toward corners
            w = (p[0] + p[1]) + (N - 1 - p[0] + N - 1 - p[1])
            weights.append(w + 1)
        shuffled = list(all_points)
        for _ in range(10):
            idx = rng.choices(range(len(shuffled)), weights=weights, k=1)[0]
            # Swap with random position
            j = rng.randint(0, len(shuffled) - 1)
            shuffled[idx], shuffled[j] = shuffled[j], shuffled[idx]
        orderings.append(shuffled)
    
    for order in orderings:
        if time.time() > deadline:
            break
        result = _greedy(order)
        if len(result) < best_size:
            best_size = len(result)
            best = result
    
    return best
# EVOLVE-BLOCK-END
