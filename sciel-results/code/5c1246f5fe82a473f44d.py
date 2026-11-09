"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 5000
_DEADLINE = time.time() + 320.0  # 330s wall-clock, 10s safety margin

# EVOLVE-BLOCK-START
def _primes(limit):
    """Return list of primes <= limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, ok in enumerate(sieve) if ok]


def _is_admissible(points, primes_to_check):
    """Check if points form an admissible tuple for the given primes."""
    point_set = set(points)
    for p in primes_to_check:
        # Check if all residue classes mod p are covered
        classes = set()
        for v in point_set:
            classes.add(v % p)
        if len(classes) == p:
            return False
    return True


def _get_violations(points, primes_to_check):
    """Return list of (prime, residue_class) that are violated (all classes covered)."""
    violations = []
    for p in primes_to_check:
        classes = {}
        for v in points:
            classes.setdefault(v % p, []).append(v)
        if len(classes) == p:
            violations.append((p, classes))
    return violations


def _repair(points, primes_to_check, max_iterations=1000):
    """Repair admissibility by removing points from most-constrained classes."""
    points = list(points)
    point_set = set(points)
    for _ in range(max_iterations):
        if time.time() > _DEADLINE:
            break
        violations = _get_violations(points, primes_to_check)
        if not violations:
            break
        # Pick the violation with the smallest class to remove
        best_p = None
        best_class = None
        best_size = float('inf')
        for p, classes in violations:
            for r, members in classes.items():
                if len(members) < best_size:
                    best_size = len(members)
                    best_p = p
                    best_class = r
        if best_p is None:
            break
        # Remove one random point from the smallest class of the most-violated prime
        candidates = [v for v in points if v % best_p == best_class]
        if candidates:
            to_remove = random.choice(candidates)
            points.remove(to_remove)
            point_set.discard(to_remove)
    return points


def construct():
    random.seed(42)
    
    # Precompute primes up to N
    all_primes = _primes(N)
    
    # Start with a random admissible set
    # Use a simple heuristic: start with random points and repair
    initial_size = 100
    points = random.sample(range(N + 1), initial_size)
    
    # We need to check primes up to len(points)
    # As we add points, we need to check more primes
    
    best_points = points
    best_size = len(points)
    
    # Initial repair
    primes_to_check = _primes(len(points))
    best_points = _repair(best_points, primes_to_check)
    best_size = len(best_points)
    
    # Simulated annealing
    temperature = 1000.0
    cooling_rate = 0.9995
    min_temperature = 1.0
    
    current_points = list(best_points)
    current_size = len(current_points)
    
    while time.time() < _DEADLINE and temperature > min_temperature:
        # Try to add a random point
        if random.random() < 0.7:
            # Add a point
            new_point = random.randint(0, N)
            if new_point not in current_points:
                trial_points = current_points + [new_point]
                trial_size = len(trial_points)
                
                # Check if still admissible
                primes_to_check = _primes(trial_size)
                if _is_admissible(trial_points, primes_to_check):
                    # Accept
                    current_points = trial_points
                    current_size = trial_size
                    if current_size > best_size:
                        best_points = list(current_points)
                        best_size = current_size
                else:
                    # Not admissible, try to repair
                    repaired = _repair(trial_points, primes_to_check, max_iterations=50)
                    if len(repaired) > current_size:
                        # Accept if larger
                        current_points = repaired
                        current_size = len(current_points)
                        if current_size > best_size:
                            best_points = list(current_points)
                            best_size = current_size
                    elif random.random() < math.exp(-(current_size - len(repaired)) / temperature):
                        # Accept worse with probability
                        current_points = repaired
                        current_size = len(current_points)
        else:
            # Try to remove a random point (to escape local optima)
            if len(current_points) > 10:
                idx = random.randint(0, len(current_points) - 1)
                trial_points = current_points[:idx] + current_points[idx+1:]
                trial_size = len(trial_points)
                
                primes_to_check = _primes(trial_size)
                if _is_admissible(trial_points, primes_to_check):
                    # Always accept removal if admissible (we want to explore)
                    if random.random() < 0.1:  # Only sometimes accept to keep size up
                        current_points = trial_points
                        current_size = trial_size
        
        # Cool down
        temperature *= cooling_rate
    
    # Final repair to ensure admissibility
    primes_to_check = _primes(len(best_points))
    best_points = _repair(best_points, primes_to_check, max_iterations=10000)
    
    return best_points
# EVOLVE-BLOCK-END
