"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 5000
_DEADLINE = time.time() + 310.0  # 330s wall-clock, 20s safety margin

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


def _is_admissible(points, all_primes):
    """Check if points form an admissible tuple."""
    L = len(points)
    for p in all_primes:
        if p > L:
            break
        # Count residue classes
        covered = [0] * p
        for v in points:
            covered[v % p] += 1
        if min(covered) > 0:
            return False
    return True


def _repair(points, all_primes):
    """Repair points to make them admissible by removing least-populated residue classes."""
    active = [True] * (N + 1)
    for v in points:
        active[v] = True
    
    changed = True
    while changed:
        if time.time() > _DEADLINE - 10.0:
            break
        changed = False
        L = len(points)
        # Only need to check primes <= L
        for p in all_primes:
            if p > L:
                break
            # Count residue classes
            class_counts = [0] * p
            for v in points:
                class_counts[v % p] += 1
            if min(class_counts) > 0:
                # All p residue classes are covered; try removing different classes
                # Sort classes by count to try least-populated first
                sorted_classes = sorted(range(p), key=lambda c: class_counts[c])
                removed_any = False
                for target_class in sorted_classes:
                    if class_counts[target_class] == 0:
                        continue
                    # Try removing this class
                    new_points = [v for v in points if v % p != target_class]
                    # Check if this resolves the conflict for this prime
                    new_counts = [0] * p
                    for v in new_points:
                        new_counts[v % p] += 1
                    if min(new_counts) == 0:
                        # This removal works
                        for v in points:
                            if v % p == target_class:
                                active[v] = False
                        points = new_points
                        changed = True
                        removed_any = True
                        break
                if not removed_any:
                    # Fallback: remove the least-populated class
                    target_class = sorted_classes[0]
                    for v in points:
                        if v % p == target_class:
                            active[v] = False
                    points = [v for v in points if active[v]]
                    changed = True
                break
    
    return points


def _local_search(points, all_primes, removed_elements):
    """Try to add back removed elements if they don't violate admissibility."""
    if not removed_elements:
        return points
    
    # Shuffle removed elements for diversity
    random.shuffle(removed_elements)
    
    improved = True
    while improved:
        if time.time() > _DEADLINE - 15.0:
            break
        improved = False
        for elem in removed_elements[:]:
            if time.time() > _DEADLINE - 15.0:
                break
            # Try adding this element
            new_points = points + [elem]
            if _is_admissible(new_points, all_primes):
                points = new_points
                removed_elements.remove(elem)
                improved = True
    
    return points


def construct():
    # Start from a good initial set: 1 and primes above sqrt(N)
    initial_points = [1] + [p for p in _primes(N) if p * p > N]
    
    # Precompute primes up to N (max possible length)
    all_primes = _primes(N)
    
    # Repair the initial set
    points = _repair(initial_points, all_primes)
    
    # Track removed elements
    removed_elements = [v for v in range(N + 1) if v not in set(points)]
    
    # Local search: try to add back removed elements
    points = _local_search(points, all_primes, removed_elements)
    
    return points
# EVOLVE-BLOCK-END
