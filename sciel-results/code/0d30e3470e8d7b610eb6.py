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


def construct():
    random.seed(42)
    
    # We need primes up to the maximum possible tuple size.
    # Max size is N+1 = 5001.
    max_size = N + 1
    primes = _primes(max_size)
    
    # Data structures for O(1) admissibility check
    # covered[p_idx][r] = True if residue r mod p is covered
    # p_idx is the index in the primes list.
    # We only need to track primes p <= current_size.
    # As size grows, we add more primes to tracking.
    
    # Let's pre-allocate for all primes up to max_size.
    # Number of primes up to 5001 is approx 669.
    # For each prime p, we need a boolean array of size p.
    # Total memory: sum of primes <= 5001.
    # Sum of primes ~ 0.5 * n^2 / log n. For n=5000, ~ 12.5M / 8.5 ~ 1.5M booleans.
    # This is very small.
    
    # However, we don't need to track all primes at once.
    # We only need to track primes p such that p <= len(points).
    # Let's maintain a list of active primes and their coverage arrays.
    
    # Alternative: Since N is small (5000), we can just use a set of points
    # and check admissibility by iterating through primes <= len(points).
    # But the bottleneck is checking admissibility.
    # With boolean arrays, adding a point is O(pi(len(points))).
    # Removing a point is O(pi(len(points))) if we maintain counts.
    
    # Let's use counts: count[p_idx][r] = number of points with residue r mod p.
    # Admissible if for all active primes p, min(count[p_idx]) > 0 is FALSE?
    # No, admissible means SOME residue class is missed.
    # So admissible if exists p such that min(count[p_idx]) == 0.
    # Wait, the definition: "for every prime p <= len(tuple) some residue class mod p is missed".
    # This means: For ALL p <= len(tuple), there exists r such that count[r] == 0.
    # So it is NOT admissible if there exists a p <= len(tuple) such that ALL r have count[r] > 0.
    
    # So we maintain:
    # active_primes: list of primes p <= current_size
    # counts: dict mapping prime p to a list/array of counts for residues 0..p-1
    # zero_counts: for each active prime p, how many residues have count 0.
    # If zero_counts[p] > 0 for all active p, it's admissible.
    # If zero_counts[p] == 0 for any active p, it's NOT admissible (violation).
    
    # Greedy Construction:
    # Start with empty set.
    # Try to add points in a way that keeps it admissible.
    # To maximize size, we want to add as many as possible.
    # A simple greedy: iterate through candidates 0..N, add if admissible.
    # But order matters.
    # Better: Randomized greedy with backtracking or local search.
    
    # Let's try a "fill" strategy.
    # Start with a small admissible set.
    # Then try to add random points.
    
    # Initialize
    points = set()
    # We will maintain state for the current set of points.
    
    # To make this efficient, let's pre-compute residues for all numbers 0..N for all primes?
    # No, that's too much memory if we store for all primes.
    # But we can compute v % p on the fly.
    
    # State:
    # current_size = len(points)
    # active_primes = [p for p in primes if p <= current_size]
    # counts = {p: [0]*p for p in active_primes}
    # zero_counts = {p: p for p in active_primes} # initially all zero
    
    def add_point(v):
        nonlocal points, current_size, active_primes, counts, zero_counts
        if v in points:
            return False
        
        # Check if adding v keeps it admissible.
        # Adding v increments count[r] for r = v % p.
        # If count[r] was 0, zero_counts[p] decreases by 1.
        # If zero_counts[p] becomes 0, it becomes a violation.
        # We need to check if any p has zero_counts[p] == 0 AFTER adding.
        # Actually, we need to ensure that for ALL p, zero_counts[p] > 0.
        
        # Let's simulate the addition.
        new_size = current_size + 1
        # New active primes might be added if new_size crosses a prime threshold.
        # If p == new_size, it becomes active. Its counts are all 0 except v%p which becomes 1.
        # So zero_counts[p] = p - 1. This is > 0 if p > 1.
        
        # Check existing active primes.
        for p in active_primes:
            r = v % p
            if counts[p][r] == 0:
                # This residue was missed, now it's covered.
                # zero_counts[p] decreases.
                # If zero_counts[p] was 1, it becomes 0 -> Violation!
                if zero_counts[p] == 1:
                    return False
                # else it's fine, zero_counts[p] > 1
            # else counts[p][r] > 0, zero_counts[p] unchanged
        
        # Check if new primes are activated.
        # Primes p such that current_size < p <= new_size.
        # Since new_size = current_size + 1, only p = new_size can be added (if it is prime).
        if new_size in prime_set:
            # p = new_size
            # This prime was not active before.
            # Now it is active.
            # Its counts are all 0.
            # We add v, so count[v % p] becomes 1.
            # zero_counts[p] = p - 1.
            # Is p - 1 > 0? Yes, for p >= 2.
            # So no violation from the new prime.
            pass
            
        # If we passed all checks, it is admissible.
        # Commit the change.
        points.add(v)
        current_size += 1
        
        # Update counts
        for p in active_primes:
            r = v % p
            if counts[p][r] == 0:
                zero_counts[p] -= 1
            counts[p][r] += 1
            
        if current_size in prime_set:
            p = current_size
            # Initialize counts for p
            counts[p] = [0] * p
            r = v % p
            counts[p][r] = 1
            zero_counts[p] = p - 1
            active_primes.append(p)
            
        return True

    def remove_point(v):
        nonlocal points, current_size, active_primes, counts, zero_counts
        if v not in points:
            return False
        
        # Check if removing v keeps it admissible.
        # Removing v decrements count[r].
        # If count[r] becomes 0, zero_counts[p] increases by 1.
        # This can only help admissibility (more missed classes).
        # So removal is ALWAYS admissible if the set was admissible before?
        # Wait. The condition is "for every prime p <= len(tuple)".
        # When we remove a point, len(tuple) decreases.
        # So the set of primes to check shrinks.
        # Primes p such that new_size < p <= old_size are no longer checked.
        # For remaining primes, zero_counts increases or stays same.
        # So if it was admissible, it remains admissible.
        # Thus, we don't need to check admissibility for removal.
        
        # Commit
        points.discard(v)
        current_size -= 1
        
        # Update counts
        for p in active_primes:
            r = v % p
            if counts[p][r] == 1:
                zero_counts[p] += 1
            counts[p][r] -= 1
            
        # Deactivate primes if size drops below them?
        # We can keep them in active_primes but ignore them if p > current_size.
        # Or we can pop them.
        # Since we only add primes when size increases, and size only decreases by 1,
        # we might have active_primes containing primes > current_size.
        # We should clean up or just filter in the check.
        # Let's just keep them but in the admissibility check, we only look at p <= current_size.
        # Actually, my add_point logic assumes active_primes are all <= current_size.
        # If we remove, current_size drops.
        # The largest active prime might now be > current_size.
        # We should remove it from active_primes to keep the list consistent.
        if active_primes and active_primes[-1] > current_size:
            # Pop primes that are now too large
            while active_primes and active_primes[-1] > current_size:
                p = active_primes.pop()
                # We can leave counts[p] and zero_counts[p] as is, they just won't be accessed.
                pass
                
        return True

    # Precompute prime set for O(1) lookup
    prime_set = set(primes)
    
    # Initialize state
    points = set()
    current_size = 0
    active_primes = []
    counts = {}
    zero_counts = {}
    
    # Greedy Fill
    # We want to add as many points as possible.
    # Strategy: Iterate through 0..N in random order? Or structured?
    # Let's try random order first.
    candidates = list(range(N + 1))
    random.shuffle(candidates)
    
    for v in candidates:
        if time.time() > _DEADLINE - 10:
            break
        if add_point(v):
            pass
        # If it fails, skip.
        
    # Now we have a maximal set (with respect to this order).
    # We can try to improve by swapping.
    # But greedy fill usually gets quite good.
    
    # Local Search:
    # Try to replace a point in the set with a point not in the set to allow more points?
    # No, we want to increase size.
    # The greedy fill is maximal for that order.
    # A different order might yield a larger set.
    # Let's do a few restarts with different random orders if time permits.
    
    best_size = current_size
    best_points = list(points)
    
    # Try a few more random restarts
    # Each restart takes O(N * pi(N)) time.
    # N=5000, pi(N)~669. 5000*669 ~ 3.3M ops. Very fast.
    # We can do many restarts.
    
    restarts = 0
    while time.time() < _DEADLINE - 5:
        restarts += 1
        # Reset state
        points = set()
        current_size = 0
        active_primes = []
        counts = {}
        zero_counts = {}
        
        candidates = list(range(N + 1))
        random.shuffle(candidates)
        
        for v in candidates:
            if time.time() > _DEADLINE - 5:
                break
            add_point(v)
            
        if current_size > best_size:
            best_size = current_size
            best_points = list(points)
            
    return best_points
# EVOLVE-BLOCK-END
