"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 5000
DEADLINE = 325  # seconds, safety margin before 330s kill

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]


def _greedy_construct():
    """Fast heuristic to get a decent starting point."""
    # Start with all numbers, then remove to make admissible? No, that's slow.
    # Better: Start with a known admissible set, e.g., {1} U {primes > sqrt(N)}
    # Then try to add numbers.
    
    # Actually, the seed's approach of starting small and repairing is O(N) removals.
    # Let's try a different greedy: Start with all numbers [0..N], and for each prime p <= N,
    # if all p classes are covered, remove the least populated class.
    # This is O(N * pi(N)) which might be too slow for N=5000? 
    # pi(5000) ~ 669. 5000 * 669 ~ 3.3M ops. This is fast.
    
    points = list(range(N + 1))
    
    # We need to ensure that for every prime p <= len(points), some class is missed.
    # Note: len(points) changes as we remove.
    # The condition is: for every prime p <= len(tuple), some residue class mod p is missed.
    
    # Let's iterate through primes in increasing order.
    # If we remove elements, len(points) decreases, so the set of relevant primes shrinks.
    # However, if we start with all points, len(points) = 5001.
    # Primes up to 5001.
    
    primes = _primes(N)
    
    # We must check primes p where p <= current_len.
    # If we remove points, current_len drops.
    
    # Strategy:
    # While there exists a prime p <= len(points) such that all p classes are covered:
    #   Find such a p.
    #   Remove the least populated class for that p.
    
    # To be efficient, we can just check primes in increasing order repeatedly?
    # Or just check all primes <= N once? 
    # If we remove a class for p, it might affect coverage for other primes q.
    # Specifically, removing points can only *increase* the number of missed classes for any modulus.
    # So if a prime p was "safe" (some class missed), it remains safe.
    # If a prime p was "unsafe" (all classes covered), removing points might make it safe.
    # So we only need to worry about primes that are currently unsafe.
    
    # Let's maintain a set of "unsafe" primes.
    # Initially, for p=2, if both even and odd are present, it's unsafe.
    # With [0..N], all primes p <= N+1 are unsafe because all classes are covered.
    
    unsafe_primes = set(primes)
    
    # We need to process them.
    # If we remove points, some unsafe primes might become safe.
    # Let's just loop through primes.
    
    # A simple approach:
    # For each prime p in sorted(unsafe_primes):
    #   If p > len(points): continue (it's no longer a constraint)
    #   Check if all classes covered.
    #   If yes, remove least populated.
    #   Mark p as safe? No, removing points for p might make other primes q unsafe?
    #   Wait. Removing points *cannot* make a prime unsafe. 
    #   If a prime q had a missed class, and we remove more points, that class is still missed (or becomes more missed).
    #   So "safety" is monotonic. Once safe, always safe.
    #   Therefore, we only need to check if a prime is unsafe.
    #   If it is unsafe, we fix it. It becomes safe. We never need to check it again.
    
    # So:
    # Iterate primes in increasing order.
    # For each p:
    #   If p > len(points): break (since primes are increasing, all subsequent are also > len)
    #   Check coverage.
    #   If all covered: remove least populated class.
    
    # This is O(pi(N) * N) roughly, but the "check coverage" can be optimized.
    # Let's just do it.
    
    for p in primes:
        if p > len(points):
            break
        
        # Check coverage
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
        
        if min(counts) > 0:
            # Find min index
            min_idx = counts.index(min(counts))
            # Remove all v in points where v % p == min_idx
            points = [v for v in points if v % p != min_idx]
            
    return points


def _try_insert(points, candidate, primes_cache):
    """Try to insert candidate into points. Returns new points if admissible, else None."""
    new_len = len(points) + 1
    # We need to check primes p <= new_len.
    # Since we only added one point, only primes p where candidate % p might have been the "missed" class are at risk.
    # Actually, adding a point can only *decrease* the number of missed classes.
    # So a prime p that was safe (had a missed class) might become unsafe if the candidate fills the last missed class.
    # Primes that were already unsafe (all classes covered) remain unsafe.
    
    # So we only need to check primes p <= new_len that were previously safe.
    # But we don't track which were safe.
    # However, note that if p was unsafe before, it's still unsafe.
    # If p was safe before, it might become unsafe.
    
    # Efficient check:
    # For each prime p <= new_len:
    #   If p > len(points) (old len): it wasn't a constraint before. 
    #      Wait, the constraint is "for every prime p <= len(tuple)".
    #      Old tuple had len L. Primes <= L were checked.
    #      New tuple has len L+1. Primes <= L+1 must be checked.
    #      So we must check all primes <= L+1.
    #      But for p <= L, if it was safe, we need to see if it's still safe.
    #      If it was unsafe, it's still unsafe.
    #      How to know if it was unsafe? 
    #      We can just check coverage for all p <= new_len.
    #      Is it fast enough?
    #      new_len ~ 5000. pi(5000) ~ 669.
    #      Checking coverage for 669 primes, each involving counting residues of ~5000 points?
    #      That's 669 * 5000 = 3.3M ops per insertion attempt.
    #      If we try 1000 insertions, that's 3.3B ops. Too slow.
    
    # Optimization:
    # Only primes p where candidate % p is the *unique* missed class are at risk.
    # If a prime p has >= 2 missed classes, adding one point cannot make it unsafe.
    # If a prime p has exactly 1 missed class, and candidate % p is that class, it becomes unsafe.
    # If a prime p has 0 missed classes (unsafe), it stays unsafe.
    
    # So we need to know, for each prime p <= new_len, how many classes are missed.
    # We can maintain this state?
    # Or, we can just check the primes that are "close to unsafe".
    
    # Let's try a simpler heuristic first:
    # Just check if the new set is admissible by verifying the condition.
    # To make it fast, we can pre-compute the "missed count" for the current points.
    # But that's expensive to update.
    
    # Alternative:
    # Randomized local search might be slow if verification is O(N*pi(N)).
    # Let's stick to the greedy removal which is fast and produces a good base.
    # Then, try to insert.
    # To speed up insertion check:
    #   Only check primes p where the number of missed classes is 1.
    #   We can find these primes by checking coverage?
    #   Actually, let's just verify admissibility of the *new* set directly but efficiently.
    
    # Let's implement a fast admissibility check for a specific set.
    # It's O(pi(L) * L) if done naively.
    # Can we do better?
    # Maybe not easily.
    
    # Given the 300s limit, and N=5000, maybe we can afford a few thousand checks?
    # 3.3M ops per check. 1000 checks = 3.3B. Python is slow. ~10-30s per 1B simple ops? 
    # 3.3B ops might take 30-100s. Risky.
    
    # Let's optimize the check.
    # We only need to check if *any* prime has 0 missed classes.
    # We can iterate primes. For each p, count residues.
    # If we find one with 0 missed, return False.
    # Early exit is likely if the set is "dense".
    
    pass


def construct():
    start_time = time.time()
    
    # 1. Get a good initial solution
    points = _greedy_construct()
    best_points = points[:]
    best_size = len(points)
    
    # 2. Try to improve by inserting points
    # Candidates: all numbers not in points
    in_set = set(points)
    candidates = [i for i in range(N + 1) if i not in in_set]
    random.shuffle(candidates)
    
    primes = _primes(N + 1)
    
    # To speed up insertion checks, we can pre-compute residue counts for current points?
    # But points changes.
    
    # Let's just try a limited number of insertions.
    # If it's too slow, we just return the greedy result.
    
    # Let's estimate time for one check.
    # Let's do a trial check.
    
    def is_admissible(pts):
        L = len(pts)
        if L == 0:
            return True
        # Check primes <= L
        for p in primes:
            if p > L:
                break
            counts = [0] * p
            for v in pts:
                counts[v % p] += 1
            if min(counts) > 0:
                return False
        return True
        
    # Verify initial
    if not is_admissible(points):
        # Should not happen if greedy is correct, but just in case
        return points

    # Try insertions
    # We will attempt to insert from candidates.
    # We'll try up to a certain time limit.
    
    # To make it faster, we can batch-check or use a smarter heuristic.
    # But let's just loop.
    
    for cand in candidates:
        if time.time() - start_time > DEADLINE - 5:
            break
            
        # Try adding cand
        # Check if adding cand keeps it admissible.
        # As discussed, only primes with exactly 1 missed class are at risk.
        # But we don't track that.
        # Let's just call is_admissible on the new list.
        # To avoid creating a new list every time, we can modify in place?
        # No, is_admissible takes a list.
        
        # Optimization: 
        # If cand is not in points, we can check admissibility of points + [cand].
        # We can write a specialized check that takes the base points and the candidate.
        
        def is_admissible_with_candidate(pts, cand):
            L = len(pts) + 1
            for p in primes:
                if p > L:
                    break
                counts = [0] * p
                for v in pts:
                    counts[v % p] += 1
                counts[cand % p] += 1
                if min(counts) > 0:
                    return False
            return True
            
        if is_admissible_with_candidate(points, cand):
            points.append(cand)
            in_set.add(cand)
            best_points = points[:]
            best_size = len(points)
            
    return best_points
# EVOLVE-BLOCK-END
