"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def _greedy_construct(all_primes, N, seed=0, order='asc'):
    random.seed(seed)
    # Start with all points
    pts = list(range(N + 1))
    
    # Pre-compute primes up to N+1
    # We process primes in a specific order.
    # The condition is: for every prime p <= len(points), some residue class mod p is missed.
    
    # Strategy: 
    # Iterate through primes. 
    # If p > len(pts), we don't need to satisfy the condition for p (since p is not <= len(pts)).
    # If p <= len(pts), we must ensure not all residues are present.
    # If all residues are present, we remove the "least harmful" residue class.
    # "Least harmful" usually means the class with the fewest elements, to minimize loss.
    # However, removing a class might affect other primes.
    # Since we only remove elements, the set of residues for any prime q can only shrink.
    # So if a prime q was NOT fully covered (had an empty class), it remains NOT fully covered.
    # If a prime q WAS fully covered, it might become NOT fully covered if we remove the last element of some class mod q.
    # Or it might remain fully covered if we don't remove the last element of any class mod q.
    # Therefore, we just need to check each prime p (in some order) and if it's fully covered, fix it.
    # We don't need to re-check primes that were already fixed, because they will remain fixed (or become "more" fixed).
    # Wait, "fixed" means "not fully covered". If it was not fully covered, it stays not fully covered.
    # So we can process each prime at most once?
    # Yes, provided we check them in an order such that if a prime is fully covered, we fix it, and this doesn't break previously fixed primes.
    # Since fixing a prime only removes points, it cannot make a previously "not fully covered" prime become "fully covered".
    # So a single pass through all primes is sufficient to ensure admissibility, IF we check p <= len(pts) at the time of processing.
    
    # However, the order matters for the FINAL SIZE.
    # If we process small primes first, we remove ~N/2, N/3, ... points.
    # This reduces len(pts) quickly.
    # Then large primes might drop out of the requirement (p > len(pts)).
    # If we process large primes first, we remove ~N/p (small) points.
    # len(pts) stays large.
    # Then we have to process small primes, removing many points.
    # Intuitively, processing small primes first seems better because it "unlocks" the skipping of large primes.
    
    primes_list = [p for p in all_primes if p <= N + 1]
    
    if order == 'asc':
        ps = primes_list
    elif order == 'desc':
        ps = primes_list[::-1]
    else:
        ps = primes_list[:]
        random.shuffle(ps)
        
    # We need to efficiently check if p is fully covered and remove a class.
    # For speed, we can maintain a list of points.
    # Checking "fully covered" requires counting residues.
    
    for p in ps:
        if p > len(pts):
            continue
            
        # Check if fully covered
        # Using a set is O(len(pts)) but with small constant?
        # Or numpy?
        # For len(pts) up to 5000, a simple set or list count might be fine.
        # But we do this for ~600 primes.
        # 600 * 5000 = 3,000,000 operations. Very fast.
        
        residues = set(pt % p for pt in pts)
        if len(residues) == p:
            # Fully covered. Remove the class with min count.
            # Count frequencies
            counts = [0] * p
            for pt in pts:
                counts[pt % p] += 1
                
            # Find min count
            min_count = min(counts)
            # Find indices with min count
            candidates = [i for i, c in enumerate(counts) if c == min_count]
            
            # Pick one
            # Heuristic: Pick the one that "helps" other primes?
            # Simple: Pick random or first.
            # Let's try picking the one that appears least often in the current set? That's what we did.
            # Maybe picking the one that is "central" or something?
            # For now, random among min counts.
            chosen_class = random.choice(candidates)
            
            # Remove points with pt % p == chosen_class
            pts = [pt for pt in pts if pt % p != chosen_class]
            
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 290 # 300s limit, leave 10s margin
    
    all_primes = _primes(N + 1)
    
    best_pts = []
    best_len = -1
    
    # Try different strategies
    # 1. Ascending
    # 2. Descending
    # 3. Random restarts
    
    # Ascending
    pts = _greedy_construct(all_primes, N, seed=0, order='asc')
    if len(pts) > best_len:
        best_len = len(pts)
        best_pts = pts
        
    # Descending
    if time.time() < deadline:
        pts = _greedy_construct(all_primes, N, seed=0, order='desc')
        if len(pts) > best_len:
            best_len = len(pts)
            best_pts = pts
            
    # Random restarts
    # How many can we afford?
    # Each run is ~0.1s?
    # 280s / 0.1s = 2800 restarts.
    # Let's try to run as many as possible.
    
    seed = 1
    while time.time() < deadline:
        pts = _greedy_construct(all_primes, N, seed=seed, order='rand')
        if len(pts) > best_len:
            best_len = len(pts)
            best_pts = pts
        seed += 1
        
    # Final verification and repair if needed
    # The greedy should produce admissible tuples, but let's be safe.
    # Also, we can try to "improve" by adding points back?
    # If we add a point, we might break admissibility.
    # We can check if adding a point x keeps the tuple admissible.
    # If yes, add it.
    # This is a local search.
    
    # Let's try a simple local search:
    # Iterate through all x in [0, N] not in best_pts.
    # Try adding x.
    # Check if new tuple is admissible.
    # If yes, keep it.
    # This is O(N * Cost_Check).
    # Cost_Check is O(P * L) where P is num primes, L is len.
    # Too slow for all x.
    
    # Better:
    # Only try adding x if it doesn't "fill" a hole for any prime p <= len(best_pts)+1.
    # This is complex.
    
    # Given the time, the greedy with random restarts is likely the main driver.
    # The "ascending" order is usually strong.
    
    # One more thing: The greedy removes the min-count class.
    # Maybe we should try removing a RANDOM class (not min count) to explore different landscapes?
    # Or removing the class that is "least used" by other primes?
    
    # Let's stick to the current logic but ensure we use the best found.
    
    return best_pts

# EVOLVE-BLOCK-END
