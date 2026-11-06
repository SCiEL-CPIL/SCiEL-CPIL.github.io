"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = False
    return np.where(sieve)[0]

def construct():
    deadline = time.time() + 280  # safety margin before 300s kill (evaluator limit)
    
    # 1. Precompute primes up to N
    primes = _primes(N)
    primes_list = primes.tolist()
    
    # 2. Start with the full set [0, N]
    # We represent the set as a boolean mask for fast NumPy operations
    mask = np.ones(N + 1, dtype=bool)
    points = np.arange(N + 1)
    
    # 3. Greedy Removal Phase (Vectorized)
    # For each prime p <= N, check if all residues are covered.
    # If yes, remove the least populated residue class.
    # Note: As we remove points, len(points) decreases, so the condition "p <= len(points)"
    # becomes harder to satisfy for larger p. However, removing points for small p
    # might not affect larger p's coverage status immediately, but it reduces the total count.
    # The condition is: for every prime p <= |S|, some residue is missed.
    # If we process primes in increasing order, we ensure small p are safe.
    # When we reach a prime p, if |S| < p, it is automatically safe (can't cover all p classes).
    # So we only need to act if |S| >= p and all residues are covered.
    
    # We iterate through primes in increasing order.
    for p in primes_list:
        if time.time() > deadline:
            break
            
        current_size = np.sum(mask)
        if p > current_size:
            # If p > current_size, it's impossible to cover all p residues.
            # Since primes are increasing and current_size only decreases,
            # for all subsequent primes p' > p, p' > current_size will also hold.
            # So we can break early? 
            # Wait. If p > current_size, then p is safe.
            # But what about a smaller prime that we skipped? We didn't skip any, we went in order.
            # What about a larger prime? If p > current_size, then any p' > p is also > current_size.
            # So all subsequent primes are automatically safe.
            # We can break the loop entirely.
            break
        
        # Check coverage for prime p
        # We need to know which residues are present in the current set of points.
        # points are indices where mask is True.
        active_points = np.where(mask)[0]
        
        # Compute residues mod p
        residues = active_points % p
        
        # Count occurrences of each residue
        # np.bincount is efficient
        counts = np.bincount(residues, minlength=p)
        
        # If all counts > 0, all residues are covered -> Violation
        if np.all(counts > 0):
            # Find the residue with the minimum count
            min_residue = np.argmin(counts)
            
            # Remove all points x such that x % p == min_residue
            # We can do this efficiently with vectorized boolean masking
            to_remove = (points % p == min_residue) & mask
            mask[to_remove] = False
            # No need to update 'points' explicitly if we use mask for future checks,
            # but np.where(mask) will be called again. 
            # To save time, we could maintain a list, but mask is fine for N=5000.
            
    # 4. Expansion Phase (Vectorized/Greedy)
    # Now we have a valid set S. We want to add back elements that were removed (or never included? we started with all, so only removed ones).
    # Actually, we started with all, so candidates are those where mask is False.
    # We want to add x if adding it doesn't violate the condition for any prime p <= |S|+1.
    # Condition for adding x:
    # For all primes p <= |S|+1, if x fills the LAST empty residue class for p, it's invalid.
    # i.e., for all p <= |S|+1, the number of empty residues for p in current S must be > 1, 
    # OR x must not be in the empty residue classes.
    # Wait, if x is in a non-empty class, it doesn't change the empty set. Safe.
    # If x is in an empty class, it reduces the empty count by 1.
    # It is unsafe if (x is in empty class for p) AND (empty count for p was 1).
    
    # We iterate through candidates. To be efficient, we can process them in batches or one by one.
    # Given N=5000, one-by-one with vectorized checks might be okay, but let's try to be smart.
    # We'll maintain the "empty residues" state for each relevant prime.
    
    # First, compute initial empty residues for all primes p <= N (or p <= max possible size)
    # Max possible size is N+1.
    # Let's precompute empty residues for all primes p <= N+1 for the CURRENT set S.
    
    max_prime_limit = N + 1
    relevant_primes = [p for p in primes_list if p <= max_prime_limit]
    
    # empty_residues[p] = set of residues r where no x in S has x%p == r
    # Using a list of sets or numpy arrays? Sets are easier for "is r in empty" checks.
    # But we have ~600 primes. 600 sets. 
    # Let's use a dictionary mapping p to a boolean array of length p (True if empty).
    
    empty_flags = {}
    for p in relevant_primes:
        active_points = np.where(mask)[0]
        if len(active_points) == 0:
            # All residues empty
            empty_flags[p] = np.ones(p, dtype=bool)
        else:
            residues = active_points % p
            present = np.zeros(p, dtype=bool)
            present[residues] = True
            empty_flags[p] = ~present
            
    # Candidates to try adding
    candidates = np.where(~mask)[0]
    
    # We will iterate through candidates. 
    # To maximize size, we should probably add as many as possible.
    # Order matters? If we add x, it might block y.
    # A simple greedy strategy: try candidates in random order or sorted order?
    # Let's try sorted order first, then maybe a second pass with random shuffle if time permits.
    
    # Actually, a better heuristic: 
    # If adding x is safe, add it.
    # Check if x is safe:
    #   For all p <= |S|+1:
    #       if empty_flags[p][x % p] is True:
    #           if count of True in empty_flags[p] == 1:
    #               return False (unsafe)
    #   return True
    
    # To make this fast, we can precompute the count of empty residues for each p.
    empty_counts = {p: np.sum(empty_flags[p]) for p in relevant_primes}
    
    # We need to update empty_flags and empty_counts as we add points.
    
    # Sort candidates? Maybe by value? 
    # Let's just iterate through them.
    # Since N is small (5000), and we might add ~100-500 points, 
    # and for each point we check ~100-200 primes, 
    # total ops ~ 500 * 200 = 100,000. Very fast.
    
    # We'll do a few passes to try to add more? 
    # No, if we add x, it only makes it harder to add others (reduces empty counts).
    # So a single greedy pass is optimal for this specific "add if safe" logic?
    # Not necessarily. Adding x might block y, but adding y might not block x.
    # However, if x is safe to add now, adding it cannot make the set invalid.
    # Does adding x prevent y from being added? Yes, if y fills the last hole for some p that x also helped fill?
    # No, x fills a hole for p. If x fills the last hole for p, x would be unsafe.
    # So if x is safe, it does NOT fill the last hole for any p.
    # Therefore, adding x reduces the number of empty holes for some p by 1, but never to 0.
    # So the set of "forbidden" candidates (those that would fill the last hole) can only grow?
    # Wait. If p has 2 empty holes, and x is in one, x is safe. After adding x, p has 1 empty hole.
    # Now any candidate y in the OTHER hole is unsafe.
    # So adding x can make new candidates unsafe.
    # This means the order of addition matters.
    # To maximize the number of added points, we might want to add points that "block" the fewest other points?
    # That's complex.
    # A simple heuristic: Add points that are "least critical"? 
    # Or just try all candidates, and if time permits, do a second pass with a different order.
    
    # Let's do a greedy pass with candidates sorted by value.
    # Then, if time permits, try a random shuffle pass on the remaining candidates?
    # Actually, if we process all candidates, we add the ones that are safe at the time of processing.
    # If we process y first, and it's safe, we add y. Then x might become unsafe.
    # If we process x first, and it's safe, we add x. Then y might become unsafe.
    # Which is better? Hard to say.
    # Let's try multiple random restarts of the expansion phase?
    # No, the removal phase is deterministic. 
    # Let's just do one greedy pass. It's fast.
    
    for x in candidates:
        if time.time() > deadline:
            break
            
        new_size = np.sum(mask) + 1
        # We only need to check primes p <= new_size
        safe = True
        for p in relevant_primes:
            if p > new_size:
                break
                
            r = x % p
            if empty_flags[p][r]:
                # x is in an empty class for p
                if empty_counts[p] == 1:
                    # x would fill the last hole
                    safe = False
                    break
            
        if safe:
            # Add x
            mask[x] = True
            
            # Update empty_flags and empty_counts
            for p in relevant_primes:
                if p > new_size:
                    break
                r = x % p
                if empty_flags[p][r]:
                    empty_flags[p][r] = False
                    empty_counts[p] -= 1
    
    # Convert mask back to list of integers
    result = np.where(mask)[0].tolist()
    
    return result
# EVOLVE-BLOCK-END
