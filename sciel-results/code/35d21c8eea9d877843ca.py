"""Narrow admissible tuple: fixed-diameter maximization in [0, N]."""
import time
import math
import random

N = 5000
DEADLINE = 315.0  # safety margin before 330s kill

def _primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(sieve[p * p::p]))
    return [p for p in range(2, limit + 1) if sieve[p]]

def construct():
    start_time = time.time()
    
    # Get primes up to N (max possible tuple size is N+1, so primes up to N+1)
    all_primes = _primes(N + 1)
    
    # We want to find the largest subset S of [0, N] such that
    # for every prime p <= |S|, S does not cover all residues mod p.
    
    # Strategy:
    # 1. Start with S = {0, ..., N}.
    # 2. Iteratively repair violations:
    #    - Find a prime p <= |S| such that S covers all residues mod p.
    #    - Remove the residue class mod p with the fewest elements.
    #    - Repeat until no violations.
    
    # To make this efficient, we track:
    # - active: boolean array of which elements are in S
    # - For each prime p we care about, we track residue counts.
    
    # However, updating residue counts after removals is expensive.
    # Instead, let's use a simpler approach:
    # - Maintain a list of active points.
    # - For each check, compute residue counts on the fly.
    # - N=5000 is small enough that this should be fast.
    
    points = list(range(N + 1))
    
    # Optimization: Pre-compute for each point and each prime, the residue.
    # But that's a lot of memory. Let's just compute on the fly.
    
    # Better optimization: Process primes in increasing order.
    # When we remove elements for a small prime, it might help with larger primes.
    # But it might also not. Let's just do a simple greedy.
    
    # Key insight: We should prioritize removing classes for small primes first,
    # because small primes have more residue classes to cover, and removing
    # elements for them might help satisfy constraints for larger primes.
    
    # Also, when removing a residue class, we should remove the one with the
    # fewest elements to minimize the damage.
    
    # Let's implement a greedy repair loop:
    
    while True:
        if time.time() - start_time > DEADLINE:
            break
        
        m = len(points)
        if m == 0:
            break
        
        # Get primes up to m
        primes_to_check = [p for p in all_primes if p <= m]
        
        changed = False
        # Try primes in increasing order
        for p in primes_to_check:
            # Count residues
            counts = [0] * p
            for v in points:
                counts[v % p] += 1
            
            # Check if all residues are covered
            if min(counts) > 0:
                # All residues covered. Need to remove one class.
                # Remove the class with minimum count.
                min_count = min(counts)
                
                # Find all residue classes with min_count
                min_residues = [r for r in range(p) if counts[r] == min_count]
                
                # Which one to remove? 
                # Heuristic: Remove the class that appears least often in other primes'
                # "critical" residue classes? Too complex.
                # For now, pick the first one, or maybe the one that causes the least
                # future violations?
                
                # Let's try: pick the residue class r such that removing it
                # results in the smallest "damage" to other primes.
                # But that's expensive to compute.
                
                # Simple heuristic: pick the first min residue.
                target_residue = min_residues[0]
                
                # Remove all points v where v % p == target_residue
                new_points = [v for v in points if v % p != target_residue]
                if len(new_points) < len(points):
                    points = new_points
                    changed = True
                    break  # Restart the while loop
        
        if not changed:
            break
    
    # Now we have an admissible tuple. Can we add more elements back?
    # Try to add elements from [0, N] that were removed, if possible.
    
    # Build a set of points for fast lookup
    point_set = set(points)
    candidates = [v for v in range(N + 1) if v not in point_set]
    
    # Try to add candidates one by one
    # But adding a point might violate admissibility.
    # We need to check: after adding v, is the tuple still admissible?
    
    # To speed up, we can check only the primes that might be affected.
    # Adding a point increases |S| by 1, so we might need to check p = |S|+1
    # if it's prime. Also, adding a point might complete a residue class
    # for some prime p <= |S|.
    
    # Let's try a greedy addition:
    # Sort candidates in some order (maybe by how "safe" they are)
    # and try to add them.
    
    # For simplicity, let's try all candidates in random order multiple times.
    
    best_points = points
    best_len = len(points)
    
    # Try multiple random orders
    num_trials = 50
    for trial in range(num_trials):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        random.shuffle(candidates)
        
        current = list(best_points)
        current_set = set(current)
        
        for v in candidates:
            if v in current_set:
                continue
            
            # Try adding v
            new_current = current + [v]
            m_new = len(new_current)
            
            # Check admissibility
            # We need to check all primes p <= m_new
            # But we can be smarter: only check primes where adding v
            # might complete a residue class, plus p = m_new if prime.
            
            # For simplicity, check all primes up to m_new.
            # N=5000, m_new <= 5001, num primes ~ 670.
            # Each check is O(m_new). Total: 670 * 5000 = 3.3M ops per candidate.
            # Too slow.
            
            # Optimization: Pre-compute residue counts for current set.
            # Then for each candidate v, check which primes are affected.
            
            # This is getting complex. Let's just check admissibility directly
            # but only for primes that could be violated.
            
            # A prime p is violated if all residues mod p are present.
            # Adding v can only complete a residue class for p if v % p was
            # the missing residue. So we only need to check primes p where
            # v % p is the unique missing residue in the current set.
            
            # But we don't know which residues are missing without checking.
            
            # Alternative: Just check all primes up to m_new, but do it fast.
            
            # Let's pre-compute residue counts for current.
            # Actually, let's just do a full check but with early exit.
            
            admissible = True
            for p in all_primes:
                if p > m_new:
                    break
                
                # Check if all residues mod p are present in new_current
                residues = set()
                for val in new_current:
                    residues.add(val % p)
                    if len(residues) == p:
                        admissible = False
                        break
                if not admissible:
                    break
            
            if admissible:
                current = new_current
                current_set.add(v)
        
        if len(current) > best_len:
            best_points = current
            best_len = len(current)
    
    return best_points
