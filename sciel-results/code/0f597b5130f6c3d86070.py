"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
import time
import math

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    deadline = time.time() + 320.0
    # Start from 1 and the primes above sqrt(N); repair any prime whose residue classes are all covered
    # by deleting the least-populated class, until the tuple is admissible.
    points = [1] + [p for p in _primes(N) if p * p > N]
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        for p in _primes(len(points)):
            classes = {}
            for v in points:
                classes.setdefault(v % p, []).append(v)
            if len(classes) == p:
                smallest = min(classes.values(), key=len)
                points = [v for v in points if v not in smallest]
                changed = True
                break
    # Now use a greedy set-cover style approach to try to remove fewer elements
    # The universe is the set of "conflict primes" (primes p <= len(points) where all p residue classes are covered)
    # Each element x in points is a set containing the primes for which removing x breaks full coverage
    # We want to find the minimum number of elements to remove to make the tuple admissible
    # Greedy: at each step, remove the element that covers the most currently active conflicts
    
    def _is_admissible(pts):
        n = len(pts)
        if n == 0:
            return True
        primes_needed = _primes(n)
        for p in primes_needed:
            residues = set()
            for v in pts:
                residues.add(v % p)
            if len(residues) == p:
                return False
        return True
    
    # First check if current points is already admissible
    if _is_admissible(points):
        return points
    
    # Build the conflict structure
    # For each prime p <= len(points), check if all p residue classes are covered
    # For each element x, determine which primes would be "fixed" by removing x
    
    n = len(points)
    primes_needed = _primes(n)
    
    # For each prime, compute residue class counts
    # res_count[p][r] = number of elements in points with v % p == r
    res_count = {}
    for p in primes_needed:
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
        res_count[p] = counts
    
    # A prime p is a "conflict" if all p residue classes have count > 0
    # Removing element x from residue class r mod p "fixes" prime p if:
    #   - p is currently a conflict (all classes non-empty)
    #   - after removing x, residue class r mod p becomes empty (count was 1)
    
    # For each element, compute which conflicts it would fix
    # conflict_set[x] = set of primes p where x is the only element in its residue class mod p
    
    # We'll use a list for points, and track removals
    removed = set()
    current_points = list(points)
    
    # Precompute for each element the list of primes where it's the unique element in its class
    # This changes as we remove elements, so we'll recompute
    
    def _compute_conflicts(pts, res_counts):
        """Return set of primes that are conflicts (all residue classes non-empty)"""
        conflicts = set()
        for p in primes_needed:
            counts = res_counts[p]
            if all(c > 0 for c in counts):
                conflicts.add(p)
        return conflicts
    
    def _element_covers(x, pts, res_counts):
        """Return set of conflicts that removing x would fix"""
        covers = set()
        for p in primes_needed:
            counts = res_counts[p]
            r = x % p
            if counts[r] == 1:
                # Check if p is currently a conflict
                if all(c > 0 for c in counts):
                    covers.add(p)
        return covers
    
    # Main greedy loop
    while time.time() < deadline:
        # Compute current conflicts
        conflicts = set()
        for p in primes_needed:
            counts = res_count[p]
            if all(c > 0 for c in counts):
                conflicts.add(p)
        
        if not conflicts:
            break
        
        # For each remaining element, compute how many conflicts it covers
        best_elem = None
        best_covers = 0
        best_cover_set = None
        
        for v in current_points:
            if v in removed:
                continue
            covers = set()
            for p in conflicts:
                counts = res_count[p]
                r = v % p
                if counts[r] == 1:
                    covers.add(p)
            if len(covers) > best_covers:
                best_covers = len(covers)
                best_elem = v
                best_cover_set = covers
        
        if best_elem is None or best_covers == 0:
            # No single element can fix any conflict - this shouldn't happen if conflicts exist
            # But it could mean we need to remove multiple elements
            # Fallback: remove the element that's in the smallest residue class for some conflict
            break
        
        # Remove best_elem
        removed.add(best_elem)
        current_points.remove(best_elem)
        
        # Update residue counts
        for p in primes_needed:
            r = best_elem % p
            res_count[p][r] -= 1
    
    # After greedy, check if we're admissible
    if _is_admissible(current_points):
        return current_points
    
    # If greedy didn't work, fall back to the original approach
    return points
# EVOLVE-BLOCK-END
