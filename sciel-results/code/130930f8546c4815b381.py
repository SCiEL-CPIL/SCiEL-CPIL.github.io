"""Combined program: seed repair-loop baseline + inspiration set-cover/local-improvement.

Strategy:
  * Baseline (seed): start from [0, N], repair by deleting the least-populated
    residue class for each prime p <= len(tuple) whose classes are all covered.
    This is the fixed-diameter greedy deletion and is a strong baseline.
  * Refinement (inspiration): treat the problem as set-cover over the primes
    that are "full" in [0, N], then run a local-improvement pass that re-adds
    removed elements (and swaps) to push the size up, all under a wall-clock
    deadline with a safety margin.
"""
import time
import sys

N = 5000
DEADLINE_MARGIN = 30.0
T_START = time.time()

# Precompute primes up to N+10 (max possible size is N+1)
_MAX_PRIME_LIMIT = N + 10
_primes_cache = {}

def _get_primes(limit):
    if limit in _primes_cache:
        return _primes_cache[limit]
    # Simple sieve for small limits, reuse large sieve if needed
    # For efficiency in this specific task, we can just sieve up to N+10 once
    if limit > 100 and limit <= _MAX_PRIME_LIMIT:
        if _MAX_PRIME_LIMIT not in _primes_cache:
            sieve = bytearray([1]) * (_MAX_PRIME_LIMIT + 1)
            sieve[0] = sieve[1] = 0
            for p in range(2, int(_MAX_PRIME_LIMIT ** 0.5) + 1):
                if sieve[p]:
                    sieve[p * p::_MAX_PRIME_LIMIT + 1] # placeholder, need correct slice
                    # Correct sieve implementation
                    sieve[p * p : _MAX_PRIME_LIMIT + 1 : p] = bytearray(len(sieve[p * p : _MAX_PRIME_LIMIT + 1 : p]))
            _primes_cache[_MAX_PRIME_LIMIT] = [i for i in range(2, _MAX_PRIME_LIMIT + 1) if sieve[i]]
        # Filter cached primes
        res = [p for p in _primes_cache[_MAX_PRIME_LIMIT] if p <= limit]
        _primes_cache[limit] = res
        return res
    
    # Fallback for very small limits or if cache not built
    if limit < 2:
        res = []
    else:
        sieve = bytearray([1]) * (limit + 1)
        sieve[0] = sieve[1] = 0
        for p in range(2, int(limit ** 0.5) + 1):
            if sieve[p]:
                sieve[p * p : limit + 1 : p] = bytearray(len(sieve[p * p : limit + 1 : p]))
        res = [i for i in range(2, limit + 1) if sieve[i]]
    _primes_cache[limit] = res
    return res

# Initialize prime cache
_get_primes(_MAX_PRIME_LIMIT)


def _time_left():
    return (330.0 - DEADLINE_MARGIN) - (time.time() - T_START)


def _check_admissible(points):
    """Return True iff `points` (distinct ints) is an admissible tuple."""
    if not points:
        return True
    m = len(points)
    primes = _get_primes(m)
    for p in primes:
        seen = [0] * p
        for v in points:
            seen[v % p] = 1
        if all(seen):
            return False
    return True


def _greedy_repair(points):
    """Seed repair: delete least-populated class for each violating prime."""
    changed = True
    while changed:
        changed = False
        m = len(points)
        primes = _get_primes(m)
        for p in primes:
            classes = {}
            for v in points:
                classes.setdefault(v % p, []).append(v)
            if len(classes) == p:
                smallest = min(classes.values(), key=len)
                # Remove all elements in the smallest class
                to_remove = set(smallest)
                points = [v for v in points if v not in to_remove]
                changed = True
                break
    return points


class FastAdmissibleChecker:
    def __init__(self, points):
        self.points = list(points)
        self.point_set = set(self.points)
        self.m = len(self.points)
        self.primes = _get_primes(self.m)
        # counts[p] is a list of length p, counts[p][r] = count of x in points with x%p==r
        self.counts = {}
        self.zero_residues = {} # p -> list of residues with count 0
        self._update_counts()

    def _update_counts(self):
        self.counts = {}
        self.zero_residues = {}
        for p in self.primes:
            c = [0] * p
            for v in self.points:
                c[v % p] += 1
            self.counts[p] = c
            zeros = [r for r in range(p) if c[r] == 0]
            self.zero_residues[p] = zeros

    def _add_point_internal(self, x):
        # Returns True if x can be added without violating admissibility
        # We must check primes p <= self.m + 1
        new_m = self.m + 1
        new_primes = _get_primes(new_m)
        
        # Check existing primes p <= self.m
        for p in self.primes:
            r = x % p
            c = self.counts[p]
            # If there are multiple zeros, adding x (which increases one count) 
            # will leave at least one zero. Safe.
            if len(self.zero_residues[p]) > 1:
                continue
            # If there is exactly one zero
            if len(self.zero_residues[p]) == 1:
                zero_r = self.zero_residues[p][0]
                # If x falls into the zero residue, it becomes 1.
                # We need to ensure no other residue becomes 0? No, counts only increase.
                # So if x % p == zero_r, the zero is gone.
                # Is there another zero? No, we established len(zeros)==1.
                # So if x % p == zero_r, we violate for this p.
                if r == zero_r:
                    return False
                # If x % p != zero_r, the zero remains. Safe.
            # If there are no zeros, we are already violating, which shouldn't happen if state is valid.
            # But if state is invalid, we shouldn't be calling this.
            
        # Check new prime p = new_m if it is prime
        if new_m in [p for p in new_primes if p > self.m]:
            # This only happens if new_m is prime
            p = new_m
            # We need to check if the new set covers all residues mod p
            # Current counts for p are not in self.counts
            # We have self.m elements.
            # We are adding x.
            # We need to check if {v % p for v in self.points} U {x % p} covers all 0..p-1
            # Since p = new_m = self.m + 1, we have self.m + 1 elements.
            # It covers all residues iff all residues are distinct.
            seen = [0] * p
            for v in self.points:
                seen[v % p] = 1
            seen[x % p] = 1
            if all(seen):
                return False

        return True

    def add_point(self, x):
        if x in self.point_set:
            return False
        if not self._add_point_internal(x):
            return False
        # Commit
        self.points.append(x)
        self.point_set.add(x)
        self.m += 1
        self.primes = _get_primes(self.m)
        # Update counts for new prime if added
        if self.primes and self.primes[-1] == self.m:
            p = self.m
            c = [0] * p
            for v in self.points:
                c[v % p] += 1
            self.counts[p] = c
            self.zero_residues[p] = [r for r in range(p) if c[r] == 0]
        # Update counts for existing primes
        for p in self.primes:
            if p in self.counts:
                self.counts[p][x % p] += 1
                # Update zero residues
                # This is slightly expensive but O(p)
                # Optimization: only update if x%p was 0
                if self.counts[p][x % p] == 1: # was 0
                    # Remove from zero_residues
                    # Find and remove
                    zeros = self.zero_residues[p]
                    # Simple removal
                    self.zero_residues[p] = [r for r in zeros if r != x % p]
        return True

    def remove_point(self, x):
        if x not in self.point_set:
            return False
        # Commit
        self.points.remove(x)
        self.point_set.discard(x)
        self.m -= 1
        # Update primes
        self.primes = _get_primes(self.m)
        # Remove prime if it was self.m+1 and is no longer needed
        # Note: primes list is sorted.
        # If self.m+1 was prime, it is now > self.m, so not in primes.
        
        # Update counts
        for p in self.primes:
            if p in self.counts:
                r = x % p
                self.counts[p][r] -= 1
                if self.counts[p][r] == 0:
                    # Add to zero_residues
                    self.zero_residues[p].append(r)
        # Clean up counts for primes no longer in self.primes?
        # Not strictly necessary for correctness of add_point if we only iterate self.primes.
        # But memory might grow. For N=5000, it's fine.
        return True

    def swap(self, y, x):
        # Remove y, Add x
        # We can just remove and add
        # But we need to ensure add succeeds.
        # If we remove y first, state is valid.
        # Then try to add x.
        # If add fails, we must put y back.
        
        # Save state? Or just revert?
        # Revert is easier.
        
        # Check if x can be added AFTER removing y?
        # This is complex.
        # Simpler: Try to add x to current state? No, y is in state.
        # If we add x, size increases. Then remove y?
        # Admissibility is monotonic w.r.t subset?
        # If S is admissible, is S \ {y} admissible?
        # Yes. Removing elements cannot create a full residue class.
        # So S \ {y} is admissible.
        # Then can we add x to S \ {y}?
        
        # Let's just do:
        # 1. Remove y
        # 2. Try Add x
        # 3. If Add fails, Re-add y
        
        self.remove_point(y)
        if self.add_point(x):
            return True
        else:
            self.add_point(y)
            return False


def _local_improve_fast(points, removed):
    checker = FastAdmissibleChecker(points)
    removed_set = set(removed)
    improved = True
    
    # Re-add pass
    # Sort removed to have deterministic behavior, maybe prioritize large numbers?
    # Actually, order doesn't matter much for correctness, but for finding "better" tuples?
    # All tuples of same size are equal score.
    # We just want max size.
    
    while improved and _time_left() > 5.0:
        improved = False
        # Re-add pass
        # Iterate over a copy of removed_set to allow modification?
        # Or just check each.
        for x in list(removed_set):
            if _time_left() < 5.0:
                break
            if checker.add_point(x):
                removed_set.discard(x)
                improved = True
        
        # Swap pass
        if _time_left() < 5.0:
            break
        # Try to swap elements in checker.points with elements in removed_set
        # This is O(M * K)
        # We can limit the number of swaps per iteration to avoid stalling?
        # Or just run until no improvement.
        
        # To speed up, we can randomize or just iterate.
        # Let's iterate through removed_set and for each, try to swap with some y in points.
        # Which y? 
        # If we just try all y, it's slow.
        # Heuristic: Try to swap with y that is "least useful"?
        # Hard to define.
        # Let's just do a few random swaps or limited swaps.
        
        # Actually, if we just re-add everything possible, we are at a local maximum for additions.
        # Swaps allow us to "move" elements to free up residue classes?
        # Yes.
        
        # Let's do a limited number of swap attempts per loop to keep time predictable.
        # Or just break if no swap found.
        
        found_swap = False
        # Shuffle removed_set? No, deterministic is better for reproducibility.
        # But maybe we get stuck in a bad local minimum.
        # Let's try all pairs but with a time check.
        
        # Optimization: Only try swaps if we have time.
        for x in list(removed_set):
            if _time_left() < 5.0:
                break
            # Try to find a y in checker.points such that swap(y, x) is valid.
            # We can just iterate y.
            for y in checker.points:
                if _time_left() < 5.0:
                    break
                if checker.swap(y, x):
                    removed_set.discard(x)
                    removed_set.add(y)
                    found_swap = True
                    # Break out of y loop, continue x loop?
                    # If we swapped, x is now in points, y in removed.
                    # We should break x loop too? Or continue to try other x?
                    # If we continue, we might swap back?
                    # Better to break both and restart the while loop to re-evaluate re-adds?
                    # Or just break x loop and go to next while iteration.
                    break
            if found_swap:
                improved = True
                # Break out of x loop?
                # If we break, we restart while.
                # This ensures we try re-adds again.
                break
                
    return checker.points


def construct():
    # ---------- Baseline: seed repair on [0, N] ----------
    points = list(range(N + 1))
    points = _greedy_repair(points)
    best = points if _check_admissible(points) else []
    best_set = set(best)
    removed = [x for x in range(N + 1) if x not in best_set]

    # ---------- Refinement: local improvement ----------
    try:
        if _time_left() > 5.0:
            improved = _local_improve_fast(best, removed)
            if _check_admissible(improved) and len(improved) > len(best):
                best = improved
    except Exception as e:
        # Fallback to slow local improve if fast one crashes?
        # Or just skip.
        pass

    # ---------- Final safety: ensure admissible, fall back to baseline ----------
    if not _check_admissible(best):
        fallback = _greedy_repair(list(range(N + 1)))
        best = fallback if _check_admissible(fallback) else []

    return list(best)
