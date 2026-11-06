import time
import math
import random
from collections import Counter

N = 5000
DEADLINE = 305.0  # seconds, keep safety margin under 330s

# EVOLVE-BLOCK-START

def _sieve_primes(limit):
    """Return list of all primes <= limit."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]


class _ResidueTracker:
    """
    Maintains residue counts for a set of points across relevant primes.
    Supports fast add and admissibility check.
    """
    def __init__(self, primes, n_max):
        self.primes = [p for p in primes if p <= n_max + 1]
        self.n_max = n_max
        self.counts = {}
        self.full_count = {}
        self.missing_residue = {} # p -> r if full_count[p] == p-1, else -1
        for p in self.primes:
            self.counts[p] = [0] * p
            self.full_count[p] = 0
            self.missing_residue[p] = -1
        self._pts = set()

    def add(self, x):
        if x in self._pts:
            return
        self._pts.add(x)
        n_pts = len(self._pts)
        for p in self.primes:
            if p > n_pts:
                break
            r = x % p
            old = self.counts[p][r]
            new = old + 1
            self.counts[p][r] = new
            
            # Update full_count and missing_residue
            if old == 0:
                self.full_count[p] += 1
                # If we just filled a hole, check if it's the last one
                if self.full_count[p] == p:
                    # Should not happen if we maintain admissibility, 
                    # but if it does, we have a violation.
                    # However, in local search, we check BEFORE adding.
                    pass
                # If full_count was p-1, missing_residue was r. Now it's -1?
                # If full_count becomes p, there is no missing residue.
                # If full_count < p-1, missing_residue is undefined/multiple.
                # We only care if full_count == p-1.
                if self.full_count[p] == p - 1:
                    # Find the missing one? 
                    # Actually, if we just added to r, and old==0, then r was missing.
                    # Now r is not missing.
                    # If full_count becomes p, no missing.
                    # If full_count < p, there might be other missing ones.
                    # We need to know if there is EXACTLY one missing.
                    # If full_count[p] == p-1, find the one with count 0.
                    # This is O(p). 
                    # Optimization: 
                    # If we add to r (old 0), full_count increases.
                    # If new full_count == p-1, then there is 1 missing.
                    # If new full_count < p-1, there are >1 missing.
                    # If new full_count == p, there are 0 missing.
                    
                    # Let's just recompute missing_residue if full_count == p-1.
                    # This is rare? 
                    # For large sets, full_count is usually p or p-1.
                    if self.full_count[p] == p - 1:
                        # Find the missing residue
                        for rr in range(p):
                            if self.counts[p][rr] == 0:
                                self.missing_residue[p] = rr
                                break
                    else:
                        self.missing_residue[p] = -1
            else:
                # old > 0, so we didn't fill a hole.
                # But did we create a hole? No, we added.
                # Did we change the "exactly one missing" status?
                # If full_count was p-1, and we added to a non-zero class, full_count stays p-1.
                # The missing residue is still the same.
                # If full_count was < p-1, it stays < p-1.
                pass

    def get_risk_primes(self):
        """Return list of primes where exactly one residue is missing."""
        n_pts = len(self._pts)
        risks = []
        for p in self.primes:
            if p > n_pts:
                break
            if self.full_count[p] == p - 1:
                risks.append(p)
        return risks

    def is_addable(self, x, risk_primes):
        """Check if x can be added without violating admissibility."""
        n_pts = len(self._pts)
        # Check against risk primes
        for p in risk_primes:
            if x % p == self.missing_residue[p]:
                return False
        # Also need to check if adding x makes full_count[p] == p for any p
        # This happens if full_count[p] == p-1 and x%p is the missing one (handled above)
        # Or if full_count[p] < p-1? No, adding one element only increases full_count by 1.
        # So if full_count[p] == p-1, and x%p is NOT missing, full_count becomes p-1 (still missing one).
        # Wait.
        # If full_count[p] == p-1, there is 1 missing residue r_miss.
        # If x % p == r_miss, we fill it. full_count becomes p. VIOLATION.
        # If x % p != r_miss, we add to an existing class. full_count stays p-1. OK.
        # If full_count[p] < p-1, there are >= 2 missing.
        # If x % p is one of them, full_count becomes full_count+1.
        # If full_count was p-2, it becomes p-1. Still admissible.
        # So we only care about the case where we fill the LAST hole.
        # Which is exactly what `risk_primes` and `missing_residue` check.
        return True


def _local_search(points, primes, n_max=N, deadline=None):
    """
    Local search: try to add elements to the set.
    """
    points_set = set(points)
    tracker = _ResidueTracker(primes, n_max)
    for x in points_set:
        tracker.add(x)
    
    best = list(points_set)
    best_size = len(best)
    
    if deadline is not None:
        start_time = time.time()
    
    failures = 0
    max_failures = 1000
    
    while failures < max_failures:
        if deadline is not None and time.time() > deadline - 5:
            break
            
        # Get current risk primes
        risk_primes = tracker.get_risk_primes()
        
        # Try to add a random element
        candidate = random.randint(0, n_max)
        if candidate in points_set:
            continue
            
        if tracker.is_addable(candidate, risk_primes):
            points_set.add(candidate)
            tracker.add(candidate)
            best = sorted(points_set)
            best_size = len(best)
            failures = 0 # Reset on success
        else:
            failures += 1
            
    return best


def _greedy_construct_random(primes, n_max=N, seed=None):
    """
    Construct an admissible set using greedy deletion with randomized prime order
    and random tie-breaking.
    """
    if seed is not None:
        random.seed(seed)
        
    points = set(range(n_max + 1))
    small_primes = [p for p in primes if p <= n_max + 1]
    
    # Shuffle primes to randomize the order of constraint application
    random.shuffle(small_primes)
    
    for p in small_primes:
        if p > len(points):
            break
        class_counts = Counter(v % p for v in points)
        if len(class_counts) == p:
            # Remove the least-populated residue class
            min_count = min(class_counts.values())
            remove_classes = [r for r, c in class_counts.items() if c == min_count]
            remove_class = random.choice(remove_classes)
            points = {v for v in points if v % p != remove_class}
    
    return sorted(points)


def construct():
    """
    Construct the largest admissible tuple within [0, 5000].
    """
    start_time = time.time()
    primes = _sieve_primes(N + 1)
    
    best = []
    best_size = 0
    
    deadline = start_time + DEADLINE
    
    # Multi-start greedy + local search
    # We allocate time for multiple starts.
    # Each start takes some time.
    # We run starts until deadline - 10s.
    
    num_starts = 0
    while time.time() < deadline - 15:
        start_iter = time.time()
        
        # Generate a candidate
        candidate = _greedy_construct_random(primes, N)
        
        # Improve with local search
        # Allocate some time for local search per candidate
        # Simple heuristic: if we have > 100s left, do more search
        ls_deadline = start_iter + 5.0 # 5 seconds per local search?
        if time.time() < deadline - 50:
            ls_deadline = min(deadline - 10, start_iter + 10.0)
            
        improved = _local_search(candidate, primes, N, deadline=ls_deadline)
        
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
            
        num_starts += 1
        # If a start takes too long, reduce future search depth?
        # For now, fixed 5-10s per LS.
        
    # Final verification and return
    n = len(best)
    small_primes = [p for p in primes if p <= n]
    for p in small_primes:
        residues = set(v % p for v in best)
        if len(residues) == p:
            # Not admissible! Remove from the smallest class
            class_counts = Counter(v % p for v in best)
            min_count = min(class_counts.values())
            min_classes = [r for r, c in class_counts.items() if c == min_count]
            remove_class = min_classes[0]
            best = [v for v in best if v % p != remove_class]
            
    return best

# EVOLVE-BLOCK-END
