import time
import math
import random
from collections import Counter

N = 5000
DEADLINE = 325.0  # seconds, keep safety margin under 330s

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
    Uses bitmask-based tracking for fast admissibility checks.
    """
    def __init__(self, primes, n_max):
        self.primes = [p for p in primes if p <= n_max + 1]
        self.n_max = n_max
        self.counts = {}
        self.full_count = {}
        self.missing_residue = {}
        self.bitmask = {}  # p -> bitmask of residues that are present
        for p in self.primes:
            self.counts[p] = [0] * p
            self.full_count[p] = 0
            self.missing_residue[p] = -1
            self.bitmask[p] = 0
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
            
            if old == 0:
                self.full_count[p] += 1
                self.bitmask[p] |= (1 << r)
                if self.full_count[p] == p - 1:
                    # Find the missing residue
                    for rr in range(p):
                        if self.counts[p][rr] == 0:
                            self.missing_residue[p] = rr
                            break
                elif self.full_count[p] == p:
                    self.missing_residue[p] = -1
                else:
                    self.missing_residue[p] = -1
            else:
                # old > 0, no change to full_count or missing_residue
                pass

    def remove(self, x):
        if x not in self._pts:
            return
        self._pts.remove(x)
        n_pts = len(self._pts)
        for p in self.primes:
            if p > n_pts:
                break
            r = x % p
            old = self.counts[p][r]
            new = old - 1
            self.counts[p][r] = new
            
            if new == 0:
                self.full_count[p] -= 1
                self.bitmask[p] &= ~(1 << r)
                if self.full_count[p] == p - 1:
                    # Find the missing residue
                    for rr in range(p):
                        if self.counts[p][rr] == 0:
                            self.missing_residue[p] = rr
                            break
                elif self.full_count[p] == p:
                    self.missing_residue[p] = -1
                else:
                    self.missing_residue[p] = -1
            else:
                # new > 0, no change to full_count or missing_residue
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
        # Check against risk primes using bitmask
        for p in risk_primes:
            if x % p == self.missing_residue[p]:
                return False
        return True

    def is_removable(self, x):
        """Check if removing x maintains admissibility."""
        n_pts = len(self._pts)
        # Removing x could make full_count[p] < p-1 for some p, but that's fine.
        # The issue is if removing x causes full_count[p] to become p-1 where it was p.
        # Actually, removing x decreases counts. If full_count[p] was p and we remove x,
        # full_count[p] becomes p-1, and the residue x%p becomes missing. That's admissible.
        # If full_count[p] was p-1 and we remove x where x%p is the missing residue, 
        # that's impossible since missing residue has count 0.
        # If full_count[p] was p-1 and we remove x where x%p != missing residue,
        # full_count stays p-1, still admissible.
        # So removing is always safe? No, wait.
        # Actually, the constraint is that for each p <= n, not all residues are present.
        # Removing elements only makes it easier to satisfy this.
        # So removal is always admissible.
        return True


def _local_search(points, primes, n_max=N, deadline=None):
    """
    Local search: try to add elements to the set.
    """
    points_set = set(points)
    tracker = _ResidueTracker(primes, n_max)
    for x in points_set:
        tracker.add(x)
    
    best = sorted(points_set)
    best_size = len(best)
    
    if deadline is not None:
        start_time = time.time()
    
    failures = 0
    max_failures = 500
    
    while failures < max_failures:
        if deadline is not None and time.time() > deadline - 2:
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
            best_size = len(points_set)
            failures = 0
        else:
            failures += 1
            
    best = sorted(points_set)
    return best


def _conflict_greedy(primes, n_max=N):
    """
    Conflict-degree greedy: build an admissible set by starting with all points
    and removing points that cause the most conflicts across primes.
    """
    points = set(range(n_max + 1))
    small_primes = [p for p in primes if p <= n_max + 1]
    
    # Process primes in increasing order (most constraining first)
    for p in small_primes:
        if p > len(points):
            break
        class_counts = Counter(v % p for v in points)
        if len(class_counts) == p:
            # All residues are present, need to remove one class
            # Remove the class with the most "conflict" - i.e., the class whose
            # removal would be least harmful. Heuristic: remove the least populated class.
            min_count = min(class_counts.values())
            remove_classes = [r for r, c in class_counts.items() if c == min_count]
            remove_class = random.choice(remove_classes)
            points = {v for v in points if v % p != remove_class}
    
    return sorted(points)


def _greedy_construct_random(primes, n_max=N, seed=None):
    """
    Construct an admissible set using greedy deletion with randomized prime order
    and random tie-breaking.
    """
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


def _is_admissible(points, primes):
    """Check if points form an admissible tuple."""
    n = len(points)
    small_primes = [p for p in primes if p <= n]
    for p in small_primes:
        residues = set(v % p for v in points)
        if len(residues) == p:
            return False
    return True


def construct():
    """
    Construct the largest admissible tuple within [0, 5000].
    """
    start_time = time.time()
    primes = _sieve_primes(N + 1)
    
    best = []
    best_size = 0
    
    deadline = start_time + DEADLINE
    
    # Multi-start: alternate between conflict greedy and random greedy
    use_conflict = False
    num_starts = 0
    
    while time.time() < deadline - 10:
        start_iter = time.time()
        
        # Alternate between conflict greedy and random greedy
        if use_conflict:
            candidate = _conflict_greedy(primes, N)
        else:
            candidate = _greedy_construct_random(primes, N)
        use_conflict = not use_conflict
        
        # Improve with local search
        remaining = deadline - time.time()
        if remaining > 100:
            ls_deadline = min(deadline - 5, start_iter + 15.0)
        elif remaining > 50:
            ls_deadline = min(deadline - 5, start_iter + 10.0)
        else:
            ls_deadline = min(deadline - 3, start_iter + 5.0)
            
        improved = _local_search(candidate, primes, N, deadline=ls_deadline)
        
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
        
        num_starts += 1
        
    # Final verification and repair
    n = len(best)
    small_primes = [p for p in primes if p <= n]
    for p in small_primes:
        residues = set(v % p for v in best)
        if len(residues) == p:
            class_counts = Counter(v % p for v in best)
            min_count = min(class_counts.values())
            min_classes = [r for r, c in class_counts.items() if c == min_count]
            remove_class = min_classes[0]
            best = [v for v in best if v % p != remove_class]
    
    return best

# EVOLVE-BLOCK-END
