import time
import math
import random
from collections import Counter

N = 5000
DEADLINE = 320.0  # seconds, keep safety margin under 330s

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
            
            if old == 0:
                self.full_count[p] += 1
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
                # If we added to an existing class, full_count doesn't change.
                # But if full_count was p-1, missing_residue is still valid.
                # If full_count was < p-1, it stays < p-1.
                pass

    def remove(self, x):
        if x not in self._pts:
            return
        self._pts.remove(x)
        n_pts = len(self._pts)
        for p in self.primes:
            if p > n_pts + 1:  # Need to check primes up to old size potentially
                break
            r = x % p
            old = self.counts[p][r]
            new = old - 1
            self.counts[p][r] = new
            
            if new == 0:
                self.full_count[p] -= 1
                if self.full_count[p] == p - 1:
                    # The removed residue r is now the missing one
                    self.missing_residue[p] = r
                else:
                    self.missing_residue[p] = -1
            else:
                # old > 1, so class is still non-empty. full_count unchanged.
                # If full_count was p-1, missing_residue is still the same (unless r was missing, but r was not missing since old>0).
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
        for p in risk_primes:
            if x % p == self.missing_residue[p]:
                return False
        return True

    def is_removable(self, x):
        """Check if x can be removed while maintaining admissibility."""
        # Removing a point always maintains admissibility for the set itself,
        # but we need to check if the *new* set is admissible.
        # Since removing a point can only reduce residue coverage, 
        # if the current set is admissible, the new set is also admissible.
        # However, we usually remove to fix violations or to enable other additions.
        # In our context, we might remove to allow adding a "better" point.
        # But standard admissibility check for the resulting set:
        # If current set is admissible, removing any point keeps it admissible.
        return True


def _local_search(points, primes, n_max=N, deadline=None):
    """
    Local search: try to add elements to the set, and if stuck, try remove-add.
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
    max_failures = 200
    
    while failures < max_failures:
        if deadline is not None and time.time() > deadline - 2:
            break
            
        # Get current risk primes
        risk_primes = tracker.get_risk_primes()
        
        # Try to add a random element
        candidate = random.randint(0, n_max)
        if candidate in points_set:
            failures += 1
            continue
            
        if tracker.is_addable(candidate, risk_primes):
            points_set.add(candidate)
            tracker.add(candidate)
            best = sorted(points_set)
            best_size = len(best)
            failures = 0
        else:
            failures += 1
            # If we fail too many times, try a remove-add move
            if failures > 50 and len(points_set) > 10:
                # Try removing a random point and then adding a different one
                # This is a simple simulated annealing-like move
                victim = random.choice(list(points_set))
                # Check if removing victim keeps admissibility (it always does)
                # But we want to see if we can add something back
                tracker.remove(victim)
                points_set.remove(victim)
                
                # Now try to add a random point
                for _ in range(10):
                    new_candidate = random.randint(0, n_max)
                    if new_candidate in points_set:
                        continue
                    risk_primes = tracker.get_risk_primes()
                    if tracker.is_addable(new_candidate, risk_primes):
                        points_set.add(new_candidate)
                        tracker.add(new_candidate)
                        if len(points_set) > best_size:
                            best = sorted(points_set)
                            best_size = len(best)
                        break
                # If we didn't manage to add anything back, we lost a point.
                # Re-add the victim to restore size? Or keep the new state?
                # For now, let's restore if we didn't improve.
                if len(points_set) < best_size:
                    # Try to re-add victim
                    risk_primes = tracker.get_risk_primes()
                    if tracker.is_addable(victim, risk_primes):
                        points_set.add(victim)
                        tracker.add(victim)
                    else:
                        # Can't re-add victim, try to find any point to fill the gap
                        # This is getting complex. Let's just accept the loss for now
                        # and hope the random restarts fix it.
                        pass
                failures = 0

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


def _conflict_greedy(primes, n_max=N):
    """
    Greedy construction based on conflict graph / intersection minimization.
    Instead of just removing the smallest class, we estimate the "cost" of removing
    a residue class as the number of points removed, but we prioritize primes
    where the removal has the least impact on future constraints.
    """
    points = list(range(n_max + 1))
    points_set = set(points)
    
    # We process primes in increasing order (smaller primes constrain more)
    # But we can randomize the choice of which class to remove.
    # Key insight: When removing a class for prime p, we should prefer classes
    # that are "less used" by other primes' constraints?
    # Actually, the standard greedy is to remove the smallest class.
    # Let's try to be smarter: remove the class that minimizes the maximum
    # future removals? That's hard.
    
    # Alternative: Process primes in order of increasing p, but for each p,
    # if multiple classes have the same count, choose one that is "aligned"
    # with previously removed classes? No, that's complex.
    
    # Let's stick to the standard greedy but with a twist:
    # Process primes in random order? No, small primes are more critical.
    # Let's process primes in increasing order, but randomly choose among
    # the minimum-count classes.
    
    small_primes = [p for p in primes if p <= n_max + 1]
    small_primes.sort() # Ensure increasing order
    
    for p in small_primes:
        if p > len(points):
            break
        class_counts = Counter(v % p for v in points)
        if len(class_counts) == p:
            min_count = min(class_counts.values())
            remove_classes = [r for r, c in class_counts.items() if c == min_count]
            remove_class = random.choice(remove_classes)
            points = [v for v in points if v % p != remove_class]
            points_set = set(points)
    
    return sorted(points)


def _multi_start_greedy(primes, n_max=N, num_starts=5, deadline=None):
    """
    Run multiple greedy constructions and return the best one.
    """
    best = []
    best_size = 0
    start_time = time.time()
    
    for i in range(num_starts):
        if deadline and time.time() > deadline - 10:
            break
        candidate = _greedy_construct_random(primes, n_max)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(best)
            
    return best


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
    
    # Phase 1: Quick greedy starts to get a baseline
    quick_candidates = []
    for i in range(10):
        if time.time() > deadline - 30:
            break
        candidate = _greedy_construct_random(primes, N)
        quick_candidates.append(candidate)
        
    # Sort by size
    quick_candidates.sort(key=lambda x: -len(x))
    
    # Phase 2: Local search on the best few candidates
    for candidate in quick_candidates[:5]:
        if time.time() > deadline - 10:
            break
        improved = _local_search(candidate, primes, N, deadline=min(deadline - 5, time.time() + 10))
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
            
    # Phase 3: More aggressive search if time permits
    # Try conflict greedy and local search
    if time.time() < deadline - 20:
        for i in range(5):
            if time.time() > deadline - 10:
                break
            candidate = _conflict_greedy(primes, N)
            improved = _local_search(candidate, primes, N, deadline=min(deadline - 5, time.time() + 15))
            if len(improved) > best_size:
                best = improved
                best_size = len(best)
                
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
