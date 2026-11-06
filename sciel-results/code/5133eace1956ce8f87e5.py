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


def _is_admissible(points, primes):
    """Check if points form an admissible tuple w.r.t. given primes."""
    n = len(points)
    for p in primes:
        if p > n:
            break
        residues = set()
        for v in points:
            residues.add(v % p)
            if len(residues) == p:
                return False
    return True


class _ResidueTracker:
    """
    Maintains residue counts for a set of points across relevant primes.
    Supports fast add/remove and admissibility check.
    """
    def __init__(self, primes, n_max):
        self.primes = [p for p in primes if p <= n_max + 1]
        self.n_max = n_max
        self.counts = {}
        self.full_count = {}
        for p in self.primes:
            self.counts[p] = [0] * p
            self.full_count[p] = 0
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
            if old == 0:
                self.full_count[p] += 1
            self.counts[p][r] = new
    
    def remove(self, x):
        if x not in self._pts:
            return
        self._pts.discard(x)
        n_pts = len(self._pts)
        for p in self.primes:
            if p > n_pts:
                break
            r = x % p
            old = self.counts[p][r]
            new = old - 1
            if new == 0:
                self.full_count[p] -= 1
            self.counts[p][r] = new
    
    def is_admissible(self):
        """Check if current set is admissible."""
        n_pts = len(self._pts)
        for p in self.primes:
            if p > n_pts:
                break
            if self.full_count[p] == p:
                return False
        return True


def _greedy_construct(primes, n_max=N):
    """
    Construct a large admissible set by greedy inclusion.
    Start with all integers in [0, n_max], then remove elements to fix
    each prime whose residue classes are all covered.
    """
    points = set(range(n_max + 1))
    small_primes = [p for p in primes if p <= n_max + 1]
    
    for p in small_primes:
        if p > len(points):
            break
        class_counts = Counter(v % p for v in points)
        if len(class_counts) == p:
            # Remove the least-populated residue class
            min_count = min(class_counts.values())
            remove_class = min(class_counts.keys(), key=lambda r: class_counts[r])
            points = {v for v in points if v % p != remove_class}
    
    return sorted(points)


def _conflict_greedy(primes, n_max=N):
    """
    Conflict-graph greedy heuristic for maximum independent set.
    
    Models the problem as: each integer x in [0, n_max] is a vertex.
    Two vertices conflict if including both would cause some prime p <= k
    to have all p residue classes covered. We greedily select vertices
    with the lowest "conflict degree" to maximize final set size.
    
    Key insight: for a target size k, a prime p <= k is "tight" if all
    p residue classes are present. To avoid tightness, for each tight prime,
    at least one residue class must be entirely absent from our set.
    
    Strategy: For each integer x, compute how many primes p would become
    "fully covered" (all p residues present) if x were the last element
    added to complete that prime's coverage. Prioritize keeping integers
    that don't complete any prime's coverage.
    """
    small_primes = [p for p in primes if p <= n_max + 1]
    
    # For each prime p, track which residue classes are "needed" to be missing
    # We'll use a different approach: iteratively decide which residue class
    # to exclude for each prime, choosing the exclusion that removes the
    # fewest elements (i.e., keeps the most).
    
    # Start with all elements
    points = set(range(n_max + 1))
    
    # For each prime in order, if all residues are covered, 
    # choose the residue class to exclude that minimizes total removals
    # considering interactions with other primes
    
    for p in small_primes:
        if p > len(points):
            break
        
        # Check if all p residue classes are present
        class_counts = Counter(v % p for v in points)
        if len(class_counts) < p:
            continue  # Already missing at least one class
        
        # All p classes present. Choose which class to remove.
        # Heuristic: remove the class whose elements have the highest
        # "conflict" with other primes, i.e., elements that would cause
        # other primes to be fully covered if kept.
        
        # For each candidate residue class r, compute the "cost" of keeping it
        # Cost = number of other primes q where the elements in class r mod p
        # are the "critical" elements that complete q's coverage
        
        best_r = None
        best_cost = float('inf')
        
        for r in range(p):
            # Elements in points with v % p == r
            elements_in_class = [v for v in points if v % p == r]
            
            # Compute how many other primes would be "saved" by removing this class
            # i.e., primes q where removing these elements would make q not fully covered
            cost = 0
            for q in small_primes:
                if q == p or q > len(points):
                    continue
                # Check if q is currently fully covered
                q_classes = Counter(v % q for v in points)
                if len(q_classes) < q:
                    continue  # q already not fully covered
                
                # If we remove elements_in_class, would q lose full coverage?
                # For each residue s mod q, check if all elements with v%q==s
                # are in elements_in_class
                for s in range(q):
                    all_in_class = True
                    for v in points:
                        if v % q == s and v % p != r:
                            all_in_class = False
                            break
                    if all_in_class:
                        # This residue s mod q is entirely in class r mod p
                        # So removing class r would make q not fully covered
                        cost += 1
                        break  # One such s is enough to "save" prime q
            
            # Lower cost means this class is more "conflicting" with other primes
            # We want to remove the class with HIGHEST cost (saves most other primes)
            # Wait, we want to KEEP elements. So we should remove the class that
            # causes the most problems for other primes.
            # Actually, we want to minimize the number of elements removed overall.
            # Let me reconsider: we must remove exactly one class. 
            # Choose the class whose removal helps with other primes the most.
            
            if cost > best_cost or (best_r is None):
                # Actually we want to maximize cost (remove class that saves most)
                pass
        
        # Simpler heuristic: just remove the smallest class
        min_count = min(class_counts.values())
        min_classes = [r for r, c in class_counts.items() if c == min_count]
        remove_class = min_classes[0]
        points = {v for v in points if v % p != remove_class}
    
    return sorted(points)


def _multi_start_greedy(primes, n_max=N, deadline=None):
    """
    Multi-start greedy with different random removals to find larger admissible sets.
    """
    best = None
    best_size = 0
    small_primes = [p for p in primes if p <= n_max + 1]
    
    if deadline is not None:
        start_time = time.time()
    
    # Start with the standard greedy
    greedy_result = _greedy_construct(primes, n_max)
    best = greedy_result
    best_size = len(best)
    
    if deadline is not None and time.time() > deadline - 10:
        return best
    
    # Try multiple random initial removals
    for attempt in range(200):
        if deadline is not None and time.time() > deadline - 10:
            break
        
        # Start with full set, but randomly remove a few elements first
        points = set(range(n_max + 1))
        # Randomly remove some elements to break symmetry
        for _ in range(random.randint(1, 8)):
            if points:
                to_remove = random.choice(list(points))
                points.discard(to_remove)
        
        # Now run greedy repair
        for p in small_primes:
            if p > len(points):
                break
            class_counts = Counter(v % p for v in points)
            if len(class_counts) == p:
                min_count = min(class_counts.values())
                # Randomly choose among min-count classes
                min_classes = [r for r, c in class_counts.items() if c == min_count]
                remove_class = random.choice(min_classes)
                points = {v for v in points if v % p != remove_class}
        
        if len(points) > best_size:
            best = sorted(points)
            best_size = len(points)
    
    return best


def _local_search(points, primes, n_max=N, max_iters=50000, deadline=None):
    """
    Local search: try to add back removed elements or swap to increase size.
    Returns improved list if found.
    """
    points_set = set(points)
    best = list(points)
    best_size = len(best)
    small_primes = [p for p in primes if p <= n_max + 1]
    
    # Build tracker
    tracker = _ResidueTracker(primes, n_max)
    for x in points_set:
        tracker.add(x)
    
    if deadline is not None:
        start_time = time.time()
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Try adding a random element from [0, n_max] not in points_set
        candidate = random.randint(0, n_max)
        if candidate in points_set:
            continue
        
        # Check if adding would violate admissibility
        n_after = len(points_set) + 1
        would_violate = False
        for p in small_primes:
            if p > n_after:
                break
            if tracker.full_count[p] == p:
                would_violate = True
                break
            r = candidate % p
            if tracker.counts[p][r] == 0:
                # This would make full_count[p] become p
                if tracker.full_count[p] == p - 1:
                    would_violate = True
                    break
        
        if not would_violate:
            points_set.add(candidate)
            tracker.add(candidate)
            best = sorted(points_set)
            best_size = len(best)
    
    return best


def construct():
    """
    Construct the largest admissible tuple within [0, 5000].
    An admissible tuple satisfies: for every prime p <= len(tuple),
    at least one residue class mod p is missed.
    """
    start_time = time.time()
    primes = _sieve_primes(N + 1)
    
    # First, get a baseline greedy solution
    baseline = _greedy_construct(primes, N)
    best = baseline
    best_size = len(best)
    
    deadline = start_time + DEADLINE
    
    # Try conflict-graph greedy
    if time.time() < deadline - 60:
        conflict_result = _conflict_greedy(primes, N)
        if len(conflict_result) > best_size:
            best = conflict_result
            best_size = len(best)
    
    # Try multi-start greedy with randomization
    if time.time() < deadline - 40:
        multi_result = _multi_start_greedy(primes, N, deadline=deadline)
        if len(multi_result) > best_size:
            best = multi_result
            best_size = len(best)
    
    # Local search to try to add elements back
    if time.time() < deadline - 20:
        local_result = _local_search(
            best, primes, N,
            max_iters=100000,
            deadline=deadline
        )
        if len(local_result) > best_size:
            best = local_result
            best_size = len(best)
    
    # Final verification
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
