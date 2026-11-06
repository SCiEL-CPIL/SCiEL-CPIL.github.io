import time
import math
import random
import numpy as np
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
    Uses a heuristic: remove the class with the minimum count, with random tie-breaking.
    """
    points = set(range(n_max + 1))
    small_primes = [p for p in primes if p <= n_max + 1]
    
    for p in small_primes:
        if p > len(points):
            break
        class_counts = Counter(v % p for v in points)
        if len(class_counts) == p:
            min_count = min(class_counts.values())
            min_classes = [r for r, c in class_counts.items() if c == min_count]
            remove_class = random.choice(min_classes)
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
    Uses a smarter candidate selection: prioritize elements that don't
    complete any prime's residue coverage. Also tries swap moves.
    Returns improved list if found.
    """
    points_set = set(points)
    best = sorted(points_set)
    best_size = len(best)
    small_primes = [p for p in primes if p <= n_max + 1]
    
    # Build tracker
    tracker = _ResidueTracker(primes, n_max)
    for x in points_set:
        tracker.add(x)
    
    if deadline is not None:
        start_time = time.time()
    
    # Pre-compute which elements are not in points_set
    all_elements = set(range(n_max + 1))
    candidates = all_elements - points_set
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        if not candidates:
            break
        
        # Score candidates: prefer those that don't complete any prime
        # Use a random subset of candidates to keep it fast
        sample_size = min(len(candidates), 50)
        sample = random.sample(list(candidates), sample_size)
        
        best_candidate = None
        best_score = float('inf')
        
        for candidate in sample:
            n_after = len(points_set) + 1
            would_violate = False
            critical_primes = 0
            
            for p in small_primes:
                if p > n_after:
                    break
                if tracker.full_count[p] == p:
                    would_violate = True
                    break
                r = candidate % p
                if tracker.counts[p][r] == 0:
                    if tracker.full_count[p] == p - 1:
                        would_violate = True
                        break
                    else:
                        critical_primes += 1
            
            if not would_violate:
                # Score: number of primes this candidate would make "critical"
                # (i.e., bring to p-1 full residues)
                if critical_primes < best_score:
                    best_score = critical_primes
                    best_candidate = candidate
        
        if best_candidate is not None:
            points_set.add(best_candidate)
            tracker.add(best_candidate)
            candidates.discard(best_candidate)
            best = sorted(points_set)
            best_size = len(best)
        
        # Also try swap moves: remove one element, add one that was previously blocked
        if iteration % 10 == 0 and candidates:
            # Try a few random swaps
            for _ in range(5):
                if not points_set:
                    break
                # Pick a random element to potentially remove
                to_remove = random.choice(list(points_set))
                
                # Temporarily remove it
                points_set.discard(to_remove)
                tracker.remove(to_remove)
                
                # Now try to add a candidate
                n_after = len(points_set) + 1
                added = False
                for candidate in random.sample(list(candidates), min(10, len(candidates))):
                    would_violate = False
                    for p in small_primes:
                        if p > n_after:
                            break
                        if tracker.full_count[p] == p:
                            would_violate = True
                            break
                        r = candidate % p
                        if tracker.counts[p][r] == 0:
                            if tracker.full_count[p] == p - 1:
                                would_violate = True
                                break
                    
                    if not would_violate:
                        points_set.add(candidate)
                        tracker.add(candidate)
                        candidates.discard(candidate)
                        added = True
                        break
                
                if added and len(points_set) > best_size:
                    best = sorted(points_set)
                    best_size = len(best)
                else:
                    # Revert the removal
                    points_set.add(to_remove)
                    tracker.add(to_remove)
    
    return best


def _vectorized_greedy(primes, n_max=N, deadline=None):
    """
    Vectorized greedy using NumPy for faster residue class computations.
    """
    small_primes = [p for p in primes if p <= n_max + 1]
    points = np.ones(n_max + 1, dtype=bool)
    
    if deadline is not None:
        start_time = time.time()
    
    for p in small_primes:
        if p > np.sum(points):
            break
        
        # Compute residue counts using vectorized operations
        residues = np.where(points)[0] % p
        class_counts = np.bincount(residues, minlength=p)
        
        if len(set(residues.tolist())) == p:
            # All classes present, remove the smallest class
            min_count = np.min(class_counts)
            min_classes = np.where(class_counts == min_count)[0]
            remove_class = min_classes[0]
            
            # Remove all elements with this residue
            points[(np.where(points)[0] % p) == remove_class] = False
    
    return sorted(np.where(points)[0].tolist())


def _smart_greedy_with_lookahead(primes, n_max=N, deadline=None):
    """
    Greedy with lookahead: when choosing which residue class to remove for prime p,
    consider how this choice affects subsequent primes.
    """
    small_primes = [p for p in primes if p <= n_max + 1]
    points = set(range(n_max + 1))
    
    if deadline is not None:
        start_time = time.time()
    
    for i, p in enumerate(small_primes):
        if p > len(points):
            break
        
        class_counts = Counter(v % p for v in points)
        if len(class_counts) < p:
            continue
        
        # All p classes present. Choose which class to remove.
        # Lookahead: for each candidate class r, estimate how many future primes
        # would still be fully covered after removing class r.
        best_r = None
        best_score = float('inf')
        
        future_primes = small_primes[i+1:i+10]  # Look ahead 10 primes
        
        for r in range(p):
            # Estimate impact: remove class r, count how many future primes remain fully covered
            remaining_full = 0
            for q in future_primes:
                if q > len(points):
                    break
                q_residues = set()
                for v in points:
                    if v % p != r:
                        q_residues.add(v % q)
                        if len(q_residues) == q:
                            break
                if len(q_residues) == q:
                    remaining_full += 1
            
            # Also consider the size of the class being removed
            score = (remaining_full, class_counts[r])
            
            if best_r is None or score < best_score:
                best_score = score
                best_r = r
        
        points = {v for v in points if v % p != best_r}
        
        if deadline is not None and time.time() > deadline - 15:
            break
    
    return sorted(points)


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
    
    # Try vectorized greedy
    if time.time() < deadline - 60:
        vec_result = _vectorized_greedy(primes, N, deadline=deadline)
        if len(vec_result) > best_size:
            best = vec_result
            best_size = len(best)
    
    # Try smart greedy with lookahead
    if time.time() < deadline - 45:
        smart_result = _smart_greedy_with_lookahead(primes, N, deadline=deadline)
        if len(smart_result) > best_size:
            best = smart_result
            best_size = len(best)
    
    # Try multi-start greedy with randomization
    if time.time() < deadline - 30:
        multi_result = _multi_start_greedy(primes, N, deadline=deadline)
        if len(multi_result) > best_size:
            best = multi_result
            best_size = len(best)
    
    # Local search to try to add elements back
    if time.time() < deadline - 15:
        local_result = _local_search(
            best, primes, N,
            max_iters=200000,
            deadline=deadline
        )
        if len(local_result) > best_size:
            best = local_result
            best_size = len(best)
    
    # Second round of local search on the improved solution
    if time.time() < deadline - 10:
        local_result2 = _local_search(
            best, primes, N,
            max_iters=100000,
            deadline=deadline
        )
        if len(local_result2) > best_size:
            best = local_result2
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
