"""Construct the largest admissible tuple within [0, 5000]."""
import time
import random
from collections import defaultdict

N = 5000
DEADLINE = 320  # safety margin before 330s kill


def _primes(limit):
    """Return list of primes up to limit using sieve."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]


def _is_admissible(points, primes):
    """Check admissibility: for every prime p <= len(points), some residue class mod p is missed."""
    n = len(points)
    if n == 0:
        return True
    for p in primes:
        if p > n:
            break
        residues = set(v % p for v in points)
        if len(residues) == p:
            return False
    return True


class _ResidueTracker:
    """
    Maintains residue class counts for a set of points across relevant primes.
    Allows O(1) per-prime update when adding/removing a point.
    """
    def __init__(self, primes, initial_points):
        self.primes = primes
        self.point_set = set(initial_points)
        self.current_len = len(self.point_set)
        # For each prime p <= current_len, track count of each residue class
        # residue_counts[p] = [count_0, count_1, ..., count_{p-1}]
        self.residue_counts = {}
        self._build()

    def _build(self):
        self.residue_counts = {}
        for p in self.primes:
            if p > self.current_len:
                break
            counts = [0] * p
            for v in self.point_set:
                counts[v % p] += 1
            self.residue_counts[p] = counts

    def _relevant_primes(self):
        """Yield primes p <= current_len."""
        for p in self.primes:
            if p > self.current_len:
                break
            yield p

    def _is_full(self, p):
        """Check if all p residue classes are present (i.e., no class is missed)."""
        counts = self.residue_counts.get(p)
        if counts is None:
            return False
        return all(c > 0 for c in counts)

    def _would_be_full_after_add(self, v, p):
        """Check if after adding v, all p residue classes would be present."""
        counts = self.residue_counts.get(p)
        if counts is None:
            return False
        r = v % p
        if counts[r] > 0:
            # Adding v doesn't create a new residue class
            return all(c > 0 for c in counts)
        else:
            # Adding v fills the last empty class
            # Check: is this the only empty class?
            empty_count = sum(1 for c in counts if c == 0)
            return empty_count == 1

    def can_add(self, v):
        """Check if adding v keeps the tuple admissible."""
        new_len = self.current_len + 1
        for p in self.primes:
            if p > new_len:
                break
            if p == new_len:
                # New prime becomes relevant; v must not make all classes present
                # Actually for a new prime p == new_len, we need at least one class missed
                # After adding v, we have new_len points. We need some class mod p to be empty.
                # Since we had current_len = p-1 points, at most p-1 classes were occupied.
                # Adding v can fill at most one more class, so at most p classes.
                # We need to check: will all p classes be filled?
                counts = [0] * p
                for u in self.point_set:
                    counts[u % p] += 1
                r = v % p
                if counts[r] == 0:
                    # v fills a new class
                    empty_after = sum(1 for c in counts if c == 0) - 1
                    if empty_after == 0:
                        return False
                # else v fills an already-occupied class, no change in class coverage
            else:
                # p < new_len, already relevant
                if self._would_be_full_after_add(v, p):
                    return False
        return True

    def add(self, v):
        """Add v to the set. Caller must ensure can_add(v) is True."""
        self.point_set.add(v)
        self.current_len += 1
        # Update residue counts for all primes p <= current_len
        for p in self.primes:
            if p > self.current_len:
                break
            if p not in self.residue_counts:
                # New prime becomes relevant
                counts = [0] * p
                for u in self.point_set:
                    counts[u % p] += 1
                self.residue_counts[p] = counts
            else:
                self.residue_counts[p][v % p] += 1

    def get_points(self):
        return list(self.point_set)

    def get_len(self):
        return self.current_len


def _construct_greedy_fast(primes, deadline, order="increasing"):
    """
    Fast greedy: start with all points [0, N]. Use residue tracking.
    For each prime in given order, if all classes present, remove least-populated class.
    """
    start = time.time()
    present = bytearray(b'\x01') * (N + 1)
    all_points = list(range(N + 1))
    
    if order == "increasing":
        prime_list = primes
    elif order == "decreasing":
        prime_list = list(reversed(primes))
    elif order == "binding":
        # Process primes that are most "binding" first: those where removing a class
        # costs the least (smallest min class count)
        prime_list = list(primes)
    else:
        prime_list = list(primes)
        random.shuffle(prime_list)
    
    # We'll work with the set of present points
    point_set = set(all_points)
    
    for p in prime_list:
        if p > N:
            break
        if time.time() - start > deadline:
            break
        
        # Count points in each residue class mod p
        class_counts = [0] * p
        for v in point_set:
            class_counts[v % p] += 1
        
        # If all p classes have at least one point, remove the least populated class
        if min(class_counts) > 0:
            min_count = min(class_counts)
            target_class = class_counts.index(min_count)
            
            # Remove all points in this class
            to_remove = [v for v in point_set if v % p == target_class]
            for v in to_remove:
                point_set.discard(v)
    
    return list(point_set)


def _refine_fast(points, primes, deadline, shuffle=False):
    """
    Fast refinement using incremental residue tracking.
    Try to add back removed elements that don't violate admissibility.
    """
    start = time.time()
    if not points:
        return points
    
    tracker = _ResidueTracker(primes, points)
    removed = [v for v in range(N + 1) if v not in tracker.point_set]
    
    if shuffle:
        random.shuffle(removed)
    
    for v in removed:
        if time.time() - start > deadline:
            break
        
        if tracker.can_add(v):
            tracker.add(v)
    
    return tracker.get_points()


def construct():
    """
    Construct the largest admissible tuple within [0, 5000].
    
    An admissible tuple is one where for every prime p <= len(tuple),
    some residue class mod p is missed.
    """
    start_time = time.time()
    
    # Generate primes up to N
    primes = _primes(N)
    
    # Try multiple greedy strategies and keep the best
    best_result = []
    best_len = 0
    
    strategies = [
        ("increasing", 0.4),
        ("decreasing", 0.3),
        ("random", 0.3),
    ]
    
    for order, time_frac in strategies:
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed - 20  # Leave time for refinement
        if remaining < 5:
            break
        
        deadline_phase1 = remaining * time_frac
        result = _construct_greedy_fast(primes, deadline_phase1, order=order)
        
        # Verify admissibility
        if not _is_admissible(result, primes):
            # Fallback: aggressive removal
            for p in primes:
                if p > len(result):
                    break
                residues = set(v % p for v in result)
                if len(residues) == p:
                    class_counts = defaultdict(list)
                    for v in result:
                        class_counts[v % p].append(v)
                    min_class = min(class_counts, key=lambda c: len(class_counts[c]))
                    result = [v for v in result if v % p != min_class]
        
        if len(result) > best_len:
            best_len = len(result)
            best_result = result
    
    # Refinement phase using fast tracker
    elapsed = time.time() - start_time
    remaining_time = DEADLINE - elapsed - 5
    
    if remaining_time > 5 and best_result:
        # Deterministic refinement
        refine_deadline = min(remaining_time * 0.35, 60)
        refined = _refine_fast(best_result, primes, refine_deadline, shuffle=False)
        if len(refined) > best_len:
            best_len = len(refined)
            best_result = refined
        
        # Random refinement passes
        elapsed = time.time() - start_time
        remaining_time = DEADLINE - elapsed - 5
        if remaining_time > 5 and best_result:
            random_deadline = min(remaining_time * 0.65, 150)
            refined_random = _refine_fast(best_result, primes, random_deadline, shuffle=True)
            if len(refined_random) > best_len:
                best_len = len(refined_random)
                best_result = refined_random
            
            # Second random pass with different seed
            elapsed = time.time() - start_time
            remaining_time = DEADLINE - elapsed - 5
            if remaining_time > 5 and best_result:
                random_deadline2 = min(remaining_time * 0.5, 60)
                refined_random2 = _refine_fast(best_result, primes, random_deadline2, shuffle=True)
                if len(refined_random2) > best_len:
                    best_len = len(refined_random2)
                    best_result = refined_random2
    
    # Final verification
    if best_result and not _is_admissible(best_result, primes):
        # Emergency fallback: remove until admissible
        for p in primes:
            if p > len(best_result):
                break
            residues = set(v % p for v in best_result)
            if len(residues) == p:
                class_counts = defaultdict(list)
                for v in best_result:
                    class_counts[v % p].append(v)
                min_class = min(class_counts, key=lambda c: len(class_counts[c]))
                best_result = [v for v in best_result if v % p != min_class]
    
    # Ensure we return distinct integers in [0, N]
    best_result = sorted(set(best_result))
    
    return best_result
