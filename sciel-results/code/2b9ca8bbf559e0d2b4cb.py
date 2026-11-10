"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000
import time
import random
import heapq

_DEADLINE = 320.0  # safety margin before 330s kill

# EVOLVE-BLOCK-START
def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


class _AdmissibleTracker:
    """Incremental tracker for admissibility using residue class counts per prime."""
    
    def __init__(self, points, primes):
        self.point_set = set(points)
        self.primes = primes
        self.residue_counts = {}
        self.nonzero_classes = {}
        self.empty_class = {} # For active primes (nonzero == p-1)
        self.active_primes = set()
        
        for p in primes:
            counts = [0] * p
            for v in self.point_set:
                counts[v % p] += 1
            self.residue_counts[p] = counts
            nz = sum(1 for c in counts if c > 0)
            self.nonzero_classes[p] = nz
            if nz == p - 1:
                self.active_primes.add(p)
                # Find the empty class
                for r in range(p):
                    if counts[r] == 0:
                        self.empty_class[p] = r
                        break
    
    def is_admissible(self):
        """Check if the current set is admissible."""
        for p in self.primes:
            if p > len(self.point_set):
                continue
            if self.nonzero_classes[p] == p:
                return False
        return True

    def can_add(self, v):
        """Check if v can be added without violating admissibility."""
        # A prime p is violated if it becomes covered (nonzero == p)
        # This happens if p was active (nonzero == p-1) and v falls in the empty class
        # OR if p was already covered (which shouldn't happen if we maintain invariant)
        
        for p in self.active_primes:
            if v % p == self.empty_class[p]:
                return False
        return True

    def add(self, v):
        """Add a point, updating all prime residue counts."""
        if v in self.point_set:
            return
        self.point_set.add(v)
        
        # Update counts and active status
        for p in self.primes:
            r = v % p
            counts = self.residue_counts[p]
            
            if counts[r] == 0:
                # This class was empty, now it's occupied
                self.nonzero_classes[p] += 1
                nz = self.nonzero_classes[p]
                
                # Check if p becomes active (nz == p-1)
                if nz == p - 1:
                    self.active_primes.add(p)
                    # Find the new empty class
                    for rr in range(p):
                        if counts[rr] == 0:
                            self.empty_class[p] = rr
                            break
                # If nz == p, it's covered, which is bad, but can_add should have prevented it
                # Unless p was not active before but became covered?
                # If p was p-2, and we add to empty, it becomes p-1 (active).
                # If p was p-1 (active), and we add to empty, it becomes p (covered). can_add prevents this.
                # So we never go from active to covered.
                
            else:
                # Class was already occupied
                # nonzero_classes[p] does not change
                # But wait, if p was active (p-1 occupied), and we add to an occupied class,
                # it remains p-1 occupied. So it stays active.
                # No change to active_primes or empty_class needed.
                pass
            
            counts[r] += 1

    def remove(self, v):
        """Remove a point, updating all prime residue counts."""
        if v not in self.point_set:
            return
        self.point_set.remove(v)
        
        for p in self.primes:
            r = v % p
            counts = self.residue_counts[p]
            
            if counts[r] == 1:
                # This class becomes empty
                self.nonzero_classes[p] -= 1
                nz = self.nonzero_classes[p]
                
                # If p was active (p-1), it becomes p-2. Remove from active.
                if p in self.active_primes:
                    self.active_primes.remove(p)
                    # empty_class[p] is now invalid/stale, will be recalculated if it becomes active again
                    # But we don't strictly need to clear it, just ignore if not in active_primes
                # If p becomes active? No, nonzero decreased.
                
            counts[r] -= 1

    def covered_primes(self):
        """Return list of primes where all residue classes are covered."""
        result = []
        for p in self.primes:
            if p > len(self.point_set):
                continue
            if self.nonzero_classes[p] == p:
                result.append(p)
        return result


def _greedy_removal(points, primes_to_check, deadline, strategy="smallest"):
    """Greedy removal using incremental tracker."""
    tracker = _AdmissibleTracker(points, primes_to_check)
    
    while time.time() < deadline:
        covered = tracker.covered_primes()
        if not covered:
            break
        
        if strategy == "smallest":
            p = min(covered)
        elif strategy == "largest":
            p = max(covered)
        elif strategy == "min_class":
            # Pick prime where the smallest class has the fewest elements
            p = min(covered, key=lambda q: min(tracker.residue_counts[q]))
        elif strategy == "random":
            p = random.choice(covered)
        else:
            p = min(covered)
        
        # Find the least populated residue class mod p
        counts = tracker.residue_counts[p]
        min_class = min(range(p), key=lambda r: counts[r])
        
        # Remove all points in that class
        to_remove = [v for v in tracker.point_set if v % p == min_class]
        for v in to_remove:
            tracker.remove(v)
    
    return list(tracker.point_set)


def _fast_add_back(points, primes_to_check, deadline):
    """Optimized add-back using active_primes tracking."""
    tracker = _AdmissibleTracker(points, primes_to_check)
    all_candidates = set(range(N + 1)) - tracker.point_set
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Find candidates that can be added
        # Condition: v % p != empty_class[p] for all p in active_primes
        valid_candidates = []
        for v in all_candidates:
            is_valid = True
            for p in tracker.active_primes:
                if v % p == tracker.empty_class[p]:
                    is_valid = False
                    break
            if is_valid:
                valid_candidates.append(v)
        
        if not valid_candidates:
            break
        
        # Pick a candidate. 
        # Heuristic: Pick the one that causes the fewest new active primes?
        # Or just pick the smallest?
        # Let's pick the smallest for determinism, or random for diversity.
        # Smallest is fine.
        v_to_add = min(valid_candidates)
        
        tracker.add(v_to_add)
        all_candidates.discard(v_to_add)
        improved = True
    
    return list(tracker.point_set)


def _full_pipeline(points, deadline, strategy="smallest"):
    """Run greedy removal then fast add-back."""
    primes_to_check = _primes(N + 1)
    
    # Phase 1: Greedy removal to make admissible
    points = _greedy_removal(points, primes_to_check, deadline, strategy)
    
    # Phase 2: Fast Add-back search
    points = _fast_add_back(points, primes_to_check, deadline)
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Get primes up to N+1 for checking
    all_primes = _primes(N + 1)
    
    best = []
    
    # Strategy 1: Start from full set [0, N] with different removal heuristics
    strategies = ["smallest", "min_class", "largest", "random"]
    
    for strat in strategies:
        if time.time() > deadline - 10:
            break
            
        init_full = list(range(N + 1))
        result = _full_pipeline(init_full, deadline, strat)
        if len(result) > len(best):
            best = result
    
    # Strategy 2: If we have time, try random restarts with different densities
    # This might find a better local optimum if the full set sieve gets stuck in a bad one
    attempts = 0
    while time.time() < deadline - 15 and attempts < 3:
        fraction = 0.7 # Start dense
        seed = attempts * 7919 + 12345
        random.seed(seed)
        init = random.sample(range(N + 1), int(fraction * (N + 1)))
        result = _full_pipeline(init, deadline, "min_class")
        if len(result) > len(best):
            best = result
        attempts += 1
    
    # Final: if we have time, do one more add-back on the best
    if time.time() < deadline - 5 and best:
        best = _fast_add_back(best, all_primes, deadline)
    
    return best
# EVOLVE-BLOCK-END
