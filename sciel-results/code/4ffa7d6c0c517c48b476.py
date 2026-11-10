"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000
import time
import random
import heapq
import math

_DEADLINE = 280.0  # reduced deadline to allow safety margin

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
        self.empty_class = {}  # For active primes (nonzero == p-1)
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
        for p in self.active_primes:
            if v % p == self.empty_class[p]:
                return False
        return True

    def would_become_active(self, v, p):
        """Return True if adding v to prime p would make p active (nonzero == p-1)."""
        r = v % p
        counts = self.residue_counts[p]
        if counts[r] == 0:
            return self.nonzero_classes[p] == p - 2
        return False

    def add(self, v):
        """Add a point, updating all prime residue counts."""
        if v in self.point_set:
            return
        self.point_set.add(v)

        for p in self.primes:
            r = v % p
            counts = self.residue_counts[p]

            if counts[r] == 0:
                self.nonzero_classes[p] += 1
                nz = self.nonzero_classes[p]

                if nz == p - 1:
                    self.active_primes.add(p)
                    for rr in range(p):
                        if counts[rr] == 0:
                            self.empty_class[p] = rr
                            break

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
                self.nonzero_classes[p] -= 1
                if p in self.active_primes:
                    self.active_primes.remove(p)

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

    def safety_score(self, v):
        """Score how 'risky' adding v is: count primes where v would become active."""
        score = 0
        for p in self.primes:
            if p > len(self.point_set) + 1:
                continue
            if self.would_become_active(v, p):
                score += 1
        return score


def _greedy_removal(points, primes_to_check, deadline, strategy="smallest"):
    """Greedy removal using incremental tracker with time checks in inner loop."""
    tracker = _AdmissibleTracker(points, primes_to_check)
    active_set = set(points)

    while time.time() < deadline:
        covered = tracker.covered_primes()
        if not covered:
            break

        if strategy == "smallest":
            p = min(covered)
        elif strategy == "largest":
            p = max(covered)
        elif strategy == "min_class":
            p = min(covered, key=lambda q: min(tracker.residue_counts[q]))
        elif strategy == "random":
            p = random.choice(covered)
        else:
            p = min(covered)

        counts = tracker.residue_counts[p]
        min_class = min(range(p), key=lambda r: counts[r])

        # Remove all points in that class
        to_remove = [v for v in active_set if v % p == min_class]
        for v in to_remove:
            if time.time() > deadline:
                break
            tracker.remove(v)
            active_set.discard(v)

    return list(active_set)


def _fast_add_back(points, primes_to_check, deadline):
    """Optimized add-back using safety-scored heap for efficient candidate selection."""
    tracker = _AdmissibleTracker(points, primes_to_check)
    active_set = set(points)
    all_candidates = set(range(N + 1)) - active_set

    while time.time() < deadline:
        if not all_candidates:
            break

        # Precompute forbidden residues for each active prime
        forbidden = {}
        for p in tracker.active_primes:
            forbidden[p] = tracker.empty_class[p]

        if not forbidden:
            # No active primes, add all remaining
            for v in sorted(all_candidates):
                if time.time() > deadline:
                    break
                tracker.add(v)
                active_set.add(v)
                all_candidates.discard(v)
            break

        # Build heap of (safety_score, value) for all valid candidates
        heap = []
        for v in all_candidates:
            # Check if v is valid (doesn't hit any forbidden residue)
            valid = True
            for p, r in forbidden.items():
                if v % p == r:
                    valid = False
                    break
            if not valid:
                continue
            s = tracker.safety_score(v)
            heapq.heappush(heap, (s, v))

        if not heap:
            break

        # Pop and add candidates with safety score 0 (safest) first
        added_any = False
        while heap:
            if time.time() > deadline:
                break
            s, v = heapq.heappop(heap)
            if v not in all_candidates:
                continue
            if s == 0:
                # Safe to add
                if tracker.can_add(v):
                    tracker.add(v)
                    active_set.add(v)
                    all_candidates.discard(v)
                    added_any = True
            else:
                # This candidate would activate a prime; stop this pass
                break

        if not added_any:
            # No more safe candidates in this pass; try adding the single best
            # even if it activates a prime, but only if it's the only option
            # and we still have time
            heap2 = []
            for v in all_candidates:
                valid = True
                for p, r in forbidden.items():
                    if v % p == r:
                        valid = False
                        break
                if not valid:
                    continue
                s = tracker.safety_score(v)
                heapq.heappush(heap2, (s, v))
            if heap2:
                s, v = heapq.heappop(heap2)
                if v in all_candidates and tracker.can_add(v):
                    tracker.add(v)
                    active_set.add(v)
                    all_candidates.discard(v)
                else:
                    break
            else:
                break

    return list(active_set)


def _polish_pass(points, primes_to_check, deadline):
    """Final high-precision polish: use soft-min surrogate to break ties more finely.
    
    The smooth surrogate considers the distribution of residue counts, not just
    which class is empty. We re-examine near-active primes (nonzero == p-2) and
    try to remove the least-impactful point to free up a residue class, then
    attempt to add back more points.
    """
    tracker = _AdmissibleTracker(points, primes_to_check)
    active_set = set(points)

    # Soft-min surrogate: for a prime p with nonzero == p-2, the "pressure" is
    # determined by the two smallest residue counts. We want to minimize the
    # maximum impact of removal.
    while time.time() < deadline:
        # Find primes that are near-active (nonzero == p-2) and could become active
        # if we add a point in the empty class
        near_active = []
        for p in tracker.primes:
            if p > len(active_set) + 1:
                continue
            if tracker.nonzero_classes[p] == p - 2:
                near_active.append(p)

        if not near_active:
            break

        # For each near-active prime, find the two empty classes
        # and try to remove a point that is in one of the non-empty classes
        # to create a third empty class (making it safe to add more)
        improved = False
        for p in near_active:
            if time.time() > deadline:
                break
            counts = tracker.residue_counts[p]
            empty_classes = [r for r in range(p) if counts[r] == 0]
            if len(empty_classes) != 2:
                continue

            # The two empty classes are e1, e2. Adding a point in e1 or e2
            # would make p active. We want to remove a point from a non-empty
            # class to create a third empty class.
            # Find the non-empty class with the smallest count (least impact)
            non_empty = [(r, counts[r]) for r in range(p) if counts[r] > 0]
            if not non_empty:
                continue
            non_empty.sort(key=lambda x: x[1])
            r_min, c_min = non_empty[0]

            # Find a point in class r_min that, when removed, doesn't break
            # other primes' admissibility
            candidates = [v for v in active_set if v % p == r_min]
            if not candidates:
                continue

            # Try removing the candidate with the highest safety score
            # (i.e., the one that covers the most "near-active" primes)
            best_v = None
            best_score = -1
            for v in candidates[:10]:  # Limit to 10 for speed
                score = 0
                for q in tracker.primes:
                    if q > len(active_set):
                        continue
                    rq = v % q
                    if tracker.residue_counts[q][rq] == 1:
                        score += 1  # Removing v would empty this class
                if score > best_score:
                    best_score = score
                    best_v = v

            if best_v is not None:
                # Remove best_v
                tracker.remove(best_v)
                active_set.discard(best_v)
                improved = True

        if not improved:
            break

    # After polish removals, do a final add-back
    points = _fast_add_back(list(active_set), primes_to_check, deadline)
    return points


def _full_pipeline(points, deadline, strategy="smallest"):
    """Run greedy removal then fast add-back."""
    primes_to_check = _primes(N + 1)

    # Phase 1: Greedy removal to make admissible
    points = _greedy_removal(points, primes_to_check, deadline, strategy)

    # Phase 2: Fast Add-back search
    points = _fast_add_back(points, primes_to_check, deadline)

    # Phase 3: Polish pass with soft-min surrogate
    if time.time() < deadline - 10:
        points = _polish_pass(points, primes_to_check, deadline)

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
    attempts = 0
    while time.time() < deadline - 15 and attempts < 5:
        fraction = 0.7 + attempts * 0.05
        if fraction > 1.0:
            fraction = 1.0
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
