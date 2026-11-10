"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000
import time
import random
import heapq
import math

_DEADLINE = 320.0  # safety margin before 330s kill

# EVOLVE-BLOCK-START

def _primes(limit):
    """Generate all primes up to limit using sieve."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


class _AdmissibleTracker:
    """Efficient incremental tracker for admissibility."""

    def __init__(self, points, primes):
        self.point_set = set(points)
        self.primes = primes
        self.residue_counts = {}
        self.nonzero_classes = {}
        self.active_primes = set()  # primes where nonzero == p-1
        self.empty_class = {}       # the empty residue class for active primes

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
        """Check if current set is admissible."""
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

    def would_become_active(self, v, p):
        """Return True if adding v would make prime p active."""
        r = v % p
        counts = self.residue_counts[p]
        if counts[r] == 0:
            return self.nonzero_classes[p] == p - 2
        return False


def _safety_score(tracker, v):
    """Score how risky adding v is: count primes where v would become active."""
    score = 0
    for p in tracker.primes:
        if p > len(tracker.point_set) + 1:
            continue
        if tracker.would_become_active(v, p):
            score += 1
    return score


def _greedy_removal(points, primes_to_check, deadline, strategy="smallest"):
    """Greedy removal to make set admissible."""
    tracker = _AdmissibleTracker(points, primes_to_check)
    active_set = set(points)

    while time.time() < deadline:
        # Find primes where all residue classes are covered
        covered = []
        for p in tracker.primes:
            if p > len(active_set):
                continue
            if tracker.nonzero_classes[p] == p:
                covered.append(p)

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


def _add_back_search(points, primes_to_check, deadline):
    """Add back points greedily, prioritizing safe additions."""
    tracker = _AdmissibleTracker(points, primes_to_check)
    active_set = set(points)
    all_candidates = set(range(N + 1)) - active_set

    while time.time() < deadline:
        if not all_candidates:
            break

        # Get forbidden residues
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

        # Find valid candidates (don't hit any forbidden residue)
        valid_candidates = []
        for v in all_candidates:
            valid = True
            for p, r in forbidden.items():
                if v % p == r:
                    valid = False
                    break
            if valid:
                valid_candidates.append(v)

        if not valid_candidates:
            break

        # Score candidates by safety (lower is better)
        scored = []
        for v in valid_candidates:
            s = _safety_score(tracker, v)
            scored.append((s, v))

        scored.sort(key=lambda x: (x[0], x[1]))

        # Add the safest candidate
        s, v = scored[0]
        if tracker.can_add(v):
            tracker.add(v)
            active_set.add(v)
            all_candidates.discard(v)
        else:
            # Shouldn't happen, but break if stuck
            break

    return list(active_set)


def _local_search(points, primes_to_check, deadline):
    """Local search with swap moves to maximize set size while maintaining admissibility."""
    tracker = _AdmissibleTracker(points, primes_to_check)
    active_set = set(points)
    excluded_set = set(range(N + 1)) - active_set

    best_size = len(active_set)
    no_improve = 0

    while time.time() < deadline - 5:
        if not excluded_set:
            break

        # Try to find a swap: remove one from active, add one from excluded
        # that maintains admissibility and doesn't decrease size
        # For pure improvement, we want to add without removing
        # But if stuck, try swaps that might enable future additions

        # First, try pure additions
        forbidden = {}
        for p in tracker.active_primes:
            forbidden[p] = tracker.empty_class[p]

        if forbidden:
            # Find candidates that don't violate constraints
            candidates = []
            for v in excluded_set:
                valid = True
                for p, r in forbidden.items():
                    if v % p == r:
                        valid = False
                        break
                if valid:
                    candidates.append(v)

            if candidates:
                # Pick the one with lowest safety score
                best_v = min(candidates, key=lambda v: _safety_score(tracker, v))
                if tracker.can_add(best_v):
                    tracker.add(best_v)
                    active_set.add(best_v)
                    excluded_set.discard(best_v)
                    no_improve = 0
                    continue
            else:
                # No valid candidates, try swaps
                pass
        else:
            # No active primes, add all
            for v in list(excluded_set):
                tracker.add(v)
                active_set.add(v)
                excluded_set.discard(v)
            break

        # Try swap moves: remove one active point, add one excluded point
        # This might free up residue classes to allow more additions later
        if no_improve > 20:
            # Try random swap
            if active_set and excluded_set:
                v_remove = random.choice(list(active_set))
                v_add = random.choice(list(excluded_set))

                # Check if this swap maintains admissibility
                # Temporarily remove v_remove
                tracker.remove(v_remove)
                active_set.discard(v_remove)
                excluded_set.add(v_remove)

                # Try to add v_add
                if tracker.can_add(v_add):
                    tracker.add(v_add)
                    active_set.add(v_add)
                    excluded_set.discard(v_add)
                    no_improve = 0
                else:
                    # Revert
                    tracker.remove(v_add) if v_add in active_set else None
                    tracker.add(v_remove)
                    active_set.add(v_remove)
                    excluded_set.discard(v_remove)
                    excluded_set.add(v_add)
                    no_improve += 1
            else:
                no_improve += 1
        else:
            no_improve += 1

        if no_improve > 100:
            break

    return list(active_set)


def _full_pipeline(points, deadline, strategy="smallest"):
    """Run greedy removal, add-back, and local search."""
    primes_to_check = _primes(N + 1)

    # Phase 1: Greedy removal to make admissible
    points = _greedy_removal(points, primes_to_check, deadline, strategy)

    # Phase 2: Add-back search
    points = _add_back_search(points, primes_to_check, deadline)

    # Phase 3: Local search
    if time.time() < deadline - 10:
        points = _local_search(points, primes_to_check, deadline)

    return points


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE

    all_primes = _primes(N + 1)
    best = []

    # Strategy 1: Start from full set with different removal heuristics
    strategies = ["smallest", "min_class", "largest"]

    for strat in strategies:
        if time.time() > deadline - 30:
            break

        init_full = list(range(N + 1))
        result = _full_pipeline(init_full, deadline, strat)
        if len(result) > len(best):
            best = result

    # Strategy 2: Try different initial densities
    attempts = 0
    while time.time() < deadline - 20 and attempts < 3:
        fraction = 0.8 + attempts * 0.06
        if fraction > 1.0:
            fraction = 1.0
        random.seed(attempts * 104729 + 42)
        init = random.sample(range(N + 1), int(fraction * (N + 1)))
        result = _full_pipeline(init, deadline, "min_class")
        if len(result) > len(best):
            best = result
        attempts += 1

    # Final add-back on best
    if time.time() < deadline - 5 and best:
        best = _add_back_search(best, all_primes, deadline)

    return best

# EVOLVE-BLOCK-END
