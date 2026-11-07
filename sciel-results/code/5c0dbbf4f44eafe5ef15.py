"""Cardinality-separated subset sums: fast incremental DP + local search.

Key improvements:
- Persistent sum_to_k map in tracker to avoid rebuilding on each add_element.
- Smarter local search: bounded swap attempts, no full shuffle of all pairs.
- Faster removal-based improvement with incremental reconstruction.
- Conservative deadline to avoid timeout.
"""
import time
import random

DEADLINE_SECONDS = 80.0  # safety margin before 90s kill


def _subset_sums_by_cardinality(S):
    """Return list: sums_by_k[k] = set of achievable sums with exactly k elements."""
    n = len(S)
    sums_by_k = [set() for _ in range(n + 1)]
    sums_by_k[0].add(0)
    for x in S:
        for k in range(n, 0, -1):
            if sums_by_k[k - 1]:
                for s in sums_by_k[k - 1]:
                    sums_by_k[k].add(s + x)
    return sums_by_k


def _is_valid(S):
    """Check no two subsets of different cardinalities share a sum."""
    if not S:
        return True
    sums_by_k = _subset_sums_by_cardinality(S)
    # For each sum, record which cardinalities achieve it
    sum_to_ks = {}
    for k, sums in enumerate(sums_by_k):
        for s in sums:
            if s in sum_to_ks:
                sum_to_ks[s].add(k)
            else:
                sum_to_ks[s] = {k}
    for s, ks in sum_to_ks.items():
        if len(ks) > 1:
            return False
    return True


class _CardinalityTracker:
    """Maintains sums_by_k incrementally for fast validity checks."""
    
    def __init__(self):
        self.sums_by_k = [set() for _ in range(61)]  # up to 60 elements
        self.sums_by_k[0].add(0)
        self.n = 0
        # sum_to_k maps each sum to the unique cardinality that achieves it
        self.sum_to_k = {0: 0}
    
    def add_element(self, x):
        """Add element x, return True if validity is preserved, False otherwise.
        If False, the state is not modified."""
        n = self.n
        sum_to_k = self.sum_to_k
        
        # Check: for each k from 1 to n+1, the new sums at k are old_sums[k-1] + x
        # We need to ensure that no sum in (old_sums[k-1] + x) appears in old_sums[j] for j != k
        for k in range(1, n + 2):
            if not self.sums_by_k[k - 1]:
                continue
            for s in self.sums_by_k[k - 1]:
                new_s = s + x
                if new_s in sum_to_k:
                    if sum_to_k[new_s] != k:
                        return False
                # If new_s is not in sum_to_k, it's a new sum at cardinality k, no conflict
        
        # All checks passed, commit the addition
        for k in range(n + 1, 0, -1):
            if self.sums_by_k[k - 1]:
                for s in self.sums_by_k[k - 1]:
                    self.sums_by_k[k].add(s + x)
                    # Update sum_to_k for the new sums
                    # But we need to be careful: if new_s was already in sums_by_k[k] (from old),
                    # it already has a mapping. If it's new, add it.
                    # Actually, since we checked validity, new_s is either:
                    # - already in sums_by_k[k] (same cardinality, fine)
                    # - new (not in any sums_by_k[j])
                    # So we can safely add to sum_to_k if not present
                    if new_s not in sum_to_k:
                        sum_to_k[new_s] = k
        self.n += 1
        return True
    
    def get_size(self):
        return self.n
    
    def reset(self):
        self.sums_by_k = [set() for _ in range(61)]
        self.sums_by_k[0].add(0)
        self.n = 0
        self.sum_to_k = {0: 0}


def _greedy_build(candidates, tracker_class=_CardinalityTracker):
    """Greedy build using the given candidate ordering."""
    tracker = tracker_class()
    result = []
    for x in candidates:
        if tracker.n >= 60:
            break
        if tracker.add_element(x):
            result.append(x)
    return result, tracker


def _try_improve_with_removal(current, deadline):
    """
    Try to improve by removing elements that block additions.
    For each element u in current, check if removing it allows adding more elements.
    """
    current_set = set(current)
    best = current[:]
    best_size = len(current)
    
    # Try removing each element and see if we can add more
    for u in current[:]:
        if time.time() > deadline - 2.0:
            break
        
        # Try removing u
        reduced = [x for x in current if x != u]
        reduced_set = set(reduced)
        
        # Now greedily add elements to the reduced set
        tracker = _CardinalityTracker()
        for x in reduced:
            tracker.add_element(x)
        
        # Try adding elements not in reduced
        for x in range(1, 257):
            if x in reduced_set:
                continue
            if len(reduced) >= 60:
                break
            if tracker.add_element(x):
                reduced.append(x)
                reduced_set.add(x)
        
        if len(reduced) > best_size:
            best = reduced[:]
            best_size = len(reduced)
    
    return best


def _local_search(current, deadline):
    """
    Local search to maximize set size using incremental tracker.
    1. Greedily add any valid element.
    2. If stuck, try targeted swaps to escape local optima.
    3. Try removal-based improvements.
    """
    current_set = set(current)
    no_improve_count = 0
    
    # Build tracker for current state
    tracker = _CardinalityTracker()
    for x in current:
        if not tracker.add_element(x):
            return current
    
    while time.time() < deadline - 2.0:
        improved = False
        
        # Phase 1: Try to add any element
        for x in range(1, 257):
            if x in current_set:
                continue
            if len(current) >= 60:
                break
            
            if tracker.add_element(x):
                current.append(x)
                current_set.add(x)
                improved = True
                no_improve_count = 0
                break  # Restart addition loop with new state
        
        if len(current) >= 60:
            break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 10:
                # Try removal-based improvement
                if time.time() < deadline - 5.0:
                    improved_set = _try_improve_with_removal(current, deadline)
                    if len(improved_set) > len(current):
                        current = improved_set
                        current_set = set(current)
                        no_improve_count = 0
                        tracker = _CardinalityTracker()
                        for x in current:
                            tracker.add_element(x)
                        continue
                break  # Give up on this local optimum
            
            # Phase 2: Try targeted swaps - sample a limited number
            # Pick random elements to remove and random candidates to add
            n_swaps_to_try = min(50, len(current) * 5)
            for _ in range(n_swaps_to_try):
                if time.time() > deadline - 2.0:
                    break
                
                u = random.choice(current)
                v = random.randint(1, 256)
                if v in current_set:
                    continue
                
                # Try the swap
                candidate = [x for x in current if x != u] + [v]
                if _is_valid(candidate):
                    current = candidate
                    current_set.remove(u)
                    current_set.add(v)
                    improved = True
                    no_improve_count = 0
                    tracker = _CardinalityTracker()
                    for x in current:
                        tracker.add_element(x)
                    break  # Restart loop to try additions again
    
    return current


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS
    
    def time_left():
        return deadline - time.time()
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Multiple greedy passes with different orderings
    # The key insight: we want elements that create wide gaps between cardinality sum ranges.
    # Small elements first, then large ones, tends to work well.
    
    orderings = [
        list(range(1, 257)),                          # Ascending
        list(range(256, 0, -1)),                       # Descending
        list(range(1, 20)) + list(range(100, 257)),   # Small then large
        [1, 2, 4, 8, 16, 32, 64, 128] + list(range(3, 257)),  # Powers of 2 first
        list(range(1, 50)) + list(range(50, 257, 2)), # Odd spacing
    ]
    
    for ordering in orderings:
        if time_left() < 2.0:
            break
        result, tracker = _greedy_build(ordering)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
    
    # Strategy 2: Random restarts with local search
    for attempt in range(500):
        if time_left() < 1.0:
            break
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        result, tracker = _greedy_build(candidates)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
            # Run local search immediately on new best if time permits
            if time_left() > 8 and best_size < 60:
                current = best_set[:]
                improved = _local_search(current, deadline)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_set = improved[:]
    
    # Strategy 3: Final local search on best set
    if time_left() > 5 and best_size > 0 and best_size < 60:
        current = best_set[:]
        improved = _local_search(current, deadline)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved[:]
    
    # Final validation
    if best_set and not _is_valid(best_set):
        best_set = [1]
    
    return list(best_set)
