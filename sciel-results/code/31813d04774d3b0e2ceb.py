"""Cardinality-separated subset sums: fast incremental DP + local search.

Key improvements:
- Maintain sums_by_k incrementally to avoid O(n^2) recomputation per candidate.
- Fast incremental validity check: only new sums need checking against existing.
- Aggressive local search with swap-based improvements.
- Descending-order bias for gap creation, combined with structured construction.
"""
import time
import random

DEADLINE_SECONDS = 85.0  # safety margin before 90s kill


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


def _build_sum_to_ks(sums_by_k):
    """Build mapping from sum -> set of cardinalities that achieve it."""
    sum_to_ks = {}
    for k, sums in enumerate(sums_by_k):
        for s in sums:
            if s in sum_to_ks:
                sum_to_ks[s].add(k)
            else:
                sum_to_ks[s] = {k}
    return sum_to_ks


def _is_valid(S):
    """Check no two subsets of different cardinalities share a sum."""
    sums_by_k = _subset_sums_by_cardinality(S)
    sum_to_ks = _build_sum_to_ks(sums_by_k)
    for s, ks in sum_to_ks.items():
        if len(ks) > 1:
            return False
    return True


class _IncrementalState:
    """Maintains sums_by_k and sum_to_ks incrementally for fast add/remove."""
    
    def __init__(self):
        self.current = []
        self.sums_by_k = [set()]  # sums_by_k[0] = {0}
        self.sum_to_ks = {0: {0}}
    
    def add(self, x):
        """Add element x. Returns True if valid, False otherwise (and reverts)."""
        if x in self.current:
            return False
        if not (1 <= x <= 256):
            return False
        
        n = len(self.current)
        # Compute new sums: for each k, new_sums[k] = old_sums[k] | {s+x for s in old_sums[k-1]}
        # We need to check that no new sum conflicts with existing sums at different cardinalities
        
        # First, compute what new sums would be added at each cardinality
        new_sums_at_k = [set() for _ in range(n + 1)]  # new_sums_at_k[k] = sums at cardinality k that involve x
        for k in range(1, n + 1):
            for s in self.sums_by_k[k - 1]:
                new_sums_at_k[k].add(s + x)
        
        # Check validity: for each new sum at cardinality k, it must not appear in sum_to_ks at any other cardinality
        for k in range(1, n + 1):
            for s in new_sums_at_k[k]:
                if s in self.sum_to_ks:
                    existing_ks = self.sum_to_ks[s]
                    if existing_ks != {k}:
                        # Conflict: s is achievable at some cardinality != k
                        return False
                # Also check: s might already be in sums_by_k[k] (that's fine, same cardinality)
        
        # Valid - commit the addition
        self.current.append(x)
        # Update sums_by_k: extend by one level
        self.sums_by_k.append(set())
        for k in range(n, 0, -1):
            for s in self.sums_by_k[k - 1]:
                self.sums_by_k[k].add(s + x)
        # Update sum_to_ks
        for k in range(1, n + 1):
            for s in new_sums_at_k[k]:
                if s in self.sum_to_ks:
                    self.sum_to_ks[s].add(k)
                else:
                    self.sum_to_ks[s] = {k}
        
        return True
    
    def remove(self, x):
        """Remove element x from the set."""
        if x not in self.current:
            return
        self.current.remove(x)
        # Rebuild from scratch (removal is complex to do incrementally)
        self.sums_by_k = _subset_sums_by_cardinality(self.current)
        self.sum_to_ks = _build_sum_to_ks(self.sums_by_k)
    
    def size(self):
        return len(self.current)
    
    def get_set(self):
        return self.current[:]


def _try_add_fast(state, candidate):
    """Try adding candidate using incremental state. Returns True if valid."""
    return state.add(candidate)


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS
    
    def time_left():
        return deadline - time.time()
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Greedy with multiple orderings
    for attempt in range(200):
        if time_left() < 1.0:
            break
        
        if attempt == 0:
            candidates = list(range(256, 0, -1))  # Descending - proven best
        elif attempt == 1:
            candidates = list(range(1, 257))  # Ascending
        elif attempt == 2:
            # Start with large, then fill in
            candidates = list(range(200, 257)) + list(range(1, 200))
        elif attempt == 3:
            # Powers of 2 style mixed
            base = [2**i for i in range(8) if 2**i <= 256]
            rest = [x for x in range(1, 257) if x not in base]
            random.shuffle(rest)
            candidates = base + rest
        elif attempt == 4:
            # Large gaps: 256, 128, 64, ... then fill
            powers = [2**i for i in range(8, -1, -1) if 2**i <= 256]
            rest = [x for x in range(1, 257) if x not in powers]
            random.shuffle(rest)
            candidates = powers + rest
        else:
            candidates = list(range(1, 257))
            random.shuffle(candidates)
        
        state = _IncrementalState()
        for x in candidates:
            if time_left() < 0.5:
                break
            if state.size() >= 60:
                break
            if _try_add_fast(state, x):
                pass  # already added
        
        if state.size() > best_size:
            best_size = state.size()
            best_set = state.get_set()
        
        # Early stop if we've done well
        if best_size >= 12 and time_left() < 20:
            break
    
    # Strategy 2: Local optimization on the best set found
    if time_left() > 5 and best_size > 0:
        state = _IncrementalState()
        # Rebuild state from best_set
        state.current = best_set[:]
        state.sums_by_k = _subset_sums_by_cardinality(best_set)
        state.sum_to_ks = _build_sum_to_ks(state.sums_by_k)
        
        # Try adding new elements
        improved = True
        while improved and time_left() > 2:
            improved = False
            for x in range(1, 257):
                if time_left() < 0.5:
                    break
                if x in state.current:
                    continue
                if _try_add_fast(state, x):
                    improved = True
                    if state.size() > best_size:
                        best_size = state.size()
                        best_set = state.get_set()
        
        # Try swapping: remove one element, try to add multiple others
        if time_left() > 10:
            for _swap_round in range(3):
                if time_left() < 5:
                    break
                for i in range(len(state.current)):
                    if time_left() < 2.0:
                        break
                    saved = state.current[i]
                    state.remove(saved)
                    
                    size_before_remove = state.size()
                    # Try to add multiple elements to compensate
                    added_count = 0
                    for x in range(256, 0, -1):  # Descending for gap creation
                        if time_left() < 1.0:
                            break
                        if x in state.current or x == saved:
                            continue
                        if _try_add_fast(state, x):
                            added_count += 1
                    
                    if state.size() > best_size:
                        best_size = state.size()
                        best_set = state.get_set()
                    else:
                        # Revert: rebuild from best_set
                        state = _IncrementalState()
                        state.current = best_set[:]
                        state.sums_by_k = _subset_sums_by_cardinality(best_set)
                        state.sum_to_ks = _build_sum_to_ks(state.sums_by_k)
    
    # Strategy 3: Fresh structured construction if we have time
    if time_left() > 15 and best_size < 15:
        # Build with descending order, focusing on creating wide gaps
        state = _IncrementalState()
        for x in range(256, 0, -1):
            if time_left() < 2.0:
                break
            if state.size() >= 60:
                break
            if _try_add_fast(state, x):
                pass
        
        if state.size() > best_size:
            best_size = state.size()
            best_set = state.get_set()
        
        # Then try to add more from ascending
        for x in range(1, 257):
            if time_left() < 1.0:
                break
            if x in state.current:
                continue
            if _try_add_fast(state, x):
                pass
        
        if state.size() > best_size:
            best_size = state.size()
            best_set = state.get_set()
    
    # Strategy 4: Multiple restarts with random perturbations if time allows
    if time_left() > 10 and best_size < 20:
        for restart in range(50):
            if time_left() < 2.0:
                break
            
            # Start from a random subset of best_set or from scratch
            if best_size > 5 and restart % 2 == 0:
                # Perturb: remove 1-2 random elements from best_set
                start_set = best_set[:]
                for _ in range(random.randint(1, 2)):
                    if start_set:
                        idx = random.randint(0, len(start_set) - 1)
                        start_set.pop(idx)
            else:
                start_set = []
            
            state = _IncrementalState()
            state.current = start_set[:]
            if start_set:
                state.sums_by_k = _subset_sums_by_cardinality(start_set)
                state.sum_to_ks = _build_sum_to_ks(state.sums_by_k)
            
            # Continue building with random order
            candidates = list(range(1, 257))
            random.shuffle(candidates)
            
            for x in candidates:
                if time_left() < 1.0:
                    break
                if state.size() >= 60:
                    break
                if x in state.current:
                    continue
                if _try_add_fast(state, x):
                    pass
            
            if state.size() > best_size:
                best_size = state.size()
                best_set = state.get_set()
    
    # Final validation
    if best_set and not _is_valid(best_set):
        best_set = [1]
    
    # Ensure all elements are Python ints
    result = []
    for x in best_set:
        if isinstance(x, bool):
            continue
        if not isinstance(x, int):
            continue
        result.append(x)
    
    return result
