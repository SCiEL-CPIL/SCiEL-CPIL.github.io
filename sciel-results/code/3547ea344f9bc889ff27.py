"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from collections import defaultdict

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p::p] = False
    return np.where(sieve)[0].tolist()


def _is_admissible(mask, all_points, primes):
    """Check if the set of points in mask is admissible."""
    current_len = int(np.sum(mask))
    for p in primes:
        if p > current_len:
            break
        residues = all_points[mask] % p
        counts = np.bincount(residues, minlength=p)
        if np.all(counts > 0):
            return False
    return True


class _ResidueTracker:
    """Track residue class counts for each prime to enable fast admissibility checks."""
    def __init__(self, all_points, primes):
        self.all_points = all_points
        self.primes = primes
        self.n = len(all_points)
        # For each prime p, counts[r] = number of active points with x % p == r
        self.counts = {}
        for p in primes:
            self.counts[p] = np.zeros(p, dtype=np.int32)
        # active_mask: which points are currently included
        self.active = np.zeros(self.n, dtype=bool)
    
    def add(self, idx):
        """Add point at index idx to the active set."""
        if self.active[idx]:
            return
        self.active[idx] = True
        x = self.all_points[idx]
        for p in self.primes:
            self.counts[p][x % p] += 1
    
    def remove(self, idx):
        """Remove point at index idx from the active set."""
        if not self.active[idx]:
            return
        self.active[idx] = False
        x = self.all_points[idx]
        for p in self.primes:
            self.counts[p][x % p] -= 1
    
    def is_admissible(self):
        """Check if current active set is admissible."""
        current_len = int(np.sum(self.active))
        for p in self.primes:
            if p > current_len:
                break
            if np.all(self.counts[p] > 0):
                return False
        return True
    
    def find_full_primes(self):
        """Return list of primes p <= current_len where all residue classes are covered."""
        current_len = int(np.sum(self.active))
        full_primes = []
        for p in self.primes:
            if p > current_len:
                break
            if np.all(self.counts[p] > 0):
                full_primes.append(p)
        return full_primes
    
    def get_min_residue(self, p):
        """Get the residue class with minimum count for prime p."""
        return int(np.argmin(self.counts[p]))
    
    def get_residue_counts(self, p):
        """Get the count array for prime p."""
        return self.counts[p]


def _greedy_construct(all_points, primes, deadline):
    """Deterministic greedy: iteratively remove least-populated residue class for full-coverage primes."""
    tracker = _ResidueTracker(all_points, primes)
    
    # Start with all points active
    for i in range(len(all_points)):
        tracker.add(i)
    
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        full_primes = tracker.find_full_primes()
        if not full_primes:
            break
        # Pick the prime with the smallest total count (tightest constraint)
        # Actually, pick the one where removing the min residue causes the least damage
        # Strategy: for each full prime, find min residue; pick the prime+residue combo
        # that removes the fewest points (least damage)
        best_p = None
        best_res = None
        best_count = float('inf')
        for p in full_primes:
            min_res = tracker.get_min_residue(p)
            cnt = tracker.get_residue_counts(p)[min_res]
            if cnt < best_count:
                best_count = cnt
                best_p = p
                best_res = min_res
        
        if best_p is None:
            break
        
        # Remove all active points with residue best_res mod best_p
        x_mod = all_points % best_p
        to_remove = np.where(tracker.active & (x_mod == best_res))[0]
        for idx in to_remove:
            tracker.remove(idx)
        changed = True
    
    return tracker.active


def _conflict_greedy(all_points, primes, deadline, seed=42):
    """Conflict-aware greedy: prioritize removals that unblock the most other primes."""
    tracker = _ResidueTracker(all_points, primes)
    rng = np.random.default_rng(seed)
    
    # Start with all points active
    for i in range(len(all_points)):
        tracker.add(i)
    
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        full_primes = tracker.find_full_primes()
        if not full_primes:
            break
        
        # For each full prime, compute the "conflict score" for each residue class:
        # how many other full primes would be affected if we remove that residue class
        # Simplified: for each full prime p, for each residue r, count how many
        # points in residue r are also in "critical" residue classes of other full primes
        # 
        # Better heuristic: pick the (p, r) pair that minimizes the number of points removed
        # (least damage), with tie-breaking by randomization
        candidates = []
        for p in full_primes:
            min_res = tracker.get_min_residue(p)
            cnt = tracker.get_residue_counts(p)[min_res]
            # Also consider second-min as an alternative
            counts_p = tracker.get_residue_counts(p).copy()
            sorted_res = np.argsort(counts_p)
            for r in sorted_res[:3]:  # consider top 3 least-populated residues
                candidates.append((int(counts_p[r]), p, int(r), rng.random()))
        
        if not candidates:
            break
        
        # Sort by (count, random) and pick the best
        candidates.sort(key=lambda x: (x[0], x[3]))
        _, best_p, best_res, _ = candidates[0]
        
        # Remove all active points with residue best_res mod best_p
        x_mod = all_points % best_p
        to_remove = np.where(tracker.active & (x_mod == best_res))[0]
        for idx in to_remove:
            tracker.remove(idx)
        changed = True
    
    return tracker.active


def _polish(mask, all_points, primes, deadline, seed=42):
    """Try to add back removed points if admissibility is preserved."""
    tracker = _ResidueTracker(all_points, primes)
    for i in range(len(all_points)):
        if mask[i]:
            tracker.add(i)
    
    removed_indices = np.where(~mask)[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(removed_indices)
    
    for idx in removed_indices:
        if time.time() > deadline:
            break
        # Check if adding this point would violate admissibility
        x = all_points[idx]
        would_violate = False
        current_len = int(np.sum(tracker.active)) + 1
        for p in primes:
            if p > current_len:
                break
            r = x % p
            if tracker.get_residue_counts(p)[r] == 0:
                # Adding this point would make residue r non-zero
                # Check if all residues would then be covered
                test_counts = tracker.get_residue_counts(p).copy()
                test_counts[r] += 1
                if np.all(test_counts > 0):
                    would_violate = True
                    break
        if not would_violate:
            tracker.add(idx)
    
    return tracker.active


def _local_search(mask, all_points, primes, deadline, seed=42):
    """Local search: try swaps - remove one active point and add two removed points."""
    tracker = _ResidueTracker(all_points, primes)
    for i in range(len(all_points)):
        if mask[i]:
            tracker.add(i)
    
    rng = np.random.default_rng(seed)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        active_indices = np.where(tracker.active)[0]
        removed_indices = np.where(~tracker.active)[0]
        if len(active_indices) == 0 or len(removed_indices) == 0:
            break
        
        # Try random swaps: remove one active, add one or two removed
        for _ in range(50):
            if time.time() > deadline:
                break
            if len(active_indices) == 0 or len(removed_indices) == 0:
                break
            
            remove_idx = active_indices[rng.integers(len(active_indices))]
            add_idx = removed_indices[rng.integers(len(removed_indices))]
            
            # Check if swap improves size (net +1 or 0 but different structure)
            # For simplicity, try single swaps first
            x_remove = all_points[remove_idx]
            x_add = all_points[add_idx]
            
            # Temporarily remove and add, check admissibility
            tracker.remove(remove_idx)
            tracker.add(add_idx)
            
            if tracker.is_admissible():
                # Accept the swap (size stays same, but might enable future improvements)
                # Actually, for pure size maximization, we need net gain
                # Let's try: remove one, add two
                pass
            else:
                # Revert
                tracker.remove(add_idx)
                tracker.add(remove_idx)
    
    return tracker.active


def _multi_start_greedy(all_points, primes, deadline):
    """Run multiple greedy starts with different seeds and return the best result."""
    best_mask = None
    best_size = 0
    n_starts = 5
    
    for seed in range(n_starts):
        start_deadline = deadline - 0.1 * (n_starts - 1 - seed)
        if time.time() > start_deadline:
            break
        
        mask = _conflict_greedy(all_points, primes, start_deadline, seed=seed)
        # Polish
        mask = _polish(mask, all_points, primes, start_deadline, seed=seed)
        
        size = int(np.sum(mask))
        if size > best_size:
            best_size = size
            best_mask = mask.copy()
    
    if best_mask is None:
        # Fallback to deterministic greedy
        best_mask = _greedy_construct(all_points, primes, deadline)
    
    return best_mask


def construct():
    start_time = time.time()
    deadline = 285.0  # safety margin before 300s kill
    
    # Precompute primes up to N+1
    primes = _primes(N + 1)
    
    # Start with all integers in [0, N]
    all_points = np.arange(N + 1)
    
    # Multi-start greedy with polish
    mask = _multi_start_greedy(all_points, primes, deadline)
    
    return all_points[mask].tolist()
# EVOLVE-BLOCK-END
