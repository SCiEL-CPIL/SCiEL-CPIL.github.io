"""
Narrow admissible tuple: maximize tuple size in [0, 5000] such that for every
prime p <= len(tuple), some residue class mod p is missed.

Strategy:
1. Greedy construction with a priority queue of "conflict pressure".
   We maintain a set S. For each prime p <= |S|, we track which residue classes
   are "full" (i.e., all p residues present). A prime p is violated if all p
   residues are present. We want to avoid that.
   
   Actually, the condition is: for every prime p <= |S|, NOT all residues mod p
   are present. So we must ensure that for each such p, at least one residue class
   is missing.

   Greedy approach: start with an empty set or a small seed, and add elements one
   by one. When adding an element x, check if it causes any prime p <= |S|+1 to
   become "full". If it does, we cannot add x (or we must remove some other element).

   A better approach: 
   - Start with a random set of size ~N/2.
   - Repair: while there is a violated prime p, remove an element from the smallest
     residue class mod p (to break the full condition).
   - Expand: try to add back elements that were removed, or new elements, if they
     don't cause violations.

   However, the repair step is tricky because removing an element to fix one prime
   might fix multiple primes, but the choice matters.

   Insight from inspirations: vectorization with NumPy allows deeper branching.
   But let's try a smarter greedy:

   Alternative: 
   - Precompute for each candidate x in [0, N] its residues for all primes p <= N+1.
   - Use a priority queue of "potential additions" scored by how few new conflicts
     they introduce.

   But checking conflicts is expensive.

   Let's use the following algorithm:
   1. Initialize S with a random subset of [0, N] of size ~N/3.
   2. Repair phase: 
      - For each prime p <= |S|, check if all residues are present.
      - If yes, remove a random element from the smallest residue class mod p.
      - Repeat until admissible.
   3. Expansion phase:
      - Maintain a list of candidate elements (those not in S).
      - For each candidate x, compute which primes p <= |S|+1 would be violated
        if x were added. Specifically, for each such p, if x's residue mod p is the
        only missing residue, then adding x would violate p.
      - If no violations, add x.
      - Use a priority queue to try candidates in an order that maximizes the chance
        of success (e.g., candidates that fill "rare" residues first? Or random?).

   To speed up, we can precompute for each prime p, the count of each residue in S.
   When adding x, for each p <= |S|+1, we check if count[p][x%p] == 0. If so, and
   if this is the only zero-count residue for p, then adding x would make p full.

   But we need to know if it's the "only" missing residue. So for each p, we track
   the number of missing residues. If missing_count[p] == 1 and x%p is the missing
   one, then adding x violates p.

   This is efficient if we maintain:
   - For each prime p <= N+1: 
        residue_count[p][r] for r in 0..p-1
        missing_count[p] = number of r with residue_count[p][r] == 0

   When adding x:
        For each p <= |S|+1:
            r = x % p
            if residue_count[p][r] == 0:
                # This would reduce missing_count[p] by 1
                # If missing_count[p] == 1, then after adding, missing_count[p] becomes 0 -> violation
                if missing_count[p] == 1:
                    # Cannot add x
                    break
        If no violation, add x and update counts.

   When removing x:
        For each p <= |S|:
            r = x % p
            residue_count[p][r] -= 1
            if residue_count[p][r] == 0:
                missing_count[p] += 1
            # Note: missing_count[p] can increase, which is good (more admissible)

   This is O(pi(N)) per add/remove, which is ~N/log(N) ~ 5000/8 ~ 600 operations.
   With N=5000, and potentially many add/remove operations, this should be fast.

   Let's implement this.

   Steps:
   1. Precompute primes up to N+1.
   2. Initialize residue_count and missing_count for all primes.
   3. Start with an empty set S.
   4. Randomly shuffle [0, N].
   5. For each x in the shuffled list:
        - Check if adding x is safe (no prime p <= |S|+1 would become full).
        - If safe, add x.
   6. This gives an admissible set. But is it maximal? Probably not, because the
      order matters. We can do multiple random restarts.

   However, the greedy "add if safe" might get stuck. We can also do:
   - After the initial pass, try to add back elements that were skipped, in a different order.
   - Or use local search: swap operations.

   Given the time limit, let's do:
   - Multiple random restarts of the greedy add-if-safe.
   - For each restart, shuffle [0, N] and do the greedy pass.
   - Keep the best result.

   To improve, we can also do a "repair" phase after the greedy pass:
   - Actually, the greedy pass ensures admissibility by construction (we never add
     an element that would violate admissibility). So the result is always admissible.

   But the size might be suboptimal. To get larger sets, we might need to be smarter
   about the order. For example, adding elements that are "compatible" with many primes
   first.

   Alternative: use a priority queue where the priority of x is the number of primes
   p <= |S|+1 for which x%p is a "rare" residue (i.e., residue_count[p][x%p] is small).
   This way, we fill rare residues first, which might allow more elements later.

   Let's try that.

   Implementation details:
   - We'll maintain a max-heap (using negative priority) of candidates.
   - Initially, all x in [0, N] are candidates.
   - At each step, pop the candidate with the highest priority, try to add it.
   - If it can be added, add it and update the priority of all other candidates?
     That's expensive. Instead, we can recompute priorities lazily or just use a
     random order with some heuristic.

   Given the complexity, let's start with multiple random restarts of the simple
   greedy add-if-safe, and see if that's enough. We can also try a few different
   initial seeds (e.g., start with a small random set and then expand).

   Let's code the core logic first.

   Note: We must be careful with the primes. The condition is for every prime p <= |S|.
   So when |S| = k, we check primes p <= k.
   When we add an element, |S| becomes k+1, so we must also check the prime p = k+1
   if k+1 is prime. But our precomputed primes list includes all primes up to N+1,
   so we can just iterate over primes p <= |S|+1 (since after adding, |S| is |S|+1,
   and we need p <= |S|+1 to be not full).

   Actually, the condition after adding is: for every prime p <= |S_new|, some residue
   is missed. So when checking if we can add x to current S (size k), we need to ensure
   that for every prime p <= k+1, after adding x, not all residues are present.
   This is equivalent to: for every prime p <= k+1, if x%p is the only missing residue
   for p in the current S, then we cannot add x.

   So in the check, we iterate over primes p <= k+1.

   Let's implement.
"""

import time
import random
from math import isqrt
import heapq

N = 5000
DEADLINE = 300.0  # seconds, safety margin from 330s kill


def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, isqrt(limit) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


# Precompute primes up to N+1 (max possible tuple size is N+1=5001)
_PRIMES = _primes(N + 1)
# Map prime to index for fast access
_PRIME_INDEX = {p: i for i, p in enumerate(_PRIMES)}


class AdmissibleSet:
    def __init__(self):
        self.elements = set()
        self.size = 0
        # residue_count[p_idx][r] for prime p_idx, residue r
        self.residue_count = []
        self.missing_count = []
        for p in _PRIMES:
            self.residue_count.append([0] * p)
            self.missing_count.append(p)  # initially all residues missing

    def _update_add(self, x):
        """Update internal state after adding x. Assumes x is not already in set."""
        self.elements.add(x)
        self.size += 1
        for i, p in enumerate(_PRIMES):
            if p > self.size:
                break
            r = x % p
            if self.residue_count[i][r] == 0:
                self.missing_count[i] -= 1
            self.residue_count[i][r] += 1

    def _update_remove(self, x):
        """Update internal state after removing x. Assumes x is in set."""
        self.elements.discard(x)
        self.size -= 1
        for i, p in enumerate(_PRIMES):
            if p > self.size + 1:  # when removing, size decreases, so p <= old_size
                break
            r = x % p
            self.residue_count[i][r] -= 1
            if self.residue_count[i][r] == 0:
                self.missing_count[i] += 1

    def can_add(self, x):
        """Check if x can be added without violating admissibility."""
        if x in self.elements:
            return False
        new_size = self.size + 1
        for i, p in enumerate(_PRIMES):
            if p > new_size:
                break
            r = x % p
            if self.residue_count[i][r] == 0:
                # This residue is currently missing. If it's the only missing one,
                # adding x would make all residues present -> violation.
                if self.missing_count[i] == 1:
                    return False
        return True

    def add(self, x):
        if not self.can_add(x):
            return False
        self._update_add(x)
        return True

    def remove(self, x):
        if x not in self.elements:
            return False
        self._update_remove(x)
        return True

    def is_admissible(self):
        """Verify admissibility (for debugging)."""
        for i, p in enumerate(_PRIMES):
            if p > self.size:
                break
            if self.missing_count[i] == 0:
                return False
        return True


def _greedy_construct(deadline):
    """
    Greedy construction: shuffle [0, N], add each element if safe.
    Returns the resulting set.
    """
    aset = AdmissibleSet()
    candidates = list(range(N + 1))
    random.shuffle(candidates)
    
    for x in candidates:
        if time.time() > deadline:
            break
        if aset.add(x):
            pass  # added
    
    return aset


def _local_search(aset, deadline):
    """
    Try to improve the set by swapping or adding elements.
    This is a placeholder for now; we'll rely on multiple random restarts.
    """
    # For now, just return the same set.
    # We could try to add elements that were skipped, but the greedy pass already
    # considered all elements in random order.
    return aset


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 5.0  # 5s safety margin
    
    best_size = 0
    best_set = set()
    
    # Multiple random restarts
    num_restarts = 0
    while time.time() < deadline - 2.0:
        aset = _greedy_construct(deadline)
        if aset.size > best_size:
            best_size = aset.size
            best_set = aset.elements.copy()
        num_restarts += 1
        # If we're getting close to deadline, break
        if time.time() > deadline - 10.0 and num_restarts > 5:
            break
    
    # If we have time, try a few more with different strategies
    # e.g., start with a small seed and expand
    if time.time() < deadline - 10.0:
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            # Start with a small random seed
            aset = AdmissibleSet()
            seed_size = random.randint(10, 50)
            seed = random.sample(range(N + 1), seed_size)
            for x in seed:
                aset.add(x)  # may fail, that's ok
            # Then expand
            candidates = [x for x in range(N + 1) if x not in aset.elements]
            random.shuffle(candidates)
            for x in candidates:
                if time.time() > deadline:
                    break
                aset.add(x)
            
            if aset.size > best_size:
                best_size = aset.size
                best_set = aset.elements.copy()
    
    return list(best_set)


if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
