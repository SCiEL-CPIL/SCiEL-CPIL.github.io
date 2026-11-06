<<<<<<< SEARCH
"""Two Sidon sets with disjoint difference sets.
Model-and-optimize approach: use a structured parameterization (arithmetic-like
progressions with controlled differences) plus greedy local improvement to maximize
combined size under the disjoint-difference constraint."""

import time
import random

def _check_sidon(els):
    """Check all pair sums a+b (a<=b) are unique."""
    n = len(els)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = els[i] + els[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _nonzero_diffs(els):
    """Set of all nonzero differences (positive and negative)."""
    d = set()
    n = len(els)
    for i in range(n):
        for j in range(n):
            if i != j:
                diff = els[j] - els[i]
                d.add(diff)
    return d

def _valid(A, B):
    """Check A,B are Sidon and their nonzero difference sets are disjoint."""
    if not _check_sidon(A):
        return False
    if not _check_sidon(B):
        return False
    if _nonzero_diffs(A) & _nonzero_diffs(B):
        return False
    return True

def _try_add_to_set(S, x, used_sums, other_diffs):
    """Try to add x to set S. Returns True if valid."""
    # Check Sidon: x+x and x+s for all s in S must not be in used_sums
    s2 = 2 * x
    if s2 in used_sums:
        return False
    for a in S:
        if (a + x) in used_sums:
            return False
    # Check difference disjointness with other set
    for a in S:
        d = x - a
        if d != 0 and d in other_diffs:
            return False
        d = a - x
        if d != 0 and d in other_diffs:
            return False
    return True

def _commit_add(S, x, used_sums, diffs_self, diffs_other):
    """Actually add x to S and update used_sums and diffs."""
    S.append(x)
    used_sums.add(2 * x)
    for a in S:
        if a != x:
            used_sums.add(a + x)
            diffs_self.add(x - a)
            diffs_self.add(a - x)

def _rollback_add(S, x, used_sums, diffs_self, diffs_other):
    """Remove x from S and update used_sums and diffs."""
    S.remove(x)
    used_sums.discard(2 * x)
    for a in S:
        used_sums.discard(a + x)
        diffs_self.discard(x - a)
        diffs_self.discard(a - x)

def construct():
    """Construct two Sidon sets with disjoint difference sets, maximizing combined size."""
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    n = 256
    max_size = 64

    best_A = [1]
    best_B = [2]
    best_score = 2

    rng = random.Random(42)

    def _attempt_build(seed_A=None, seed_B=None):
        """Build A and B from optional seeds using greedy strategy."""
        A = list(seed_A) if seed_A else []
        B = list(seed_B) if seed_B else []

        if not A:
            A.append(rng.randint(1, n))
        if not B:
            B.append(rng.randint(1, n))

        A = list(set(A))
        B = list(set(B))

        if not _valid(A, B):
            return None, None, 0

        used_sums_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                used_sums_A.add(A[i] + A[j])

        used_sums_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                used_sums_B.add(B[i] + B[j])

        diffs_A = _nonzero_diffs(A)
        diffs_B = _nonzero_diffs(B)

        # Pre-compute candidates
        candidates = list(range(1, n + 1))
        rng.shuffle(candidates)

        # Greedy phase
        progress = True
        while progress and time.time() < deadline - 5:
            progress = False
            
            # Try to add to A
            if len(A) < max_size:
                for x in candidates:
                    if x in A:
                        continue
                    if _try_add_to_set(A, x, used_sums_A, diffs_B):
                        _commit_add(A, x, used_sums_A, diffs_A, diffs_B)
                        progress = True
                        break

            # Try to add to B
            if len(B) < max_size:
                for x in candidates:
                    if x in B:
                        continue
                    if _try_add_to_set(B, x, used_sums_B, diffs_A):
                        _commit_add(B, x, used_sums_B, diffs_B, diffs_A)
                        progress = True
                        break

        if _valid(A, B):
            return A, B, len(A) + len(B)
        return None, None, 0

    def _local_search(A, B, time_limit):
        """Perform local search with swaps to improve the solution."""
        A = list(A)
        B = list(B)
        
        # Reconstruct state
        used_sums_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                used_sums_A.add(A[i] + A[j])
        used_sums_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                used_sums_B.add(B[i] + B[j])
        diffs_A = _nonzero_diffs(A)
        diffs_B = _nonzero_diffs(B)
        
        current_score = len(A) + len(B)
        improved = True
        t_end = time.time() + time_limit
        
        while improved and time.time() < t_end:
            improved = False
            
            # Try swaps in A: remove x, add y
            if len(A) > 1:
                for i, x in enumerate(A):
                    if time.time() > t_end:
                        break
                    # Remove x
                    _rollback_add(A, x, used_sums_A, diffs_A, diffs_B)
                    
                    # Try to add any y
                    for y in range(1, n + 1):
                        if y in A:
                            continue
                        if _try_add_to_set(A, y, used_sums_A, diffs_B):
                            _commit_add(A, y, used_sums_A, diffs_A, diffs_B)
                            # Check if score increased (it shouldn't for 1-for-1 swap unless we allow growth)
                            # Actually, 1-for-1 swap keeps score same. 
                            # We want to try to add MORE. 
                            # So: Remove x, try to add y AND z?
                            # That's expensive. 
                            # Let's try: Remove x, add y. If valid, keep it if it allows future growth?
                            # No, local search should only accept if score increases or same but "better" (hard to define).
                            # Let's stick to: Remove x, try to add y. If valid, accept swap (score same).
                            # Then try to add z to A or B.
                            # This is getting complex. 
                            
                            # Simpler: Try to add 2 elements after removing 1.
                            # Remove x.
                            # Try to add y.
                            #   If success, try to add z.
                            #     If success, score +1. Accept.
                            #   If success, but z fails, we have A' with same size. 
                            #     We can accept this swap if it opens up possibilities?
                            #     Let's just accept any valid swap (score same) to move in search space.
                            #     But we need to ensure we don't cycle.
                            
                            # Let's just do: 
                            # 1. Try to add to A (if space)
                            # 2. Try to add to B (if space)
                            # 3. Try Swap: Remove x from A, Add y to A (if valid). 
                            #    Then try to Add z to A (if space) or B.
                            #    If score increases, accept.
                            
                            pass # Placeholder for logic below
                            
                    # If we didn't add anything, put x back
                    if x not in A:
                        _commit_add(A, x, used_sums_A, diffs_A, diffs_B)
                        
            # This logic is getting messy inside the loop. Let's simplify.
            # We will implement a specific "Grow" move:
            # 1. If len(A) < 64, try to add. (Done in greedy)
            # 2. If stuck, try to remove one element from A, then try to add two.
            
            # Let's implement a simpler local search:
            # Try all possible single additions (already done in greedy, but maybe we missed some due to order?)
            # Greedy is order dependent. 
            # Let's try "Random Restart" with better initialization.
            
            # Actually, the current greedy is quite good. The main issue is getting stuck.
            # Let's try to improve by trying to add elements in a different order.
            # The current loop breaks on the first valid candidate.
            # We can try all candidates and pick the one that minimizes "conflicts" for the future?
            # Or just try all and pick the first one that allows the most further additions?
            
            # Let's try a "Lookahead" greedy:
            # For each candidate x in A:
            #   Add x tentatively.
            #   Count how many more can be added to A (greedy).
            #   Remove x.
            #   Pick x with max count.
            # This is O(N * GreedyCost) which is expensive.
            
            # Given time constraints, let's just do more random restarts with different seeds.
            # And a simple hill climbing: 
            # Try to swap x in A with y not in A.
            # If valid, accept.
            # Then try to add z to A or B.
            
            # Let's just return the current best from this local search attempt.
            # The main improvement will be in the number of restarts and better seed generation.
            
            break # Disable complex local search for now to avoid bugs, rely on more restarts.
            
        return A, B, len(A) + len(B)

    # Main search loop
    for it in range(50000):
        if time.time() > deadline - 3:
            break

        # Different strategies
        strat = it % 5

        if strat == 0:
            # Random single seeds
            A0 = [rng.randint(1, n)]
            B0 = [rng.randint(1, n)]
            A, B, score = _attempt_build(A0, B0)
        elif strat == 1:
            # A first
            A0 = [rng.randint(1, n)]
            A, B, score = _attempt_build(A0, None)
        elif strat == 2:
            # B first
            B0 = [rng.randint(1, n)]
            A, B, score = _attempt_build(None, B0)
        elif strat == 3:
            # Small random starts
            sa = rng.randint(1, 3)
            sb = rng.randint(1, 3)
            A0 = sorted(rng.sample(range(1, n + 1), sa))
            B0 = sorted(rng.sample(range(1, n + 1), sb))
            A, B, score = _attempt_build(A0, B0)
        else:
            # Wider spacing seeds
            A0 = [rng.randint(1, 128)]
            B0 = [rng.randint(129, n)]
            A, B, score = _attempt_build(A0, B0)

        if A is not None and score > best_score:
            best_score = score
            best_A = list(A)
            best_B = list(B)

            # Try local improvement: simple random additions
            if time.time() < deadline - 8:
                for _ls in range(50):
                    if time.time() > deadline - 5:
                        break
                    if len(best_A) < max_size:
                        x = rng.randint(1, n)
                        if x not in best_A:
                            test_A = best_A + [x]
                            if _valid(test_A, best_B):
                                best_A = test_A
                                best_score += 1
                                continue
                    if len(best_B) < max_size:
                        x = rng.randint(1, n)
                        if x not in best_B:
                            test_B = best_B + [x]
                            if _valid(best_A, test_B):
                                best_B = test_B
                                best_score += 1

    # Final validation
    if not _valid(best_A, best_B):
        best_A = [1]
        best_B = [2]
        best_score = 2

    return [best_A, best_B]
=======
"""Two Sidon sets with disjoint difference sets.
Bitmask-based fast collision checks enable thousands of randomized greedy
restarts within the time limit, escaping local minima of simple greedy."""

import time
import random
import sys

# n=256, so values in [1,256]
# Pair sums: a+b where a,b in [1,256], so sums in [2,512]
# Differences: a-b where a,b in [1,256], so diffs in [-255,255]
# We use two bitmasks:
#   sums_mask: bit s set if sum s is used. Index by s (2..512)
#   diffs_mask: bit d set if diff d is used. Index by d+255 (0..510)

N = 256
MAX_SIZE = 64
SUM_MIN = 2
SUM_MAX = 512
DIFF_MIN = -255
DIFF_MAX = 255
DIFF_OFFSET = 255  # diff + DIFF_OFFSET gives index 0..510


def _check_sidon(els):
    """Check all pair sums a+b (a<=b) are unique."""
    n = len(els)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = els[i] + els[j]
            if s in seen:
                return False
            seen.add(s)
    return True


def _nonzero_diffs(els):
    """Set of all nonzero differences (positive and negative)."""
    d = set()
    n = len(els)
    for i in range(n):
        for j in range(n):
            if i != j:
                diff = els[j] - els[i]
                d.add(diff)
    return d


def _valid(A, B):
    """Check A,B are Sidon and their nonzero difference sets are disjoint."""
    if not _check_sidon(A):
        return False
    if not _check_sidon(B):
        return False
    if _nonzero_diffs(A) & _nonzero_diffs(B):
        return False
    return True


class FastSidonState:
    """Fast state tracking using bitmasks for O(1) collision checks."""

    __slots__ = ('A', 'B', 'sums_A', 'sums_B', 'diffs_A', 'diffs_B',
                 'in_A', 'in_B')

    def __init__(self, A_init=None, B_init=None):
        self.A = list(A_init) if A_init else []
        self.B = list(B_init) if B_init else []
        self.sums_A = 0  # bitmask for pair sums in A
        self.sums_B = 0  # bitmask for pair sums in B
        self.diffs_A = 0  # bitmask for nonzero diffs in A
        self.diffs_B = 0  # bitmask for nonzero diffs in B
        self.in_A = [False] * (N + 1)
        self.in_B = [False] * (N + 1)

        # Initialize from given sets
        for a in self.A:
            self.in_A[a] = True
        for b in self.B:
            self.in_B[b] = True

        self._rebuild_sums()
        self._rebuild_diffs()

    def _rebuild_sums(self):
        self.sums_A = 0
        A = self.A
        for i in range(len(A)):
            ai = A[i]
            # a+a
            self.sums_A |= (1 << ai)  # sum = 2*ai, but we index by sum value
            # Wait, we need to index by sum value, not element.
            # Let me redo: sum s = a_i + a_j. We want bit s set.
            # But s can be up to 512, so we need a big integer.
            pass

        # Redo properly
        self.sums_A = 0
        self.sums_B = 0
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                self.sums_A |= (1 << s)

        B = self.B
        for i in range(len(B)):
            for j in range(i, len(B)):
                s = B[i] + B[j]
                self.sums_B |= (1 << s)

    def _rebuild_diffs(self):
        self.diffs_A = 0
        self.diffs_B = 0
        A = self.A
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    d = A[j] - A[i]
                    if d != 0:
                        self.diffs_A |= (1 << (d + DIFF_OFFSET))

        B = self.B
        for i in range(len(B)):
            for j in range(len(B)):
                if i != j:
                    d = B[j] - B[i]
                    if d != 0:
                        self.diffs_B |= (1 << (d + DIFF_OFFSET))

    def can_add_to_A(self, x):
        """Check if x can be added to A (Sidon + diff disjoint with B)."""
        if self.in_A[x]:
            return False
        # Check Sidon: 2*x and x+a for all a in A must not conflict
        if (self.sums_A >> (2 * x)) & 1:
            return False
        for a in self.A:
            if (self.sums_A >> (x + a)) & 1:
                return False
        # Check diff disjointness with B
        for b in self.B:
            d1 = x - b
            if d1 != 0 and (self.diffs_B >> (d1 + DIFF_OFFSET)) & 1:
                return False
            d2 = b - x
            if d2 != 0 and (self.diffs_B >> (d2 + DIFF_OFFSET)) & 1:
                return False
        return True

    def can_add_to_B(self, x):
        """Check if x can be added to B (Sidon + diff disjoint with A)."""
        if self.in_B[x]:
            return False
        if (self.sums_B >> (2 * x)) & 1:
            return False
        for b in self.B:
            if (self.sums_B >> (x + b)) & 1:
                return False
        for a in self.A:
            d1 = x - a
            if d1 != 0 and (self.diffs_A >> (d1 + DIFF_OFFSET)) & 1:
                return False
            d2 = a - x
            if d2 != 0 and (self.diffs_A >> (d2 + DIFF_OFFSET)) & 1:
                return False
        return True

    def add_to_A(self, x):
        """Add x to A, updating all state."""
        self.A.append(x)
        self.in_A[x] = True
        self.sums_A |= (1 << (2 * x))
        for a in self.A:
            if a != x:
                self.sums_A |= (1 << (x + a))
                d1 = x - a
                d2 = a - x
                self.diffs_A |= (1 << (d1 + DIFF_OFFSET))
                self.diffs_A |= (1 << (d2 + DIFF_OFFSET))

    def add_to_B(self, x):
        """Add x to B, updating all state."""
        self.B.append(x)
        self.in_B[x] = True
        self.sums_B |= (1 << (2 * x))
        for b in self.B:
            if b != x:
                self.sums_B |= (1 << (x + b))
                d1 = x - b
                d2 = b - x
                self.diffs_B |= (1 << (d1 + DIFF_OFFSET))
                self.diffs_B |= (1 << (d2 + DIFF_OFFSET))

    def remove_from_A(self, x):
        """Remove x from A, updating all state."""
        self.A.remove(x)
        self.in_A[x] = False
        # Rebuild sums_A and diffs_A from scratch (expensive but safe)
        self._rebuild_sums()
        self._rebuild_diffs()

    def remove_from_B(self, x):
        """Remove x from B, updating all state."""
        self.B.remove(x)
        self.in_B[x] = False
        self._rebuild_sums()
        self._rebuild_diffs()


def _greedy_build(state, rng, deadline, max_steps=200):
    """Greedy build: repeatedly try to add to A or B in random order."""
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)

    steps = 0
    while steps < max_steps and time.time() < deadline:
        steps += 1
        added = False

        # Try to add to A
        if len(state.A) < MAX_SIZE:
            for x in candidates:
                if state.in_A[x]:
                    continue
                if state.can_add_to_A(x):
                    state.add_to_A(x)
                    added = True
                    break

        # Try to add to B
        if len(state.B) < MAX_SIZE:
            for x in candidates:
                if state.in_B[x]:
                    continue
                if state.can_add_to_B(x):
                    state.add_to_B(x)
                    added = True
                    break

        if not added:
            break

    return state


def _local_search_improve(state, rng, deadline):
    """Local search: try random swaps and additions to improve."""
    best_score = len(state.A) + len(state.B)
    best_A = list(state.A)
    best_B = list(state.B)

    while time.time() < deadline:
        # Try random addition
        if len(state.A) < MAX_SIZE:
            x = rng.randint(1, N)
            if not state.in_A[x] and state.can_add_to_A(x):
                state.add_to_A(x)
                best_score = len(state.A) + len(state.B)
                best_A = list(state.A)
                best_B = list(state.B)
                continue

        if len(state.B) < MAX_SIZE:
            x = rng.randint(1, N)
            if not state.in_B[x] and state.can_add_to_B(x):
                state.add_to_B(x)
                best_score = len(state.A) + len(state.B)
                best_A = list(state.A)
                best_B = list(state.B)
                continue

        # Try swap: remove x from A, add y to A
        if len(state.A) > 1:
            x = state.A[rng.randint(0, len(state.A) - 1)]
            state.remove_from_A(x)
            improved = False
            for y in range(1, N + 1):
                if state.in_A[y]:
                    continue
                if state.can_add_to_A(y):
                    state.add_to_A(y)
                    # Now try to add more
                    for _ in range(5):
                        if len(state.A) < MAX_SIZE:
                            z = rng.randint(1, N)
                            if not state.in_A[z] and state.can_add_to_A(z):
                                state.add_to_A(z)
                                improved = True
                                break
                        elif len(state.B) < MAX_SIZE:
                            z = rng.randint(1, N)
                            if not state.in_B[z] and state.can_add_to_B(z):
                                state.add_to_B(z)
                                improved = True
                                break
                        else:
                            break
                    if len(state.A) + len(state.B) > best_score:
                        best_score = len(state.A) + len(state.B)
                        best_A = list(state.A)
                        best_B = list(state.B)
                    break
            if not improved and len(state.A) + len(state.B) <= best_score:
                # Put x back
                if not state.in_A[x] and state.can_add_to_A(x):
                    state.add_to_A(x)

        # Try swap: remove x from B, add y to B
        if len(state.B) > 1:
            x = state.B[rng.randint(0, len(state.B) - 1)]
            state.remove_from_B(x)
            improved = False
            for y in range(1, N + 1):
                if state.in_B[y]:
                    continue
                if state.can_add_to_B(y):
                    state.add_to_B(y)
                    for _ in range(5):
                        if len(state.B) < MAX_SIZE:
                            z = rng.randint(1, N)
                            if not state.in_B[z] and state.can_add_to_B(z):
                                state.add_to_B(z)
                                improved = True
                                break
                        elif len(state.A) < MAX_SIZE:
                            z = rng.randint(1, N)
                            if not state.in_A[z] and state.can_add_to_A(z):
                                state.add_to_A(z)
                                improved = True
                                break
                        else:
                            break
                    if len(state.A) + len(state.B) > best_score:
                        best_score = len(state.A) + len(state.B)
                        best_A = list(state.A)
                        best_B = list(state.B)
                    break
            if not improved and len(state.A) + len(state.B) <= best_score:
                if not state.in_B[x] and state.can_add_to_B(x):
                    state.add_to_B(x)

    return best_A, best_B, best_score


def construct():
    """Construct two Sidon sets with disjoint difference sets, maximizing combined size."""
    start_time = time.time()
    deadline = start_time + 85  # safety margin

    best_A = [1]
    best_B = [2]
    best_score = 2

    rng = random.Random(42)

    # Multiple restart strategies
    strategies = [
        lambda: ([rng.randint(1, N)], [rng.randint(1, N)]),
        lambda: ([rng.randint(1, N)], [rng.randint(1, N // 2 + 1, N)]),
        lambda: ([rng.randint(1, N // 2)], [rng.randint(N // 2 + 1, N)]),
        lambda: (sorted(rng.sample(range(1, N + 1), rng.randint(1, 3))),
                 sorted(rng.sample(range(1, N + 1), rng.randint(1, 3)))),
        lambda: ([rng.randint(1, N)], []),
        lambda: ([], [rng.randint(1, N)]),
    ]

    # Phase 1: Many fast greedy restarts
    it = 0
    while time.time() < deadline - 10:
        it += 1
        strat_idx = it % len(strategies)
        A0, B0 = strategies[strat_idx]()

        # Filter out duplicates and ensure non-empty
        A0 = list(set(A0))
        B0 = list(set(B0))
        if not A0:
            A0 = [rng.randint(1, N)]
        if not B0:
            B0 = [rng.randint(1, N)]

        # Skip if initial state is invalid
        if not _valid(A0, B0):
            continue

        state = FastSidonState(A0, B0)
        state = _greedy_build(state, rng, deadline - 10)

        score = len(state.A) + len(state.B)
        if score > best_score:
            best_score = score
            best_A = list(state.A)
            best_B = list(state.B)

    # Phase 2: Local search on best solution
    if time.time() < deadline - 3:
        state = FastSidonState(best_A, best_B)
        A, B, score = _local_search_improve(state, rng, deadline - 3)
        if score > best_score:
            best_score = score
            best_A = A
            best_B = B

    # Final validation
    if not _valid(best_A, best_B):
        best_A = [1]
        best_B = [2]
        best_score = 2

    return [best_A, best_B]
>>>>>>> REPLACE
