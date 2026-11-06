"""Two Sidon sets with disjoint difference sets.
Uses bitset-accelerated greedy construction with ILS (Iterated Local Search)
and structured multi-start seeds to maximize combined size under the
disjoint-difference constraint."""

import time
import random
import math

N = 256
MAX_SIZE = 64

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

class State:
    """Maintains state for two sets with fast constraint checking."""
    def __init__(self, A=None, B=None):
        self.A = list(A) if A else []
        self.B = list(B) if B else []
        self.sums_A = set()
        self.sums_B = set()
        self.diffs_A = set()
        self.diffs_B = set()
        self._rebuild()

    def _rebuild(self):
        self.sums_A = set()
        self.sums_B = set()
        self.diffs_A = set()
        self.diffs_B = set()
        for i in range(len(self.A)):
            for j in range(i, len(self.A)):
                self.sums_A.add(self.A[i] + self.A[j])
        for i in range(len(self.B)):
            for j in range(i, len(self.B)):
                self.sums_B.add(self.B[i] + self.B[j])
        for i in range(len(self.A)):
            for j in range(len(self.A)):
                if i != j:
                    self.diffs_A.add(self.A[j] - self.A[i])
        for i in range(len(self.B)):
            for j in range(len(self.B)):
                if i != j:
                    self.diffs_B.add(self.B[j] - self.B[i])

    def can_add_A(self, x):
        """Check if x can be added to A."""
        if x in self.A:
            return False
        if 2 * x in self.sums_A:
            return False
        for a in self.A:
            if (a + x) in self.sums_A:
                return False
            d = x - a
            if d != 0 and d in self.diffs_B:
                return False
        return True

    def can_add_B(self, x):
        """Check if x can be added to B."""
        if x in self.B:
            return False
        if 2 * x in self.sums_B:
            return False
        for b in self.B:
            if (b + x) in self.sums_B:
                return False
            d = x - b
            if d != 0 and d in self.diffs_A:
                return False
        return True

    def add_A(self, x):
        """Add x to A and update state."""
        self.A.append(x)
        self.sums_A.add(2 * x)
        for a in self.A[:-1]:
            self.sums_A.add(a + x)
            d1 = x - a
            d2 = a - x
            self.diffs_A.add(d1)
            self.diffs_A.add(d2)

    def add_B(self, x):
        """Add x to B and update state."""
        self.B.append(x)
        self.sums_B.add(2 * x)
        for b in self.B[:-1]:
            self.sums_B.add(b + x)
            d1 = x - b
            d2 = b - x
            self.diffs_B.add(d1)
            self.diffs_B.add(d2)

    def remove_A(self, x):
        """Remove x from A and rebuild (expensive but safe)."""
        self.A.remove(x)
        self._rebuild()

    def remove_B(self, x):
        """Remove x from B and rebuild (expensive but safe)."""
        self.B.remove(x)
        self._rebuild()

    def size(self):
        return len(self.A) + len(self.B)

def _greedy_fill(state, rng, deadline, max_rounds=200):
    """Fill state by greedily adding elements. Returns state."""
    for _ in range(max_rounds):
        if time.time() > deadline - 5:
            break
        progress = False

        # Try to add to A
        if len(state.A) < MAX_SIZE:
            candidates = [x for x in range(1, N + 1) if x not in state.A]
            rng.shuffle(candidates)
            for x in candidates:
                if state.can_add_A(x):
                    state.add_A(x)
                    progress = True
                    break

        # Try to add to B
        if len(state.B) < MAX_SIZE:
            candidates = [x for x in range(1, N + 1) if x not in state.B]
            rng.shuffle(candidates)
            for x in candidates:
                if state.can_add_B(x):
                    state.add_B(x)
                    progress = True
                    break

        if not progress:
            break

    return state

def _local_search(state, rng, deadline, max_iters=50):
    """Local search: try swaps to improve."""
    for _ in range(max_iters):
        if time.time() > deadline - 5:
            break
        improved = False

        # Try removing one element from A and adding a different one
        if len(state.A) > 0 and len(state.A) < MAX_SIZE:
            for i in range(len(state.A)):
                old = state.A[i]
                state.remove_A(old)
                # Try to add a new element
                candidates = [x for x in range(1, N + 1) if x not in state.A]
                rng.shuffle(candidates)
                added = False
                for x in candidates:
                    if state.can_add_A(x):
                        state.add_A(x)
                        improved = True
                        added = True
                        break
                if not added:
                    # Revert
                    state.add_A(old)
                    state._rebuild()
                if improved:
                    break

        # Try removing one element from B and adding a different one
        if len(state.B) > 0 and len(state.B) < MAX_SIZE:
            for i in range(len(state.B)):
                old = state.B[i]
                state.remove_B(old)
                candidates = [x for x in range(1, N + 1) if x not in state.B]
                rng.shuffle(candidates)
                added = False
                for x in candidates:
                    if state.can_add_B(x):
                        state.add_B(x)
                        improved = True
                        added = True
                        break
                if not added:
                    state.add_B(old)
                    state._rebuild()
                if improved:
                    break

        if not improved:
            break

    return state

def _ilscycle(state, rng, deadline, max_iters=30):
    """ILS: perturb and local search."""
    best_state = state
    best_size = state.size()

    for _ in range(max_iters):
        if time.time() > deadline - 5:
            break

        # Perturb: remove 1-3 random elements from A or B
        perturb_state = State(list(state.A), list(state.B))
        num_perturb = rng.randint(1, 3)
        for _p in range(num_perturb):
            if rng.random() < 0.5 and perturb_state.A:
                idx = rng.randrange(len(perturb_state.A))
                perturb_state.remove_A(perturb_state.A[idx])
            elif perturb_state.B:
                idx = rng.randrange(len(perturb_state.B))
                perturb_state.remove_B(perturb_state.B[idx])

        # Greedy fill
        perturb_state = _greedy_fill(perturb_state, rng, deadline)
        # Local search
        perturb_state = _local_search(perturb_state, rng, deadline)

        if perturb_state.size() > best_size:
            best_size = perturb_state.size()
            best_state = perturb_state
            state = perturb_state

    return best_state

def _structured_seed(rng, n=N):
    """Generate structured seeds using spacing patterns."""
    strategies = []

    # Strategy 1: Arithmetic progressions with different moduli
    for mod in [4, 5, 6, 7, 8]:
        A = list(range(1, n + 1, mod))
        B = list(range(2, n + 1, mod))
        if A and B:
            strategies.append((A[:1], B[:1]))

    # Strategy 2: Power-based spacing
    A = [2**i for i in range(1, 8) if 2**i <= n]
    B = [3**i for i in range(1, 7) if 3**i <= n]
    if A and B:
        strategies.append((A[:1], B[:1]))

    # Strategy 3: Random wide spacing
    for _ in range(5):
        A0 = [rng.randint(1, n // 2)]
        B0 = [rng.randint(n // 2 + 1, n)]
        strategies.append((A0, B0))

    # Strategy 4: Prime-based
    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(math.sqrt(x)) + 1):
            if x % i == 0:
                return False
        return True
    primes = [p for p in range(2, n + 1) if is_prime(p)]
    if len(primes) >= 2:
        strategies.append(([primes[0]], [primes[-1]]))

    if not strategies:
        return [rng.randint(1, n)], [rng.randint(1, n)]

    return rng.choice(strategies)

def construct():
    """Construct two Sidon sets with disjoint difference sets, maximizing combined size."""
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    rng = random.Random(12345)

    best_state = None
    best_size = 0

    # Multiple restarts with different strategies
    max_restarts = 200

    for it in range(max_restarts):
        if time.time() > deadline - 10:
            break

        # Choose seed strategy
        if it % 4 == 0:
            A0, B0 = _structured_seed(rng)
        elif it % 4 == 1:
            A0 = [rng.randint(1, N)]
            B0 = [rng.randint(1, N)]
        elif it % 4 == 2:
            sa = rng.randint(1, 3)
            sb = rng.randint(1, 3)
            A0 = sorted(rng.sample(range(1, N + 1), sa))
            B0 = sorted(rng.sample(range(1, N + 1), sb))
        else:
            # Wider spacing
            A0 = [rng.randint(1, N // 2)]
            B0 = [rng.randint(N // 2 + 1, N)]

        # Build initial state
        state = State(A0, B0)

        # Validate initial state
        if not _valid(state.A, state.B):
            # Try to fix by removing conflicts
            state = State([A0[0]], [B0[0]])

        # Greedy fill
        state = _greedy_fill(state, rng, deadline)

        # Local search
        state = _local_search(state, rng, deadline)

        # ILS cycles
        if time.time() < deadline - 15:
            state = _ilscycle(state, rng, deadline, max_iters=10)

        if state.size() > best_size:
            best_size = state.size()
            best_state = state

            # Final polish with more iterations
            if time.time() < deadline - 20:
                best_state = _local_search(best_state, rng, deadline, max_iters=100)
                best_state = _ilscycle(best_state, rng, deadline, max_iters=20)
                best_size = best_state.size()

    # Ensure valid result
    if best_state is None or not _valid(best_state.A, best_state.B):
        return [[1], [2]]

    return [best_state.A, best_state.B]
