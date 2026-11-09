import time
import random

N = 256
MAX_EACH = 64
DEADLINE = 85.0  # safety margin before the 90s kill

class State:
    __slots__ = ('A', 'B', 'sumA', 'sumB', 'diffA', 'diffB')

    def __init__(self):
        self.A = []
        self.B = []
        self.sumA = set()
        self.sumB = set()
        self.diffA = set()
        self.diffB = set()

    def size(self):
        return len(self.A) + len(self.B)

    def can_add(self, side, x):
        """Vectorized feasibility check for adding x to side (0=A, 1=B)."""
        if side == 0:
            elems, sums, diffs, other_diffs = self.A, self.sumA, self.diffA, self.diffB
        else:
            elems, sums, diffs, other_diffs = self.B, self.sumB, self.diffB, self.diffA
        if len(elems) >= MAX_EACH:
            return False
        # New pair sums: {a+x for a in elems} U {2x}; must not hit existing sums
        for a in elems:
            if (a + x) in sums:
                return False
        if (2 * x) in sums:
            return False
        # New signed differences: {x-a, a-x for a in elems}; must not hit other_diffs
        for a in elems:
            if (x - a) in other_diffs or (a - x) in other_diffs:
                return False
        return True

    def add(self, side, x):
        if side == 0:
            elems, sums, diffs = self.A, self.sumA, self.diffA
        else:
            elems, sums, diffs = self.B, self.sumB, self.diffB
        for a in elems:
            sums.add(a + x)
            diffs.add(x - a)
            diffs.add(a - x)
        sums.add(2 * x)
        elems.append(x)
        elems.sort()

    def remove(self, side, x):
        if side == 0:
            elems, sums, diffs = self.A, self.sumA, self.diffA
        else:
            elems, sums, diffs = self.B, self.sumB, self.diffB
        for a in elems:
            if a != x:
                sums.discard(a + x)
                diffs.discard(x - a)
                diffs.discard(a - x)
        sums.discard(2 * x)
        elems.remove(x)

def greedy_build(rng, order):
    """Greedy construction following a given candidate order."""
    st = State()
    for x in order:
        if st.can_add(0, x):
            st.add(0, x)
        elif st.can_add(1, x):
            st.add(1, x)
    return st

def greedy_build_with_deadline(rng, order, t0, deadline):
    st = State()
    for x in order:
        if time.time() - t0 > deadline:
            break
        if st.can_add(0, x):
            st.add(0, x)
        elif st.can_add(1, x):
            st.add(1, x)
    return st

def try_add_anywhere(st, x):
    """Try to add x to whichever side it fits. Return True if added."""
    if st.can_add(0, x):
        st.add(0, x)
        return True
    if st.can_add(1, x):
        st.add(1, x)
        return True
    return False

def hill_climb(st, t0, deadline):
    """Local search: add any free element; swap to enable a strictly better addition."""
    improved = True
    while improved and (time.time() - t0) < deadline:
        improved = False
        # Phase 1: add any free element that fits
        for x in range(1, N + 1):
            if x in st.A or x in st.B:
                continue
            if try_add_anywhere(st, x):
                improved = True
                break
        # Phase 2: swap an element for another to enable a strictly better addition
        if (time.time() - t0) > deadline:
            break
        for side in (0, 1):
            if (time.time() - t0) > deadline:
                break
            elems = list(st.A if side == 0 else st.B)
            for a in elems:
                if (time.time() - t0) > deadline:
                    break
                st.remove(side, a)
                # Find the best x we could add now
                best_x = None
                for x in range(1, N + 1):
                    if x == a or x in st.A or x in st.B:
                        continue
                    if try_add_anywhere(st, x):
                        best_x = x
                        break
                if best_x is not None:
                    # We added best_x after removing a -> net +1 size, keep it
                    improved = True
                else:
                    # No addition possible; restore a
                    st.add(side, a)
    return st

def construct():
    t0 = time.time()
    rng = random.Random(12345)
    best = None
    best_size = -1

    # First a deterministic-ish greedy to get a baseline quickly
    order = list(range(1, N + 1))
    st = greedy_build_with_deadline(rng, order, t0, DEADLINE)
    hill_climb(st, t0, DEADLINE)
    best = st
    best_size = st.size()

    # Randomized restarts with varied orderings
    while (time.time() - t0) < (DEADLINE - 0.5):
        # Build a random order, occasionally with a bias
        order = list(range(1, N + 1))
        rng.shuffle(order)
        st = greedy_build_with_deadline(rng, order, t0, DEADLINE)
        hill_climb(st, t0, DEADLINE)
        if st.size() > best_size:
            best = st
            best_size = st.size()
            if best_size >= 2 * MAX_EACH:
                break

    return [list(best.A), list(best.B)]
