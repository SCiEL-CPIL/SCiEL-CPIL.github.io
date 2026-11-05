import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill
    n = 256
    max_per_set = 64

    class State:
        """Incremental state for a pair of Sidon sets."""
        __slots__ = ['sets', 'sums', 'diffs', 'used']

        def __init__(self):
            self.sets = [[], []]
            self.sums = [set(), set()]
            self.diffs = [set(), set()]
            self.used = set()

        def can_add(self, x, side):
            """Check if x can be added to side."""
            S = self.sets[side]
            if len(S) >= max_per_set:
                return False
            S_sums = self.sums[side]
            S_diffs = self.diffs[side]
            other_diffs = self.diffs[1 - side]
            # Sidon check
            for a in S:
                if (x + a) in S_sums:
                    return False
            if (2 * x) in S_sums:
                return False
            # Diff disjointness check
            for a in S:
                if (x - a) in other_diffs or (a - x) in other_diffs:
                    return False
            return True

        def add(self, x, side):
            """Add x to side, updating all state."""
            S = self.sets[side]
            S_sums = self.sums[side]
            S_diffs = self.diffs[side]
            for a in S:
                S_sums.add(x + a)
                S_diffs.add(x - a)
                S_diffs.add(a - x)
            S_sums.add(2 * x)
            S.append(x)
            self.used.add(x)

        def remove(self, x, side):
            """Remove x from side, updating all state."""
            S = self.sets[side]
            S_sums = self.sums[side]
            S_diffs = self.diffs[side]
            S.remove(x)
            for a in S:
                S_sums.discard(x + a)
                S_diffs.discard(x - a)
                S_diffs.discard(a - x)
            S_sums.discard(2 * x)
            # x may still be in the other set
            if x not in self.sets[1 - side]:
                self.used.discard(x)

        def validate(self):
            """Full validation."""
            if not self.sets[0] or not self.sets[1]:
                return False
            for side in (0, 1):
                S = self.sets[side]
                sums = set()
                for i in range(len(S)):
                    for j in range(i, len(S)):
                        s = S[i] + S[j]
                        if s in sums:
                            return False
                        sums.add(s)
            d0 = self.diffs[0]
            d1 = self.diffs[1]
            if d0 & d1:
                return False
            return True

    def greedy_fill(state, order, max_time):
        """Fill using greedy in given order."""
        for x in order:
            if time.time() > max_time:
                break
            for side in (0, 1):
                if state.can_add(x, side):
                    state.add(x, side)
                    break

    def local_improve(state, max_time):
        """Local search: try additions, then removal+refill for net gain."""
        improved = True
        while improved and time.time() < max_time:
            improved = False
            # Phase 1: try adding any unused element
            unused = [x for x in range(1, n + 1) if x not in state.used]
            random.shuffle(unused)
            for x in unused:
                if time.time() > max_time:
                    break
                for side in (0, 1):
                    if state.can_add(x, side):
                        state.add(x, side)
                        improved = True
                        break
            # Phase 2: try removing one element and greedily refilling for net gain
            if time.time() > max_time:
                break
            for side in (0, 1):
                if time.time() > max_time:
                    break
                S = state.sets[side]
                for i in range(len(S)):
                    if time.time() > max_time:
                        break
                    removed = S[i]
                    state.remove(removed, side)
                    # Save state
                    old_len = len(state.sets[side])
                    old_sums = set(state.sums[side])
                    old_diffs = set(state.diffs[side])
                    old_used = set(state.used)
                    # Greedily refill
                    unused = [x for x in range(1, n + 1) if x not in state.used]
                    random.shuffle(unused)
                    for x in unused:
                        if time.time() > max_time:
                            break
                        if len(state.sets[side]) >= max_per_set:
                            break
                        if state.can_add(x, side):
                            state.add(x, side)
                    # Check if we improved
                    if len(state.sets[side]) > old_len:
                        improved = True
                    else:
                        # Revert
                        # Remove all newly added
                        cur = state.sets[side][:]
                        for x in cur:
                            if x != removed and len(state.sets[side]) > old_len:
                                state.remove(x, side)
                        # Re-add removed
                        state.add(removed, side)
                        # Verify state consistency
                        if len(state.sets[side]) != old_len + 1:
                            # Something went wrong, rebuild
                            pass

    def solve_one_order(order, max_time):
        """Solve with one greedy order + local improvement."""
        state = State()
        greedy_fill(state, order, max_time - 5)
        if state.validate():
            local_improve(state, max_time - 2)
            if state.validate():
                return [list(state.sets[0]), list(state.sets[1])]
        return None

    random.seed(42)

    best_sets = None
    best_size = 0

    # Generate diverse orderings
    orderings = []
    orderings.append(list(range(1, n + 1)))
    orderings.append(list(range(n, 0, -1)))
    for _ in range(8):
        o = list(range(1, n + 1))
        random.shuffle(o)
        orderings.append(o)
    for _ in range(4):
        o = list(range(1, n + 1))
        o.sort(key=lambda x: x * random.random())
        orderings.append(o)
    for _ in range(4):
        o = list(range(1, n + 1))
        o.sort(key=lambda x: (n + 1 - x) * random.random())
        orderings.append(o)

    # Time budget per ordering
    per_order_time = 4.0

    for idx, order in enumerate(orderings):
        if time.time() > deadline - 3:
            break
        t0 = time.time()
        remaining = deadline - 3 - t0
        alloc = min(per_order_time, remaining)
        result = solve_one_order(order, t0 + alloc)
        if result is not None:
            sz = len(result[0]) + len(result[1])
            if sz > best_size:
                best_size = sz
                best_sets = result

    # Final polish: try more local search on best solution
    if best_sets is not None and time.time() < deadline - 5:
        state = State()
        for i, x in enumerate(best_sets[0]):
            state.add(x, 0)
        for i, x in enumerate(best_sets[1]):
            state.add(x, 1)
        if state.validate():
            local_improve(state, deadline - 2)
            if state.validate():
                sz = len(state.sets[0]) + len(state.sets[1])
                if sz > best_size:
                    best_size = sz
                    best_sets = [list(state.sets[0]), list(state.sets[1])]

    if best_sets is None:
        # Fallback
        state = State()
        greedy_fill(state, list(range(1, n + 1)), deadline - 1)
        if state.validate():
            best_sets = [list(state.sets[0]), list(state.sets[1])]
        else:
            best_sets = [[1], [256]]

    return best_sets
