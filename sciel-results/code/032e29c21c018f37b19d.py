import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin within 90s limit

    n = 256

    def get_diffs(S):
        """Get all nonzero absolute differences."""
        S = list(S)
        d = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d.add(abs(S[i] - S[j]))
        return d

    class SetState:
        __slots__ = ['elements', 'sums', 'diffs']

        def __init__(self):
            self.elements = []
            self.sums = set()
            self.diffs = set()

        def try_add(self, x, other_diffs):
            """Try to add x. Returns (success, new_sums, new_diffs)."""
            new_sums = set()
            for s in self.elements:
                ns = x + s
                if ns in self.sums or ns in new_sums:
                    return False, None, None
                new_sums.add(ns)
            two_x = 2 * x
            if two_x in self.sums or two_x in new_sums:
                return False, None, None
            new_sums.add(two_x)

            new_diffs = set()
            for s in self.elements:
                d = abs(x - s)
                if d in other_diffs or d in self.diffs or d in new_diffs:
                    return False, None, None
                new_diffs.add(d)

            return True, new_sums, new_diffs

        def commit(self, x, new_sums, new_diffs):
            self.elements.append(x)
            self.sums.update(new_sums)
            self.diffs.update(new_diffs)

        def remove(self, x):
            """Remove x and recompute sums/diffs."""
            self.elements.remove(x)
            self.sums = set()
            self.diffs = set()
            for i in range(len(self.elements)):
                for j in range(i, len(self.elements)):
                    self.sums.add(self.elements[i] + self.elements[j])
            self.diffs = get_diffs(self.elements)

        def rebuild(self, elements):
            self.elements = list(elements)
            self.sums = set()
            self.diffs = set()
            for i in range(len(self.elements)):
                for j in range(i, len(self.elements)):
                    self.sums.add(self.elements[i] + self.elements[j])
            self.diffs = get_diffs(self.elements)

    def build_with_order(ordering, prefer_A_first=True):
        A = SetState()
        B = SetState()

        for x in ordering:
            if prefer_A_first:
                order = [(A, B), (B, A)]
            else:
                order = [(B, A), (A, B)]

            for target, other in order:
                success, new_sums, new_diffs = target.try_add(x, other.diffs)
                if success:
                    target.commit(x, new_sums, new_diffs)
                    break

        return A, B

    def local_search(A_init, B_init, max_time=10.0):
        """Improve via adding unused elements and swapping."""
        A = SetState()
        B = SetState()
        A.rebuild(A_init)
        B.rebuild(B_init)

        best_total = len(A.elements) + len(B.elements)
        best_A = list(A.elements)
        best_B = list(B.elements)

        all_elements = set(range(1, n + 1))
        used = set(A.elements) | set(B.elements)
        unused = list(all_elements - used)
        random.shuffle(unused)

        # Phase 1: Try adding unused elements
        for x in unused:
            if time.time() > max_time:
                break
            for target, other in [(A, B), (B, A)]:
                success, new_sums, new_diffs = target.try_add(x, other.diffs)
                if success:
                    target.commit(x, new_sums, new_diffs)
                    if len(target.elements) + len(other.elements) > best_total:
                        best_total = len(target.elements) + len(other.elements)
                        best_A = list(A.elements)
                        best_B = list(B.elements)

        # Phase 2: Random swaps (move element from one set to other)
        attempts = 0
        while time.time() < max_time and attempts < 2000:
            attempts += 1
            if not A.elements or not B.elements:
                break
            src, dst = (A, B) if random.random() < 0.5 else (B, A)
            idx = random.randrange(len(src.elements))
            x = src.elements[idx]
            src.remove(x)
            success, new_sums, new_diffs = dst.try_add(x, src.diffs)
            if success:
                dst.commit(x, new_sums, new_diffs)
                total = len(A.elements) + len(B.elements)
                if total > best_total:
                    best_total = total
                    best_A = list(A.elements)
                    best_B = list(B.elements)
            else:
                # Revert
                success2, ns2, nd2 = src.try_add(x, dst.diffs)
                if success2:
                    src.commit(x, ns2, nd2)
                else:
                    src.rebuild(src.elements + [x])

        return best_A, best_B

    # Generate orderings
    orderings = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
        list(range(1, n + 1, 2)) + list(range(2, n + 1, 2)),
        list(range(2, n + 1, 2)) + list(range(1, n + 1, 2)),
    ]

    # Interleaved low-high
    inter = []
    for i in range(1, n // 2 + 1):
        inter.append(i)
        inter.append(n + 1 - i)
    orderings.append(inter[:n])

    rng = random.Random(42)
    for _ in range(12):
        s = list(range(1, n + 1))
        rng.shuffle(s)
        orderings.append(s)

    best_total = 0
    best_A = None
    best_B = None

    for ordering in orderings:
        if time.time() > deadline - 25:
            break
        for prefer_A in [True, False]:
            if time.time() > deadline - 25:
                break
            A_state, B_state = build_with_order(ordering, prefer_A)
            total = len(A_state.elements) + len(B_state.elements)
            if total > best_total:
                best_total = total
                best_A = list(A_state.elements)
                best_B = list(B_state.elements)

    # Local search polish
    if best_A is not None and best_B is not None:
        remaining = deadline - time.time() - 10
        if remaining > 5:
            polished_A, polished_B = local_search(best_A, best_B, max_time=min(remaining, 30.0))
            if len(polished_A) + len(polished_B) > best_total:
                best_A = polished_A
                best_B = polished_B
                best_total = len(best_A) + len(best_B)

    if best_A is None or best_B is None:
        best_A = [1]
        best_B = [2]

    return [best_A, best_B]
