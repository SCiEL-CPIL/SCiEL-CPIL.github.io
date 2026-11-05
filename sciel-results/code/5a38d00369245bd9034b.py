"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit
    n = 256

    def _can_add_to_set(x, set_elems, sums, diffs_self, diffs_other):
        """Check if x can be added to a set with given elements, sums, and diffs."""
        for a in set_elems:
            s = a + x
            if s in sums:
                return False
            d = abs(a - x)
            if d in diffs_self or d in diffs_other:
                return False
        if 2 * x in sums:
            return False
        return True

    def _add_to_set(x, set_elems, sums, diffs):
        """Add x to set, updating sums and diffs."""
        for a in set_elems:
            sums.add(a + x)
            diffs.add(abs(a - x))
        sums.add(2 * x)
        set_elems.append(x)

    def _remove_from_set(x, set_elems, sums, diffs):
        """Remove x from set, updating sums and diffs."""
        set_elems.remove(x)
        for a in set_elems:
            sums.discard(a + x)
            diffs.discard(abs(a - x))
        sums.discard(2 * x)

    def _greedy_simultaneous(order, deadline, rng):
        """
        Simultaneous greedy: process elements in order, try to add to either set.
        Uses balance heuristic and random tie-breaking.
        """
        setA = []
        setB = []
        sumsA = set()
        sumsB = set()
        diffsA = set()
        diffsB = set()

        for x in order:
            if time.time() > deadline:
                break

            can_add_A = _can_add_to_set(x, setA, sumsA, diffsA, diffsB)
            can_add_B = _can_add_to_set(x, setB, sumsB, diffsB, diffsA)

            if can_add_A and can_add_B:
                # Both possible: choose based on balance with some randomness
                if len(setA) < len(setB):
                    target = 'A'
                elif len(setB) < len(setA):
                    target = 'B'
                else:
                    target = 'A' if rng.random() < 0.5 else 'B'
                if target == 'A':
                    _add_to_set(x, setA, sumsA, diffsA)
                else:
                    _add_to_set(x, setB, sumsB, diffsB)
            elif can_add_A:
                _add_to_set(x, setA, sumsA, diffsA)
            elif can_add_B:
                _add_to_set(x, setB, sumsB, diffsB)

        return sorted(setA), sorted(setB)

    def _local_search(setA, setB, deadline, rng, max_iters=2000):
        """
        Local search: try removing one element from A or B and adding a new element
        to maximize total size. Also try swaps between sets.
        """
        setA = list(setA)
        setB = list(setB)

        # Build sums and diffs
        sumsA = set()
        sumsB = set()
        diffsA = set()
        diffsB = set()
        for i, a in enumerate(setA):
            for j in range(i + 1):
                sumsA.add(setA[i] + setA[j])
        for i, b in enumerate(setB):
            for j in range(i + 1):
                sumsB.add(setB[i] + setB[j])
        for a in setA:
            for b in setA:
                if a != b:
                    d = abs(a - b)
                    if d > 0:
                        diffsA.add(d)
        for b in setB:
            for c in setB:
                if b != c:
                    d = abs(b - c)
                    if d > 0:
                        diffsB.add(d)

        best_size = len(setA) + len(setB)
        improved = True
        iter_count = 0

        while improved and time.time() < deadline and iter_count < max_iters:
            improved = False
            iter_count += 1

            # Try: remove one element from A, then try to add a new element to A or B
            if len(setA) > 0 and time.time() < deadline:
                idx = rng.randint(0, len(setA) - 1)
                removed = setA[idx]

                _remove_from_set(removed, setA, sumsA, diffsA)

                # Try to add some element to A or B
                candidates = [x for x in range(1, n + 1) if x not in setA and x not in setB]
                rng.shuffle(candidates)

                added = False
                for x in candidates:
                    if time.time() > deadline:
                        break
                    if _can_add_to_set(x, setA, sumsA, diffsA, diffsB):
                        _add_to_set(x, setA, sumsA, diffsA)
                        added = True
                        break
                    if _can_add_to_set(x, setB, sumsB, diffsB, diffsA):
                        _add_to_set(x, setB, sumsB, diffsB)
                        added = True
                        break

                if added and len(setA) + len(setB) > best_size:
                    best_size = len(setA) + len(setB)
                    improved = True
                else:
                    # Revert
                    if added:
                        # Remove what we just added
                        if x in setA:
                            _remove_from_set(x, setA, sumsA, diffsA)
                        elif x in setB:
                            _remove_from_set(x, setB, sumsB, diffsB)
                    _add_to_set(removed, setA, sumsA, diffsA)

            # Try: remove one element from B, then try to add a new element to A or B
            if len(setB) > 0 and time.time() < deadline:
                idx = rng.randint(0, len(setB) - 1)
                removed = setB[idx]

                _remove_from_set(removed, setB, sumsB, diffsB)

                candidates = [x for x in range(1, n + 1) if x not in setA and x not in setB]
                rng.shuffle(candidates)

                added = False
                for x in candidates:
                    if time.time() > deadline:
                        break
                    if _can_add_to_set(x, setA, sumsA, diffsA, diffsB):
                        _add_to_set(x, setA, sumsA, diffsA)
                        added = True
                        break
                    if _can_add_to_set(x, setB, sumsB, diffsB, diffsA):
                        _add_to_set(x, setB, sumsB, diffsB)
                        added = True
                        break

                if added and len(setA) + len(setB) > best_size:
                    best_size = len(setA) + len(setB)
                    improved = True
                else:
                    if added:
                        if x in setA:
                            _remove_from_set(x, setA, sumsA, diffsA)
                        elif x in setB:
                            _remove_from_set(x, setB, sumsB, diffsB)
                    _add_to_set(removed, setB, sumsB, diffsB)

        return sorted(setA), sorted(setB)

    # Multiple random restarts with simultaneous greedy + local search
    best_sets = ([1], [2])
    best_size = 2

    rng = random.Random(42)

    for restart in range(200):
        if time.time() > deadline - 5:
            break

        order = list(range(1, n + 1))
        rng.shuffle(order)

        setA, setB = _greedy_simultaneous(order, deadline - 5, rng)
        current_size = len(setA) + len(setB)

        # Apply local search
        if time.time() < deadline - 2:
            setA, setB = _local_search(setA, setB, deadline - 1, rng, max_iters=500)
            current_size = len(setA) + len(setB)

        if current_size > best_size:
            best_sets = (setA, setB)
            best_size = current_size

    return [best_sets[0], best_sets[1]]
# EVOLVE-BLOCK-END
