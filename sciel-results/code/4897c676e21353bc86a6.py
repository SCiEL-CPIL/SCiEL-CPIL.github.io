"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit
    n = 256
    max_size = 64

    best_total = 0
    best_A = []
    best_B = []

    def _greedy_one_ordering(order, deadline):
        """
        Single forward pass over the given ordering.
        Returns (setA, setB) as sorted lists.
        """
        setA = []
        setB = []
        sumsA = set()
        sumsB = set()
        diffsA = set()
        diffsB = set()

        for x in order:
            # Try adding to A
            if len(setA) < max_size:
                # Check Sidon condition for A
                ok = True
                for a in setA:
                    if (x + a) in sumsA:
                        ok = False
                        break
                if ok:
                    # Check difference disjointness
                    for a in setA:
                        d = x - a
                        if d in diffsB or (-d) in diffsB:
                            ok = False
                            break
                if ok:
                    # Update state
                    for a in setA:
                        sumsA.add(x + a)
                    sumsA.add(2 * x)
                    for a in setA:
                        diffsA.add(x - a)
                        diffsA.add(a - x)
                    setA.append(x)
                    continue

            # Try adding to B
            if len(setB) < max_size:
                # Check Sidon condition for B
                ok = True
                for b in setB:
                    if (x + b) in sumsB:
                        ok = False
                        break
                if ok:
                    # Check difference disjointness
                    for b in setB:
                        d = x - b
                        if d in diffsA or (-d) in diffsA:
                            ok = False
                            break
                if ok:
                    # Update state
                    for b in setB:
                        sumsB.add(x + b)
                    sumsB.add(2 * x)
                    for b in setB:
                        diffsB.add(x - b)
                        diffsB.add(b - x)
                    setB.append(x)

        return sorted(setA), sorted(setB)

    # Multiple random restarts with different orderings
    while time.time() < deadline:
        # Create a random ordering of [1, n]
        order = list(range(1, n + 1))
        random.shuffle(order)

        A, B = _greedy_one_ordering(order, deadline)
        total = len(A) + len(B)

        if total > best_total:
            best_total = total
            best_A = A
            best_B = B

        # Early termination if we hit the maximum possible
        if best_total >= 2 * max_size:
            break

    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]

    return [best_A, best_B]
# EVOLVE-BLOCK-END
