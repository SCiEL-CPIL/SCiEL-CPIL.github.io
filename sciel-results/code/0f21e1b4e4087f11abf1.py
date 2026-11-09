import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85

    n = 256
    max_size = 64

    def greedy_with_bitmask(order):
        """
        Use bitmask-like tracking via sets but with early termination.
        sums_A: set of pair sums in A
        sums_B: set of pair sums in B
        diffs_A: set of nonzero differences in A
        diffs_B: set of nonzero differences in B
        """
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()

        for x in order:
            if len(A) >= max_size and len(B) >= max_size:
                break

            # Try A
            if len(A) < max_size:
                feasible = True
                if (x + x) in sums_A:
                    feasible = False
                else:
                    for a in A:
                        s = x + a
                        if s in sums_A:
                            feasible = False
                            break
                        d1 = x - a
                        d2 = a - x
                        if d1 in diffs_B or d2 in diffs_B:
                            feasible = False
                            break
                if feasible:
                    A.append(x)
                    for a in A[:-1]:
                        sums_A.add(x + a)
                        diffs_A.add(x - a)
                        diffs_A.add(a - x)
                    sums_A.add(x + x)
                    continue

            # Try B
            if len(B) < max_size:
                feasible = True
                if (x + x) in sums_B:
                    feasible = False
                else:
                    for b in B:
                        s = x + b
                        if s in sums_B:
                            feasible = False
                            break
                        d1 = x - b
                        d2 = b - x
                        if d1 in diffs_A or d2 in diffs_A:
                            feasible = False
                            break
                if feasible:
                    B.append(x)
                    for b in B[:-1]:
                        sums_B.add(x + b)
                        diffs_B.add(x - b)
                        diffs_B.add(b - x)
                    sums_B.add(x + x)

        return A, B

    def polish(A, B, max_iters=50):
        """
        Repeatedly try to insert any element from [1..256] not already in A or B.
        Recalculate constraints each pass.
        """
        for _ in range(max_iters):
            if time.time() > deadline - 2:
                break

            in_A = set(A)
            in_B = set(B)

            # Build constraint sets
            sums_A = set()
            sums_B = set()
            diffs_A = set()
            diffs_B = set()

            for i in range(len(A)):
                for j in range(i, len(A)):
                    sums_A.add(A[i] + A[j])
            for i in range(len(A)):
                for j in range(len(A)):
                    if i != j:
                        diffs_A.add(A[j] - A[i])

            for i in range(len(B)):
                for j in range(i, len(B)):
                    sums_B.add(B[i] + B[j])
            for i in range(len(B)):
                for j in range(len(B)):
                    if i != j:
                        diffs_B.add(B[j] - B[i])

            added = False
            candidates = [x for x in range(1, n + 1) if x not in in_A and x not in in_B]
            random.shuffle(candidates)

            for x in candidates:
                if time.time() > deadline - 2:
                    break

                # Try A
                if len(A) < max_size:
                    ok = True
                    if (x + x) in sums_A:
                        ok = False
                    else:
                        for a in A:
                            if (x + a) in sums_A or (x - a) in diffs_B or (a - x) in diffs_B:
                                ok = False
                                break
                    if ok:
                        A.append(x)
                        added = True
                        break

                # Try B
                if len(B) < max_size:
                    ok = True
                    if (x + x) in sums_B:
                        ok = False
                    else:
                        for b in B:
                            if (x + b) in sums_B or (x - b) in diffs_A or (b - x) in diffs_A:
                                ok = False
                                break
                    if ok:
                        B.append(x)
                        added = True
                        break

            if not added:
                break

        return A, B

    def validate(A, B):
        """Check all constraints are satisfied."""
        if not A or not B:
            return False
        if len(A) > max_size or len(B) > max_size:
            return False

        # Check Sidon for A
        sums_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums_A:
                    return False
                sums_A.add(s)

        # Check Sidon for B
        sums_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                s = B[i] + B[j]
                if s in sums_B:
                    return False
                sums_B.add(s)

        # Check disjoint nonzero differences
        diffs_A = set()
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    diffs_A.add(A[j] - A[i])

        diffs_B = set()
        for i in range(len(B)):
            for j in range(len(B)):
                if i != j:
                    diffs_B.add(B[j] - B[i])

        if diffs_A & diffs_B:
            return False

        return True

    best_A, best_B = [], []
    best_score = 0

    num_trials = 500

    for trial in range(num_trials):
        if time.time() > deadline - 3:
            break

        order = list(range(1, n + 1))
        random.shuffle(order)

        A, B = greedy_with_bitmask(order)
        A, B = polish(A, B)

        if validate(A, B):
            score = len(A) + len(B)
            if score > best_score:
                best_score = score
                best_A, best_B = A[:], B[:]

        if best_score >= 32:
            break

    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]

    return [best_A, best_B]
