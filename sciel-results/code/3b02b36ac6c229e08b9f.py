"""New design: multi-trial greedy with random orderings + local search swaps.
   Two Sidon subsets of [1,256], maximize |A|+|B|, disjoint nonzero differences."""

import time
import random
from itertools import combinations, combinations_with_replacement

N = 256
MAX_SIZE = 64
DEADLINE = 85.0  # safety margin before 90s kill


def _pair_sums_ok(candidate):
    """Check that all pair sums (with replacement) in candidate are unique."""
    sums = set()
    for i in range(len(candidate)):
        for j in range(i, len(candidate)):
            s = candidate[i] + candidate[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def _diffs_of(sets_list):
    """Return set of nonzero differences for a list."""
    d = set()
    for i in range(len(sets_list)):
        for j in range(i + 1, len(sets_list)):
            diff = sets_list[j] - sets_list[i]
            if diff != 0:
                d.add(diff)
    return d


def _can_add_x_to_set(x, current_set, current_sums, current_diffs, other_diffs):
    """Check if x can be added to current_set maintaining Sidon and disjoint diffs."""
    # Check pair sums
    for a in current_set:
        s = x + a
        if s in current_sums:
            return False
    s2 = x + x
    if s2 in current_sums:
        return False
    # Check differences
    new_diffs = set()
    for a in current_set:
        d = x - a
        if d != 0:
            new_diffs.add(d)
    if new_diffs & other_diffs:
        return False
    if new_diffs & current_diffs:
        return False
    return True


def _greedy_build(order, deadline):
    """Build two sets greedily following a random ordering of [1..256]."""
    A = []
    B = []
    sums_A = set()
    sums_B = set()
    diffs_A = set()
    diffs_B = set()

    for x in order:
        if time.time() > deadline:
            break
        # Try adding to A
        if len(A) < MAX_SIZE:
            ok = True
            for a in A:
                s = x + a
                if s in sums_A:
                    ok = False
                    break
            if ok and (x + x) in sums_A:
                ok = False
            if ok:
                new_diffs = set()
                for a in A:
                    d = x - a
                    if d != 0:
                        new_diffs.add(d)
                if new_diffs & diffs_B or new_diffs & diffs_A:
                    ok = False
            if ok:
                # Add to A
                A.append(x)
                for a in A[:-1]:
                    sums_A.add(x + a)
                sums_A.add(x + x)
                for a in A[:-1]:
                    d = x - a
                    if d != 0:
                        diffs_A.add(d)
                continue

        # Try adding to B
        if len(B) < MAX_SIZE:
            ok = True
            for b in B:
                s = x + b
                if s in sums_B:
                    ok = False
                    break
            if ok and (x + x) in sums_B:
                ok = False
            if ok:
                new_diffs = set()
                for b in B:
                    d = x - b
                    if d != 0:
                        new_diffs.add(d)
                if new_diffs & diffs_A or new_diffs & diffs_B:
                    ok = False
            if ok:
                B.append(x)
                for b in B[:-1]:
                    sums_B.add(x + b)
                sums_B.add(x + x)
                for b in B[:-1]:
                    d = x - b
                    if d != 0:
                        diffs_B.add(d)

    return A, B


def _validate(A, B):
    """Validate that A and B are valid Sidon sets with disjoint nonzero differences."""
    for S in (A, B):
        if not S:
            return False
        if len(S) > MAX_SIZE:
            return False
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
    dA = set()
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            dA.add(A[j] - A[i])
    dB = set()
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            dB.add(B[j] - B[i])
    if dA & dB:
        return False
    return True


def _local_search(A, B, deadline):
    """Try to improve by swapping elements or adding new ones."""
    A = list(A)
    B = list(B)
    best_size = len(A) + len(B)
    improved = True

    while improved and time.time() < deadline:
        improved = False

        # Try adding new elements
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if x in A or x in B:
                continue

            # Try adding to A
            if len(A) < MAX_SIZE:
                new_A = A + [x]
                ok = True
                sums = set()
                for i in range(len(new_A)):
                    for j in range(i, len(new_A)):
                        s = new_A[i] + new_A[j]
                        if s in sums:
                            ok = False
                            break
                        sums.add(s)
                    if not ok:
                        break
                if ok:
                    dA_new = set()
                    for i in range(len(new_A)):
                        for j in range(i + 1, len(new_A)):
                            dA_new.add(new_A[j] - new_A[i])
                    dB = set()
                    for i in range(len(B)):
                        for j in range(i + 1, len(B)):
                            dB.add(B[j] - B[i])
                    if not (dA_new & dB):
                        A = new_A
                        best_size = len(A) + len(B)
                        improved = True
                        continue

            # Try adding to B
            if len(B) < MAX_SIZE:
                new_B = B + [x]
                ok = True
                sums = set()
                for i in range(len(new_B)):
                    for j in range(i, len(new_B)):
                        s = new_B[i] + new_B[j]
                        if s in sums:
                            ok = False
                            break
                        sums.add(s)
                    if not ok:
                        break
                if ok:
                    dB_new = set()
                    for i in range(len(new_B)):
                        for j in range(i + 1, len(new_B)):
                            dB_new.add(new_B[j] - new_B[i])
                    dA = set()
                    for i in range(len(A)):
                        for j in range(i + 1, len(A)):
                            dA.add(A[j] - A[i])
                    if not (dA & dB_new):
                        B = new_B
                        best_size = len(A) + len(B)
                        improved = True
                        continue

        # Try swapping: remove an element from A and add a different one
        if time.time() > deadline:
            break
        for i in range(len(A)):
            if time.time() > deadline:
                break
            for x in range(1, N + 1):
                if x in A or x in B:
                    continue
                new_A = [a for j, a in enumerate(A) if j != i] + [x]
                new_A.sort()
                # Check Sidon
                ok = True
                sums = set()
                for ii in range(len(new_A)):
                    for jj in range(ii, len(new_A)):
                        s = new_A[ii] + new_A[jj]
                        if s in sums:
                            ok = False
                            break
                        sums.add(s)
                    if not ok:
                        break
                if not ok:
                    continue
                dA_new = set()
                for ii in range(len(new_A)):
                    for jj in range(ii + 1, len(new_A)):
                        dA_new.add(new_A[jj] - new_A[ii])
                dB = set()
                for ii in range(len(B)):
                    for jj in range(ii + 1, len(B)):
                        dB.add(B[jj] - B[ii])
                if dA_new & dB:
                    continue
                # Check that removing old element doesn't break B's constraints
                # (it doesn't, since B is unchanged)
                A = new_A
                best_size = len(A) + len(B)
                improved = True
                break
            if improved:
                break

        # Try swapping: remove from B and add different
        if time.time() > deadline:
            break
        if not improved:
            for i in range(len(B)):
                if time.time() > deadline:
                    break
                for x in range(1, N + 1):
                    if x in A or x in B:
                        continue
                    new_B = [b for j, b in enumerate(B) if j != i] + [x]
                    new_B.sort()
                    ok = True
                    sums = set()
                    for ii in range(len(new_B)):
                        for jj in range(ii, len(new_B)):
                            s = new_B[ii] + new_B[jj]
                            if s in sums:
                                ok = False
                                break
                            sums.add(s)
                        if not ok:
                            break
                    if not ok:
                        continue
                    dB_new = set()
                    for ii in range(len(new_B)):
                        for jj in range(ii + 1, len(new_B)):
                            dB_new.add(new_B[jj] - new_B[ii])
                    dA = set()
                    for ii in range(len(A)):
                        for jj in range(ii + 1, len(A)):
                            dA.add(A[jj] - A[ii])
                    if dA & dB_new:
                        continue
                    B = new_B
                    best_size = len(A) + len(B)
                    improved = True
                    break
                if improved:
                    break

    return A, B


def construct():
    start = time.time()
    deadline = start + DEADLINE

    best_A, best_B = [], []
    best_score = 0

    # Multiple random trials
    rng = random.Random(42)
    num_trials = 50

    for trial in range(num_trials):
        if time.time() > deadline - 5.0:
            break

        order = list(range(1, N + 1))
        rng.shuffle(order)

        A, B = _greedy_build(order, deadline - 5.0)

        # Local search
        if time.time() < deadline - 3.0:
            A, B = _local_search(A, B, deadline - 3.0)

        if _validate(A, B):
            score = len(A) + len(B)
            if score > best_score:
                best_score = score
                best_A, best_B = A, B

    # If nothing found, use deterministic fallback
    if not best_A or not best_B:
        best_A, best_B = _greedy_build(list(range(1, N + 1)), deadline)

    # Final safety: ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]

    return [best_A, best_B]
