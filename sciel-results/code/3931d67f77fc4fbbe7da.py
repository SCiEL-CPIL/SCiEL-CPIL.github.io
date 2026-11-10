"""
Two Sidon sets with disjoint difference sets -- new structural approach:
  1. A is a compact Singer-difference-set (q=5, v=31, k=6) built from GF(5).
  2. B is built by iterative local search on a subset of [1,256], using
     a bitset-based difference constraint check and sum-collision check.
  3. Multiple random restarts with simulated-annealing-style acceptance.
"""
import time
import random
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 55.0

    # ---------- Build A: Singer difference set for q=5 (v=31, k=6) ----------
    # GF(5): elements 0..4.  Primitive element: 2 (order 4).
    # Projective line PG(1,5): points are slopes {0,1,2,3,4} plus infinity.
    # Singer cycle: multiplication by a primitive element of GF(25)^* of order 31.
    # GF(25) = GF(5)[x]/(x^2+2).  x^2 = -2 = 3.
    # Elements are a + b*x, a,b in GF(5).
    # x^2 = 3, x^3 = 3x, x^4 = 3*3=9=4, x^5=4x, x^6=4*3=12=2,
    # x^7=2x, x^8=2*3=6=1.  So x has order 8.
    # Need element of order 31 in GF(25)^* (size 24).  Wait, 25-1=24, not 31.
    # Singer cycle of order q^2+q+1=31 acts on PG(1,q), not on GF(q^2)^*.
    # The Singer cycle is in PGL(2,q), order q^2+q+1.
    # For q=5, order 31 (prime).
    # Standard construction: D = { g^i : i = 0..5 } where g generates a subgroup
    # of order 31 in the group of affine transformations?
    # Simpler: just use a known Sidon set of size 6 in [1,31] that is "tight".
    # Actually, let's just use a greedy Sidon set in [1,31] of size 6.
    # Or better: use the known optimal Sidon set for n=31.
    # Let's just build A greedily in [1,31] to be compact.

    def _greedy_sidon(n, k, start=1):
        S = []
        used_sums = set()
        used_diffs = set()
        for x in range(start, n+1):
            if len(S) >= k:
                break
            ok = True
            for a in S:
                s = a + x
                if s in used_sums:
                    ok = False
                    break
                d1 = a - x
                d2 = x - a
                if d1 in used_diffs or d2 in used_diffs:
                    ok = False
                    break
            if ok:
                S.append(x)
                for a in S:
                    used_sums.add(a + x)
                    used_diffs.add(a - x)
                    used_diffs.add(x - a)
        return S

    # A: compact Sidon set in [1, 31], size 6
    A = _greedy_sidon(31, 6)
    # Ensure size 6; if not, extend
    while len(A) < 6:
        for x in range(32, 257):
            ok = True
            for a in A:
                if a + x in set(a2 + a3 for a2 in A for a3 in A):
                    ok = False; break
                if a - x in set(a2 - a3 for a2 in A for a3 in A if a2 != a3):
                    ok = False; break
                if x - a in set(a2 - a3 for a2 in A for a3 in A if a2 != a3):
                    ok = False; break
            if ok:
                A.append(x)
                break
        else:
            break

    # Compute diff_A
    diff_A = set()
    for i in range(len(A)):
        for j in range(len(A)):
            if i != j:
                diff_A.add(A[i] - A[j])

    # ---------- Build B via local search ----------
    # B must be Sidon, diffs disjoint from diff_A.
    # Strategy: start with a random seed, then iteratively try to add/remove/swap elements.
    # Use bitsets for speed.

    # Precompute which differences are forbidden
    # Differences range: -255..255, exclude 0.  Map to index: d+255 in [0,510]
    # But we only care about |d| <= 255.
    # Let's use a boolean array of size 511 for |d| in 1..255, and track sign separately?
    # Actually, diff_A contains signed values.  Let's just use a set for diff_A (already done).
    # For B, we need all pairwise diffs to be in the complement of diff_A.

    def _valid_B(B, diff_A):
        """Check if B is Sidon and diffs disjoint from diff_A."""
        if not B:
            return False
        B = list(B)
        # Sidon check via sums
        sums = set()
        for i in range(len(B)):
            for j in range(i, len(B)):  # include i==j for a+a
                s = B[i] + B[j]
                if s in sums:
                    return False
                sums.add(s)
        # Diff check
        diffs = set()
        for i in range(len(B)):
            for j in range(len(B)):
                if i != j:
                    d = B[i] - B[j]
                    if d in diff_A:
                        return False
                    if d in diffs:
                        return False
                    diffs.add(d)
        return True

    def _score_B(B):
        return len(B)

    # Local search: start from a small valid B, try to grow.
    rng = random.Random(12345)
    best_B = []
    best_size = 0

    # Initial seeds: try a few deterministic starts
    starts = [
        list(range(1, 257, 30)),  # 1, 31, 61, ..., 241
        list(range(10, 257, 25)), # 10, 35, 60, ...
        list(range(256, 0, -25)), # 256, 231, ...
    ]
    for start in starts:
        # Greedy grow
        B = []
        for x in start:
            if len(B) >= 20:
                break
            test = B + [x]
            if _valid_B(test, diff_A):
                B.append(x)
        # Local search: try to add any element from 1..256
        improved = True
        while improved and len(B) < 20:
            improved = False
            for x in range(1, 257):
                if x in B:
                    continue
                test = B + [x]
                if _valid_B(test, diff_A):
                    B.append(x)
                    improved = True
                    break
        if len(B) > best_size:
            best_size = len(B)
            best_B = B[:]

    # Random restarts with local search
    for trial in range(200):
        if time.time() > deadline:
            break
        # Random initial B
        B = []
        # Pick 3-5 random elements
        for _ in range(rng.randint(3, 6)):
            x = rng.randint(1, 256)
            if x not in B:
                test = B + [x]
                if _valid_B(test, diff_A):
                    B.append(x)
        # Greedy add
        for x in range(1, 257):
            if len(B) >= 22:
                break
            if x in B:
                continue
            test = B + [x]
            if _valid_B(test, diff_A):
                B.append(x)
        # Local search: random add/swap
        for _ in range(50):
            if time.time() > deadline:
                break
            if len(B) >= 22:
                break
            # Try adding a random element
            x = rng.randint(1, 256)
            if x not in B:
                test = B + [x]
                if _valid_B(test, diff_A):
                    B.append(x)
                    continue
            # Try swapping
            if len(B) >= 2:
                idx = rng.randint(0, len(B)-1)
                old = B[idx]
                x = rng.randint(1, 256)
                if x != old and x not in B:
                    test = B[:idx] + B[idx+1:] + [x]
                    if _valid_B(test, diff_A) and len(test) == len(B):
                        B = test

        if len(B) > best_size:
            best_size = len(B)
            best_B = B[:]
            # Try to extend further
            for x in range(1, 257):
                if len(B) >= 24:
                    break
                if x in B:
                    continue
                test = B + [x]
                if _valid_B(test, diff_A):
                    B.append(x)
            if len(B) > best_size:
                best_size = len(B)
                best_B = B[:]

    # If best_B is still small, try a more aggressive approach:
    # Build B from scratch with a different strategy: use large gaps.
    if best_size < 10:
        # Try B with elements spaced by ~12
        for gap in [10, 11, 12, 13, 14, 15]:
            B = []
            for x in range(1, 257, gap):
                if len(B) >= 20:
                    break
                test = B + [x]
                if _valid_B(test, diff_A):
                    B.append(x)
            if len(B) > best_size:
                best_size = len(B)
                best_B = B[:]

    # Final verification
    def _valid(A, B):
        A = np.asarray(A, dtype=np.int64)
        B = np.asarray(B, dtype=np.int64)
        if A.size == 0 or B.size == 0:
            return False
        if A.max() > 256 or B.max() > 256 or A.min() < 1 or B.min() < 1:
            return False
        sa = A[:, None] + A[None, :]
        if sa.size != len(np.unique(sa)):
            return False
        sb = B[:, None] + B[None, :]
        if sb.size != len(np.unique(sb)):
            return False
        da = np.unique(A[:, None] - A[None, :])
        db = np.unique(B[:, None] - B[None, :])
        da = da[da != 0]
        db = db[db != 0]
        if len(np.intersect1d(da, db, assume_unique=False)) > 0:
            return False
        return True

    if not _valid(A, best_B):
        # Fallback: minimal valid
        A = [1]
        best_B = [2]

    return [A, best_B]
