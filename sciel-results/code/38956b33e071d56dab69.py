"""Rewritten program for two Sidon sets with disjoint difference sets."""
# EVOLVE-BLOCK-START
import time
import random

N = 256
MAX_EACH = 64
DEADLINE = 50.0  # seconds (Safe margin for 60s/90s limits)


def _validate(A, B):
    """Validate that A and B are valid Sidon sets with disjoint nonzero differences."""
    if not A or not B:
        return False
    if len(A) > MAX_EACH or len(B) > MAX_EACH:
        return False
    # Check Sidon property for A
    sumsA = set()
    for i in range(len(A)):
        for j in range(i, len(A)):
            s = A[i] + A[j]
            if s in sumsA:
                return False
            sumsA.add(s)
    # Check Sidon property for B
    sumsB = set()
    for i in range(len(B)):
        for j in range(i, len(B)):
            s = B[i] + B[j]
            if s in sumsB:
                return False
            sumsB.add(s)
    # Check difference disjointness
    diffsA = set()
    for i in range(len(A)):
        for j in range(len(A)):
            if i != j:
                diffsA.add(A[j] - A[i])
    diffsB = set()
    for i in range(len(B)):
        for j in range(len(B)):
            if i != j:
                diffsB.add(B[j] - B[i])
    if diffsA & diffsB:
        return False
    # Check all elements in [1, N]
    for x in A:
        if x < 1 or x > N:
            return False
    for x in B:
        if x < 1 or x > N:
            return False
    return True


def _fast_greedy(t0, rng, seed_offset=0):
    """
    Fast greedy builder using incremental sum sets and bitmask differences.
    """
    A = []
    B = []
    sumsA = set()
    sumsB = set()
    diffsA = 0  # Bitmask for differences in A
    diffsB = 0  # Bitmask for differences in B

    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)

    # To add variation, we can bias the preference
    prefer_A = rng.random() < 0.5

    for x in candidates:
        if time.time() - t0 > DEADLINE - 2:
            break

        added = False

        # Determine order of sides to try
        if prefer_A:
            sides = [0, 1]
        else:
            sides = [1, 0]

        for side in sides:
            if side == 0:
                if len(A) >= MAX_EACH:
                    continue
                # Check Sidon condition for A with x
                valid_sidon_A = True
                if (x + x) in sumsA:
                    valid_sidon_A = False
                else:
                    for a in A:
                        if (x + a) in sumsA:
                            valid_sidon_A = False
                            break

                if valid_sidon_A:
                    # Check difference constraints
                    new_diffs_mask = 0
                    for a in A:
                        d = abs(a - x)
                        if d > 0:
                            new_diffs_mask |= (1 << d)

                    if not (new_diffs_mask & diffsB):
                        # Valid! Add x to A
                        A.append(x)
                        sumsA.add(x + x)
                        for a in A[:-1]:
                            sumsA.add(x + a)
                        diffsA |= new_diffs_mask
                        added = True
                        break
            else:
                if len(B) >= MAX_EACH:
                    continue
                # Check Sidon condition for B with x
                valid_sidon_B = True
                if (x + x) in sumsB:
                    valid_sidon_B = False
                else:
                    for b in B:
                        if (x + b) in sumsB:
                            valid_sidon_B = False
                            break

                if valid_sidon_B:
                    new_diffs_mask = 0
                    for b in B:
                        d = abs(b - x)
                        if d > 0:
                            new_diffs_mask |= (1 << d)

                    if not (new_diffs_mask & diffsA):
                        # Valid! Add x to B
                        B.append(x)
                        sumsB.add(x + x)
                        for b in B[:-1]:
                            sumsB.add(x + b)
                        diffsB |= new_diffs_mask
                        added = True
                        break

        # If not added, skip x

    return A, B


def _local_improve(A, B, t0):
    """
    Try to improve A and B by adding new ones.
    Uses bitmask diffs and incremental sums for speed.
    """
    if not A or not B:
        return A, B

    # Recompute state
    sumsA = set()
    sumsB = set()
    diffsA = 0
    diffsB = 0

    for x in A:
        for y in A:
            if x <= y:
                sumsA.add(x + y)
            if x != y:
                d = abs(x - y)
                diffsA |= (1 << d)
    for x in B:
        for y in B:
            if x <= y:
                sumsB.add(x + y)
            if x != y:
                d = abs(x - y)
                diffsB |= (1 << d)

    used = set(A) | set(B)

    # Try to add more elements
    for x in range(1, N + 1):
        if time.time() - t0 > DEADLINE - 2:
            break
        if x in used:
            continue

        # Try A
        if len(A) < MAX_EACH:
            valid = True
            if (x + x) in sumsA:
                valid = False
            else:
                for a in A:
                    if (x + a) in sumsA:
                        valid = False
                        break
            if valid:
                new_diffs_mask = 0
                for a in A:
                    d = abs(a - x)
                    if d > 0:
                        new_diffs_mask |= (1 << d)
                if not (new_diffs_mask & diffsB):
                    A.append(x)
                    used.add(x)
                    sumsA.add(x + x)
                    for a in A[:-1]:
                        sumsA.add(x + a)
                    diffsA |= new_diffs_mask
                    continue

        # Try B
        if len(B) < MAX_EACH:
            valid = True
            if (x + x) in sumsB:
                valid = False
            else:
                for b in B:
                    if (x + b) in sumsB:
                        valid = False
                        break
            if valid:
                new_diffs_mask = 0
                for b in B:
                    d = abs(b - x)
                    if d > 0:
                        new_diffs_mask |= (1 << d)
                if not (new_diffs_mask & diffsA):
                    B.append(x)
                    used.add(x)
                    sumsB.add(x + x)
                    for b in B[:-1]:
                        sumsB.add(x + b)
                    diffsB |= new_diffs_mask
                    continue

    return A, B


def construct():
    t0 = time.time()
    rng = random.Random(42)

    best_A, best_B = [], []
    best_size = 0

    # Run multiple fast greedy restarts
    # Since each restart is fast, we can run many more.
    # Let's loop until deadline.
    
    i = 0
    while time.time() - t0 < DEADLINE - 2:
        # Use a different seed for each restart to vary the shuffle
        r = random.Random(42 + i)
        A, B = _fast_greedy(t0, r)
        
        if not A or not B:
            i += 1
            continue
        
        # Local improvement
        A, B = _local_improve(A, B, t0)
        
        if _validate(A, B) and len(A) + len(B) > best_size:
            best_size = len(A) + len(B)
            best_A, best_B = list(A), list(B)
            
            # If we hit the max possible, we can stop early? Max is 128.
            if best_size == 128:
                break
        
        i += 1

    # Ensure nonempty and valid
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]

    if not _validate(best_A, best_B):
        # Fallback
        best_A = [1]
        best_B = [2]

    return [best_A, best_B]
# EVOLVE-BLOCK-END
