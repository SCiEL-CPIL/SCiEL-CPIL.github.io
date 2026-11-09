import time
import random

N = 256
MAX_EACH = 64

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    best_A = []
    best_B = []
    best_score = 0

    def is_valid(A, B):
        """Verify the sets are valid Sidon sets with disjoint differences."""
        if not A or not B:
            return False
        if len(A) > MAX_EACH or len(B) > MAX_EACH:
            return False

        sums_A = set()
        diffs_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums_A:
                    return False
                sums_A.add(s)
                if i != j:
                    d = abs(A[i] - A[j])
                    diffs_A.add(d)

        sums_B = set()
        diffs_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                s = B[i] + B[j]
                if s in sums_B:
                    return False
                sums_B.add(s)
                if i != j:
                    d = abs(B[i] - B[j])
                    diffs_B.add(d)

        if diffs_A & diffs_B:
            return False

        return True

    def _can_add(x, sums, diffs_own, diffs_other, elements):
        """Check if x can be added to a set with given state."""
        # Diagonal sum
        if (sums >> (2 * x)) & 1:
            return False
        for e in elements:
            if (sums >> (x + e)) & 1:
                return False
            d = abs(x - e)
            if d > 0 and (((diffs_own >> d) & 1) or ((diffs_other >> d) & 1)):
                return False
        return True

    def greedy_build(seed, strategy=0):
        """Run a single randomized greedy construction using bitsets."""
        rng = random.Random(seed)
        A = []
        B = []

        sums_A = 0
        sums_B = 0
        diffs_A = 0
        diffs_B = 0

        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)

        for x in candidates:
            if time.time() > deadline:
                break

            # Determine order based on strategy
            if strategy == 0:  # balanced
                if len(A) < len(B):
                    order = [0, 1]
                elif len(B) < len(A):
                    order = [1, 0]
                else:
                    order = [0, 1] if rng.random() < 0.5 else [1, 0]
            elif strategy == 1:  # fill A first
                if len(A) < MAX_EACH:
                    order = [0, 1]
                else:
                    order = [1, 0]
            elif strategy == 2:  # fill B first
                if len(B) < MAX_EACH:
                    order = [1, 0]
                else:
                    order = [0, 1]
            elif strategy == 3:  # random
                order = [0, 1] if rng.random() < 0.5 else [1, 0]
            else:  # density-based: prefer the set with fewer used differences
                if diffs_A.bit_count() < diffs_B.bit_count():
                    order = [0, 1]
                elif diffs_B.bit_count() < diffs_A.bit_count():
                    order = [1, 0]
                else:
                    order = [0, 1] if rng.random() < 0.5 else [1, 0]

            added = False
            for side in order:
                if side == 0:
                    if len(A) >= MAX_EACH:
                        continue
                    if _can_add(x, sums_A, diffs_A, diffs_B, A):
                        A.append(x)
                        for a in A:
                            sums_A |= (1 << (x + a))
                        for a in A:
                            d = abs(x - a)
                            if d > 0:
                                diffs_A |= (1 << d)
                        added = True
                        break
                else:
                    if len(B) >= MAX_EACH:
                        continue
                    if _can_add(x, sums_B, diffs_B, diffs_A, B):
                        B.append(x)
                        for b in B:
                            sums_B |= (1 << (x + b))
                        for b in B:
                            d = abs(x - b)
                            if d > 0:
                                diffs_B |= (1 << d)
                        added = True
                        break

        return A, B

    def local_search_polish(A, B, max_iters=1000):
        """Try to add more elements and try swaps via local search."""
        A = list(A)
        B = list(B)
        if not A or not B:
            return A, B

        # Rebuild bitsets from current A, B
        sums_A = 0
        sums_B = 0
        diffs_A = 0
        diffs_B = 0
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums_A |= (1 << (A[i] + A[j]))
                if i != j:
                    diffs_A |= (1 << abs(A[i] - A[j]))
        for i in range(len(B)):
            for j in range(i, len(B)):
                sums_B |= (1 << (B[i] + B[j]))
                if i != j:
                    diffs_B |= (1 << abs(B[i] - B[j]))

        in_A = set(A)
        in_B = set(B)
        rng = random.Random()

        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            candidates = [x for x in range(1, N + 1) if x not in in_A and x not in in_B]
            if not candidates:
                break

            # Try adding
            for _ in range(30):
                if time.time() > deadline:
                    break
                x = rng.choice(candidates)

                if len(A) < MAX_EACH and _can_add(x, sums_A, diffs_A, diffs_B, A):
                    A.append(x)
                    in_A.add(x)
                    for a in A:
                        sums_A |= (1 << (x + a))
                    for a in A:
                        d = abs(x - a)
                        if d > 0:
                            diffs_A |= (1 << d)
                    improved = True
                    break

                if len(B) < MAX_EACH and _can_add(x, sums_B, diffs_B, diffs_A, B):
                    B.append(x)
                    in_B.add(x)
                    for b in B:
                        sums_B |= (1 << (x + b))
                    for b in B:
                        d = abs(x - b)
                        if d > 0:
                            diffs_B |= (1 << d)
                    improved = True
                    break

            # Try swaps: remove one element, add another
            if not improved and (len(A) > 1 or len(B) > 1):
                for _ in range(8):
                    if time.time() > deadline:
                        break

                    # Try swapping in A
                    if len(A) > 1:
                        idx = rng.randrange(len(A))
                        removed = A[idx]
                        A.pop(idx)
                        in_A.discard(removed)

                        # Rebuild sums/diffs for A
                        sums_A_new = 0
                        diffs_A_new = 0
                        for i in range(len(A)):
                            for j in range(i, len(A)):
                                sums_A_new |= (1 << (A[i] + A[j]))
                                if i != j:
                                    diffs_A_new |= (1 << abs(A[i] - A[j]))

                        # Try to find a replacement
                        found = False
                        for x in candidates:
                            if _can_add(x, sums_A_new, diffs_A_new, diffs_B, A):
                                A.append(x)
                                in_A.add(x)
                                sums_A = 0
                                diffs_A = 0
                                for i in range(len(A)):
                                    for j in range(i, len(A)):
                                        sums_A |= (1 << (A[i] + A[j]))
                                        if i != j:
                                            diffs_A |= (1 << abs(A[i] - A[j]))
                                improved = True
                                found = True
                                break

                        if not found:
                            # Restore
                            A.insert(idx, removed)
                            in_A.add(removed)
                            sums_A = 0
                            diffs_A = 0
                            for i in range(len(A)):
                                for j in range(i, len(A)):
                                    sums_A |= (1 << (A[i] + A[j]))
                                    if i != j:
                                        diffs_A |= (1 << abs(A[i] - A[j]))

                    # Try swapping in B
                    if not improved and len(B) > 1:
                        idx = rng.randrange(len(B))
                        removed = B[idx]
                        B.pop(idx)
                        in_B.discard(removed)

                        sums_B_new = 0
                        diffs_B_new = 0
                        for i in range(len(B)):
                            for j in range(i, len(B)):
                                sums_B_new |= (1 << (B[i] + B[j]))
                                if i != j:
                                    diffs_B_new |= (1 << abs(B[i] - B[j]))

                        found = False
                        for x in candidates:
                            if _can_add(x, sums_B_new, diffs_B_new, diffs_A, B):
                                B.append(x)
                                in_B.add(x)
                                sums_B = 0
                                diffs_B = 0
                                for i in range(len(B)):
                                    for j in range(i, len(B)):
                                        sums_B |= (1 << (B[i] + B[j]))
                                        if i != j:
                                            diffs_B |= (1 << abs(B[i] - B[j]))
                                improved = True
                                found = True
                                break

                        if not found:
                            B.insert(idx, removed)
                            in_B.add(removed)
                            sums_B = 0
                            diffs_B = 0
                            for i in range(len(B)):
                                for j in range(i, len(B)):
                                    sums_B |= (1 << (B[i] + B[j]))
                                    if i != j:
                                        diffs_B |= (1 << abs(B[i] - B[j]))

                    if improved:
                        break

            if not improved:
                break

        return A, B

    def final_exhaustive_pass(A, B):
        """Try every remaining element in order, adding if possible."""
        A = list(A)
        B = list(B)
        if not A or not B:
            return A, B

        sums_A = 0
        sums_B = 0
        diffs_A = 0
        diffs_B = 0
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums_A |= (1 << (A[i] + A[j]))
                if i != j:
                    diffs_A |= (1 << abs(A[i] - A[j]))
        for i in range(len(B)):
            for j in range(i, len(B)):
                sums_B |= (1 << (B[i] + B[j]))
                if i != j:
                    diffs_B |= (1 << abs(B[i] - B[j]))

        in_A = set(A)
        in_B = set(B)

        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if x in in_A or x in in_B:
                continue
            if len(A) < MAX_EACH and _can_add(x, sums_A, diffs_A, diffs_B, A):
                A.append(x)
                in_A.add(x)
                for a in A:
                    sums_A |= (1 << (x + a))
                for a in A:
                    d = abs(x - a)
                    if d > 0:
                        diffs_A |= (1 << d)
            elif len(B) < MAX_EACH and _can_add(x, sums_B, diffs_B, diffs_A, B):
                B.append(x)
                in_B.add(x)
                for b in B:
                    sums_B |= (1 << (x + b))
                for b in B:
                    d = abs(x - b)
                    if d > 0:
                        diffs_B |= (1 << d)

        return A, B

    # Run multiple restarts with different strategies
    strategies = [0, 0, 0, 1, 2, 3, 4]  # More balanced, some specialized, density-based
    num_restarts = 300
    for i in range(num_restarts):
        if time.time() > deadline:
            break

        strategy = strategies[i % len(strategies)]
        A, B = greedy_build(i, strategy)

        # Polish
        A2, B2 = local_search_polish(A, B)

        # Final exhaustive pass
        A2, B2 = final_exhaustive_pass(A2, B2)

        # Verify
        if is_valid(A2, B2):
            score = len(A2) + len(B2)
            if score > best_score:
                best_score = score
                best_A = A2
                best_B = B2

    # Fallback
    if not best_A:
        best_A = [1]
        best_B = [3]

    return [best_A, best_B]
