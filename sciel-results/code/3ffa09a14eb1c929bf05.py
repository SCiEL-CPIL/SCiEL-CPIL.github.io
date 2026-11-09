import time
import random

N = 256
MAX_EACH = 64

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

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
        if (sums >> (x + x)) & 1:
            return False
        for e in elements:
            if (sums >> (x + e)) & 1:
                return False
            d = abs(x - e)
            if d > 0 and (((diffs_own >> d) & 1) or ((diffs_other >> d) & 1)):
                return False
        return True

    def _rebuild_sums_diffs(S):
        """Rebuild sums and diffs bitsets from a set."""
        sums = 0
        diffs = 0
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums |= (1 << (S[i] + S[j]))
                if i != j:
                    diffs |= (1 << abs(S[i] - S[j]))
        return sums, diffs

    def greedy_build(seed, strategy=0):
        """Run a single randomized greedy construction using bitsets.
        strategy: 0=balanced, 1=fill_A_first, 2=fill_B_first, 3=random_order,
                  4=spaced (prefer large gaps), 5=low-first
        """
        rng = random.Random(seed)
        A = []
        B = []

        sums_A = 0
        sums_B = 0
        diffs_A = 0
        diffs_B = 0

        if strategy == 5:
            candidates = list(range(1, N + 1))
        elif strategy == 4:
            # Prefer elements spaced out: sort by value but with random tie-break
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
            # Sort to favor spread: interleave low and high
            half = N // 2
            low = [x for x in candidates if x <= half]
            high = [x for x in candidates if x > half]
            candidates = []
            i = j = 0
            while i < len(low) or j < len(high):
                if i < len(low):
                    candidates.append(low[i])
                    i += 1
                if j < len(high):
                    candidates.append(high[j])
                    j += 1
        else:
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)

        for x in candidates:
            if time.time() > deadline:
                break

            # Determine order based on strategy
            if strategy in (0, 4):  # balanced
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
            elif strategy == 5:  # low-first, balanced
                if len(A) < len(B):
                    order = [0, 1]
                elif len(B) < len(A):
                    order = [1, 0]
                else:
                    order = [0, 1] if rng.random() < 0.5 else [1, 0]
            else:  # random
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

    def local_search_polish(A, B, max_iters=300):
        """Try to add more elements and try swaps via local search with incremental updates."""
        A = list(A)
        B = list(B)
        if not A or not B:
            return A, B

        # Rebuild bitsets from current A, B
        sums_A, diffs_A = _rebuild_sums_diffs(A)
        sums_B, diffs_B = _rebuild_sums_diffs(B)

        in_A = set(A)
        in_B = set(B)
        all_elements = in_A | in_B
        rng = random.Random()

        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            candidates = [x for x in range(1, N + 1) if x not in all_elements]
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
                    all_elements.add(x)
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
                    all_elements.add(x)
                    for b in B:
                        sums_B |= (1 << (x + b))
                    for b in B:
                        d = abs(x - b)
                        if d > 0:
                            diffs_B |= (1 << d)
                    improved = True
                    break

            # Try swaps: remove one element from A or B, add another
            if not improved:
                for _ in range(10):
                    if time.time() > deadline:
                        break

                    # Choose which side to swap
                    if len(A) > 1 and (len(B) <= 1 or rng.random() < 0.5):
                        # Try swapping in A
                        idx = rng.randrange(len(A))
                        removed = A[idx]
                        A.pop(idx)
                        in_A.discard(removed)
                        all_elements.discard(removed)

                        # Rebuild sums/diffs for A after removal
                        sums_A_new, diffs_A_new = _rebuild_sums_diffs(A)

                        # Try to find a replacement
                        found = False
                        for x in candidates:
                            if x not in all_elements and _can_add(x, sums_A_new, diffs_A_new, diffs_B, A):
                                A.append(x)
                                in_A.add(x)
                                all_elements.add(x)
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
                            all_elements.add(removed)
                            sums_A, diffs_A = _rebuild_sums_diffs(A)

                    elif len(B) > 1:
                        # Try swapping in B
                        idx = rng.randrange(len(B))
                        removed = B[idx]
                        B.pop(idx)
                        in_B.discard(removed)
                        all_elements.discard(removed)

                        sums_B_new, diffs_B_new = _rebuild_sums_diffs(B)

                        found = False
                        for x in candidates:
                            if x not in all_elements and _can_add(x, sums_B_new, diffs_B_new, diffs_A, B):
                                B.append(x)
                                in_B.add(x)
                                all_elements.add(x)
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
                            all_elements.add(removed)
                            sums_B, diffs_B = _rebuild_sums_diffs(B)

                    if improved:
                        break

            # Try 2-swap: remove two from one side, add two new
            if not improved and (len(A) > 2 or len(B) > 2):
                for _ in range(3):
                    if time.time() > deadline:
                        break

                    if len(A) > 2 and (len(B) <= 2 or rng.random() < 0.5):
                        # Remove two from A
                        idx1 = rng.randrange(len(A))
                        removed1 = A[idx1]
                        A.pop(idx1)
                        in_A.discard(removed1)
                        all_elements.discard(removed1)
                        idx2 = rng.randrange(len(A))
                        removed2 = A[idx2]
                        A.pop(idx2)
                        in_A.discard(removed2)
                        all_elements.discard(removed2)

                        sums_A_new, diffs_A_new = _rebuild_sums_diffs(A)

                        # Try to find two replacements
                        found1 = False
                        found2 = False
                        for x1 in candidates:
                            if x1 not in all_elements and _can_add(x1, sums_A_new, diffs_A_new, diffs_B, A):
                                A.append(x1)
                                in_A.add(x1)
                                all_elements.add(x1)
                                found1 = True
                                # Update sums/diffs
                                for a in A:
                                    sums_A_new |= (1 << (x1 + a))
                                for a in A:
                                    d = abs(x1 - a)
                                    if d > 0:
                                        diffs_A_new |= (1 << d)
                                break
                        if found1:
                            for x2 in candidates:
                                if x2 not in all_elements and x2 != x1 and _can_add(x2, sums_A_new, diffs_A_new, diffs_B, A):
                                    A.append(x2)
                                    in_A.add(x2)
                                    all_elements.add(x2)
                                    found2 = True
                                    sums_A = 0
                                    diffs_A = 0
                                    for i in range(len(A)):
                                        for j in range(i, len(A)):
                                            sums_A |= (1 << (A[i] + A[j]))
                                            if i != j:
                                                diffs_A |= (1 << abs(A[i] - A[j]))
                                    improved = True
                                    break
                        if not (found1 and found2):
                            # Restore
                            A.append(removed1)
                            in_A.add(removed1)
                            all_elements.add(removed1)
                            A.append(removed2)
                            in_A.add(removed2)
                            all_elements.add(removed2)
                            sums_A, diffs_A = _rebuild_sums_diffs(A)

                    elif len(B) > 2:
                        # Remove two from B
                        idx1 = rng.randrange(len(B))
                        removed1 = B[idx1]
                        B.pop(idx1)
                        in_B.discard(removed1)
                        all_elements.discard(removed1)
                        idx2 = rng.randrange(len(B))
                        removed2 = B[idx2]
                        B.pop(idx2)
                        in_B.discard(removed2)
                        all_elements.discard(removed2)

                        sums_B_new, diffs_B_new = _rebuild_sums_diffs(B)

                        found1 = False
                        found2 = False
                        for x1 in candidates:
                            if x1 not in all_elements and _can_add(x1, sums_B_new, diffs_B_new, diffs_A, B):
                                B.append(x1)
                                in_B.add(x1)
                                all_elements.add(x1)
                                found1 = True
                                for b in B:
                                    sums_B_new |= (1 << (x1 + b))
                                for b in B:
                                    d = abs(x1 - b)
                                    if d > 0:
                                        diffs_B_new |= (1 << d)
                                break
                        if found1:
                            for x2 in candidates:
                                if x2 not in all_elements and x2 != x1 and _can_add(x2, sums_B_new, diffs_B_new, diffs_A, B):
                                    B.append(x2)
                                    in_B.add(x2)
                                    all_elements.add(x2)
                                    found2 = True
                                    sums_B = 0
                                    diffs_B = 0
                                    for i in range(len(B)):
                                        for j in range(i, len(B)):
                                            sums_B |= (1 << (B[i] + B[j]))
                                            if i != j:
                                                diffs_B |= (1 << abs(B[i] - B[j]))
                                    improved = True
                                    break
                        if not (found1 and found2):
                            B.append(removed1)
                            in_B.add(removed1)
                            all_elements.add(removed1)
                            B.append(removed2)
                            in_B.add(removed2)
                            all_elements.add(removed2)
                            sums_B, diffs_B = _rebuild_sums_diffs(B)

                    if improved:
                        break

            if not improved:
                break

        return A, B

    # Run multiple restarts with different strategies
    strategies = [0, 0, 0, 1, 2, 3, 4, 5]
    num_restarts = 300
    for i in range(num_restarts):
        if time.time() > deadline:
            break

        strategy = strategies[i % len(strategies)]
        A, B = greedy_build(i, strategy)

        # Polish
        A2, B2 = local_search_polish(A, B)

        # Verify
        if is_valid(A2, B2):
            score = len(A2) + len(B2)
            if score > best_score:
                best_score = score
                best_A = A2
                best_B = B2

    # Final polish on best found
    if best_A and best_B:
        A_final, B_final = local_search_polish(best_A, best_B, max_iters=200)
        if is_valid(A_final, B_final):
            if len(A_final) + len(B_final) > best_score:
                best_A = A_final
                best_B = B_final
                best_score = len(A_final) + len(B_final)

    # Fallback
    if not best_A:
        best_A = [1]
        best_B = [3]

    return [best_A, best_B]
