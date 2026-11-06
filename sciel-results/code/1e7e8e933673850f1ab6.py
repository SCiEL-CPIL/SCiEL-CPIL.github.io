import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    N = 256
    MAX_SUM = 2 * N  # max sum is 512
    MAX_DIFF = N - 1  # max diff is 255

    best_size = 0
    best_A = []
    best_B = []

    def build_set(order, forbidden_diffs_mask, max_size=16, existing_sums_mask=0, existing_set=None):
        """
        Greedily builds a Sidon set from 'order'
        such that its differences do not intersect with forbidden_diffs_mask.
        Uses bitset for sum conflicts.
        """
        S = list(existing_set) if existing_set else []
        sums_mask = existing_sums_mask
        diffs_mask = 0

        # Precompute diffs for existing set
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d = abs(S[i] - S[j])
                diffs_mask |= (1 << d)

        for x in order:
            if len(S) >= max_size:
                break
            if x in S:
                continue

            # Check Sidon condition using bitmask
            # New sums introduced by x: x+a for all a in S, plus 2*x
            # A sum s is a conflict if (sums_mask >> s) & 1
            # We need: for all a in S, (x+a) not in sums_mask, AND 2*x not in sums_mask
            # AND: for all a in S, x+a != x+b for a!=b (but since x is fixed and a,b distinct, x+a != x+b automatically)
            # The only intra-new conflicts are: x+a == 2*x => a == x (impossible since x not in S)
            # So we just check: no (x+a) in sums_mask for any a, and 2*x not in sums_mask

            valid = True

            # Check 2*x
            if (sums_mask >> (2 * x)) & 1:
                valid = False

            # Check x+a for all a in S
            if valid:
                for a in S:
                    s = x + a
                    if (sums_mask >> s) & 1:
                        valid = False
                        break

            # Check diff conflicts with forbidden
            if valid:
                for a in S:
                    d = abs(x - a)
                    if d == 0:
                        continue
                    if (forbidden_diffs_mask >> d) & 1:
                        valid = False
                        break

            if valid:
                S.append(x)
                # Update sums_mask
                for a in S[:-1]:
                    s = x + a
                    sums_mask |= (1 << s)
                sums_mask |= (1 << (2 * x))
                # Update diffs_mask
                for a in S[:-1]:
                    d = abs(x - a)
                    if d > 0:
                        diffs_mask |= (1 << d)

        return S, sums_mask, diffs_mask

    def compute_diffs_mask(S):
        mask = 0
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d = abs(S[i] - S[j])
                mask |= (1 << d)
        return mask

    def compute_sums_mask(S):
        mask = 0
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                mask |= (1 << s)
        return mask

    def polish(A, B, forbidden_A_mask, forbidden_B_mask):
        """
        Try to add elements to A and B.
        Returns (A, B, forbidden_A_mask, forbidden_B_mask).
        """
        improved = True
        while improved:
            improved = False

            # Try to add to A
            sums_A = compute_sums_mask(A)
            diffs_A = compute_diffs_mask(A)
            for x in range(1, N + 1):
                if x in A:
                    continue
                if len(A) >= 16:
                    break
                # Check Sidon
                valid = True
                if (sums_A >> (2 * x)) & 1:
                    valid = False
                if valid:
                    for a in A:
                        if (sums_A >> (x + a)) & 1:
                            valid = False
                            break
                # Check diff conflicts with B
                if valid:
                    for a in A:
                        d = abs(x - a)
                        if d > 0 and (forbidden_B_mask >> d) & 1:
                            valid = False
                            break
                if valid:
                    A.append(x)
                    improved = True
                    break

            # Try to add to B
            if improved:
                continue
            sums_B = compute_sums_mask(B)
            diffs_B = compute_diffs_mask(B)
            for x in range(1, N + 1):
                if x in B:
                    continue
                if len(B) >= 16:
                    break
                valid = True
                if (sums_B >> (2 * x)) & 1:
                    valid = False
                if valid:
                    for b in B:
                        if (sums_B >> (x + b)) & 1:
                            valid = False
                            break
                if valid:
                    for b in B:
                        d = abs(x - b)
                        if d > 0 and (forbidden_A_mask >> d) & 1:
                            valid = False
                            break
                if valid:
                    B.append(x)
                    improved = True
                    break

        return A, B

    while time.time() < deadline:
        # Generate random A
        order_A = list(range(1, N + 1))
        random.shuffle(order_A)
        A, sums_A, diffs_A = build_set(order_A, 0, 16)

        if len(A) == 0:
            continue

        # Generate random B avoiding D_A
        order_B = list(range(1, N + 1))
        random.shuffle(order_B)
        B, sums_B, diffs_B = build_set(order_B, diffs_A, 16)

        current_size = len(A) + len(B)

        # Polish
        if current_size > best_size or (current_size >= best_size - 2 and random.random() < 0.3):
            forbidden_B = diffs_B
            forbidden_A = diffs_A
            A_pol, B_pol = polish(A, B, forbidden_A, forbidden_B)
            current_size = len(A_pol) + len(B_pol)
            if current_size > best_size:
                best_size = current_size
                best_A = A_pol
                best_B = B_pol

        if best_size >= 32:
            break

    # Fallback
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]

    return [best_A, best_B]
