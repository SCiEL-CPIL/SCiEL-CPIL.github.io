import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    N = 256
    MAX_SUM = 2 * N  # 512
    MAX_DIFF = N - 1  # 255

    best_size = 0
    best_A = []
    best_B = []

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

    def build_set(order, forbidden_diffs_mask, max_size, existing_sums_mask=0, existing_set=None):
        """
        Greedily builds a Sidon set from 'order'
        """
        S = list(existing_set) if existing_set else []
        sums_mask = existing_sums_mask

        # Precompute existing diffs
        diffs_mask = 0
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d = abs(S[i] - S[j])
                diffs_mask |= (1 << d)

        S_set = set(S)

        for x in order:
            if len(S) >= max_size:
                break
            if x in S_set:
                continue

            valid = True

            # Check 2x sum
            if (sums_mask >> (2 * x)) & 1:
                valid = False

            if valid:
                # Check x+a sums
                for a in S:
                    s = x + a
                    if (sums_mask >> s) & 1:
                        valid = False
                        break

            if valid:
                # Check forbidden diffs
                for a in S:
                    d = abs(x - a)
                    if d == 0:
                        continue
                    if (forbidden_diffs_mask >> d) & 1:
                        valid = False
                        break

            if valid:
                S.append(x)
                S_set.add(x)
                for a in S[:-1]:
                    s = x + a
                    sums_mask |= (1 << s)
                sums_mask |= (1 << (2 * x))
                for a in S[:-1]:
                    d = abs(x - a)
                    if d > 0:
                        diffs_mask |= (1 << d)

        return S, sums_mask, diffs_mask

    def try_add_to_set(S, sums_mask, forbidden_mask, x):
        """
        Try to add x to S. Returns (success, new_sums_mask, new_diffs_mask)
        """
        if x in S:
            return False, sums_mask, 0

        # Check 2x
        if (sums_mask >> (2 * x)) & 1:
            return False, sums_mask, 0

        # Check x+a sums
        for a in S:
            s = x + a
            if (sums_mask >> s) & 1:
                return False, sums_mask, 0

        # Check forbidden diffs
        for a in S:
            d = abs(x - a)
            if d > 0 and (forbidden_mask >> d) & 1:
                return False, sums_mask, 0

        # Update sums
        new_sums_mask = sums_mask
        for a in S:
            new_sums_mask |= (1 << (x + a))
        new_sums_mask |= (1 << (2 * x))

        # Update diffs
        new_diffs_mask = 0
        for a in S:
            d = abs(x - a)
            if d > 0:
                new_diffs_mask |= (1 << d)

        return True, new_sums_mask, new_diffs_mask

    def polish(A, B, forbidden_A_mask, forbidden_B_mask, max_rounds=10):
        """
        Try to add elements to A and B with random orders.
        """
        A = list(A)
        B = list(B)
        A_set = set(A)
        B_set = set(B)

        sums_A = compute_sums_mask(A)
        sums_B = compute_sums_mask(B)
        diffs_A = compute_diffs_mask(A)
        diffs_B = compute_diffs_mask(B)

        for rnd in range(max_rounds):
            if time.time() > deadline:
                break

            improved = False

            # Try to add to A
            candidates_A = [x for x in range(1, N + 1) if x not in A_set]
            random.shuffle(candidates_A)
            for x in candidates_A:
                if len(A) >= 16:
                    break
                success, new_sums_A, new_diffs_A = try_add_to_set(A, sums_A, diffs_B, x)
                if success:
                    A.append(x)
                    A_set.add(x)
                    sums_A = new_sums_A
                    diffs_A |= new_diffs_A
                    improved = True
                    break

            # Try to add to B
            if not improved:
                candidates_B = [x for x in range(1, N + 1) if x not in B_set]
                random.shuffle(candidates_B)
                for x in candidates_B:
                    if len(B) >= 16:
                        break
                    success, new_sums_B, new_diffs_B = try_add_to_set(B, sums_B, diffs_A, x)
                    if success:
                        B.append(x)
                        B_set.add(x)
                        sums_B = new_sums_B
                        diffs_B |= new_diffs_B
                        improved = True
                        break

            if not improved:
                break

        return A, B

    def local_search_remove_and_readd(A, B, forbidden_A_mask, forbidden_B_mask):
        """
        Try removing an element from A or B and re-adding a different one.
        """
        A = list(A)
        B = list(B)
        A_set = set(A)
        B_set = set(B)

        sums_A = compute_sums_mask(A)
        sums_B = compute_sums_mask(B)
        diffs_A = compute_diffs_mask(A)
        diffs_B = compute_diffs_mask(B)

        improved = False

        # Try removing from A and re-adding
        for i in range(len(A)):
            x_remove = A[i]
            A_temp = A[:i] + A[i+1:]
            A_temp_set = set(A_temp)
            sums_A_temp = compute_sums_mask(A_temp)
            diffs_A_temp = compute_diffs_mask(A_temp)

            # Try to add a new element
            candidates = [x for x in range(1, N + 1) if x not in A_temp_set]
            random.shuffle(candidates)
            for x_add in candidates:
                success, _, _ = try_add_to_set(A_temp, sums_A_temp, diffs_B, x_add)
                if success:
                    A = A_temp
                    A_set = A_temp_set
                    A.append(x_add)
                    A_set.add(x_add)
                    sums_A = compute_sums_mask(A)
                    diffs_A = compute_diffs_mask(A)
                    improved = True
                    break

            if improved:
                break

        # Try removing from B and re-adding
        if not improved:
            for i in range(len(B)):
                x_remove = B[i]
                B_temp = B[:i] + B[i+1:]
                B_temp_set = set(B_temp)
                sums_B_temp = compute_sums_mask(B_temp)
                diffs_B_temp = compute_diffs_mask(B_temp)

                candidates = [x for x in range(1, N + 1) if x not in B_temp_set]
                random.shuffle(candidates)
                for x_add in candidates:
                    success, _, _ = try_add_to_set(B_temp, sums_B_temp, diffs_A, x_add)
                    if success:
                        B = B_temp
                        B_set = B_temp_set
                        B.append(x_add)
                        B_set.add(x_add)
                        sums_B = compute_sums_mask(B)
                        diffs_B = compute_diffs_mask(B)
                        improved = True
                        break

                if improved:
                    break

        return A, B, improved

    while time.time() < deadline:
        order_A = list(range(1, N + 1))
        random.shuffle(order_A)
        A, sums_A, diffs_A = build_set(order_A, 0, 16)

        if len(A) == 0:
            continue

        order_B = list(range(1, N + 1))
        random.shuffle(order_B)
        B, sums_B, diffs_B = build_set(order_B, diffs_A, 16)

        current_size = len(A) + len(B)

        if current_size > best_size or (current_size >= best_size - 1 and random.random() < 0.7):
            forbidden_B = diffs_B
            forbidden_A = diffs_A
            A_pol, B_pol = polish(A, B, forbidden_A, forbidden_B)
            current_size = len(A_pol) + len(B_pol)

            # Try local search
            A_ls, B_ls, ls_improved = local_search_remove_and_readd(A_pol, B_pol, forbidden_A, forbidden_B)
            if ls_improved:
                current_size = len(A_ls) + len(B_ls)
                A_pol = A_ls
                B_pol = B_ls

            if current_size > best_size:
                best_size = current_size
                best_A = A_pol
                best_B = B_pol

        if best_size >= 32:
            break

    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]

    return [best_A, best_B]
