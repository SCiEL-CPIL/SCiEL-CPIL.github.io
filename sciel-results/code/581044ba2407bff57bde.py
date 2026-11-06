import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    N = 256
    MAX_EACH = 64

    def is_sidon(S):
        """Check if S is a Sidon set (all pair sums including a+a unique)."""
        if not S:
            return True
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def diff_set(S):
        """Compute set of nonzero absolute differences."""
        ds = set()
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                ds.add(abs(S[i] - S[j]))
        return ds

    def can_add_to_sidon(S, x, sums_set):
        """Check if x can be added to Sidon set S with existing sums_set."""
        if 2*x in sums_set:
            return False
        for a in S:
            if (x + a) in sums_set:
                return False
        return True

    def can_add_with_diffs(S, x, sums_set, forbidden_diffs):
        """Check if x can be added considering both Sidon and diff constraints."""
        if not can_add_to_sidon(S, x, sums_set):
            return False
        for a in S:
            d = abs(x - a)
            if d > 0 and d in forbidden_diffs:
                return False
        return True

    def greedy_build(order, forbidden_diffs, max_size, existing_set=None, existing_sums=None):
        """Greedily build a Sidon set respecting forbidden diffs."""
        if existing_set is not None:
            S = list(existing_set)
            sums_set = set(existing_sums) if existing_sums else set()
        else:
            S = []
            sums_set = set()

        S_set = set(S)

        for x in order:
            if len(S) >= max_size:
                break
            if x in S_set:
                continue

            # Check Sidon condition
            valid = True
            if 2*x in sums_set:
                valid = False
            else:
                for a in S:
                    if (x + a) in sums_set:
                        valid = False
                        break

            if valid:
                # Check diff constraint
                for a in S:
                    d = abs(x - a)
                    if d > 0 and d in forbidden_diffs:
                        valid = False
                        break

            if valid:
                S.append(x)
                S_set.add(x)
                for a in S[:-1]:
                    sums_set.add(x + a)
                sums_set.add(2*x)

        return S, sums_set

    def polish_add(A, B, diffs_A, diffs_B, max_rounds=20):
        """Try to add elements to A and B using exhaustive candidate scan."""
        A = list(A)
        B = list(B)
        A_set = set(A)
        B_set = set(B)

        sums_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums_A.add(A[i] + A[j])

        sums_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                sums_B.add(B[i] + B[j])

        for rnd in range(max_rounds):
            if time.time() > deadline:
                break

            improved = False

            # Try to add to A
            for x in range(1, N+1):
                if len(A) >= MAX_EACH:
                    break
                if x in A_set:
                    continue
                if can_add_with_diffs(A, x, sums_A, diffs_B):
                    A.append(x)
                    A_set.add(x)
                    for a in A[:-1]:
                        sums_A.add(x + a)
                    sums_A.add(2*x)
                    diffs_A = diff_set(A)
                    improved = True
                    break

            # Try to add to B
            for x in range(1, N+1):
                if len(B) >= MAX_EACH:
                    break
                if x in B_set:
                    continue
                if can_add_with_diffs(B, x, sums_B, diffs_A):
                    B.append(x)
                    B_set.add(x)
                    for a in B[:-1]:
                        sums_B.add(x + a)
                    sums_B.add(2*x)
                    diffs_B = diff_set(B)
                    improved = True
                    break

            if not improved:
                break

        return A, B, diffs_A, diffs_B

    def swap_search(A, B, diffs_A, diffs_B, max_swaps=50):
        """Try 1-swap moves: remove one from A, add one to A (or vice versa)."""
        A = list(A)
        B = list(B)
        A_set = set(A)
        B_set = set(B)

        for _ in range(max_swaps):
            if time.time() > deadline:
                break

            improved = False

            # Try removing from A and adding a different element
            for i in range(len(A)):
                x_remove = A[i]
                A_temp = A[:i] + A[i+1:]
                A_temp_set = set(A_temp)

                sums_A_temp = set()
                for j in range(len(A_temp)):
                    for k in range(j, len(A_temp)):
                        sums_A_temp.add(A_temp[j] + A_temp[k])

                # Try to add a new element to A_temp
                for x_add in range(1, N+1):
                    if x_add in A_temp_set:
                        continue
                    if can_add_with_diffs(A_temp, x_add, sums_A_temp, diffs_B):
                        # Check that the new diff set doesn't conflict with B
                        new_diffs_A = diff_set(A_temp + [x_add])
                        if new_diffs_A.isdisjoint(diffs_B):
                            A = A_temp + [x_add]
                            A_set = set(A)
                            improved = True
                            break

                if improved:
                    break

            if not improved:
                # Try removing from B and adding a different element
                for i in range(len(B)):
                    x_remove = B[i]
                    B_temp = B[:i] + B[i+1:]
                    B_temp_set = set(B_temp)

                    sums_B_temp = set()
                    for j in range(len(B_temp)):
                        for k in range(j, len(B_temp)):
                            sums_B_temp.add(B_temp[j] + B_temp[k])

                    for x_add in range(1, N+1):
                        if x_add in B_temp_set:
                            continue
                        if can_add_with_diffs(B_temp, x_add, sums_B_temp, diffs_A):
                            new_diffs_B = diff_set(B_temp + [x_add])
                            if new_diffs_B.isdisjoint(diffs_A):
                                B = B_temp + [x_add]
                                B_set = set(B)
                                improved = True
                                break

                    if improved:
                        break

            if not improved:
                break

        diffs_A = diff_set(A)
        diffs_B = diff_set(B)
        return A, B, diffs_A, diffs_B

    def two_swap(A, B, diffs_A, diffs_B, max_iters=30):
        """Try 2-swap: remove one from A and one from B, add one to each."""
        A = list(A)
        B = list(B)
        A_set = set(A)
        B_set = set(B)

        for _ in range(max_iters):
            if time.time() > deadline:
                break

            improved = False

            for i in range(len(A)):
                for j in range(len(B)):
                    A_temp = A[:i] + A[i+1:]
                    B_temp = B[:j] + B[j+1:]
                    A_temp_set = set(A_temp)
                    B_temp_set = set(B_temp)

                    sums_A_temp = set()
                    for k in range(len(A_temp)):
                        for m in range(k, len(A_temp)):
                            sums_A_temp.add(A_temp[k] + A_temp[m])

                    sums_B_temp = set()
                    for k in range(len(B_temp)):
                        for m in range(k, len(B_temp)):
                            sums_B_temp.add(B_temp[k] + B_temp[m])

                    new_diffs_A = diff_set(A_temp)
                    new_diffs_B = diff_set(B_temp)

                    # Try to add to A_temp
                    for x_add in range(1, N+1):
                        if x_add in A_temp_set:
                            continue
                        if can_add_with_diffs(A_temp, x_add, sums_A_temp, new_diffs_B):
                            new_diffs_A2 = diff_set(A_temp + [x_add])
                            if new_diffs_A2.isdisjoint(new_diffs_B):
                                # Try to add to B_temp
                                for y_add in range(1, N+1):
                                    if y_add in B_temp_set:
                                        continue
                                    if can_add_with_diffs(B_temp, y_add, sums_B_temp, new_diffs_A2):
                                        new_diffs_B2 = diff_set(B_temp + [y_add])
                                        if new_diffs_B2.isdisjoint(new_diffs_A2):
                                            A = A_temp + [x_add]
                                            B = B_temp + [y_add]
                                            improved = True
                                            break
                                if improved:
                                    break
                    if improved:
                        break

            if not improved:
                break

        diffs_A = diff_set(A)
        diffs_B = diff_set(B)
        return A, B, diffs_A, diffs_B

    best_size = 0
    best_A = []
    best_B = []

    # Multiple restarts with different strategies
    num_restarts = 200

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Vary the ordering strategy
        if restart % 4 == 0:
            order_A = list(range(1, N+1))
            random.shuffle(order_A)
            order_B = list(range(1, N+1))
            random.shuffle(order_B)
        elif restart % 4 == 1:
            # Start from low values
            order_A = list(range(1, N+1))
            random.shuffle(order_A)
            order_B = list(range(1, N+1))
            random.shuffle(order_B)
        elif restart % 4 == 2:
            # Try to build A first with preference for spread-out values
            order_A = list(range(1, N+1))
            random.shuffle(order_A)
            # Sort by some hash to get variety
            order_A.sort(key=lambda x: (x * 7 + restart) % 13)
            order_B = list(range(1, N+1))
            random.shuffle(order_B)
        else:
            order_A = list(range(1, N+1))
            random.shuffle(order_A)
            order_B = list(range(1, N+1))
            random.shuffle(order_B)

        # Build A
        A, sums_A = greedy_build(order_A, set(), MAX_EACH)
        if not A:
            continue
        diffs_A = diff_set(A)

        # Build B respecting A's diffs
        B, sums_B = greedy_build(order_B, diffs_A, MAX_EACH)
        if not B:
            continue
        diffs_B = diff_set(B)

        # Polish
        A, B, diffs_A, diffs_B = polish_add(A, B, diffs_A, diffs_B)

        current_size = len(A) + len(B)

        if current_size > best_size:
            best_size = current_size
            best_A = list(A)
            best_B = list(B)

            # Try swap search on the best
            if time.time() < deadline - 10:
                A, B, diffs_A, diffs_B = swap_search(A, B, diffs_A, diffs_B)
                current_size = len(A) + len(B)
                if current_size > best_size:
                    best_size = current_size
                    best_A = list(A)
                    best_B = list(B)

                # Try two-swap
                if time.time() < deadline - 5:
                    A, B, diffs_A, diffs_B = two_swap(A, B, diffs_A, diffs_B)
                    current_size = len(A) + len(B)
                    if current_size > best_size:
                        best_size = current_size
                        best_A = list(A)
                        best_B = list(B)

        if best_size >= 40:
            break

    # Final polish on the best solution
    if best_A and best_B:
        diffs_A = diff_set(best_A)
        diffs_B = diff_set(best_B)
        best_A, best_B, diffs_A, diffs_B = polish_add(best_A, best_B, diffs_A, diffs_B)
        best_size = len(best_A) + len(best_B)

    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]

    return [best_A, best_B]
