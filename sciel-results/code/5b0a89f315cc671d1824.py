"""Direct construction of two Sidon sets with disjoint difference sets."""
import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s limit
    n = 256
    # Bitset indices: sums range [2, 512], diffs range [1, 255]
    # Use bit i to represent value (i+1) for diffs, value i for sums
    SUM_OFFSET = 0  # sum value s -> bit s
    DIFF_OFFSET = 1  # diff value d -> bit d

    def elapsed():
        return time.time() - start_time

    def is_valid_solution(A, B):
        if not A or not B:
            return False
        if len(A) > 64 or len(B) > 64:
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

        # Check disjoint differences
        diffs_A = set()
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                d = abs(A[j] - A[i])
                diffs_A.add(d)
        diffs_B = set()
        for i in range(len(B)):
            for j in range(i + 1, len(B)):
                d = abs(B[j] - B[i])
                diffs_B.add(d)

        if diffs_A & diffs_B:
            return False

        return True

    def try_insert(x, A, B, sums_A, sums_B, diffs_A, diffs_B):
        """Try to insert x into A or B, returning True if successful.
        sums_A, sums_B: bitsets where bit s is set if sum s is used
        diffs_A, diffs_B: bitsets where bit d is set if diff d is used
        """
        # Try A first if smaller, else B first
        if len(A) < len(B):
            order = [0, 1]
        elif len(B) < len(A):
            order = [1, 0]
        else:
            order = random.sample([0, 1], 2)

        for side in order:
            if side == 0:
                if len(A) >= 64:
                    continue
                # Check Sidon: for each a in A, x+a must not be in sums_A, and 2*x must not be in sums_A
                valid_sidon = True
                for a in A:
                    if (sums_A >> (x + a)) & 1:
                        valid_sidon = False
                        break
                if valid_sidon and ((sums_A >> (2 * x)) & 1):
                    valid_sidon = False
                if valid_sidon:
                    # Check diffs: for each a in A, |x-a| must not be in diffs_B
                    valid_diffs = True
                    for a in A:
                        d = abs(x - a)
                        if d != 0 and ((diffs_B >> d) & 1):
                            valid_diffs = False
                            break
                    if valid_diffs:
                        A.append(x)
                        for a in A[:-1]:
                            sums_A |= (1 << (x + a))
                            d = abs(x - a)
                            if d != 0:
                                diffs_A |= (1 << d)
                        sums_A |= (1 << (2 * x))
                        return True
            else:
                if len(B) >= 64:
                    continue
                valid_sidon = True
                for b in B:
                    if (sums_B >> (x + b)) & 1:
                        valid_sidon = False
                        break
                if valid_sidon and ((sums_B >> (2 * x)) & 1):
                    valid_sidon = False
                if valid_sidon:
                    valid_diffs = True
                    for b in B:
                        d = abs(x - b)
                        if d != 0 and ((diffs_A >> d) & 1):
                            valid_diffs = False
                            break
                    if valid_diffs:
                        B.append(x)
                        for b in B[:-1]:
                            sums_B |= (1 << (x + b))
                            d = abs(x - b)
                            if d != 0:
                                diffs_B |= (1 << d)
                        sums_B |= (1 << (2 * x))
                        return True
        return False

    def rebuild_states(A, B):
        """Rebuild bitset states from A and B."""
        sums_A = 0
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums_A |= (1 << (A[i] + A[j]))
        sums_B = 0
        for i in range(len(B)):
            for j in range(i, len(B)):
                sums_B |= (1 << (B[i] + B[j]))
        diffs_A = 0
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                d = abs(A[j] - A[i])
                diffs_A |= (1 << d)
        diffs_B = 0
        for i in range(len(B)):
            for j in range(i + 1, len(B)):
                d = abs(B[j] - B[i])
                diffs_B |= (1 << d)
        return sums_A, sums_B, diffs_A, diffs_B

    def polish(A, B):
        """Local search: try inserting skipped elements into the solution."""
        sums_A, sums_B, diffs_A, diffs_B = rebuild_states(A, B)

        in_A = set(A)
        in_B = set(B)
        used = in_A | in_B
        remaining = [x for x in range(1, n + 1) if x not in used]

        # Try multiple passes of insertion
        for pass_num in range(5):
            if elapsed() > deadline - 3.0:
                break
            improved = False
            for x in remaining[:]:
                if elapsed() > deadline - 3.0:
                    break
                if try_insert(x, A, B, sums_A, sums_B, diffs_A, diffs_B):
                    remaining.remove(x)
                    improved = True
            if not improved:
                break

        return A, B

    def swap_remove(A, B, sums_A, sums_B, diffs_A, diffs_B):
        """Try to remove an element from the larger set and insert into the smaller."""
        if not A or not B:
            return False
        if len(A) <= len(B):
            src, dst = A, B
            src_is_A = True
        else:
            src, dst = B, A
            src_is_A = False

        for idx in range(len(src) - 1, -1, -1):
            x = src[idx]
            # Check if x can be added to dst (Sidon + diff constraints)
            valid_sidon = True
            for d in dst:
                if (sums_B if src_is_A else sums_A) and False:
                    pass
            # Determine which sums/diffs belong to dst
            if src_is_A:
                dst_sums = sums_B
                src_diffs = diffs_A
            else:
                dst_sums = sums_A
                src_diffs = diffs_B

            for d in dst:
                if (dst_sums >> (x + d)) & 1:
                    valid_sidon = False
                    break
            if valid_sidon and ((dst_sums >> (2 * x)) & 1):
                valid_sidon = False
            if not valid_sidon:
                continue

            valid_diffs = True
            for d in dst:
                diff = abs(x - d)
                if diff != 0 and ((src_diffs >> diff) & 1):
                    valid_diffs = False
                    break
            if not valid_diffs:
                continue

            # Compute new diffs for dst after adding x
            if src_is_A:
                dst_diffs = diffs_B
            else:
                dst_diffs = diffs_A

            new_dst_diffs = dst_diffs
            for d in dst:
                diff = abs(x - d)
                if diff != 0:
                    new_dst_diffs |= (1 << diff)

            # Compute new diffs for src after removing x
            new_src_diffs = 0
            src_list = list(src)
            for i in range(len(src_list)):
                for j in range(i + 1, len(src_list)):
                    if i == idx or j == idx:
                        continue
                    new_src_diffs |= (1 << abs(src_list[j] - src_list[i]))

            # Check that new diffs are still disjoint
            if new_src_diffs & new_dst_diffs:
                continue

            # Perform the swap
            src.pop(idx)
            # Recompute src sums
            new_src_sums = 0
            for i in range(len(src)):
                for j in range(i, len(src)):
                    new_src_sums |= (1 << (src[i] + src[j]))

            dst.append(x)
            if src_is_A:
                new_dst_sums = sums_B
            else:
                new_dst_sums = sums_A
            for d in dst[:-1]:
                new_dst_sums |= (1 << (x + d))
            new_dst_sums |= (1 << (2 * x))

            # Update the shared sets
            if src_is_A:
                sums_A = new_src_sums
                sums_B = new_dst_sums
                diffs_A = new_src_diffs
                diffs_B = new_dst_diffs
            else:
                sums_B = new_src_sums
                sums_A = new_dst_sums
                diffs_B = new_src_diffs
                diffs_A = new_dst_diffs

            return True, sums_A, sums_B, diffs_A, diffs_B
        return False, sums_A, sums_B, diffs_A, diffs_B

    best_score = 0
    best_sets = ([1], [2])  # Fallback

    # Randomized Greedy with restarts
    for restart in range(5000):
        if elapsed() > deadline - 5.0:
            break

        A = []
        B = []
        sums_A = 0
        sums_B = 0
        diffs_A = 0
        diffs_B = 0

        # Candidate ordering: mix of random and low-first strategies
        if restart % 3 == 0:
            candidates = list(range(1, n + 1))
            random.shuffle(candidates)
        elif restart % 3 == 1:
            # Low-first with some randomization
            candidates = list(range(1, n + 1))
            random.shuffle(candidates)
            candidates.sort(key=lambda x: random.random() * x)
        else:
            # Pure random
            candidates = list(range(1, n + 1))
            random.shuffle(candidates)

        for x in candidates:
            if elapsed() > deadline - 5.0:
                break

            try_insert(x, A, B, sums_A, sums_B, diffs_A, diffs_B)

        if A and B:
            # Polish pass
            A, B = polish(A, B)

            # Rebuild states for swap operations
            sums_A, sums_B, diffs_A, diffs_B = rebuild_states(A, B)

            # Local search: try swaps to rebalance
            if elapsed() < deadline - 5.0:
                for _ in range(20):
                    if elapsed() > deadline - 5.0:
                        break
                    result = swap_remove(A, B, sums_A, sums_B, diffs_A, diffs_B)
                    if not result[0]:
                        break
                    sums_A, sums_B, diffs_A, diffs_B = result[1], result[2], result[3], result[4]
                # Re-polish after swaps
                A, B = polish(A, B)

            if is_valid_solution(A, B):
                score = len(A) + len(B)
                if score > best_score:
                    best_score = score
                    best_sets = (A[:], B[:])

    return list(best_sets)
