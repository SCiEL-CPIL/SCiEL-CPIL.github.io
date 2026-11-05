"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 58.0
    n = 256
    max_sum = 2 * n  # 512
    max_diff = n     # 256

    def _greedy_build(order, excluded_diff_mask, deadline):
        """Greedily build a Sidon set using bitmask checks."""
        elems = []
        sum_mask = 0  # bits for sums 2..512
        diff_mask = 0  # bits for diffs 1..256

        for x in order:
            if time.time() > deadline:
                break

            # Check Sidon: for each a in elems, a+x must not be in sum_mask
            ok = True
            for a in elems:
                s = a + x
                if (sum_mask >> (s - 2)) & 1:
                    ok = False
                    break
            if ok and ((sum_mask >> (2 * x - 2)) & 1):
                ok = False
            if not ok:
                continue

            # Check diff exclusion
            for a in elems:
                d = abs(a - x)
                if d > 0 and (excluded_diff_mask >> (d - 1)) & 1:
                    ok = False
                    break
            if not ok:
                continue

            # Add x
            for a in elems:
                sum_mask |= 1 << (a + x - 2)
                d = abs(a - x)
                if d > 0:
                    diff_mask |= 1 << (d - 1)
            sum_mask |= 1 << (2 * x - 2)
            elems.append(x)

        return sorted(elems), sum_mask, diff_mask

    def _local_search(setA, setB, deadline, max_iters=5000):
        """Fast local search with inline checks."""
        A = sorted(setA)
        B = sorted(setB)
        nA = len(A)
        nB = len(B)

        # Build masks
        sumA = 0
        for i in range(nA):
            for j in range(i, nA):
                sumA |= 1 << (A[i] + A[j] - 2)
        sumB = 0
        for i in range(nB):
            for j in range(i, nB):
                sumB |= 1 << (B[i] + B[j] - 2)
        diffA = 0
        for i in range(nA):
            for j in range(i + 1, nA):
                d = abs(A[i] - A[j])
                if d > 0:
                    diffA |= 1 << (d - 1)
        diffB = 0
        for i in range(nB):
            for j in range(i + 1, nB):
                d = abs(B[i] - B[j])
                if d > 0:
                    diffB |= 1 << (d - 1)

        setA_elems = set(A)
        setB_elems = set(B)
        best_A = A[:]
        best_B = B[:]
        best_size = nA + nB

        for it in range(max_iters):
            if time.time() > deadline:
                break

            op = random.random()
            if op < 0.35:
                # Try adding to A
                x = random.randint(1, n)
                if x in setA_elems:
                    continue
                ok = True
                for a in A:
                    s = a + x
                    if (sumA >> (s - 2)) & 1:
                        ok = False
                        break
                    d = abs(a - x)
                    if d > 0 and (diffB >> (d - 1)) & 1:
                        ok = False
                        break
                if ok and ((sumA >> (2 * x - 2)) & 1):
                    ok = False
                if ok:
                    for a in A:
                        sumA |= 1 << (a + x - 2)
                        d = abs(a - x)
                        if d > 0:
                            diffA |= 1 << (d - 1)
                    sumA |= 1 << (2 * x - 2)
                    A.append(x)
                    setA_elems.add(x)
                    nA += 1
                    if nA + nB > best_size:
                        best_size = nA + nB
                        best_A = A[:]
                        best_B = B[:]
            elif op < 0.70:
                # Try adding to B
                x = random.randint(1, n)
                if x in setB_elems:
                    continue
                ok = True
                for b in B:
                    s = b + x
                    if (sumB >> (s - 2)) & 1:
                        ok = False
                        break
                    d = abs(b - x)
                    if d > 0 and (diffA >> (d - 1)) & 1:
                        ok = False
                        break
                if ok and ((sumB >> (2 * x - 2)) & 1):
                    ok = False
                if ok:
                    for b in B:
                        sumB |= 1 << (b + x - 2)
                        d = abs(b - x)
                        if d > 0:
                            diffB |= 1 << (d - 1)
                    sumB |= 1 << (2 * x - 2)
                    B.append(x)
                    setB_elems.add(x)
                    nB += 1
                    if nA + nB > best_size:
                        best_size = nA + nB
                        best_A = A[:]
                        best_B = B[:]
            elif op < 0.85 and nA > 0:
                # Move from A to B
                idx = random.randint(0, nA - 1)
                x = A[idx]
                # Remove from A
                A[idx] = A[-1]
                A.pop()
                setA_elems.discard(x)
                nA -= 1
                # Rebuild sumA and diffA (expensive but needed for correctness)
                sumA = 0
                for i in range(nA):
                    for j in range(i, nA):
                        sumA |= 1 << (A[i] + A[j] - 2)
                diffA = 0
                for i in range(nA):
                    for j in range(i + 1, nA):
                        d = abs(A[i] - A[j])
                        if d > 0:
                            diffA |= 1 << (d - 1)
                # Try add to B
                ok = True
                for b in B:
                    s = b + x
                    if (sumB >> (s - 2)) & 1:
                        ok = False
                        break
                    d = abs(b - x)
                    if d > 0 and (diffA >> (d - 1)) & 1:
                        ok = False
                        break
                if ok and ((sumB >> (2 * x - 2)) & 1):
                    ok = False
                if ok:
                    for b in B:
                        sumB |= 1 << (b + x - 2)
                        d = abs(b - x)
                        if d > 0:
                            diffB |= 1 << (d - 1)
                    sumB |= 1 << (2 * x - 2)
                    B.append(x)
                    setB_elems.add(x)
                    nB += 1
                    if nA + nB > best_size:
                        best_size = nA + nB
                        best_A = A[:]
                        best_B = B[:]
                else:
                    # Revert: put x back in A
                    A.append(x)
                    setA_elems.add(x)
                    nA += 1
                    sumA = 0
                    for i in range(nA):
                        for j in range(i, nA):
                            sumA |= 1 << (A[i] + A[j] - 2)
                    diffA = 0
                    for i in range(nA):
                        for j in range(i + 1, nA):
                            d = abs(A[i] - A[j])
                            if d > 0:
                                diffA |= 1 << (d - 1)
            else:
                # Move from B to A
                if nB == 0:
                    continue
                idx = random.randint(0, nB - 1)
                x = B[idx]
                B[idx] = B[-1]
                B.pop()
                setB_elems.discard(x)
                nB -= 1
                sumB = 0
                for i in range(nB):
                    for j in range(i, nB):
                        sumB |= 1 << (B[i] + B[j] - 2)
                diffB = 0
                for i in range(nB):
                    for j in range(i + 1, nB):
                        d = abs(B[i] - B[j])
                        if d > 0:
                            diffB |= 1 << (d - 1)
                # Try add to A
                ok = True
                for a in A:
                    s = a + x
                    if (sumA >> (s - 2)) & 1:
                        ok = False
                        break
                    d = abs(a - x)
                    if d > 0 and (diffB >> (d - 1)) & 1:
                        ok = False
                        break
                if ok and ((sumA >> (2 * x - 2)) & 1):
                    ok = False
                if ok:
                    for a in A:
                        sumA |= 1 << (a + x - 2)
                        d = abs(a - x)
                        if d > 0:
                            diffA |= 1 << (d - 1)
                    sumA |= 1 << (2 * x - 2)
                    A.append(x)
                    setA_elems.add(x)
                    nA += 1
                    if nA + nB > best_size:
                        best_size = nA + nB
                        best_A = A[:]
                        best_B = B[:]
                else:
                    B.append(x)
                    setB_elems.add(x)
                    nB += 1
                    sumB = 0
                    for i in range(nB):
                        for j in range(i, nB):
                            sumB |= 1 << (B[i] + B[j] - 2)
                    diffB = 0
                    for i in range(nB):
                        for j in range(i + 1, nB):
                            d = abs(B[i] - B[j])
                            if d > 0:
                                diffB |= 1 << (d - 1)

        return best_A, best_B

    best_sets = ([1], [2])
    best_size = 2

    rng = random.Random(42)
    full_range = list(range(1, n + 1))

    for restart in range(2000):
        if time.time() > deadline:
            break

        # Build A greedily with random order
        order_A = full_range[:]
        rng.shuffle(order_A)
        setA, _, diffA_mask = _greedy_build(order_A, 0, deadline)

        if not setA:
            continue

        # Build B avoiding A's differences
        order_B = full_range[:]
        rng.shuffle(order_B)
        setB, _, _ = _greedy_build(order_B, diffA_mask, deadline)

        if not setB:
            continue

        current_size = len(setA) + len(setB)

        if current_size > best_size:
            # Run local search to try to improve
            setA_ls, setB_ls = _local_search(setA, setB, deadline, max_iters=3000)
            ls_size = len(setA_ls) + len(setB_ls)
            if ls_size > best_size:
                best_sets = (setA_ls, setB_ls)
                best_size = ls_size
            elif current_size > best_size:
                best_sets = (setA, setB)
                best_size = current_size
        elif current_size >= best_size - 2 and restart % 3 == 0:
            # Occasionally try local search on near-best solutions
            setA_ls, setB_ls = _local_search(setA, setB, deadline, max_iters=1500)
            ls_size = len(setA_ls) + len(setB_ls)
            if ls_size > best_size:
                best_sets = (setA_ls, setB_ls)
                best_size = ls_size

    return [best_sets[0], best_sets[1]]
# EVOLVE-BLOCK-END
