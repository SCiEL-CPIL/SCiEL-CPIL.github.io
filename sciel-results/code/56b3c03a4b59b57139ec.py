# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit
    n = 256
    elements = list(range(1, n + 1))

    def _check_sidon(s):
        """Check if set s is a valid Sidon set (pair sums unique including a+a)."""
        seen = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sm = s[i] + s[j]
                if sm in seen:
                    return False
                seen.add(sm)
        return True

    def _check_diffs_disjoint(A, B):
        """Check if nonzero difference sets of A and B are disjoint."""
        diffs_a = set()
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    diffs_a.add(A[j] - A[i])
        diffs_b = set()
        for i in range(len(B)):
            for j in range(len(B)):
                if i != j:
                    diffs_b.add(B[j] - B[i])
        return not (diffs_a & diffs_b)

    def _greedy_one_ordering(order, deadline):
        """
        Single forward pass over the given ordering.
        Uses in-place tracking for O(n^2) per pass.
        Returns (setA, setB) as lists.
        """
        setA = []
        setB = []
        sumsA = set()
        sumsB = set()
        diffsA = set()
        diffsB = set()

        for x in order:
            if time.time() > deadline:
                break
            for side in (0, 1):
                if side == 0:
                    s = setA
                    sums = sumsA
                    diffs_own = diffsA
                    diffs_other = diffsB
                else:
                    s = setB
                    sums = sumsB
                    diffs_own = diffsB
                    diffs_other = diffsA

                # Check Sidon: all pair sums with x must be unique and not in existing sums
                ok = True
                new_sums = []
                for a in s:
                    sm = a + x
                    if sm in sums:
                        ok = False
                        break
                    new_sums.append(sm)
                # Also x + x
                sm = x + x
                if sm in sums:
                    ok = False
                else:
                    new_sums.append(sm)

                if not ok:
                    continue

                # Check differences: new diffs from x with existing elements must not conflict
                new_diffs = set()
                for a in s:
                    new_diffs.add(x - a)
                    new_diffs.add(a - x)

                # Check against own diffs (should be empty since we track all)
                if new_diffs & diffs_own:
                    continue

                # Check against other set's diffs
                if new_diffs & diffs_other:
                    continue

                # Add x to this set
                s.append(x)
                sums.update(new_sums)
                diffs_own.update(new_diffs)

        return setA, setB

    def _local_search(setA, setB, deadline):
        """
        Try to improve by adding elements using smarter candidate selection.
        Returns (setA, setB).
        """
        setA = list(setA)
        setB = list(setB)
        
        # Maintain incremental state
        sumsA = set()
        for i in range(len(setA)):
            for j in range(i, len(setA)):
                sumsA.add(setA[i] + setA[j])
        sumsB = set()
        for i in range(len(setB)):
            for j in range(i, len(setB)):
                sumsB.add(setB[i] + setB[j])
        
        diffsA = set()
        for i in range(len(setA)):
            for j in range(len(setA)):
                if i != j:
                    diffsA.add(setA[j] - setA[i])
        diffsB = set()
        for i in range(len(setB)):
            for j in range(len(setB)):
                if i != j:
                    diffsB.add(setB[j] - setB[i])
        
        improved = True
        max_iterations = 500
        iteration = 0
        
        while improved and time.time() < deadline and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Alternate between trying to add to A and B
            for side in (0, 1):
                if time.time() > deadline:
                    return setA, setB
                
                if side == 0:
                    s = setA
                    sums = sumsA
                    diffs_own = diffsA
                    diffs_other = diffsB
                else:
                    s = setB
                    sums = sumsB
                    diffs_own = diffsB
                    diffs_other = diffsA

                # Find valid candidates
                in_s = set(s)
                candidates = []
                for x in elements:
                    if x in in_s:
                        continue
                    
                    # Check Sidon: all pair sums with x must be unique and not in existing sums
                    ok = True
                    new_sums = []
                    for a in s:
                        sm = a + x
                        if sm in sums:
                            ok = False
                            break
                        new_sums.append(sm)
                    sm = x + x
                    if sm in sums:
                        ok = False
                    else:
                        new_sums.append(sm)

                    if not ok:
                        continue

                    # Check differences: new diffs from x with existing elements must not conflict
                    new_diffs = set()
                    for a in s:
                        new_diffs.add(x - a)
                        new_diffs.add(a - x)

                    if new_diffs & diffs_own:
                        continue
                    if new_diffs & diffs_other:
                        continue

                    # Score: prefer candidates that minimize future conflicts
                    # Heuristic: count how many other elements would become invalid
                    # if we add x (approximate by checking a few random others)
                    conflict_count = 0
                    for y in elements:
                        if y in in_s or y == x:
                            continue
                        # Quick check: would y conflict with x in terms of sums?
                        # (This is expensive, so we use a lighter heuristic)
                        # Just count if y-x or x-y is already a diff
                        if (x - y) in diffs_own or (y - x) in diffs_own:
                            conflict_count += 1
                        if (x - y) in diffs_other or (y - x) in diffs_other:
                            conflict_count += 1
                    
                    candidates.append((conflict_count, x))

                if not candidates:
                    continue

                # Sort by conflict count (ascending)
                candidates.sort(key=lambda t: t[0])
                
                # Try the best candidate
                for _, x in candidates[:3]:
                    if time.time() > deadline:
                        break
                    
                    # Re-verify before adding
                    in_s = set(s)
                    if x in in_s:
                        continue
                    
                    # Check Sidon
                    ok = True
                    new_sums = []
                    for a in s:
                        sm = a + x
                        if sm in sums:
                            ok = False
                            break
                        new_sums.append(sm)
                    sm = x + x
                    if sm in sums:
                        ok = False
                    else:
                        new_sums.append(sm)

                    if not ok:
                        continue

                    # Check diffs
                    new_diffs = set()
                    for a in s:
                        new_diffs.add(x - a)
                        new_diffs.add(a - x)
                    if new_diffs & diffs_own:
                        continue
                    if new_diffs & diffs_other:
                        continue

                    # Add it
                    s.append(x)
                    sums.update(new_sums)
                    diffs_own.update(new_diffs)
                    improved = True
                    break

                if improved:
                    break

        return setA, setB

    best_total = -1
    best_result = ([1], [2])

    # Try many random orderings with different strategies
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        order = elements[:]
        
        # Use different shuffling strategies
        if attempt % 3 == 0:
            random.shuffle(order)
        elif attempt % 3 == 1:
            # Sort by some hash to get diverse orderings
            random.shuffle(order)
            order.sort(key=lambda x: random.random())
        else:
            # Reverse then shuffle
            order.reverse()
            random.shuffle(order)

        setA, setB = _greedy_one_ordering(order, deadline)
        if time.time() > deadline:
            break

        # Local search to try to improve
        setA, setB = _local_search(setA, setB, deadline)

        total = len(setA) + len(setB)
        if total > best_total:
            best_total = total
            best_result = (sorted(setA), sorted(setB))

        # Early stop if we hit the max possible
        if best_total >= 128:
            break

        # If we're making progress, continue; if stuck, try a different approach
        if attempt % 10 == 0 and time.time() < deadline - 5:
            # Occasionally try a denser initial greedy
            order2 = elements[:]
            random.shuffle(order2)
            setA2, setB2 = _greedy_one_ordering(order2, deadline)
            setA2, setB2 = _local_search(setA2, setB2, deadline)
            total2 = len(setA2) + len(setB2)
            if total2 > best_total:
                best_total = total2
                best_result = (sorted(setA2), sorted(setB2))

    return list(best_result)

# EVOLVE-BLOCK-END
