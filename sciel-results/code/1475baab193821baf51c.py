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
        Improved local search with systematic candidate scanning and better kicks.
        """
        setA = list(setA)
        setB = list(setB)
        
        # State tracking
        sumsA = set()
        sumsB = set()
        diffsA = set()
        diffsB = set()
        
        def _update_state(s, sums, diffs):
            sums.clear()
            diffs.clear()
            for i in range(len(s)):
                for j in range(i, len(s)):
                    sums.add(s[i] + s[j])
            for i in range(len(s)):
                for j in range(s):
                    if i != j:
                        diffs.add(s[j] - s[i])
        
        _update_state(setA, sumsA, diffsA)
        _update_state(setB, sumsB, diffsB)

        best_total = len(setA) + len(setB)
        best_A = list(setA)
        best_B = list(setB)
        
        max_kicks = 30
        kick_count = 0
        no_improve_count = 0
        max_no_improve = 100

        while time.time() < deadline and kick_count < max_kicks:
            improved = False
            no_improve_count = 0
            
            # Systematic scan through all candidates for both sides
            candidates = list(range(1, 257))
            random.shuffle(candidates)
            
            for x in candidates:
                if time.time() > deadline:
                    break
                
                # Try adding x to A
                if x not in setA:
                    ok = True
                    new_sums = set()
                    for a in setA:
                        sm = a + x
                        if sm in sumsA or sm in new_sums:
                            ok = False
                            break
                        new_sums.add(sm)
                    sm = x + x
                    if sm in sumsA or sm in new_sums:
                        ok = False
                    else:
                        new_sums.add(sm)
                    
                    if ok:
                        new_diffs = set()
                        for a in setA:
                            new_diffs.add(x - a)
                            new_diffs.add(a - x)
                        if not (new_diffs & diffsA) and not (new_diffs & diffsB):
                            # Add to A
                            setA.append(x)
                            sumsA.update(new_sums)
                            diffsA.update(new_diffs)
                            improved = True
                            no_improve_count = 0
                            continue  # Continue scanning with updated state
                
                # Try adding x to B
                if x not in setB:
                    ok = True
                    new_sums = set()
                    for b in setB:
                        sm = b + x
                        if sm in sumsB or sm in new_sums:
                            ok = False
                            break
                        new_sums.add(sm)
                    sm = x + x
                    if sm in sumsB or sm in new_sums:
                        ok = False
                    else:
                        new_sums.add(sm)
                    
                    if ok:
                        new_diffs = set()
                        for b in setB:
                            new_diffs.add(x - b)
                            new_diffs.add(b - x)
                        if not (new_diffs & diffsB) and not (new_diffs & diffsA):
                            # Add to B
                            setB.append(x)
                            sumsB.update(new_sums)
                            diffsB.update(new_diffs)
                            improved = True
                            no_improve_count = 0
                            continue  # Continue scanning with updated state
                
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break

            total = len(setA) + len(setB)
            if total > best_total:
                best_total = total
                best_A = list(setA)
                best_B = list(setB)
                
            if not improved:
                # Kick: remove elements strategically
                kick_count += 1
                
                # Strategy: remove elements that create the most conflicts
                # For simplicity, remove random elements but try different counts
                if kick_count % 3 == 1:
                    num_remove_A = random.randint(1, 2)
                    num_remove_B = random.randint(0, 1)
                elif kick_count % 3 == 2:
                    num_remove_A = random.randint(0, 1)
                    num_remove_B = random.randint(1, 2)
                else:
                    num_remove_A = random.randint(2, 4)
                    num_remove_B = random.randint(2, 4)
                
                # Remove from A
                for _ in range(min(num_remove_A, len(setA))):
                    if setA:
                        idx = random.randint(0, len(setA) - 1)
                        setA.pop(idx)
                
                # Remove from B
                for _ in range(min(num_remove_B, len(setB))):
                    if setB:
                        idx = random.randint(0, len(setB) - 1)
                        setB.pop(idx)
                        
                # Rebuild state
                _update_state(setA, sumsA, diffsA)
                _update_state(setB, sumsB, diffsB)

        return best_A, best_B

    best_total = -1
    best_result = ([1], [2])

    # Try many random orderings with different strategies
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        
        # Different ordering strategies
        if attempt % 5 == 1:
            # Random order
            order = elements[:]
            random.shuffle(order)
        elif attempt % 5 == 2:
            # Sorted ascending
            order = elements[:]
        elif attempt % 5 == 3:
            # Sorted descending
            order = elements[::-1]
        elif attempt % 5 == 4:
            # Even then odd
            order = [x for x in elements if x % 2 == 0] + [x for x in elements if x % 2 == 1]
        else:
            # Random with bias toward spread
            order = elements[:]
            random.shuffle(order)
            # Sort by distance from median to spread out
            median = 128.5
            order.sort(key=lambda x: abs(x - median), reverse=True)

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

    return list(best_result)

# EVOLVE-BLOCK-END
