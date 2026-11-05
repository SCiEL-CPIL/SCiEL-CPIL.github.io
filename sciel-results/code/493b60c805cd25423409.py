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
        Improved local search with incremental state tracking and restarts.
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
                for j in range(len(s)):
                    if i != j:
                        diffs.add(s[j] - s[i])
        
        _update_state(setA, sumsA, diffsA)
        _update_state(setB, sumsB, diffsB)

        best_total = len(setA) + len(setB)
        best_A = list(setA)
        best_B = list(setB)
        
        max_kicks = 20
        kick_count = 0
        no_improve_count = 0
        max_no_improve = 50

        while time.time() < deadline and kick_count < max_kicks:
            improved = False
            no_improve_count = 0
            
            # Try to add elements
            while time.time() < deadline:
                added = False
                # Randomly pick a side to try adding to
                # Or try both? Let's try the one with fewer elements first, or random.
                # To be fair and effective, let's iterate through candidates for both sides.
                
                # We can pick a random candidate x and try to add it to A or B
                # But scanning all candidates is O(N). 
                # Let's pick a random candidate from [1..256] not in A or B? 
                # Actually, we can add an element to A even if it's in B (sets can overlap).
                # But usually distinct is better for differences? No, overlap is allowed.
                
                # Let's try to add a random element from the full range [1, 256]
                # This allows re-adding elements if we kick them out.
                
                x = random.randint(1, 256)
                
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
                            added = True
                            improved = True
                            no_improve_count = 0
                            break
                            
                # Try adding x to B
                if not added and x not in setB:
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
                            added = True
                            improved = True
                            no_improve_count = 0
                            break
                            
                if not added:
                    no_improve_count += 1
                    if no_improve_count > max_no_improve:
                        break

            total = len(setA) + len(setB)
            if total > best_total:
                best_total = total
                best_A = list(setA)
                best_B = list(setB)
                
            if not improved:
                # Kick: remove some elements to allow reshuffling
                kick_count += 1
                # Remove 1-3 random elements from A and B
                num_remove_A = random.randint(1, 3)
                num_remove_B = random.randint(1, 3)
                
                # Remove from A
                for _ in range(min(num_remove_A, len(setA))):
                    if setA:
                        idx = random.randint(0, len(setA) - 1)
                        x = setA.pop(idx)
                        # Update state: remove x from sumsA and diffsA
                        # This is expensive. Let's just rebuild state after kick.
                        pass 
                
                # Remove from B
                for _ in range(min(num_remove_B, len(setB))):
                    if setB:
                        idx = random.randint(0, len(setB) - 1)
                        x = setB.pop(idx)
                        pass
                        
                # Rebuild state
                _update_state(setA, sumsA, diffsA)
                _update_state(setB, sumsB, diffsB)
                
                # If we kicked and didn't improve, we might want to restart greedy from scratch?
                # Or just continue local search.
                # Given the complexity of removing from state, rebuilding is safer.

        return best_A, best_B

    best_total = -1
    best_result = ([1], [2])

    # Try many random orderings
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        order = elements[:]
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

    return list(best_result)

# EVOLVE-BLOCK-END
