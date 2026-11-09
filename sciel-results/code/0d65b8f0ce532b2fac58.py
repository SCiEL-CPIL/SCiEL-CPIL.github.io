import time
import random
import math


def construct():
    start = time.time()
    DEADLINE = start + 115.0

    prescribed = [1, 2, 5, 11]
    pres_set = set(prescribed)
    N = 284
    random.seed(777)

    def remaining():
        return DEADLINE - time.time()

    def is_valid_set(s):
        if len(s) != len(set(s)):
            return False
        if not pres_set.issubset(set(s)):
            return False
        for v in s:
            if v < 1 or v > N:
                return False
        seen = set()
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                v = s[i] + s[j]
                if v in seen:
                    return False
                seen.add(v)
        return True

    # ---------- Greedy baseline ----------
    def greedy_build():
        arr = list(prescribed)
        used = set(arr)
        sums = set()
        for a in arr:
            for b in arr:
                if a <= b:
                    sums.add(a + b)
        for x in range(1, N + 1):
            if x in used:
                continue
            ok = True
            new_sums = []
            for a in arr:
                s = a + x
                if s in sums:
                    ok = False
                    break
                new_sums.append(s)
            if ok and (x + x) in sums:
                ok = False
            if ok:
                sums.update(new_sums)
                sums.add(x + x)
                used.add(x)
                arr.append(x)
        return arr

    def greedy_build_reversed():
        """Try greedy from high to low."""
        arr = list(prescribed)
        used = set(arr)
        sums = set()
        for a in arr:
            for b in arr:
                if a <= b:
                    sums.add(a + b)
        for x in range(N, 0, -1):
            if x in used:
                continue
            ok = True
            new_sums = []
            for a in arr:
                s = a + x
                if s in sums:
                    ok = False
                    break
                new_sums.append(s)
            if ok and (x + x) in sums:
                ok = False
            if ok:
                sums.update(new_sums)
                sums.add(x + x)
                used.add(x)
                arr.append(x)
        return arr

    try:
        g1 = greedy_build()
        g2 = greedy_build_reversed()
        best = g1 if len(g1) >= len(g2) else g2
        best_size = len(best)

        # ---------- Backtracking Search ----------
        def backtrack_search(target_size, time_budget):
            """
            Attempts to find a Sidon set of size target_size containing prescribed elements.
            Uses randomized backtracking with forward checking.
            """
            end_time = time.time() + time_budget
            current_set = list(prescribed)
            current_set.sort()
            used_sums = set()
            for i in range(len(current_set)):
                for j in range(i, len(current_set)):
                    used_sums.add(current_set[i] + current_set[j])
            
            # Precompute candidates? No, dynamic is better.
            # We need to find target_size - len(current_set) more elements.
            needed = target_size - len(current_set)
            if needed <= 0:
                return list(current_set)
            
            # Candidates are 1..N not in current_set
            # We will use a list of available candidates, shuffled at each step?
            # Or just iterate 1..N and check.
            # To be efficient, we can filter candidates that are "likely" to work.
            
            def is_compatible(x):
                if x in current_set:
                    return False
                # Check sums with all existing elements
                for a in current_set:
                    if a + x in used_sums:
                        return False
                if 2 * x in used_sums:
                    return False
                return True

            def add_element(x):
                current_set.append(x)
                current_set.sort()
                for a in current_set:
                    used_sums.add(a + x)
                used_sums.add(2 * x)

            def remove_element(x):
                current_set.remove(x)
                for a in current_set:
                    used_sums.discard(a + x)
                used_sums.discard(2 * x)

            # We use a recursive function, but we need to handle time limits.
            # Python recursion depth is fine for 20.
            
            def search(depth):
                if time.time() > end_time:
                    return False
                if depth == target_size:
                    return True
                
                # Generate candidates
                # Optimization: We can sort candidates by a heuristic.
                # "Least Constraining Value": pick x that leaves the most options for future?
                # Or "Most Constrained Variable": pick x that is most restricted?
                # For now, random order is a good baseline to break symmetry.
                
                # We can pre-filter candidates to those compatible with current_set.
                candidates = [x for x in range(1, N + 1) if is_compatible(x)]
                
                if not candidates:
                    return False
                
                # Shuffle to randomize search path
                random.shuffle(candidates)
                
                for x in candidates:
                    if time.time() > end_time:
                        return False
                    # Add x
                    add_element(x)
                    if search(depth + 1):
                        return True
                    remove_element(x)
                    
                return False

            # Initial call
            if search(len(current_set)):
                return list(current_set)
            return None

        # Try to find larger sets using backtracking
        # We start from the greedy best size and try to go higher.
        # Since greedy is often suboptimal, we might try to find size 20 directly.
        # Or we can try to extend the greedy set.
        
        # Strategy:
        # 1. Try to find a set of size 20 (the theoretical max).
        # 2. If that fails, try 19, 18, etc.
        # 3. Use time budget wisely.
        
        current_best = best
        current_best_size = best_size
        
        # Allocate time for backtracking
        # We have ~100s left.
        # Let's try target 20 first with a generous budget.
        
        targets = [20, 19, 18, 17, 16]
        
        for target in targets:
            if target <= current_best_size:
                continue
            if remaining() < 5:
                break
            
            # Budget: if target is 20, give it more time.
            # If we fail, we drop the target.
            budget = min(remaining() - 2, 40.0)
            if budget < 2:
                break
                
            # We can run multiple restarts of backtracking if we have time.
            # But backtracking is deterministic if we don't shuffle.
            # We shuffle inside, so each call is different.
            
            found = backtrack_search(target, budget)
            if found is not None and is_valid_set(found):
                current_best = found
                current_best_size = target
                # If we found max size (20), we can stop.
                if target == 20:
                    break
            else:
                # If we didn't find it, maybe try again with smaller budget?
                # Or just move to next target.
                # Since backtracking is complete, if it didn't find it in budget,
                # it might be hard. But randomized restarts help.
                pass

        # Final validation
        if not is_valid_set(current_best):
            current_best = greedy_build()

        return current_best

    except Exception:
        try:
            return greedy_build()
        except Exception:
            return [1, 2, 5, 11]
