"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, leave 10s margin for 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        Sset = set(S)
        for a in S:
            for b in S:
                if a + b in Sset:
                    return False
        return True

    def count_violations(S):
        Sset = set(S)
        count = 0
        for a in S:
            for b in S:
                if a + b in Sset:
                    count += 1
        return count

    def greedy_add(S, candidates):
        """Try to add each candidate to S if it doesn't create a violation."""
        Sset = set(S)
        for x in candidates:
            if x in Sset:
                continue
            valid = True
            for a in Sset:
                if a + x in Sset:
                    valid = False
                    break
            if valid:
                for a in Sset:
                    if x - a in Sset:
                        valid = False
                        break
            if valid:
                Sset.add(x)
                S.append(x)
        return S

    def local_search(S, all_elements, max_time=10):
        """Improve S by trying to add excluded elements."""
        Sset = set(S)
        excluded = [x for x in all_elements if x not in Sset]
        S = list(S)
        best = len(S)
        while time.time() - start_time < DEADLINE - 5:
            improved = False
            for x in excluded:
                if time.time() - start_time > DEADLINE - 5:
                    break
                has_violation = False
                for a in Sset:
                    if a + x in Sset:
                        has_violation = True
                        break
                if not has_violation:
                    for a in Sset:
                        if x - a in Sset:
                            has_violation = True
                            break
                if not has_violation:
                    Sset.add(x)
                    S.append(x)
                    excluded.remove(x)
                    improved = True
                    best = len(S)
            if not improved:
                break
        return S

    def swap_search(S, all_elements, max_time=5):
        """Try swapping elements out and in to improve size."""
        Sset = set(S)
        S = list(S)
        excluded = [x for x in all_elements if x not in Sset]
        improved = True
        while improved and time.time() - start_time < DEADLINE - 5:
            improved = False
            for x_out in list(S):
                if time.time() - start_time > DEADLINE - 5:
                    break
                # Try removing x_out
                Sset.remove(x_out)
                # Now try to add multiple excluded elements
                added_any = False
                for x_in in excluded[:]:
                    if time.time() - start_time > DEADLINE - 5:
                        break
                    has_violation = False
                    for a in Sset:
                        if a + x_in in Sset:
                            has_violation = True
                            break
                    if not has_violation:
                        for a in Sset:
                            if x_in - a in Sset:
                                has_violation = True
                                break
                    if not has_violation:
                        Sset.add(x_in)
                        S.remove(x_out)
                        S.append(x_in)
                        excluded.remove(x_in)
                        added_any = True
                        improved = True
                        break
                if not added_any:
                    Sset.add(x_out)
                    S.append(x_out)
        return S

    def max_2sat_heuristic(elements):
        """
        Formulate as Max-2-SAT: variable x_i for each element.
        For each triple (a,b,c) with a+b=c, add clause (not x_a or not x_b or not x_c).
        Actually this is 3-SAT. Let's use a different approach.
        
        Instead, use a greedy approach based on conflict graph coloring.
        Build a graph where edges connect elements that conflict.
        Then find a large independent set.
        """
        n = len(elements)
        elem_set = set(elements)
        
        # Build conflict graph: edge between a and b if a+b is in elem_set, 
        # or if there exists c in elem_set such that a+b=c (same thing)
        # Actually, the constraint is: for any a,b,c in S, a+b != c.
        # So if we include a and b, we cannot include a+b (if a+b is in elements).
        # If we include a and a+b, we cannot include b.
        # If we include b and a+b, we cannot include a.
        
        # This is a 3-uniform hypergraph independent set problem.
        # Let's use a greedy approach: repeatedly pick the element with fewest conflicts.
        
        # Precompute for each element, which pairs (a,b) have a+b = element
        # And for each element, which pairs (a,b) have element + b = a or element + a = b
        
        # Greedy: start with all elements, remove elements that cause conflicts
        S = set(elements)
        
        # Iteratively remove elements that are part of the most conflicts
        for iteration in range(100):
            if time.time() - start_time > DEADLINE - 5:
                break
            # Count conflicts for each element
            conflict_count = {x: 0 for x in S}
            for a in S:
                for b in S:
                    c = a + b
                    if c in S:
                        conflict_count[a] += 1
                        conflict_count[b] += 1
                        conflict_count[c] += 1
            
            # Remove the element with the most conflicts
            if not S:
                break
            max_conflict = max(conflict_count.values())
            if max_conflict == 0:
                break
            # Remove one element with max conflicts (random tie-break)
            candidates = [x for x in S if conflict_count[x] == max_conflict]
            x_to_remove = random.choice(candidates)
            S.remove(x_to_remove)
        
        return list(S)

    def solve_instance(elements):
        n = len(elements)
        best_S = None
        best_len = 0

        # Multiple initializations
        initializations = []
        
        # Init 1: all odds
        S1 = [x for x in elements if x % 2 != 0]
        initializations.append(S1)
        
        # Init 2: all positives > max/2
        max_val = max(elements)
        S2 = [x for x in elements if x > max_val / 2]
        initializations.append(S2)
        
        # Init 3: all negatives < min/2
        min_val = min(elements)
        S3 = [x for x in elements if x < min_val / 2]
        initializations.append(S3)
        
        # Init 4: top n/3 largest
        S4 = sorted(elements, reverse=True)[:max(1, n // 3)]
        initializations.append(S4)
        
        # Init 5: bottom n/3 smallest
        S5 = sorted(elements)[:max(1, n // 3)]
        initializations.append(S5)
        
        # Init 6: all positives
        S6 = [x for x in elements if x > 0]
        initializations.append(S6)
        
        # Init 7: all negatives
        S7 = [x for x in elements if x < 0]
        initializations.append(S7)
        
        # Init 8: random subset of size n/2
        rng = random.Random(42)
        S8 = rng.sample(elements, n // 2)
        initializations.append(S8)
        
        # Init 9: modular approach - pick elements in a residue class mod 3
        # Elements where x % 3 == 1 (sum of two 1s is 2, not 1)
        S9 = [x for x in elements if x % 3 == 1]
        initializations.append(S9)
        
        # Init 10: modular approach - pick elements where x % 3 == 2
        S10 = [x for x in elements if x % 3 == 2]
        initializations.append(S10)
        
        for S0 in initializations:
            if not S0:
                continue
            # Verify and fix sum-free
            S0_set = set(S0)
            valid = True
            for a in S0:
                for b in S0:
                    if a + b in S0_set:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                S0_set = set(S0)
                changed = True
                while changed:
                    changed = False
                    to_remove = set()
                    for a in list(S0_set):
                        for b in S0_set:
                            if a + b in S0_set:
                                to_remove.add(a)
                                to_remove.add(b)
                                to_remove.add(a + b)
                    if to_remove:
                        S0_set -= to_remove
                        changed = True
                S0 = list(S0_set)
            
            # Greedy add
            S = list(S0)
            S = greedy_add(S, [x for x in elements if x not in set(S)])
            
            # Local search
            S = local_search(S, elements)
            
            # Swap search
            S = swap_search(S, elements)
            
            if len(S) > best_len:
                best_len = len(S)
                best_S = S[:]
        
        # Also try the Max-2-SAT heuristic
        S_maxsat = max_2sat_heuristic(elements)
        if S_maxsat:
            S_maxsat = greedy_add(S_maxsat, [x for x in elements if x not in set(S_maxsat)])
            S_maxsat = local_search(S_maxsat, elements)
            if len(S_maxsat) > best_len:
                best_len = len(S_maxsat)
                best_S = S_maxsat[:]
        
        return best_S if best_S else []

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() - start_time > DEADLINE - 10:
            elements = instance(size, seed)
            results.append([x for x in elements if x % 2 != 0])
            continue
        elements = instance(size, seed)
        S = solve_instance(elements)
        # Final verification
        Sset = set(S)
        for a in S:
            for b in S:
                if a + b in Sset:
                    S = [x for x in S if x not in (a, b, a+b)]
                    Sset = set(S)
                    break
        results.append(S)

    return results
# EVOLVE-BLOCK-END
