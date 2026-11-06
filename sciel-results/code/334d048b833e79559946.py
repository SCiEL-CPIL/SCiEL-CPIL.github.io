import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin within 90s limit

    n = 256
    
    # Strategy: Build two Sidon sets using structured constructions.
    # A Sidon set can be built using a modular construction:
    # If we take elements a + k*m where m > max element, sums are unique.
    # More practically, for small sets we can use greedy with backtracking.
    # 
    # Key structural idea: Use two "bands" of numbers.
    # Band 1: low numbers (for set A)
    # Band 2: high numbers (for set B)
    # Differences within band 1 are small, within band 2 are small,
    # but cross differences would be large. Since we only care about
    # each set's own differences, this helps.
    #
    # For a Sidon set, a sufficient condition: if |a - b| >= |set_size| 
    # for distinct elements, it helps avoid sum collisions.
    # 
    # Let's try: A uses elements spaced by a step to ensure Sidon property,
    # B uses a different step, and we verify difference disjointness.
    #
    # Better: use the known construction where if S is Sidon, 
    # then {2*x + f(x)} can be used to separate.
    #
    # Practical approach: Greedy with multiple strategies and local search.
    
    def is_sidon(S):
        """Check if S is a Sidon set."""
        if len(S) <= 1:
            return True
        S = list(S)
        seen = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in seen:
                    return False
                seen.add(s)
        return True
    
    def get_diffs(S):
        """Get all nonzero absolute differences."""
        S = list(S)
        d = set()
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                d.add(abs(S[i] - S[j]))
        return d

    def try_add(x, S, sums, diffs, other_diffs, max_size):
        """Try to add x to set S. Returns (success, new_sums, new_diffs)."""
        if len(S) >= max_size:
            return (False, None, None)
        # Check Sidon: new sums x+s for s in S (including x+x)
        valid = True
        new_sums = set()
        for s in S:
            ns = x + s
            if ns in sums or ns in new_sums:
                valid = False
                break
            new_sums.add(ns)
        if not valid:
            return (False, None, None)
        if 2 * x in sums:
            return (False, None, None)
        new_sums.add(2 * x)
        # Check diff disjointness: new diffs |x - s| for s in S
        new_diffs = set()
        for s in S:
            d = abs(x - s)
            if d in other_diffs or d in new_diffs:
                return (False, None, None)
            new_diffs.add(d)
        return (True, new_sums, new_diffs)
    
    def greedy_build(start_set, other_diffs, all_numbers, max_size):
        """Greedily add elements maintaining Sidon property and diff disjointness."""
        S = list(start_set)
        sums = set()
        # Precompute sums for initial set
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        diffs = get_diffs(S)
        
        for x in all_numbers:
            if x in S:
                continue
            if len(S) >= max_size:
                break
            # Check Sidon: new sums x+s for s in S (including x+x)
            valid = True
            new_sums = set()
            for s in S:
                ns = x + s
                if ns in sums or ns in new_sums:
                    valid = False
                    break
                new_sums.add(ns)
            if not valid:
                continue
            # Check diff disjointness: new diffs |x - s| for s in S
            new_diffs = set()
            for s in S:
                d = abs(x - s)
                if d in other_diffs or d in diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if not valid:
                continue
            # Also check new_diffs don't collide with each other (they don't since they're all |x-s|)
            S.append(x)
            sums.update(new_sums)
            diffs.update(new_diffs)
        
        return S
    
    # Strategy 1: Alternate assignment with smart ordering
    # Use numbers in an order that maximizes packing: interleave low and high
    
    best_total = 0
    best_A = None
    best_B = None
    
    # Predefine several ordering strategies
    strategies = [
        list(range(1, 257)),           # ascending
        list(range(256, 0, -1)),       # descending
        # Interleaved: small, large, small, large...
        [1, 256, 2, 255, 3, 254, 4, 253] + list(range(5, 129)) + list(range(128, 257, -1))[::-1],
        # Odd then even
        list(range(1, 257, 2)) + list(range(2, 257, 2)),
        # Even then odd
        list(range(2, 257, 2)) + list(range(1, 257, 2)),
    ]
    
    # Try multiple random shuffles as well
    rng = random.Random(42)
    for _ in range(10):
        s = list(range(1, 257))
        rng.shuffle(s)
        strategies.append(s)
    
    # Add more structured strategies
    # Primes-based ordering
    primes = [p for p in range(2, 257) if all(p % q != 0 for q in range(2, int(math.isqrt(p)) + 1))]
    strategies.append(primes + [x for x in range(1, 257) if x not in primes])
    # Multiples of 4, then 2, then odd
    strategies.append([x for x in range(4, 257, 4)] + [x for x in range(2, 257, 4)] + [x for x in range(1, 257, 2)])
    # Reverse multiples
    strategies.append([x for x in range(256, 0, -4)] + [x for x in range(255, 0, -4)] + [x for x in range(255, 0, -2)])
    
    for strategy in strategies:
        if time.time() > deadline - 15:
            break
        
        # Try both orderings: A-first and B-first
        for prefer_A in [True, False]:
            if time.time() > deadline - 15:
                break
            
            A = []
            B = []
            sums_A = set()
            sums_B = set()
            diffs_A = set()
            diffs_B = set()
            
            for x in strategy:
                if prefer_A:
                    first_S, first_sums, first_diffs = A, sums_A, diffs_A
                    second_S, second_sums, second_diffs = B, sums_B, diffs_B
                    first_other = diffs_B
                    second_other = diffs_A
                else:
                    first_S, first_sums, first_diffs = B, sums_B, diffs_B
                    second_S, second_sums, second_diffs = A, sums_A, diffs_A
                    first_other = diffs_A
                    second_other = diffs_B
                
                added_first = False
                ok, ns, nd = try_add(x, first_S, first_sums, first_diffs, first_other, 64)
                if ok:
                    first_S.append(x)
                    first_sums.update(ns)
                    first_diffs.update(nd)
                    added_first = True
                
                if not added_first:
                    ok, ns, nd = try_add(x, second_S, second_sums, second_diffs, second_other, 64)
                    if ok:
                        second_S.append(x)
                        second_sums.update(ns)
                        second_diffs.update(nd)
            
            # Ensure A and B are the right sets
            if prefer_A:
                A, B = A, B
            else:
                A, B = B, A
            
            total = len(A) + len(B)
            if total > best_total:
                best_total = total
                best_A = list(A)
                best_B = list(B)
    
    # Additional: try a "balanced" greedy where we alternate more carefully
    # and use a priority-based approach for element ordering
    rng2 = random.Random(123)
    for trial in range(5):
        if time.time() > deadline - 10:
            break
        
        # Create a dynamic ordering: at each step, pick the element that
        # has the fewest conflicts (can be added to either set)
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        remaining = set(range(1, 257))
        
        for _ in range(256):
            if time.time() > deadline - 10:
                break
            if not remaining:
                break
            
            # Find the element with minimum total conflict score
            best_x = None
            best_score = float('inf')
            # Sample a subset for speed
            sample = list(remaining)
            if len(sample) > 30:
                sample = rng2.sample(sample, 30)
            
            for x in sample:
                score = 0
                # Check if can add to A
                ok_a = True
                new_sums_a = set()
                for s in A:
                    ns = x + s
                    if ns in sums_A or ns in new_sums_a:
                        ok_a = False
                        break
                    new_sums_a.add(ns)
                if ok_a and 2*x in sums_A:
                    ok_a = False
                if ok_a:
                    new_diffs_a = set()
                    for s in A:
                        d = abs(x - s)
                        if d in diffs_B or d in new_diffs_a:
                            ok_a = False
                            break
                        new_diffs_a.add(d)
                
                # Check if can add to B
                ok_b = True
                new_sums_b = set()
                for s in B:
                    ns = x + s
                    if ns in sums_B or ns in new_sums_b:
                        ok_b = False
                        break
                    new_sums_b.add(ns)
                if ok_b and 2*x in sums_B:
                    ok_b = False
                if ok_b:
                    new_diffs_b = set()
                    for s in B:
                        d = abs(x - s)
                        if d in diffs_A or d in new_diffs_b:
                            ok_b = False
                            break
                        new_diffs_b.add(d)
                
                # Score: prefer elements that can be added to the smaller set
                if ok_a and ok_b:
                    # Can go either way; prefer the smaller set
                    if len(A) <= len(B):
                        score = 0
                    else:
                        score = 1
                elif ok_a:
                    score = 2
                elif ok_b:
                    score = 2
                else:
                    score = 100  # Can't be added anywhere
                
                if score < best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None or best_score >= 100:
                # Try all remaining
                for x in remaining:
                    ok_a = True
                    new_sums_a = set()
                    for s in A:
                        ns = x + s
                        if ns in sums_A or ns in new_sums_a:
                            ok_a = False
                            break
                        new_sums_a.add(ns)
                    if ok_a and 2*x in sums_A:
                        ok_a = False
                    if ok_a:
                        new_diffs_a = set()
                        for s in A:
                            d = abs(x - s)
                            if d in diffs_B or d in new_diffs_a:
                                ok_a = False
                                break
                            new_diffs_a.add(d)
                    if ok_a:
                        best_x = x
                        break
                
                if best_x is None:
                    break
            
            # Add best_x to the appropriate set
            if len(A) <= len(B) and best_x not in A:
                ok, ns, nd = try_add(best_x, A, sums_A, diffs_A, diffs_B, 64)
                if ok:
                    A.append(best_x)
                    sums_A.update(ns)
                    diffs_A.update(nd)
                    remaining.discard(best_x)
                else:
                    ok, ns, nd = try_add(best_x, B, sums_B, diffs_B, diffs_A, 64)
                    if ok:
                        B.append(best_x)
                        sums_B.update(ns)
                        diffs_B.update(nd)
                        remaining.discard(best_x)
            else:
                ok, ns, nd = try_add(best_x, B, sums_B, diffs_B, diffs_A, 64)
                if ok:
                    B.append(best_x)
                    sums_B.update(ns)
                    diffs_B.update(nd)
                    remaining.discard(best_x)
                else:
                    ok, ns, nd = try_add(best_x, A, sums_A, diffs_A, diffs_B, 64)
                    if ok:
                        A.append(best_x)
                        sums_A.update(ns)
                        diffs_A.update(nd)
                        remaining.discard(best_x)
        
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = list(A)
            best_B = list(B)
    
    # If nothing found, use trivial solution
    if best_A is None or best_B is None:
        best_A = [1]
        best_B = [2]
        best_total = 2
    
    # Local search polish: try to add more elements or swap to improve
    def polish(A, B, max_iters=500):
        """Local search: try adding elements, or swapping elements between sets."""
        A = list(A)
        B = list(B)
        sums_A = set()
        sums_B = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums_A.add(A[i] + A[j])
        for i in range(len(B)):
            for j in range(i, len(B)):
                sums_B.add(B[i] + B[j])
        diffs_A = get_diffs(A)
        diffs_B = get_diffs(B)
        
        all_nums = set(range(1, 257))
        in_A = set(A)
        in_B = set(B)
        
        improved = True
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            
            # Try adding unassigned elements
            unassigned = all_nums - in_A - in_B
            for x in unassigned:
                ok, ns, nd = try_add(x, A, sums_A, diffs_A, diffs_B, 64)
                if ok:
                    A.append(x)
                    sums_A.update(ns)
                    diffs_A.update(nd)
                    in_A.add(x)
                    improved = True
                    break
                ok, ns, nd = try_add(x, B, sums_B, diffs_B, diffs_A, 64)
                if ok:
                    B.append(x)
                    sums_B.update(ns)
                    diffs_B.update(nd)
                    in_B.add(x)
                    improved = True
                    break
            
            # If no additions, try 1-swap moves: remove one element, try to add two
            if not improved:
                # Try removing from A and adding to A
                for i, a in enumerate(A):
                    if time.time() > deadline - 5:
                        return A, B
                    if len(A) <= 1:
                        break
                    A_test = A[:i] + A[i+1:]
                    sums_test = set()
                    for k in range(len(A_test)):
                        for m in range(k, len(A_test)):
                            sums_test.add(A_test[k] + A_test[m])
                    diffs_test = get_diffs(A_test)
                    
                    # Try adding from unassigned or from B
                    candidates = list(unassigned) + [b for b in B if b not in A_test]
                    added = False
                    for x in candidates:
                        if x in A_test:
                            continue
                        ok, ns, nd = try_add(x, A_test, sums_test, diffs_test, diffs_B, 64)
                        if ok:
                            A = A_test + [x]
                            sums_A = sums_test | ns
                            diffs_A = diffs_test | nd
                            in_A = set(A)
                            unassigned = all_nums - in_A - in_B
                            improved = True
                            added = True
                            break
                    if improved:
                        break
                
                # Try removing from B and adding to B
                if not improved:
                    for i, b in enumerate(B):
                        if time.time() > deadline - 5:
                            return A, B
                        if len(B) <= 1:
                            break
                        B_test = B[:i] + B[i+1:]
                        sums_test = set()
                        for k in range(len(B_test)):
                            for m in range(k, len(B_test)):
                                sums_test.add(B_test[k] + B_test[m])
                        diffs_test = get_diffs(B_test)
                        
                        candidates = list(unassigned) + [a for a in A if a not in B_test]
                        added = False
                        for x in candidates:
                            if x in B_test:
                                continue
                            ok, ns, nd = try_add(x, B_test, sums_test, diffs_test, diffs_A, 64)
                            if ok:
                                B = B_test + [x]
                                sums_B = sums_test | ns
                                diffs_B = diffs_test | nd
                                in_B = set(B)
                                unassigned = all_nums - in_A - in_B
                                improved = True
                                added = True
                                break
                        if improved:
                            break
                
                # Try 2-swap: remove one from A and one from B, add different ones
                if not improved:
                    for i, a in enumerate(A):
                        if time.time() > deadline - 5:
                            return A, B
                        if len(A) <= 1:
                            break
                        A_test = A[:i] + A[i+1:]
                        sums_A_test = set()
                        for k in range(len(A_test)):
                            for m in range(k, len(A_test)):
                                sums_A_test.add(A_test[k] + A_test[m])
                        diffs_A_test = get_diffs(A_test)
                        
                        for j, b in enumerate(B):
                            if time.time() > deadline - 5:
                                return A, B
                            if len(B) <= 1:
                                break
                            B_test = B[:j] + B[j+1:]
                            sums_B_test = set()
                            for k in range(len(B_test)):
                                for m in range(k, len(B_test)):
                                    sums_B_test.add(B_test[k] + B_test[m])
                            diffs_B_test = get_diffs(B_test)
                            
                            # Try to add to A from unassigned or B
                            candidates_A = list(unassigned) + [x for x in B if x not in A_test]
                            for x in candidates_A:
                                if x in A_test:
                                    continue
                                ok, ns, nd = try_add(x, A_test, sums_A_test, diffs_A_test, diffs_B_test, 64)
                                if ok:
                                    A_new = A_test + [x]
                                    # Now try to add to B
                                    candidates_B = list(unassigned - {x}) + [y for y in A if y not in B_test and y != x]
                                    for y in candidates_B:
                                        if y in B_test:
                                            continue
                                        ok2, ns2, nd2 = try_add(y, B_test, sums_B_test, diffs_B_test, diffs_A_test | nd, 64)
                                        if ok2:
                                            B_new = B_test + [y]
                                            if len(A_new) + len(B_new) > len(A) + len(B):
                                                A = A_new
                                                B = B_new
                                                sums_A = sums_A_test | ns
                                                sums_B = sums_B_test | ns2
                                                diffs_A = diffs_A_test | nd
                                                diffs_B = diffs_B_test | nd2
                                                in_A = set(A)
                                                in_B = set(B)
                                                unassigned = all_nums - in_A - in_B
                                                improved = True
                                                break
                                    if improved:
                                        break
                            if improved:
                                break
                        if improved:
                            break
        
        return A, B
    
    if time.time() < deadline - 5:
        best_A, best_B = polish(best_A, best_B)
        best_total = len(best_A) + len(best_B)
    
    return [best_A, best_B]
