import time
import math
import random

def construct():
    deadline = time.time() + 55  # 55s margin
    
    # Candidate pool: 2..256
    all_nums = list(range(2, 257))
    
    # Initial set: Upper half
    S = list(range(129, 257))
    S_set = set(S)
    
    def is_valid(S, S_set):
        """
        Check if S is valid.
        Condition: For all distinct a,b,c in S, a does not divide b*c.
        Optimized check:
        For each a in S:
          For each b in S, b != a:
            r = a // gcd(a, b)
            If r == 1, continue (a divides b, so a divides b*c? No, a|b implies a|bc. 
            Wait. If a|b, then a|bc. So if a|b for b != a, it's a violation?
            Let's re-read: "a does not divide b*c".
            If a|b, then b = k*a. bc = k*ac. a divides k*ac. Yes.
            So if a divides b, the condition is violated for ANY c.
            So S cannot contain two elements where one divides the other?
            Let's check: a=2, b=4, c=3. a|b. a|bc? 2 | 12. Yes.
            So if a|b, then a|bc is always true.
            Therefore, S must be an "antichain" under divisibility?
            No, just that for any pair (a,b), if a|b, we can't have any third element c.
            Actually, if a|b, then for ANY c distinct from a,b, a|bc.
            So if S contains a and b with a|b, S must have size <= 2.
            But we want large S. So we should avoid divisibility pairs entirely.
            
            So, a necessary condition: No a in S divides b in S for a != b.
            This is much stronger and easier to check.
            
            Is it sufficient?
            If no a|b, does it imply no a|bc?
            No. Example: 6, 2, 3. 6 does not divide 2, 6 does not divide 3.
            But 6 divides 2*3=6.
            So we must check the product condition.
            
            However, the divisibility check (a|b) is a quick filter.
            If a|b, it's a conflict.
            """
        # Precompute multiples sets for current S?
        # For a fixed S, we can just check pairs.
        
        # Optimization: 
        # For each a, we want to ensure no pair (b,c) exists such that a|bc.
        # This is equivalent to: for each a, the set {b in S\{a} | gcd(a,b) > 1} ... no.
        
        # Let's use the r = a/gcd(a,b) method.
        # For a given a, and a candidate b, we need to know if there is a c in S\{a,b} such that c % r == 0.
        
        # To make this O(N^2) or better:
        # We can precompute for each divisor d of numbers in S, how many numbers in S are divisible by d.
        # But S changes.
        
        # For a single check, O(N^2) is fine if N~130.
        # 130^2 = 16900 iterations.
        
        for a in S:
            for b in S:
                if a == b:
                    continue
                g = math.gcd(a, b)
                r = a // g
                if r == 1:
                    # a divides b. 
                    # Then a divides b*c for any c.
                    # So if there is ANY c in S distinct from a,b, it's invalid.
                    # If len(S) > 2, it's invalid.
                    # If len(S) == 2, it's valid (no c exists).
                    if len(S) > 2:
                        return False
                    else:
                        continue
                
                # Check if there exists c in S, c != a, c != b, such that c % r == 0
                # We can iterate c, but that's O(N^3).
                # We can precompute a list of S.
                # Since N is small, O(N^3) with early exit might be okay, 
                # but O(N^2) with precomputed counts is better.
                
                # Let's just iterate c for now, it might be fast enough with early exit.
                found = False
                for c in S:
                    if c == a or c == b:
                        continue
                    if c % r == 0:
                        found = True
                        break
                if found:
                    return False
        return True

    def count_conflicts(S, S_set):
        conflicts = {x: 0 for x in S}
        for a in S:
            for b in S:
                if a == b:
                    continue
                g = math.gcd(a, b)
                r = a // g
                if r == 1:
                    if len(S) > 2:
                        conflicts[a] += 1
                        conflicts[b] += 1
                        # Increment all others?
                        for c in S:
                            if c != a and c != b:
                                conflicts[c] += 1
                    continue
                
                for c in S:
                    if c == a or c == b:
                        continue
                    if c % r == 0:
                        conflicts[a] += 1
                        conflicts[b] += 1
                        conflicts[c] += 1
        return conflicts

    # Phase 1: Ensure validity of initial set
    # The set [129, 256] is likely valid or close to it.
    # Check validity
    if not is_valid(S, S_set):
        while not is_valid(S, S_set) and time.time() < deadline:
            conflicts = count_conflicts(S, S_set)
            max_conf = max(conflicts.values()) if conflicts else 0
            if max_conf == 0:
                break
            # Remove element with max conflicts
            worst = None
            for x in S:
                if conflicts[x] == max_conf:
                    worst = x
                    break
            if worst is not None:
                S.remove(worst)
                S_set.remove(worst)
            else:
                break

    # Phase 2: Try to add smaller numbers
    candidates = list(range(2, 129))
    random.shuffle(candidates)
    
    for x in candidates:
        if time.time() > deadline:
            break
        if x in S_set:
            continue
        # Try adding x
        S_test = S + [x]
        S_test_set = S_set | {x}
        if is_valid(S_test, S_test_set):
            S = S_test
            S_set = S_test_set

    # Phase 3: Local Search (Swaps)
    # Try to swap elements to increase size or maintain size while allowing better additions?
    # Actually, we just want max size.
    # If we can't add any more, we are stuck.
    # Swapping a large number for a small number doesn't increase size.
    # But maybe removing a "bad" large number allows adding TWO small numbers?
    # Net gain +1.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        removed = [x for x in range(2, 257) if x not in S_set]
        random.shuffle(removed)
        
        for x in removed:
            if time.time() > deadline:
                break
            # Try to add x by removing one y from S
            for y in S:
                S_test = [z for z in S if z != y] + [x]
                S_test_set = (S_set - {y}) | {x}
                if is_valid(S_test, S_test_set):
                    # Accept swap
                    S = S_test
                    S_set = S_test_set
                    improved = True
                    break

    # Final check
    if not is_valid(S, S_set):
        # Fallback
        while not is_valid(S, S_set) and time.time() < deadline:
            conflicts = count_conflicts(S, S_set)
            max_conf = max(conflicts.values()) if conflicts else 0
            if max_conf == 0:
                break
            worst = None
            for x in S:
                if conflicts[x] == max_conf:
                    worst = x
                    break
            if worst is not None:
                S.remove(worst)
                S_set.remove(worst)
            else:
                break

    return S
