"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    # Precompute conflict structure:
    # For a set S, the condition is: for all a < b <= c in S, a does NOT divide (b+c).
    # This means: if we have a in S, then for any b, c in S with b > a and c >= b,
    # we need (b+c) % a != 0.
    
    # Key insight: The constraint only involves a as the smallest element in the triple.
    # So for each a in S, we need to ensure that no pair (b,c) in S with b,c > a
    # has b+c divisible by a.
    
    # Let's think about this differently. For a fixed set S, define for each a in S:
    # forbidden_pairs[a] = {(b,c) : b > a, c >= b, b,c in S, (b+c) % a == 0}
    # The set S is valid iff forbidden_pairs[a] is empty for all a in S.
    
    # Greedy approach with backtracking:
    # We'll try to build a large set. Start with a good initial solution, then try to improve.
    
    # Initial solution: upper tail
    # For a set [k, n], the condition is: for a in [k,n], b,c in [k,n] with a < b <= c,
    # a does not divide b+c. Since b+c >= 2k and a <= n, we need 2k > n for safety? 
    # Actually, b+c can be as large as 2n, so a could divide b+c if a <= 2n.
    # But if all elements are in [k, n] and a < b, then b+c >= b + b = 2b > 2a.
    # So b+c > 2a. For a to divide b+c, we need b+c >= 2a, which is always true.
    # But we need b+c to be a multiple of a. The smallest multiple > 2a is 3a.
    # So we need 3a > b+c for all a < b <= c in S.
    # The worst case is a=k, b=k+1, c=n (if k+1 <= n).
    # Then b+c = k+1+n. We need 3k > k+1+n, i.e., 2k > n+1, i.e., k > (n+1)/2.
    # Wait, that's not right. Let me reconsider.
    
    # For a in S, b,c in S with a < b <= c:
    # We need (b+c) % a != 0.
    # The maximum b+c is when b=c=n (if n in S), giving 2n.
    # The minimum b+c is when b=a+1, c=a+1 (if a+1 in S), giving 2(a+1).
    # For a to divide b+c, b+c must be a multiple of a: a, 2a, 3a, ...
    # Since b > a, b+c > 2a, so b+c >= 3a (next multiple after 2a).
    # So we need 3a > max possible b+c for given a.
    # If S = [k, n], then for a=k, max b+c = 2n (when b=c=n).
    # We need 3k > 2n, so k > 2n/3.
    # For n=256, k > 512/3 = 170.67, so k >= 171.
    # This gives |S| = 256 - 171 + 1 = 86.
    
    # Can we do better by including some numbers below 171?
    # Let's use a greedy + local search approach.
    
    # Precompute for each pair (b,c) with b<=c, the set of a < b such that a | (b+c)
    # conflict[a] = set of (b,c) pairs that conflict with a
    # Actually, let's precompute: for each a, which values of s = b+c (where b,c > a) are divisible by a.
    # s ranges from 2(a+1) to 2n.
    # s is divisible by a iff s = ma for some integer m >= 3 (since s > 2a).
    # So s = 3a, 4a, 5a, ..., up to 2n.
    
    # For a given a, the forbidden sums are {3a, 4a, 5a, ..., floor(2n/a)*a}
    # A pair (b,c) with b > a, c >= b is forbidden if b+c is in this set.
    
    # Let's build a greedy solution:
    # Start with empty set, add numbers one by one.
    # For each candidate, check if adding it violates the constraint with existing elements.
    
    def is_valid_with_existing(existing, candidate):
        """Check if adding candidate to existing maintains the property."""
        if candidate in existing:
            return False
        # For each a in existing where a < candidate:
        #   Check if there exists b,c in existing with b > a, c >= b, and (b+c) % a == 0
        #   where either b or c is candidate (since we're adding candidate)
        # Actually, we need to check all triples in existing ∪ {candidate}.
        # The new triples involving candidate are:
        #   (a, b, candidate) where a < b < candidate or a < candidate <= b... wait
        #   a < b <= c, and one of b,c is candidate.
        #   Case 1: b = candidate, c >= candidate. c in existing, c >= candidate.
        #     Then a < candidate, and we need (candidate + c) % a != 0 for all a in existing, a < candidate.
        #   Case 2: c = candidate, b <= candidate. b in existing, a < b <= candidate.
        #     Then a < b, and we need (b + candidate) % a != 0.
        #   Also case 1 and 2 overlap when b = c = candidate.
        
        for a in existing:
            if a >= candidate:
                continue
            # Check triples where a is the smallest
            # b and c are from existing ∪ {candidate}, with b > a, c >= b
            # New pairs (b,c) involving candidate:
            # (candidate, c) for c in existing, c >= candidate
            # (b, candidate) for b in existing, a < b <= candidate
            # (candidate, candidate)
            
            # (candidate, candidate): (2*candidate) % a
            if (2 * candidate) % a == 0:
                return False
            # (candidate, c) for c in existing, c >= candidate
            for c in existing:
                if c < candidate:
                    continue
                if c == candidate:
                    continue  # already checked
                if (candidate + c) % a == 0:
                    return False
            # (b, candidate) for b in existing, a < b < candidate
            for b in existing:
                if b <= a or b >= candidate:
                    continue
                if (b + candidate) % a == 0:
                    return False
        return True
    
    # Greedy: try to add numbers in order of increasing size
    # But also try different orderings
    
    best_set = set()
    
    # Strategy: greedy with multiple orderings
    orderings = [
        list(range(1, n+1)),          # ascending
        list(range(n, 0, -1)),        # descending
        list(range(1, n+1, 2)) + list(range(2, n+1, 2)),  # odds then evens
        list(range(n//2+1, n+1)) + list(range(1, n//2+1)),  # upper half then lower
    ]
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        current = set()
        for num in ordering:
            if time.time() > deadline:
                break
            if is_valid_with_existing(current, num):
                current.add(num)
        if len(current) > len(best_set):
            best_set = set(current)
    
    # Local search: try to swap elements to increase size
    # Try removing one element and adding two others
    improved = True
    while improved and time.time() < deadline:
        improved = False
        removed_candidates = list(best_set)
        for to_remove in removed_candidates:
            if time.time() > deadline:
                break
            trial = best_set - {to_remove}
            # Try adding pairs of numbers not in trial
            for add1 in range(1, n+1):
                if add1 in trial or time.time() > deadline:
                    continue
                trial2 = trial | {add1}
                for add2 in range(add1+1, n+1):
                    if add2 in trial2 or time.time() > deadline:
                        continue
                    if is_valid_with_existing(trial2, add2):
                        new_set = trial2 | {add2}
                        if len(new_set) > len(best_set):
                            best_set = new_set
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    
    # Try single swaps: remove one, add one (if it increases size, which it won't, but maybe enables further improvements)
    # Actually, let's try a different local search: hill climbing on the set
    
    # Final validation
    def validate(s):
        s = sorted(s)
        for i in range(len(s)):
            a = s[i]
            for j in range(i+1, len(s)):
                b = s[j]
                for k in range(j, len(s)):
                    c = s[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    if not validate(best_set):
        # Fallback to known valid set
        best_set = set(range(171, n+1))
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
