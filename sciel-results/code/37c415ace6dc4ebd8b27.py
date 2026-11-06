import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 75.0  # safety margin before 90s kill
    
    N = 256
    
    def is_valid(S):
        """Check if S satisfies the property: for all a<b<=c in S, a does not divide b+c."""
        S_sorted = sorted(S)
        n = len(S_sorted)
        for i in range(n):
            a = S_sorted[i]
            for j in range(i+1, n):
                b = S_sorted[j]
                for k in range(j, n):
                    c = S_sorted[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def count_violations(S):
        """Count number of violating triples (a,b,c) with a<b<=c in S where a|(b+c)."""
        S_sorted = sorted(S)
        n = len(S_sorted)
        violations = 0
        for i in range(n):
            a = S_sorted[i]
            for j in range(i+1, n):
                b = S_sorted[j]
                for k in range(j, n):
                    c = S_sorted[k]
                    if (b + c) % a == 0:
                        violations += 1
        return violations
    
    def fast_violation_count(S, limit=1000):
        """Faster violation count using set lookups, with early termination."""
        S_set = set(S)
        S_sorted = sorted(S)
        n = len(S_sorted)
        violations = 0
        for i in range(n):
            a = S_sorted[i]
            # For each pair (b, c) with b <= c, check if b+c is a multiple of a
            # and b, c are both in S with b > a
            # b+c must be in range (2a+1, 2*max(S)]
            # Actually b > a, so b >= a+1, c >= b
            # min b+c = 2(a+1), max b+c = 2*max(S)
            for multiple in range(2*(a+1), 2*max(S)+1, a):
                # Find pairs (b, c) in S with b > a, b <= c, b+c = multiple
                # b ranges from max(a+1, multiple - max(S)) to multiple//2
                for b in S_sorted:
                    if b <= a:
                        continue
                    c = multiple - b
                    if c < b:
                        continue
                    if c in S_set:
                        violations += 1
                        if violations > limit:
                            return violations
        return violations
    
    # Start with the known good solution: [171, 256], size 86
    base_set = list(range(171, 257))
    best_set = list(base_set)
    best_size = len(best_set)
    
    # Try to improve by adding elements from [1, 170]
    # and removing minimal number of elements from [171, 256]
    
    # Greedy approach: try adding each candidate from [1, 170] in decreasing order
    # (larger candidates are "cheaper" to include)
    
    candidates = list(range(170, 0, -1))
    
    current_set = list(base_set)
    
    for cand in candidates:
        if time.time() > DEADLINE:
            break
        
        if cand in current_set:
            continue
        
        # Try adding cand to current_set
        test_set = current_set + [cand]
        
        # Count violations
        violations = fast_violation_count(test_set, limit=50)
        
        if violations == 0:
            # No violations, add it
            current_set = test_set
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
    
    # Now try local search: swap elements
    # Try removing one element and adding two from outside
    if time.time() < DEADLINE - 5:
        outside = [x for x in range(1, N+1) if x not in current_set]
        
        # Try single additions again more carefully
        for cand in sorted(outside, reverse=True):
            if time.time() > DEADLINE:
                break
            test_set = current_set + [cand]
            v = fast_violation_count(test_set, limit=20)
            if v == 0:
                current_set = test_set
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
        
        # Try removing one element to allow adding two
        for i in range(len(current_set)):
            if time.time() > DEADLINE - 2:
                break
            remaining = current_set[:i] + current_set[i+1:]
            # Try adding each outside element
            for cand in sorted(outside, reverse=True):
                if time.time() > DEADLINE:
                    break
                test_set = remaining + [cand]
                if len(test_set) <= best_size:
                    continue
                v = fast_violation_count(test_set, limit=20)
                if v == 0:
                    current_set = test_set
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = list(current_set)
                    outside = [x for x in range(1, N+1) if x not in current_set]
                    break
    
    # Final validation
    if is_valid(best_set):
        return best_set
    else:
        # Fallback to known valid set
        return list(range(171, 257))
