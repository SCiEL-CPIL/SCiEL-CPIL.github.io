import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 87.0  # 3s safety margin
    
    n = 256
    
    def is_valid_set(S):
        """Check: for all a < b <= c in S, a does not divide b+c"""
        S_list = sorted(S)
        k = len(S_list)
        for i in range(k):
            a = S_list[i]
            for j in range(i + 1, k):
                b = S_list[j]
                # b + c must be divisible by a
                # c >= b
                # min sum = 2*b
                # max sum = b + S_list[-1]
                # We need to check if there exists c in S_list[j:] such that (b+c)%a == 0
                # This is equivalent to checking if (b+c) is a multiple of a.
                # c = m*a - b.
                # We need c in S_list[j:] and c >= b.
                # Instead of iterating c, we can iterate multiples of a.
                # But iterating c is O(K). Total O(K^3).
                # For K=100, 10^6. 100 restarts -> 10^8. Might be too slow.
                # Let's stick to the loop but break early if possible?
                # Or just use the triple loop but optimize inner.
                for l in range(j, k):
                    c = S_list[l]
                    if (b + c) % a == 0:
                        return False
        return True

    def find_conflict(S):
        """Returns (a, b, c) if conflict exists, else None"""
        S_list = sorted(S)
        k = len(S_list)
        for i in range(k):
            a = S_list[i]
            for j in range(i + 1, k):
                b = S_list[j]
                for l in range(j, k):
                    c = S_list[l]
                    if (b + c) % a == 0:
                        return (a, b, c)
        return None

    def can_add_element(S, x):
        """Check if x can be added to S without violating constraints.
        x is assumed to be the smallest element in S U {x} for this check?
        No, x can be anywhere.
        If x is smallest, it acts as 'a'.
        If x is middle or largest, it acts as 'b' or 'c'.
        
        Case 1: x is smallest. x < all b in S.
        Check if x divides b+c for any b<=c in S.
        
        Case 2: x is not smallest.
        Let a < x. Then x can be b or c.
        If x is b: a < x <= c. Check a | x+c.
        If x is c: a < b <= x. Check a | b+x.
        
        This is complex. 
        Simpler: Just check all triples involving x.
        """
        S_list = sorted(S)
        # x as 'a' (smallest)
        # We need x < b <= c.
        # If x is smaller than all in S, then all b,c in S are candidates.
        if x < S_list[0]:
            for i in range(len(S_list)):
                b = S_list[i]
                for j in range(i, len(S_list)):
                    c = S_list[j]
                    if (b + c) % x == 0:
                        return False
        else:
            # x is not the smallest.
            # There exists a in S such that a < x.
            # x can be 'b' or 'c'.
            
            # 1. x is 'b' or 'c' with some a < x.
            # Triples (a, b, c) where x is b or c.
            
            # If x is 'b': a < x <= c.
            # c must be in S and c >= x.
            # Check if there is a < x in S and c >= x in S such that a | x+c.
            for a in S_list:
                if a >= x: break # a must be < x
                # Check c in S such that c >= x and (x+c) % a == 0
                # c = k*a - x
                for c in S_list:
                    if c < x: continue
                    if (x + c) % a == 0:
                        return False
            
            # 2. x is 'c': a < b <= x.
            # b must be in S and b <= x.
            # Check if there is a < b <= x in S (and b != x? no, b can be x? No, a<b<=c. If c=x, b<=x. If b=x, a<x=x=c. So a<b=c=x. Check a | 2x.)
            # Actually, if x is c, then b <= x.
            for a in S_list:
                if a >= x: break # a < b <= x implies a < x
                for b in S_list:
                    if b > x: break # b <= x
                    if b == x:
                        # a < x = b = c
                        if (x + x) % a == 0:
                            return False
                    else: # b < x
                        if (b + x) % a == 0:
                            return False
        return True

    def greedy_purge(S, deadline):
        """Remove elements to make S valid. Prioritize removing smallest 'a' in conflict."""
        S_set = set(S)
        while True:
            if time.time() > deadline:
                break
            conflict = find_conflict(S_set)
            if not conflict:
                break
            a, b, c = conflict
            # Remove 'a' as it is the divisor and likely causes many conflicts
            S_set.remove(a)
        return S_set

    def greedy_add(S, candidates, deadline):
        """Add elements from candidates to S."""
        S_set = set(S)
        # Try to add in increasing order? Or random?
        # Small numbers are dangerous. Large numbers are safer.
        # But we want max cardinality.
        # Let's try adding in random order or increasing order.
        # If we add small numbers first, they might block large ones?
        # No, small numbers are divisors. Large numbers are sums.
        # Adding a small number $x$ makes it a divisor. It forbids pairs $(b,c)$ in current set where $x | b+c$.
        # Adding a large number $y$ makes it a summand. It forbids pairs $(a,b)$ where $a | b+y$ or $a | y+b$ (same) or if $y$ is smallest? No, $y$ is large.
        # So adding large numbers is "cheaper" in terms of constraint propagation?
        # Actually, adding a large number $y$ only creates conflicts if there are small divisors $a$ in S.
        # Adding a small number $x$ creates conflicts with ALL pairs in S.
        # So we should add large numbers first, then small numbers.
        
        for x in sorted(candidates, reverse=True):
            if time.time() > deadline:
                break
            if can_add_element(S_set, x):
                S_set.add(x)
        return S_set

    best_result = set()
    best_size = 0
    
    # Strategy: Start with intervals [L, 256], purge, then add smaller elements.
    for L in range(100, 257):
        if time.time() > deadline - 5:
            break
            
        S_init = set(range(L, n + 1))
        # Purge conflicts
        S_purged = greedy_purge(S_init, deadline)
        
        # Now try to add elements 1..L-1
        candidates = [x for x in range(1, L) if x not in S_purged]
        S_final = greedy_add(S_purged, candidates, deadline)
        
        if len(S_final) > best_size:
            best_size = len(S_final)
            best_result = S_final
            
    # Also try starting with empty set and adding?
    # Or starting with a random set?
    # The interval strategy is strong because large numbers have fewer divisors.
    
    if not best_result:
        return list(range(192, 257))
        
    return list(best_result)
