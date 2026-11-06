"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    n = 256
    deadline = time.time() + 85.0  # safety margin
    
    def is_valid(S):
        """Check if set S satisfies the property."""
        Sset = set(S)
        for a in S:
            for b in S:
                if b <= a:
                    continue
                for c in S:
                    if c < b:
                        continue
                    # a < b <= c, check a does not divide b+c
                    if (b + c) % a == 0:
                        return False
        return True
    
    def count_violations(S):
        """Count number of violating triples (a,b,c) with a<b<=c."""
        Sset = set(S)
        count = 0
        for a in S:
            for b in S:
                if b <= a:
                    continue
                for c in S:
                    if c < b:
                        continue
                    if (b + c) % a == 0:
                        count += 1
        return count
    
    def can_add(S, x):
        """Check if adding x to S maintains validity."""
        Sset = set(S)
        for a in S:
            for b in S:
                if b <= a:
                    continue
                for c in S:
                    if c < b:
                        continue
                # New triples involving x
                # Case 1: x is 'a' (smallest)
                if x < a:
                    # x < a, need to check x doesn't divide a+b and a+c and b+c for b,c > x
                    pass
                # Simpler: just check all triples with x
            # Check triples where x is the smallest: x < b <= c
            for b in S:
                if b <= x:
                    continue
                for c in S:
                    if c < b:
                        continue
                    if (b + c) % x == 0:
                        return False
            # Check triples where x is middle: a < x <= c  
            for a in S:
                if a >= x:
                    continue
                for c in S:
                    if c < x:
                        continue
                    if (x + c) % a == 0:
                        return False
            # Check triples where x is largest: a < b <= x
            for a in S:
                if a >= x:
                    continue
                for b in S:
                    if b <= a or b > x:
                        continue
                    if (b + x) % a == 0:
                        return False
        return True
    
    # Phase 1: Build initial set greedily from large to small
    S = set()
    for x in range(n, 0, -1):
        if can_add(S, x):
            S.add(x)
    
    best_S = set(S)
    best_size = len(S)
    
    # Phase 2: Local search - try to add elements by removing conflicting ones
    for restart in range(50):
        if time.time() > deadline - 5.0:
            break
            
        current = set(best_S)
        
        # Try each element not in current set
        for x in range(1, n + 1):
            if time.time() > deadline - 2.0:
                break
            if x in current:
                continue
            
            # Find which elements in current conflict with x
            conflicts = set()
            Slist = list(current)
            for a in Slist:
                for b in Slist:
                    if b <= a:
                        continue
                    for c in Slist:
                        if c < b:
                            continue
                    # Check if adding x creates a violation
                    # We need to check all triples involving x
                    pass
            
            # Simpler: just try adding and see if valid
            test_S = current | {x}
            if is_valid(test_S):
                current = test_S
                if len(current) > best_size:
                    best_S = set(current)
                    best_size = len(current)
        
        # Try removal-replacement: remove one element, try to add two
        for _ in range(20):
            if time.time() > deadline - 2.0:
                break
            if len(current) < 2:
                break
            remove_candidate = random.choice(list(current))
            test_S = current - {remove_candidate}
            # Try to add back different elements
            for x in range(1, n + 1):
                if x in test_S or x == remove_candidate:
                    continue
                if is_valid(test_S | {x}):
                    test_S = test_S | {x}
                    break
            if len(test_S) > len(current):
                current = test_S
                if len(current) > best_size:
                    best_S = set(current)
                    best_size = len(current)
    
    return list(best_S)
# EVOLVE-BLOCK-END
