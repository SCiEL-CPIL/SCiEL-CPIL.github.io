import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    N = 128
    L = set(range(65, N + 1))
    
    def is_valid(R):
        S = L.union(R)
        # Check Case 3: One in L, two in R
        # For all a in L, b != c in R:
        # If a+b in S and a+c in S and b+c in S -> Invalid
        # Optimization: 
        # a+b in S implies a+b <= 128 (always true for a>=65, b>=1? No, a+b can be >128)
        # a+b in S implies a+b in L or a+b in R.
        # a+b in L iff 65 <= a+b <= 128.
        # a+b in R iff a+b in R.
        
        # We can iterate over pairs b,c in R
        R_list = list(R)
        k = len(R_list)
        
        # Precompute sums in R?
        # For each pair b,c in R:
        # 1. Check if b+c in S. If not, this pair is "safe" for the third condition?
        #    No, we need a+b in S AND a+c in S AND b+c in S.
        #    So if b+c not in S, the triple (a,b,c) is safe for any a.
        
        # Let's just iterate triples?
        # k is small (~17). k^3 is ~5000.
        # We need to check for each triple (a,b,c) in R?
        # No, Case 4 is all in R.
        # Case 3 is one in L, two in R.
        
        # Case 4: a,b,c in R distinct.
        # Check if a+b in S and a+c in S and b+c in S.
        for i in range(k):
            a = R_list[i]
            for j in range(i+1, k):
                b = R_list[j]
                ab = a + b
                ab_in_S = ab in S
                if not ab_in_S:
                    # If a+b not in S, then for any c, (a,b,c) is safe regarding a+b.
                    # But we need to check other pairs?
                    # The condition is specific to the triple.
                    # If a+b not in S, the conjunction is false.
                    continue
                for l in range(j+1, k):
                    c = R_list[l]
                    ac = a + c
                    bc = b + c
                    if ac in S and bc in S:
                        return False
        
        # Case 3: a in L, b,c in R distinct.
        # Condition: a+b in S and a+c in S and b+c in S.
        # We can iterate pairs b,c in R.
        # For a fixed pair b,c:
        # We need to know if there exists a in L such that a+b in S and a+c in S.
        # a+b in S implies a+b <= 128 (since a+b >= 66).
        # So a <= 128-b.
        # Also a >= 65.
        # So a in [65, 128-b].
        # We need a such that a+b in S and a+c in S.
        # Note: b+c must be in S.
        # So first check if b+c in S. If not, this pair is safe for all a.
        
        for i in range(k):
            b = R_list[i]
            for j in range(i+1, k):
                c = R_list[j]
                bc = b + c
                if bc not in S:
                    continue
                # Now check if there is a in L such that a+b in S and a+c in S.
                # a must be in L.
                # a+b in S and a+c in S.
                # Since a in L, a+b and a+c are > 65.
                # So a+b in S iff a+b in L (since a+b > 65 and if a+b in R, it's small, but a>=65,b>=1 => a+b>=66, so a+b not in R unless R has large numbers? No, R subset 1..64. So a+b > 64, so a+b not in R. Thus a+b in S iff a+b in L).
                # Similarly a+c in S iff a+c in L.
                # So we need a in L such that a+b in L and a+c in L.
                # a+b in L iff a+b <= 128.
                # a+c in L iff a+c <= 128.
                # So a <= 128-b and a <= 128-c.
                # a <= 128 - max(b,c).
                # Also a >= 65.
                # If such an 'a' exists in L, and L is the full range 65..128, then yes.
                # So if 65 <= 128 - max(b,c), i.e., max(b,c) <= 63, then such 'a' exists.
                # Specifically, a=65 is always a candidate if 65+b <= 128 and 65+c <= 128.
                if max(b, c) <= 63:
                    # There exists a=65 in L.
                    # 65+b <= 128 and 65+c <= 128 is true.
                    # So a=65 works.
                    return False
        return True

    best_R = set()
    best_size = 0
    
    # Heuristic initialization
    # Try upper half of 1..64
    R_init = set(range(33, 65)) # 33..64
    # Check validity
    if is_valid(R_init):
        best_R = R_init
        best_size = len(R_init)
        
    # Greedy add
    R = set(R_init)
    for x in range(1, 65):
        if x not in R:
            R_test = set(R)
            R_test.add(x)
            if is_valid(R_test):
                R = R_test
                if len(R) > best_size:
                    best_size = len(R)
                    best_R = set(R)
    if time.time() > deadline:
        return list(L.union(best_R))
        
    # Local search: try to swap
    # We can try to replace elements in R with smaller ones to allow more?
    # Or just random restarts.
    
    for _ in range(1000):
        if time.time() > deadline:
            break
        # Random R
        R_rand = set()
        # Bias towards smaller numbers?
        # Or random subset of 1..64
        # Let's try random subset of size ~20
        candidates = list(range(1, 65))
        random.shuffle(candidates)
        R_rand = set(candidates[:20])
        if is_valid(R_rand):
            # Greedy add
            for x in range(1, 65):
                if x not in R_rand:
                    R_test = set(R_rand)
                    R_test.add(x)
                    if is_valid(R_test):
                        R_rand = R_test
            if len(R_rand) > best_size:
                best_size = len(R_rand)
                best_R = set(R_rand)
                
    return list(L.union(best_R))
