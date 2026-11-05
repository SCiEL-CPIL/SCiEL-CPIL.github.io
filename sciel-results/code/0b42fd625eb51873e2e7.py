import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 256
    
    def check_sidon(S):
        """Check if S is a Sidon set."""
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def get_diffs(S):
        """Get set of nonzero positive differences."""
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                diffs.add(abs(S[i] - S[j]))
        return diffs

    def can_add_to_set(S, O_diffs, x):
        """
        Check if x can be added to S such that:
        1. S U {x} is Sidon.
        2. Diffs(S U {x}) is disjoint from O_diffs.
        """
        # 1. Check Sidon for S U {x}
        # New sums: x + s for s in S, and 2x
        # These must not collide with existing sums in S, nor each other (x+s1 != x+s2 implies s1!=s2, ok)
        # But x+s could equal 2x? No, s!=x.
        # We need to check if any new sum is in existing sums of S.
        # Existing sums of S are in a set for efficiency, but here we pass S.
        # For speed, we should maintain sums in the main loop.
        # For this helper, we assume S is small.
        
        # Efficient check:
        # Let sums_S be the set of all a+b for a,b in S, a<=b.
        # We don't have sums_S passed here, so we compute it or pass it.
        # Let's refactor to pass sums_S.
        pass

    # Better approach: Maintain state (A, B, sums_A, sums_B, diffs_A, diffs_B)
    
    best_size = 0
    best_A = []
    best_B = []
    
    def try_greedy(order):
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for x in order:
            added = False
            
            # Try adding to A
            # Check Sidon for A U {x}
            ok_sidon_A = True
            new_sums_A = set()
            for s in A:
                ns = x + s
                if ns in sums_A:
                    ok_sidon_A = False
                    break
                new_sums_A.add(ns)
            if 2*x in sums_A:
                ok_sidon_A = False
            # Note: 2x could also collide with x+s? No, s != x.
            # Also need to check if new sums collide with each other? 
            # x+s1 == x+s2 => s1==s2. No.
            # So just check against old sums.
            
            if ok_sidon_A:
                # Check Diffs for A U {x} against diffs_B
                ok_diffs = True
                new_diffs_A = set()
                for s in A:
                    nd = abs(x - s)
                    if nd in diffs_B:
                        ok_diffs = False
                        break
                    # Also, do new diffs collide with existing diffs in A?
                    # If A is Sidon, diffs_A are unique.
                    # If we add x, new diffs are |x-s|.
                    # If |x-s1| == |s2-s3|, does it violate Sidon?
                    # Yes, unique sums implies unique diffs.
                    # So if the resulting set is Sidon, diffs are unique.
                    # We already checked Sidon validity via sums.
                    # So we only need to check disjointness with B.
                    new_diffs_A.add(nd)
                
                if ok_diffs:
                    # Add x to A
                    A.append(x)
                    sums_A.update(new_sums_A)
                    sums_A.add(2*x)
                    diffs_A.update(new_diffs_A)
                    added = True
            
            if not added:
                # Try adding to B
                ok_sidon_B = True
                new_sums_B = set()
                for s in B:
                    ns = x + s
                    if ns in sums_B:
                        ok_sidon_B = False
                        break
                    new_sums_B.add(ns)
                if 2*x in sums_B:
                    ok_sidon_B = False
                
                if ok_sidon_B:
                    ok_diffs = True
                    new_diffs_B = set()
                    for s in B:
                        nd = abs(x - s)
                        if nd in diffs_A:
                            ok_diffs = False
                            break
                        new_diffs_B.add(nd)
                    
                    if ok_diffs:
                        B.append(x)
                        sums_B.update(new_sums_B)
                        sums_B.add(2*x)
                        diffs_B.update(new_diffs_B)
        
        return len(A) + len(B), A, B

    # Run multiple restarts
    while time.time() < deadline:
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        # Simple greedy
        size, A, B = try_greedy(candidates)
        
        if size > best_size:
            best_size = size
            best_A = A
            best_B = B
            
    return [best_A, best_B]
