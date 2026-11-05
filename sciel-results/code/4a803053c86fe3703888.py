import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = 256
    
    # State for incremental checks
    # We need to check if adding x to S (with sums_S, diffs_S) is valid w.r.t O (with diffs_O)
    # and maintains Sidon property for S U {x}.
    
    def can_add_incremental(S, sums_S, diffs_S, O, diffs_O, x):
        # 1. Check Sidon for S U {x}
        # New sums: x+s for s in S, 2x
        # Check against sums_S
        for s in S:
            ns = x + s
            if ns in sums_S:
                return False
        if 2*x in sums_S:
            return False
        
        # 2. Check Diffs for S U {x} against diffs_O
        # New diffs: |x-s| for s in S
        # Check against diffs_O
        for s in S:
            nd = abs(x - s)
            if nd in diffs_O:
                return False
        
        # Note: We don't need to check new diffs against diffs_S 
        # because S is Sidon, so all diffs in S are unique.
        # And we just checked new diffs against diffs_O.
        # Do we need to check if new diff collides with existing diff in S?
        # Yes, Sidon property requires ALL diffs to be unique?
        # Wait. Sidon = unique sums.
        # Unique sums implies unique positive differences?
        # Yes, as proven before.
        # So if S is Sidon, diffs_S has no duplicates.
        # When we add x, we create new diffs.
        # Do new diffs collide with old diffs in S?
        # If |x-s1| == |s2-s3|, does that violate Sidon?
        # Not necessarily directly, but if sums are unique, diffs are unique.
        # So if the resulting set is Sidon, the diffs will be unique.
        # My check `if ns in sums_S` ensures the new set is Sidon.
        # If the new set is Sidon, its diffs are unique.
        # So we don't need to explicitly check diff collisions within S U {x}.
        # The sum check is sufficient for Sidon validity.
        
        return True

    best_A = []
    best_B = []
    best_score = 0
    
    while time.time() < deadline:
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for x in candidates:
            added = False
            # Try A
            if can_add_incremental(A, sums_A, diffs_A, B, diffs_B, x):
                A.append(x)
                # Update sums_A
                for s in A[:-1]: # s is in old A
                    sums_A.add(x + s)
                sums_A.add(2*x)
                # Update diffs_A
                for s in A[:-1]:
                    diffs_A.add(abs(x - s))
                added = True
            # Try B
            if not added:
                if can_add_incremental(B, sums_B, diffs_B, A, diffs_A, x):
                    B.append(x)
                    for s in B[:-1]:
                        sums_B.add(x + s)
                    sums_B.add(2*x)
                    for s in B[:-1]:
                        diffs_B.add(abs(x - s))
                    
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A = A
            best_B = B
            
    return [best_A, best_B]
