import random
import time

def construct():
    n = 256
    best_A = []
    best_B = []
    best_score = 0
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer

    def try_add(A, sums_A, diffs_A, B, sums_B, diffs_B, x):
        """
        Try to add x to A or B.
        Returns (success, A, sums_A, diffs_A, B, sums_B, diffs_B)
        """
        # Check A
        valid_A = True
        # Check sums: x + a for all a in A, and x + x
        # If x is in A, x+x is already checked? No, we assume x is not in A yet.
        # Sums to check: {x + a for a in A} U {x + x}
        # Diffs to check: {x - a for a in A} U {a - x for a in A}
        
        # Optimization: Precompute potential conflicts
        # For A:
        for a in A:
            if (x + a) in sums_A:
                valid_A = False
                break
            if (x - a) in diffs_B: # Disjointness with B's diffs
                valid_A = False
                break
            if (a - x) in diffs_B:
                valid_A = False
                break
        if valid_A and (2 * x) in sums_A:
            valid_A = False
        if valid_A and (x - x) in diffs_B: # 0 diff? Task says "nonzero difference sets". 
             # Usually Sidon difference sets include 0? "nonzero difference sets must be disjoint".
             # So 0 is ignored for disjointness? Or is 0 in the set?
             # "nonzero difference sets" implies we only care about d != 0.
             # So we don't need to check 0.
             pass

        # Wait, the check above is slightly flawed. 
        # If x+a conflicts with an existing sum, it's invalid.
        # If x-a conflicts with a diff in B, it's invalid.
        # Also, the new diffs x-a and a-x must not conflict with each other? No, they are just added to diffs_A.
        # But they must not be in diffs_B.
        # And the new sums x+a and x+x must not be in sums_A.
        
        # Let's refine the check to be safe and fast.
        
        # Check A validity
        valid_A = True
        for a in A:
            if (x + a) in sums_A:
                valid_A = False
                break
            if (x - a) in diffs_B:
                valid_A = False
                break
            if (a - x) in diffs_B:
                valid_A = False
                break
        if valid_A and (2 * x) in sums_A:
            valid_A = False
            
        # Check B validity
        valid_B = True
        for b in B:
            if (x + b) in sums_B:
                valid_B = False
                break
            if (x - b) in diffs_A:
                valid_B = False
                break
            if (b - x) in diffs_A:
                valid_B = False
                break
        if valid_B and (2 * x) in sums_B:
            valid_B = False

        if valid_A and not valid_B:
            # Add to A
            new_A = A + [x]
            new_sums_A = sums_A | {(x + a) for a in A} | {2 * x}
            new_diffs_A = diffs_A | {(x - a) for a in A} | {(a - x) for a in A}
            return True, new_A, new_sums_A, new_diffs_A, B, sums_B, diffs_B
        elif valid_B and not valid_A:
            # Add to B
            new_B = B + [x]
            new_sums_B = sums_B | {(x + b) for b in B} | {2 * x}
            new_diffs_B = diffs_B | {(x - b) for b in B} | {(b - x) for b in B}
            return True, A, sums_A, diffs_A, new_B, new_sums_B, new_diffs_B
        elif valid_A and valid_B:
            # Heuristic: Add to the set with fewer elements to balance?
            # Or add to the set where it creates "less clutter"?
            # Let's try balancing size.
            if len(A) <= len(B):
                new_A = A + [x]
                new_sums_A = sums_A | {(x + a) for a in A} | {2 * x}
                new_diffs_A = diffs_A | {(x - a) for a in A} | {(a - x) for a in A}
                return True, new_A, new_sums_A, new_diffs_A, B, sums_B, diffs_B
            else:
                new_B = B + [x]
                new_sums_B = sums_B | {(x + b) for b in B} | {2 * x}
                new_diffs_B = diffs_B | {(x - b) for b in B} | {(b - x) for b in B}
                return True, A, sums_A, diffs_A, new_B, new_sums_B, new_diffs_B
        else:
            return False, A, sums_A, diffs_A, B, sums_B, diffs_B

    def greedy_construct(order):
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for x in order:
            success, A, sums_A, diffs_A, B, sums_B, diffs_B = try_add(
                A, sums_A, diffs_A, B, sums_B, diffs_B, x
            )
            # If success is False, x is discarded
            
        return A, B

    # 1. Deterministic orders
    orders = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
        list(range(1, n + 1, 2)) + list(range(2, n + 1, 2)), # Odds then Evens
        list(range(2, n + 1, 2)) + list(range(1, n + 1, 2)), # Evens then Odds
    ]
    
    for order in orders:
        A, B = greedy_construct(order)
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A, best_B = A, B

    # 2. Random restarts
    # With the optimized O(k) check, we can run many more.
    # n=256, k~64. 256 * 64 ~ 16k ops per greedy pass.
    # 1000 restarts ~ 16M ops. Should be well under 1s in Python if optimized, but sets are slow.
    # Let's aim for ~200-500 restarts depending on speed.
    
    num_restarts = 0
    while time.time() - start_time < time_limit:
        order = list(range(1, n + 1))
        random.shuffle(order)
        A, B = greedy_construct(order)
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A, best_B = A, B
        num_restarts += 1
        
        # Also try a "biased" random shuffle?
        # Maybe weight towards smaller numbers?
        # Or just keep it simple.
        
    # Ensure non-empty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]

    return [best_A, best_B]
