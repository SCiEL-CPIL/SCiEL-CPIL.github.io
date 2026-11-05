import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    n = 256
    
    def can_add(x, S, sums_S, D_S, D_Other):
        """
        Check if x can be added to S.
        S: list of elements
        sums_S: set of all pair sums a+b (a<=b) in S
        D_S: set of all absolute differences |a-b| in S
        D_Other: set of all absolute differences in the other set
        """
        # Check Sidon property for S (sums)
        for a in S:
            s = x + a
            if s in sums_S:
                return False
        if 2 * x in sums_S:
            return False
        
        # Check Sidon property for S (differences)
        # And Disjoint Difference constraint
        for a in S:
            d = abs(x - a)
            if d == 0: continue
            # d must not be in D_S (to keep S Sidon)
            if d in D_S:
                return False
            # d must not be in D_Other (to keep differences disjoint)
            if d in D_Other:
                return False
        
        return True

    def update_structs(x, S, sums_S, D_S):
        """
        Add x to S and update sums_S and D_S.
        Returns new S, sums_S, D_S (as sets/lists)
        Note: We should copy or modify in place carefully. 
        Since we are in a greedy loop, we can modify in place and revert if needed, 
        but here we just build forward.
        """
        new_S = S + [x]
        new_sums = set(sums_S)
        new_D = set(D_S)
        
        for a in S:
            new_sums.add(x + a)
            d = abs(x - a)
            new_D.add(d)
        new_sums.add(2 * x)
        
        return new_S, new_sums, new_D

    def greedy_construct(order):
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        D_A = set()
        D_B = set()
        
        for x in order:
            can_A = can_add(x, A, sums_A, D_A, D_B)
            can_B = can_add(x, B, sums_B, D_B, D_A)
            
            if can_A and can_B:
                # Choose randomly or balance sizes
                if len(A) <= len(B):
                    A, sums_A, D_A = update_structs(x, A, sums_A, D_A)
                else:
                    B, sums_B, D_B = update_structs(x, B, sums_B, D_B)
            elif can_A:
                A, sums_A, D_A = update_structs(x, A, sums_A, D_A)
            elif can_B:
                B, sums_B, D_B = update_structs(x, B, sums_B, D_B)
        
        return A, B

    best_A = []
    best_B = []
    best_size = 0
    
    # Initial deterministic run to have a baseline
    base_order = list(range(1, n + 1))
    A0, B0 = greedy_construct(base_order)
    best_A, best_B = A0, B0
    best_size = len(A0) + len(B0)
    
    # Randomized restarts
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Random permutation
        order = list(range(1, n + 1))
        rng.shuffle(order)
        
        A, B = greedy_construct(order)
        size = len(A) + len(B)
        
        if size > best_size:
            best_size = size
            best_A = A
            best_B = B
            
            # If we hit a very high score, we might want to continue to see if we can do better
            # Theoretical max is around 32-34. If we reach 34, it's likely near optimal.
            # But let's keep running to be safe.

    return [best_A, best_B]
