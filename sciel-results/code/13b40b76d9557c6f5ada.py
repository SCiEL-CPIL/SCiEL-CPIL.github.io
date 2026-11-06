import time
import random
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    best_A = []
    best_B = []
    best_score = 0
    
    n = 256
    candidates = list(range(1, n + 1))
    
    # Helper to check if adding x to set S is valid given other set's diffs
    # S: list of elements
    # sums_S: set of sums a+b for a,b in S (a<=b)
    # diffs_S: set of non-zero differences a-b for a,b in S
    # diffs_other: set of non-zero differences of the other set
    
    def try_add(x, S, sums_S, diffs_S, diffs_other):
        # Check Sidon condition for S U {x}
        # New sums: x+x, x+s for s in S
        new_sums = [x + x] + [x + s for s in S]
        if any(s in sums_S for s in new_sums):
            return False
        
        # Check difference disjointness
        # New diffs: x-s, s-x for s in S
        new_diffs = set()
        for s in S:
            d1 = x - s
            d2 = s - x
            if d1 != 0 and d1 in diffs_other:
                return False
            if d2 != 0 and d2 in diffs_other:
                return False
            new_diffs.add(d1)
            new_diffs.add(d2)
            
        # If we pass, we need to update sums_S and diffs_S
        # But for the check, we just need to know if it's valid.
        return True

    def update_state(x, S, sums_S, diffs_S):
        sums_S.add(x + x)
        diffs_S.update() # Will be updated in caller
        for s in S:
            sums_S.add(x + s)
            diffs_S.add(x - s)
            diffs_S.add(s - x)
        S.append(x)

    for _ in range(1000): # Max iterations, controlled by time
        if time.time() > deadline:
            break
            
        random.shuffle(candidates)
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for x in candidates:
            if time.time() > deadline:
                break
                
            # Try adding to A
            if try_add(x, A, sums_A, diffs_A, diffs_B):
                # Commit to A
                A.append(x)
                sums_A.add(x + x)
                for a in A[:-1]:
                    sums_A.add(x + a)
                    diffs_A.add(x - a)
                    diffs_A.add(a - x)
            # Try adding to B
            elif try_add(x, B, sums_B, diffs_B, diffs_A):
                # Commit to B
                B.append(x)
                sums_B.add(x + x)
                for b in B[:-1]:
                    sums_B.add(x + b)
                    diffs_B.add(x - b)
                    diffs_B.add(b - x)
        
        current_score = len(A) + len(B)
        if current_score > best_score:
            best_score = current_score
            best_A = A[:]
            best_B = B[:]
            
            # If we reached a very high score, we might stop early?
            # Max possible is ~32. If we get 32, it's likely optimal.
            if best_score >= 32:
                break

    return [best_A, best_B]
