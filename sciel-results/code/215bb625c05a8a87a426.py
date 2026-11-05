import time
import random
from itertools import combinations_with_replacement

def construct():
    # Deadline guard: 90s limit, use 55s for safety
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_score = 0
    best_A = [1]
    best_B = [2]
    
    # Helper to check and add x to a set
    # Returns True if added, False otherwise
    # Modifies the sets in place if successful
    def try_add(x, A, B, sums_A, sums_B, diffs_A, diffs_B):
        # Check Sidon condition for A
        # New sums: x + a for a in A, and x + x
        valid_A = True
        # Check sums
        if (x + x) in sums_A:
            valid_A = False
        else:
            for a in A:
                if (x + a) in sums_A:
                    valid_A = False
                    break
        
        # Check differences for A
        if valid_A:
            for a in A:
                d = x - a
                if d != 0:
                    if d in diffs_A or d in diffs_B:
                        valid_A = False
                        break
                    if -d in diffs_A or -d in diffs_B:
                        valid_A = False
                        break
        
        # If A is valid, we can add to A
        # But we also need to check if we *should* add to A or B
        # The caller decides. Here we just check validity.
        
        # Check Sidon condition for B
        valid_B = True
        if (x + x) in sums_B:
            valid_B = False
        else:
            for b in B:
                if (x + b) in sums_B:
                    valid_B = False
                    break
        
        if valid_B:
            for b in B:
                d = x - b
                if d != 0:
                    if d in diffs_A or d in diffs_B:
                        valid_B = False
                        break
                    if -d in diffs_A or -d in diffs_B:
                        valid_B = False
                        break
        
        return valid_A, valid_B

    def add_to_set(x, A, sums, diffs, diffs_other):
        # Add x to A, update sums and diffs
        sums.add(x + x)
        for a in A:
            sums.add(x + a)
            d = x - a
            diffs.add(d)
            diffs.add(-d)
        A.append(x)

    # We will run multiple greedy attempts with different random orderings
    # and different tie-breaking heuristics.
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        random.seed(attempts)
        
        # Shuffle the candidate numbers
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # We can initialize with a small random seed or just start empty
        # Starting empty is fine, but we must ensure non-empty at the end.
        
        for x in candidates:
            valid_A, valid_B = try_add(x, A, B, sums_A, sums_B, diffs_A, diffs_B)
            
            if valid_A and valid_B:
                # Tie-breaking heuristic:
                # 1. Balance sizes: add to smaller set
                if len(A) < len(B):
                    add_to_set(x, A, sums_A, diffs_A, diffs_B)
                elif len(B) < len(A):
                    add_to_set(x, B, sums_B, diffs_B, diffs_A)
                else:
                    # If equal size, pick randomly to avoid bias
                    if random.random() < 0.5:
                        add_to_set(x, A, sums_A, diffs_A, diffs_B)
                    else:
                        add_to_set(x, B, sums_B, diffs_B, diffs_A)
            elif valid_A:
                add_to_set(x, A, sums_A, diffs_A, diffs_B)
            elif valid_B:
                add_to_set(x, B, sums_B, diffs_B, diffs_A)
            # else: skip x
            
        # Ensure non-empty
        if not A:
            # If A is empty, try to force a valid element
            # This shouldn't happen often with 256 candidates, but just in case
            for x in range(1, 257):
                valid_A, _ = try_add(x, A, B, sums_A, sums_B, diffs_A, diffs_B)
                if valid_A:
                    add_to_set(x, A, sums_A, diffs_A, diffs_B)
                    break
        if not B:
            for x in range(1, 257):
                _, valid_B = try_add(x, A, B, sums_A, sums_B, diffs_A, diffs_B)
                if valid_B:
                    add_to_set(x, B, sums_B, diffs_B, diffs_A)
                    break
        
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A = A.copy()
            best_B = B.copy()
            
            # If we hit the theoretical max (which is hard to know exactly but 64+64=128 is the cap), we can stop
            # But 128 is likely not achievable due to difference constraints.
            # A good score is around 80-100.
            if best_score >= 100:
                # We can continue to try to improve, but if we are close to limit, break?
                # No, let's keep running until deadline to see if we can find a better one.
                pass

    return [best_A, best_B]
