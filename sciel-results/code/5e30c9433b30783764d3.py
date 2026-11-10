import time
import random

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin
    best_size = 0
    best_A = []
    best_B = []
    
    # We will try multiple random restarts
    max_restarts = 5000 # Adjust based on speed
    
    for _ in range(max_restarts):
        if time.time() - start_time > timeout:
            break
            
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # Candidates are 1..256
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() - start_time > timeout:
                break
            
            added = False
            
            # Try adding to A
            # Check if x can be added to A
            valid_A = True
            # Check sums: x+x and x+a for a in A
            if x + x in sums_A:
                valid_A = False
            else:
                for a in A:
                    if x + a in sums_A:
                        valid_A = False
                        break
            
            if valid_A:
                # Check diffs: x-a and a-x for a in A
                new_diffs = set()
                for a in A:
                    d1 = x - a
                    d2 = a - x
                    new_diffs.add(d1)
                    new_diffs.add(d2)
                if new_diffs & diffs_B:
                    valid_A = False
                else:
                    # It's valid, but we might prefer B if B is smaller? 
                    # Or just take it. Let's try A first.
                    pass
            
            # Try adding to B
            valid_B = True
            if x + x in sums_B:
                valid_B = False
            else:
                for b in B:
                    if x + b in sums_B:
                        valid_B = False
                        break
            
            if valid_B:
                new_diffs = set()
                for b in B:
                    d1 = x - b
                    d2 = b - x
                    new_diffs.add(d1)
                    new_diffs.add(d2)
                if new_diffs & diffs_A:
                    valid_B = False
            
            # Decision logic:
            # If both valid, choose randomly or the one that leads to more future options?
            # Simple heuristic: choose the set with fewer elements to balance.
            # Or just pick one.
            
            if valid_A and valid_B:
                if len(A) < len(B):
                    side = 'A'
                elif len(B) < len(A):
                    side = 'B'
                else:
                    side = random.choice(['A', 'B'])
            elif valid_A:
                side = 'A'
            elif valid_B:
                side = 'B'
            else:
                continue # Cannot add x to either
            
            if side == 'A':
                # Add x to A
                A.append(x)
                sums_A.add(x + x)
                for a in A[:-1]:
                    sums_A.add(x + a)
                for a in A[:-1]:
                    diffs_A.add(x - a)
                    diffs_A.add(a - x)
            else:
                # Add x to B
                B.append(x)
                sums_B.add(x + x)
                for b in B[:-1]:
                    sums_B.add(x + b)
                for b in B[:-1]:
                    diffs_B.add(x - b)
                    diffs_B.add(b - x)
            
        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A = A
            best_B = B

    return [best_A, best_B]
