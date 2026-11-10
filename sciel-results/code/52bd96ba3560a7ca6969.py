import time
import random

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin
    best_size = 0
    best_A = []
    best_B = []
    
    max_restarts = 3000
    
    for _ in range(max_restarts):
        if time.time() - start_time > timeout - 2.0:
            break
        
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # Use a mix of ordering strategies
        strategy = random.random()
        if strategy < 0.4:
            candidates = list(range(1, 257))
            random.shuffle(candidates)
        elif strategy < 0.7:
            candidates = list(range(1, 257))
            # Sort with some randomization - prefer smaller numbers
            candidates.sort(key=lambda x: x + random.random() * 50)
        else:
            candidates = list(range(1, 257))
            # Random permutation but biased toward middle
            candidates.sort(key=lambda x: abs(x - 128) + random.random() * 100)
        
        for x in candidates:
            if time.time() - start_time > timeout - 2.0:
                break
            
            valid_A = True
            if x + x in sums_A:
                valid_A = False
            else:
                for a in A:
                    if x + a in sums_A:
                        valid_A = False
                        break
            
            if valid_A:
                new_diffs = set()
                for a in A:
                    d1 = x - a
                    d2 = a - x
                    new_diffs.add(d1)
                    new_diffs.add(d2)
                if new_diffs & diffs_B:
                    valid_A = False
            
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
            
            if valid_A and valid_B:
                # Balance with some randomness
                if len(A) < len(B) - 1:
                    side = 'A'
                elif len(B) < len(A) - 1:
                    side = 'B'
                else:
                    # When balanced, try the one that creates fewer new differences
                    if valid_A and valid_B:
                        n_diffs_A = 0
                        for a in A:
                            if (x - a) not in diffs_A and (a - x) not in diffs_A:
                                n_diffs_A += 2
                        n_diffs_B = 0
                        for b in B:
                            if (x - b) not in diffs_B and (b - x) not in diffs_B:
                                n_diffs_B += 2
                        if n_diffs_A < n_diffs_B:
                            side = 'A'
                        elif n_diffs_B < n_diffs_A:
                            side = 'B'
                        else:
                            side = random.choice(['A', 'B'])
                    else:
                        side = random.choice(['A', 'B'])
            elif valid_A:
                side = 'A'
            elif valid_B:
                side = 'B'
            else:
                continue
            
            if side == 'A':
                A.append(x)
                sums_A.add(x + x)
                for a in A[:-1]:
                    sums_A.add(x + a)
                for a in A[:-1]:
                    diffs_A.add(x - a)
                    diffs_A.add(a - x)
            else:
                B.append(x)
                sums_B.add(x + x)
                for b in B[:-1]:
                    sums_B.add(x + b)
                for b in B[:-1]:
                    diffs_B.add(x - b)
                    diffs_B.add(b - x)
        
        # Local improvement: try to add more elements
        if time.time() - start_time < timeout - 5.0:
            remaining = set(range(1, 257)) - set(A) - set(B)
            # Try adding elements that fit
            for x in sorted(remaining, key=lambda v: random.random()):
                if time.time() - start_time > timeout - 5.0:
                    break
                valid_A = True
                if x + x in sums_A:
                    valid_A = False
                else:
                    for a in A:
                        if x + a in sums_A:
                            valid_A = False
                            break
                if valid_A:
                    new_diffs = set()
                    for a in A:
                        new_diffs.add(x - a)
                        new_diffs.add(a - x)
                    if new_diffs & diffs_B:
                        valid_A = False
                
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
                        new_diffs.add(x - b)
                        new_diffs.add(b - x)
                    if new_diffs & diffs_A:
                        valid_B = False
                
                if valid_A and not valid_B:
                    A.append(x)
                    sums_A.add(x + x)
                    for a in A[:-1]:
                        sums_A.add(x + a)
                    for a in A[:-1]:
                        diffs_A.add(x - a)
                        diffs_A.add(a - x)
                elif valid_B and not valid_A:
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
            best_A = list(A)
            best_B = list(B)

    return [best_A, best_B]
