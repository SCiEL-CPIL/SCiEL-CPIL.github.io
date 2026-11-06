import time
import random
from itertools import combinations, combinations_with_replacement

N = 256
MAX_EACH = 64
DEADLINE = 55.0  # Safety margin

def construct():
    start_time = time.time()
    best_score = 0
    best_A = []
    best_B = []
    
    # We will run multiple restarts
    while time.time() - start_time < DEADLINE:
        # Randomize the order of elements to try
        elements = list(range(1, N + 1))
        random.shuffle(elements)
        
        A = []
        B = []
        sum_bits_A = 0
        diff_bits_A = 0
        sum_bits_B = 0
        diff_bits_B = 0
        
        # Phase 1: Greedy build A
        for x in elements:
            if len(A) >= MAX_EACH:
                break
            # Check if x can be added to A
            # 1. Sum collisions with A
            ok = True
            if (sum_bits_A >> (x + x)) & 1:
                ok = False
            else:
                for a in A:
                    if (sum_bits_A >> (x + a)) & 1:
                        ok = False
                        break
            if not ok:
                continue
            
            # 2. Diff collisions with A
            if ok:
                for a in A:
                    d = abs(x - a)
                    if (diff_bits_A >> d) & 1:
                        ok = False
                        break
            if not ok:
                continue
            
            # 3. Diff disjointness with B
            if ok:
                for a in A:
                    d = abs(x - a)
                    if (diff_bits_B >> d) & 1:
                        ok = False
                        break
            if not ok:
                continue
            
            # Add x to A
            A.append(x)
            sum_bits_A |= (1 << (x + x))
            for a in A[:-1]:
                sum_bits_A |= (1 << (x + a))
                d = abs(x - a)
                diff_bits_A |= (1 << d)
        
        # Phase 2: Greedy build B
        for x in elements:
            if len(B) >= MAX_EACH:
                break
            # Check if x can be added to B
            ok = True
            # 1. Sum collisions with B
            if (sum_bits_B >> (x + x)) & 1:
                ok = False
            else:
                for b in B:
                    if (sum_bits_B >> (x + b)) & 1:
                        ok = False
                        break
            if not ok:
                continue
            
            # 2. Diff collisions with B
            if ok:
                for b in B:
                    d = abs(x - b)
                    if (diff_bits_B >> d) & 1:
                        ok = False
                        break
            if not ok:
                continue
            
            # 3. Diff disjointness with A
            if ok:
                for b in B:
                    d = abs(x - b)
                    if (diff_bits_A >> d) & 1:
                        ok = False
                        break
            if not ok:
                continue
            
            # Add x to B
            B.append(x)
            sum_bits_B |= (1 << (x + x))
            for b in B[:-1]:
                sum_bits_B |= (1 << (x + b))
                d = abs(x - b)
                diff_bits_B |= (1 << d)
        
        # Phase 3: Local Search / Refinement
        # Try to add more elements to A and B in random order
        improved = True
        while improved and time.time() - start_time < DEADLINE:
            improved = False
            # Try adding to A
            for x in elements:
                if x in A or len(A) >= MAX_EACH:
                    continue
                ok = True
                if (sum_bits_A >> (x + x)) & 1:
                    ok = False
                else:
                    for a in A:
                        if (sum_bits_A >> (x + a)) & 1:
                            ok = False
                            break
                if not ok: continue
                
                if ok:
                    for a in A:
                        d = abs(x - a)
                        if (diff_bits_A >> d) & 1 or (diff_bits_B >> d) & 1:
                            ok = False
                            break
                if not ok: continue
                
                A.append(x)
                sum_bits_A |= (1 << (x + x))
                for a in A[:-1]:
                    sum_bits_A |= (1 << (x + a))
                    d = abs(x - a)
                    diff_bits_A |= (1 << d)
                improved = True
                break
            
            # Try adding to B
            if not improved:
                for x in elements:
                    if x in B or len(B) >= MAX_EACH:
                        continue
                    ok = True
                    if (sum_bits_B >> (x + x)) & 1:
                        ok = False
                    else:
                        for b in B:
                            if (sum_bits_B >> (x + b)) & 1:
                                ok = False
                                break
                    if not ok: continue
                    
                    if ok:
                        for b in B:
                            d = abs(x - b)
                            if (diff_bits_B >> d) & 1 or (diff_bits_A >> d) & 1:
                                ok = False
                                break
                    if not ok: continue
                    
                    B.append(x)
                    sum_bits_B |= (1 << (x + x))
                    for b in B[:-1]:
                        sum_bits_B |= (1 << (x + b))
                        d = abs(x - b)
                        diff_bits_B |= (1 << d)
                    improved = True
                    break
        
        current_score = len(A) + len(B)
        if current_score > best_score:
            best_score = current_score
            best_A = list(A)
            best_B = list(B)
            
            # If we hit the theoretical max, we can stop early?
            # Max sum is likely around 32-34.
            if best_score >= 34:
                break

    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2] if 2 not in best_A else [3]
        
    return [best_A, best_B]
