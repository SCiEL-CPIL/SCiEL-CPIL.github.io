import time
import random

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    n = 256
    candidates = list(range(1, n + 1))
    
    best_size = 0
    best_A = []
    best_B = []
    
    # Theoretical max is 32.
    # If we find 32, we can stop.
    
    max_restarts = 50000
    
    def try_greedy(order, balance_threshold=0):
        """Greedy construction with given order. balance_threshold: prefer smaller set if size diff > threshold"""
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for x in order:
            # Check if we can add to A
            ok_A = True
            if 2 * x in sums_A:
                ok_A = False
            else:
                for a in A:
                    s = x + a
                    if s in sums_A:
                        ok_A = False
                        break
                    d = abs(x - a)
                    if d in diffs_A or d in diffs_B:
                        ok_A = False
                        break
            
            # Check if we can add to B
            ok_B = True
            if 2 * x in sums_B:
                ok_B = False
            else:
                for a in B:
                    s = x + a
                    if s in sums_B:
                        ok_B = False
                        break
                    d = abs(x - a)
                    if d in diffs_B or d in diffs_A:
                        ok_B = False
                        break
            
            if ok_A and ok_B:
                # Both possible: prefer the smaller set to balance
                if len(A) <= len(B):
                    # Add to A
                    sums_A.add(2 * x)
                    for a in A:
                        sums_A.add(x + a)
                        diffs_A.add(abs(x - a))
                    A.append(x)
                else:
                    # Add to B
                    sums_B.add(2 * x)
                    for a in B:
                        sums_B.add(x + a)
                        diffs_B.add(abs(x - a))
                    B.append(x)
            elif ok_A:
                # Add to A
                sums_A.add(2 * x)
                for a in A:
                    sums_A.add(x + a)
                    diffs_A.add(abs(x - a))
                A.append(x)
            elif ok_B:
                # Add to B
                sums_B.add(2 * x)
                for a in B:
                    sums_B.add(x + a)
                    diffs_B.add(abs(x - a))
                B.append(x)
            # else: skip x
        
        return A, B
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Use different ordering strategies
        strategy = restart % 4
        if strategy == 0:
            order = candidates[:]
            random.shuffle(order)
        elif strategy == 1:
            order = candidates[:]
            random.shuffle(order)
            order.reverse()
        elif strategy == 2:
            # Modular ordering for diversity
            order = [i for i in range(1, n+1) if i % 2 == 1]
            order2 = [i for i in range(1, n+1) if i % 2 == 0]
            random.shuffle(order)
            random.shuffle(order2)
            # Interleave
            merged = []
            for i in range(max(len(order), len(order2))):
                if i < len(order):
                    merged.append(order[i])
                if i < len(order2):
                    merged.append(order2[i])
            order = merged
        else:
            # Random with slight bias toward middle values
            order = candidates[:]
            random.shuffle(order)
            # Sort by distance from center, then shuffle within groups
            center = (n + 1) / 2.0
            order.sort(key=lambda x: (abs(x - center) // 32, random.random()))
        
        A, B = try_greedy(order)
        
        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A = A[:]
            best_B = B[:]
            
            if best_size >= 32:
                return [best_A, best_B]
        
        # Local improvement: try to add any missed elements
        if current_size < 32:
            A_set = set(A)
            B_set = set(B)
            sums_A = set()
            sums_B = set()
            diffs_A = set()
            diffs_B = set()
            for a in A:
                sums_A.add(2*a)
            for i in range(len(A)):
                for j in range(i+1, len(A)):
                    sums_A.add(A[i]+A[j])
                    diffs_A.add(abs(A[i]-A[j]))
            for b in B:
                sums_B.add(2*b)
            for i in range(len(B)):
                for j in range(i+1, len(B)):
                    sums_B.add(B[i]+B[j])
                    diffs_B.add(abs(B[i]-B[j]))
            
            # Try adding each candidate not in A or B
            for x in candidates:
                if x in A_set or x in B_set:
                    continue
                
                # Try adding to A
                ok_A = True
                if 2*x in sums_A:
                    ok_A = False
                else:
                    for a in A:
                        if (x+a) in sums_A:
                            ok_A = False
                            break
                        d = abs(x-a)
                        if d in diffs_A or d in diffs_B:
                            ok_A = False
                            break
                
                if ok_A:
                    sums_A.add(2*x)
                    for a in A:
                        sums_A.add(x+a)
                        diffs_A.add(abs(x-a))
                    A.append(x)
                    A_set.add(x)
                    current_size += 1
                    continue
                
                # Try adding to B
                ok_B = True
                if 2*x in sums_B:
                    ok_B = False
                else:
                    for b in B:
                        if (x+b) in sums_B:
                            ok_B = False
                            break
                        d = abs(x-b)
                        if d in diffs_B or d in diffs_A:
                            ok_B = False
                            break
                
                if ok_B:
                    sums_B.add(2*x)
                    for b in B:
                        sums_B.add(x+b)
                        diffs_B.add(abs(x-b))
                    B.append(x)
                    B_set.add(x)
                    current_size += 1
            
            if current_size > best_size:
                best_size = current_size
                best_A = A[:]
                best_B = B[:]
                
                if best_size >= 32:
                    return [best_A, best_B]
    
    if best_size == 0:
        return [[1], [2]]
        
    return [best_A, best_B]
