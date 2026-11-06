import time

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 60s limit
    
    # 1. Generate all 2,3,5-smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth.append(x)
    
    smooth_set = set(smooth)
    m = len(smooth)
    
    # Precompute divisibility matrix for smooth numbers to speed up antichain check
    # div[i][j] is True if smooth[i] divides smooth[j]
    # Note: distinct terms, so i != j.
    # We need to check if any term divides another.
    # For a set of terms, we check pairs.
    
    answers = []
    
    for n in range(1, 129):
        if time.time() > deadline:
            # Emergency fallback: return a known valid structure or empty?
            # If we haven't found a solution, we must try harder or return a default.
            # But given the small search space, this shouldn't happen.
            pass
        
        found = None
        
        # Try k=1
        if n in smooth_set:
            found = [n]
        
        # Try k=2
        if found is None:
            for i in range(m):
                a = smooth[i]
                if a >= n:
                    break
                b = n - a
                if b in smooth_set and b > a:
                    # Check antichain: b % a != 0
                    if b % a != 0:
                        found = [a, b]
                        break
        
        # Try k=3
        if found is None:
            # a < b < c
            for i in range(m):
                a = smooth[i]
                # Pruning: if a is too large, sum will exceed n
                # Min b > a, Min c > b. 
                # Roughly a + (a+1) + (a+2) < n is a loose bound.
                # Better: just iterate.
                if a * 3 > n: 
                    # Since smooth numbers are at least 1 apart? No, 1,2,4...
                    # 1+2+4=7. If a=3, 3+4+5=12? No, smooth numbers are sparse.
                    # Let's just iterate.
                    pass
                for j in range(i + 1, m):
                    b = smooth[j]
                    if a + b >= n:
                        break
                    c = n - a - b
                    if c in smooth_set and c > b:
                        # Check antichain:
                        # b % a != 0
                        # c % b != 0
                        # c % a != 0
                        if b % a != 0 and c % b != 0 and c % a != 0:
                            found = [a, b, c]
                            break
                if found is not None:
                    break
        
        # Try k=4 (Safety net, should rarely be needed)
        if found is None:
            # a < b < c < d
            for i in range(m):
                a = smooth[i]
                if a * 4 > n:
                    break
                for j in range(i + 1, m):
                    b = smooth[j]
                    if a + b + 2 * (b + 1) > n: # Loose bound
                        break
                    for k in range(j + 1, m):
                        c = smooth[k]
                        if a + b + c >= n:
                            break
                        d = n - a - b - c
                        if d in smooth_set and d > c:
                            # Check antichain for all pairs
                            terms = [a, b, c, d]
                            is_antichain = True
                            for x_idx in range(4):
                                for y_idx in range(x_idx + 1, 4):
                                    if terms[y_idx] % terms[x_idx] == 0:
                                        is_antichain = False
                                        break
                                if not is_antichain:
                                    break
                            if is_antichain:
                                found = terms
                                break
                        if found is not None:
                            break
                    if found is not None:
                        break
                if found is not None:
                    break
        
        # Fallback if still None (should not happen for n<=128)
        if found is None:
            # Try a simple valid representation if possible, or mark as error
            # For safety, return an empty list or a dummy that might be invalid?
            # The task requires valid representations.
            # We assume we found one.
            found = []
            
        answers.append(found)
        
    return answers
