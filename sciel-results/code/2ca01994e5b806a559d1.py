import time

def construct():
    n = 12
    start_time = time.time()
    deadline = start_time + 50.0
    
    best_seq = None
    best_max = float('inf')
    
    # Try different starting values for the second element
    # a1 is always 1
    for a2 in range(2, 10): 
        a = [1, a2]
        sums = {0, 1, a2, 1+a2}
        
        valid = True
        for i in range(2, n):
            last_val = a[-1]
            x = last_val + 1
            # Find next valid x
            while x in sums:
                x += 1
                if time.time() > deadline:
                    valid = False
                    break
            if not valid:
                break
            
            a.append(x)
            new_sums = set(sums)
            for s in sums:
                new_sums.add(s + x)
            sums = new_sums
        
        if valid:
            current_max = a[-1]
            if current_max < best_max:
                best_max = current_max
                best_seq = a
                
    # If no sequence found (should not happen), fallback to powers of 2
    if best_seq is None:
        best_seq = [2**i for i in range(n)]
        
    return best_seq
