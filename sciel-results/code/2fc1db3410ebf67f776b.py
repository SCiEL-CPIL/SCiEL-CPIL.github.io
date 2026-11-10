import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit
    
    LCM = 30030  # LCM of all divisors
    
    best_result = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    best_span = 33  # 39 - 6
    
    # Minimum possible span is 9 (10 distinct integers)
    for W in range(9, 200):
        if time.time() > deadline:
            break
        
        found = False
        for L in range(LCM):
            if time.time() > deadline:
                break
            
            available = []
            valid = True
            for d in divisors:
                r = L % d
                offset = (d - r) % d
                if offset > W:
                    valid = False
                    break
                count = (W - offset) // d + 1
                multiples = [L + offset + j * d for j in range(count)]
                available.append(multiples)
            
            if not valid:
                continue
            
            # Most constrained first
            indices = list(range(n))
            indices.sort(key=lambda i: len(available[i]))
            
            choices = [None] * n
            used = set()
            
            def backtrack(idx):
                if idx == n:
                    return True
                i = indices[idx]
                for val in available[i]:
                    if val not in used:
                        used.add(val)
                        choices[i] = val
                        if backtrack(idx + 1):
                            return True
                        choices[i] = None
                        used.remove(val)
                return False
            
            if backtrack(0):
                result = [int(choices[i]) for i in range(n)]
                span = max(result) - min(result)
                if span < best_span:
                    best_span = span
                    best_result = result
                found = True
                break  # Found valid solution for this W
        
        if found:
            return best_result
    
    return best_result
