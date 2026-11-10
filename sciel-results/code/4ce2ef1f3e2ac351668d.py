import time
import sys

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    LCM = 30030
    
    start_time = time.time()
    timeout = 85
    
    best_result = None
    best_W = 33
    
    for L in range(1, LCM + 1):
        if time.time() - start_time > timeout:
            break
        
        firsts = []
        for d in divisors:
            k = (L + d - 1) // d
            firsts.append(k * d)
        
        min_W = max(firsts) - L
        if min_W >= best_W:
            continue
        
        for W in range(min_W, best_W):
            available = []
            valid = True
            for d in divisors:
                k = (L + d - 1) // d
                first = k * d
                if first > L + W:
                    valid = False
                    break
                multiples = []
                x = first
                while x <= L + W:
                    multiples.append(x)
                    x += d
                available.append(multiples)
            
            if not valid:
                continue
            
            indices = sorted(range(n), key=lambda i: len(available[i]))
            used = set()
            choices = [0] * n
            
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
                        used.remove(val)
                return False
            
            if backtrack(0):
                result = choices[:]
                if len(set(result)) == n and all(0 < x <= 1000000 for x in result) and all(result[i] % divisors[i] == 0 for i in range(n)):
                    best_W = W
                    best_result = [int(x) for x in result]
                    break
    
    if best_result:
        return best_result
    return [int(x) for x in divisors]
