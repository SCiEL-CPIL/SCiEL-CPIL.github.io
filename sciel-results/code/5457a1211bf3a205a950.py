import numpy as np
from itertools import product

def construct():
    best_A = list(range(1, 21))
    best_score = float('inf')
    
    def score(A):
        if len(set(A)) != 20:
            return float('inf')
        if max(A) > 10000:
            return float('inf')
        sums = set()
        prods = set()
        for a in A:
            for b in A:
                sums.add(a + b)
                prods.add(a * b)
        return max(len(sums), len(prods))
    
    # Update best with initial guess
    best_score = score(best_A)
    
    # Search over AP + GP unions
    # A = AP union GP
    # AP: a, a+d, ..., a+(k-1)d
    # GP: b, br, ..., br^(m-1)
    # Total 20 distinct elements.
    
    # Coarse search
    for k in range(5, 16):
        m = 20 - k
        if m < 1: continue
        
        # Try various APs
        for a in [1, 2, 3]:
            for d in [1, 2, 3, 5]:
                ap = [a + i*d for i in range(k)]
                if max(ap) > 10000: continue
                
                # Try various GPs
                for b in [1, 2, 3, 5, 10, 20, 50]:
                    for r in [2, 3]:
                        gp = []
                        val = b
                        for _ in range(m):
                            if val > 10000: break
                            gp.append(val)
                            val *= r
                        if len(gp) < m: continue
                        
                        A = sorted(list(set(ap + gp)))
                        if len(A) != 20: continue
                        if max(A) > 10000: continue
                        
                        s = score(A)
                        if s < best_score:
                            best_score = s
                            best_A = A
                            
    # Also try pure AP and pure GP
    for d in [1, 2, 3]:
        A = list(range(1, 21, d))[:20]
        if len(A) == 20:
            s = score(A)
            if s < best_score:
                best_score = s
                best_A = A
                
    return best_A

def score(A):
    sums = set()
    prods = set()
    for a in A:
        for b in A:
            sums.add(a + b)
            prods.add(a * b)
    return max(len(sums), len(prods))
