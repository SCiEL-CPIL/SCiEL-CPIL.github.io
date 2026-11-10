import time

def construct():
    p = 1009
    m = 5
    
    # Precompute quadratic residues and their roots
    is_res = [False] * p
    root_of = [0] * p
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            is_res[r] = True
            if root_of[r] == 0:
                root_of[r] = x
                
    # Find the smallest s such that s..s+m-1 are all non-zero QRs
    # s ranges from 1 to p-m
    for s in range(1, p - m + 1):
        # Check if all s+i are residues
        if all(is_res[s + i] for i in range(m)):
            roots = [root_of[s + i] for i in range(m)]
            return {'start': s, 'roots': roots}
            
    raise RuntimeError("No solution found")
