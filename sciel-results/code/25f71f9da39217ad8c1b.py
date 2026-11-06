import time

def construct():
    p = 1009
    
    # 1. Precompute quadratic residues
    # is_qr[r] = True if r is a non-zero quadratic residue mod p
    # root[r] = smallest x in [1, p-1] such that x^2 % p == r
    is_qr = [False] * p
    root = [0] * p
    
    for x in range(1, p):
        r = (x * x) % p
        # r is non-zero because x is not divisible by p
        if not is_qr[r]:
            is_qr[r] = True
            root[r] = x
        
    # 2. Find minimal s
    # s must be such that s, s+1, s+2, s+3, s+4 are all in [1, p-1]
    # Max s is p - 5 = 1004
    for s in range(1, p - 4):
        # Check if s..s+4 are all quadratic residues
        if all(is_qr[s + i] for i in range(5)):
            # Construct roots
            roots = [root[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}
            
    raise RuntimeError('No run found')
