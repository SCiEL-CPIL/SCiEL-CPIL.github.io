import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 (F_49 = F_7[alpha], alpha^2 = 3).
    
    Strategy:
    Uses the standard algebraic construction for Nikodym sets in F_{p^2}^2.
    1. Define the Hermitian unital U = {(x, y) : Tr(x) + Norm(y) = 0}.
       Tr(x) = 2*x_a mod p.
       Norm(y) = y_a^2 - y_b^2*k mod p, where k=3 (non-residue mod 7).
    2. Define special y-coordinates Y = {(u^2 mod p, v) : u, v in F_p}.
       This means y_a must be a quadratic residue mod p.
    3. Exclude points S = { (x,y) in U : y not in Y }.
    4. Return N = F_49^2 \ S.
    
    This set N is a valid Nikodym set.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    p = 7
    k = 3  # alpha^2 = 3
    
    # Precompute squares mod p
    squares = set((u * u) % p for u in range(p))
    # Squares mod 7 are {0, 1, 2, 4}
    
    # Build the set of excluded points S
    # A point (xa, xb, ya, yb) is excluded if:
    # 1. (ya, yb) is NOT in Y. Since Y is defined by ya being a square, 
    #    this means ya is NOT in squares.
    # 2. The point is in the unital U.
    #    Unital condition: 2*xa + ya^2 - yb^2*k == 0 (mod p)
    #    => 2*xa == -ya^2 + yb^2*k (mod p)
    #    => xa == inv2 * (-ya^2 + yb^2*k) (mod p)
    
    inv2 = pow(2, p - 2, p)  # 4
    
    excluded = set()
    
    # Iterate over y coordinates
    for ya in range(p):
        # If ya is a square, (ya, yb) is in Y for all yb, so no points with this ya are excluded
        if ya in squares:
            continue
        
        for yb in range(p):
            # Calculate the unique xa that satisfies the unital equation
            xa = (inv2 * (-ya * ya + yb * yb * k)) % p
            
            # All xb values (0 to p-1) for this xa, ya, yb are excluded
            for xb in range(p):
                excluded.add((xa, xb, ya, yb))
    
    # Construct the Nikodym set N
    # N contains all points NOT in excluded
    points = []
    
    # To be safe against timeouts, we can break early, but this loop is very fast
    # 2401 iterations is trivial.
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (xa, xb, ya, yb) not in excluded:
                        points.append([[xa, xb], [ya, yb]])
                    
                    # Safety check, though unlikely to trigger for p=7
                    if time.time() > deadline - 5.0:
                        break
                if time.time() > deadline - 5.0:
                    break
            if time.time() > deadline - 5.0:
                break
        if time.time() > deadline - 5.0:
            break
    
    if not points:
        # Fallback, should never happen
        return np.array([[[0, 0], [0, 0]]], dtype=np.int64)
    
    return np.array(points, dtype=np.int64)
