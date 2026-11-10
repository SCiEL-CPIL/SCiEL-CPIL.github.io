# EVOLVE-BLOCK-START
import numpy as np

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_{49}^2 by taking the complement of the Hermitian unital.
    
    The Hermitian unital U in F_{49}^2 is defined by Tr(x) + Norm(y) = 0.
    For x = xa + xb*alpha, Tr(x) = 2*xa.
    For y = ya + yb*alpha, Norm(y) = ya^2 - 3*yb^2.
    So U = {(x, y) : 2*xa + ya^2 - 3*yb^2 = 0 mod 7}.
    
    It is a known property that N = F_{49}^2 \ U is a Nikodym set.
    - For x in U, there exists a tangent line through x meeting U only at x.
    - For x not in U, there exists a line through x disjoint from U.
    
    This yields |S| = |U| = 49^(3/2) + 1 = 344.
    Score = 344.
    """
    p = 7
    q = 49
    k = 3  # QNR for p=7 is 3 (since 3^3 = 27 = 6 = -1 mod 7)
    
    # Total points
    total = q * q
    n_points = total
    
    # We will store indices of points in N
    N_indices = []
    
    for idx in range(n_points):
        # Decode index to (xa, xb, ya, yb)
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        
        # Check if point is in Unital U
        # Tr(x) + Norm(y) = 2*xa + (ya^2 - k*yb^2) == 0
        val = (2 * xa + ya * ya - k * yb * yb) % p
        
        if val != 0:
            # Point is NOT in U, so it is in N
            N_indices.append(idx)
            
    # Convert to (k, 2, 2) array
    result = np.zeros((len(N_indices), 2, 2), dtype=np.int64)
    for i, idx in enumerate(N_indices):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        result[i] = [[xa, xb], [ya, yb]]
        
    return result

# EVOLVE-BLOCK-END
