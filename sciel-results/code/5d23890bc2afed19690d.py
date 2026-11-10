# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 as the complement of the affine Hermitian curve.
    
    The Hermitian curve in P^2(F_49) is x^(q+1) + y^(q+1) + z^(q+1) = 0.
    The affine part (z=1) is x^(q+1) + y^(q+1) + 1 = 0.
    This curve has q^3 - q = 336 points for q=7.
    The complement of a unital is a Nikodym set.
    """
    p = 7
    q = p * p  # 49
    
    # Field F_49 = F_7[alpha] with alpha^2 = 3
    # Element is (a, b) representing a + b*alpha
    
    def norm_f49(elem):
        a, b = elem
        return (a*a - 3*b*b) % p
    
    # Precompute norm for all elements in F_49
    # Index 0..48 corresponds to (a, b) where index = a*7 + b
    norms = [0] * q
    elements = []
    for a in range(p):
        for b in range(p):
            idx = a * p + b
            elements.append((a, b))
            norms[idx] = norm_f49((a, b))
    
    # Precompute solutions to N(x) = c for c in 0..6
    # sol[c] is a list of indices in 0..48 such that N(x) = c
    sols = [[] for _ in range(p)]
    for i in range(q):
        c = norms[i]
        sols[c].append(i)
        
    # Generate points of the affine Hermitian curve H_aff
    # Equation: N(x) + N(y) + 1 = 0 => N(x) = -N(y) - 1
    # We mark these points as excluded from N
    
    # Create a set of excluded points (as indices in 0..2400)
    # Point index = (xa, xb, ya, yb) -> (xa*7+xb)*49 + (ya*7+yb)
    excluded = set()
    
    for y_idx in range(q):
        ya, yb = elements[y_idx]
        ny = norms[y_idx]
        cx = (-ny - 1) % p
        
        # Get all x such that N(x) = cx
        for x_idx in sols[cx]:
            xa, xb = elements[x_idx]
            # Point is (x, y)
            # x index = xa*7 + xb = x_idx
            # y index = ya*7 + yb = y_idx
            # Global index = x_idx * 49 + y_idx
            global_idx = x_idx * q + y_idx
            excluded.add(global_idx)
            
    # Build the result array
    # Total points = 2401
    # in_N[i] = True if i not in excluded
    
    # We need to return points in N
    # Point i corresponds to:
    # x_idx = i // 49
    # y_idx = i % 49
    # xa, xb = elements[x_idx]
    # ya, yb = elements[y_idx]
    
    result_list = []
    for i in range(q * q):
        if i in excluded:
            continue
        x_idx = i // q
        y_idx = i % q
        xa, xb = elements[x_idx]
        ya, yb = elements[y_idx]
        result_list.append([xa, xb, ya, yb])
        
    result = np.array(result_list, dtype=np.int64)
    # Reshape to (k, 2, 2)
    result = result.reshape(-1, 2, 2)
    
    return result

# EVOLVE-BLOCK-END
