# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 with q = 49, p = 7.
    Field: F_49 = F_7[alpha] with alpha^2 = 3.
    A point is [[x_a, x_b], [y_a, y_b]] representing (x_a + x_b*alpha, y_a + y_b*alpha).
    
    We use the baseline construction:
      Unital U = {(x, y) : Tr(x) + Norm(y) = 0}
      Special Y = {(u^2 mod p, v) : u, v in F_p}
      Excluded S = U \\ {points with y in Y}
      Nikodym set N = F_49^2 \\ S
    
    Returns:
        np.ndarray of shape (k, 2, 2) with dtype int64.
    """
    start_time = time.time()
    deadline = start_time + 600.0

    p = 7
    k = 3  # alpha^2 = 3 (smallest quadratic non-residue mod 7)

    # Precompute y_special: y_a must be a square mod p
    # Squares mod 7: 0^2=0, 1^2=1, 2^2=4, 3^2=2, 4^2=2, 5^2=4, 6^2=1
    # So squares are {0, 1, 2, 4}
    squares = set((u * u) % p for u in range(p))
    
    # Build the set of excluded points: in unital U and y_a NOT in squares
    # Unital condition: 2*xa + ya^2 - yb^2*k ≡ 0 (mod p)
    # So xa ≡ (-ya^2 + yb^2*k) / 2 (mod p)
    # 2^{-1} mod 7 = 4 since 2*4=8≡1
    
    inv2 = pow(2, p - 2, p)  # = 4
    
    # For each (ya, yb) where ya is NOT a square, compute the unique xa in the unital
    # Then all xb values (0..6) give excluded points
    # This is much faster than iterating all 2401 points
    
    excluded = set()
    for ya in range(p):
        if ya in squares:
            continue
        for yb in range(p):
            # xa = inv2 * (-ya^2 + yb^2 * k) mod p
            xa = (inv2 * (-ya * ya + yb * yb * k)) % p
            # All xb values are excluded
            for xb in range(p):
                excluded.add((xa, xb, ya, yb))
    
    # Build the Nikodym set: all points NOT in excluded
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (xa, xb, ya, yb) not in excluded:
                        points.append([[xa, xb], [ya, yb]])
                    
                    if time.time() > deadline - 5.0:
                        break
                if time.time() > deadline - 5.0:
                    break
            if time.time() > deadline - 5.0:
                break
        if time.time() > deadline - 5.0:
            break
    
    if not points:
        return np.array([[[0, 0], [0, 0]]], dtype=np.int64)
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
