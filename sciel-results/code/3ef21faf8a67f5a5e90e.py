# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Returns:
        points: np.ndarray of shape (k, 2, 2) integer array.
    """
    start_time = time.time()
    time_limit = 600.0
    margin = 10.0
    
    p = 7
    k = 3  # alpha^2 = 3 in F_49 = F_7[alpha]
    
    # y_special: y_a is a square mod p
    # Squares mod 7: 0^2=0, 1^2=1, 2^2=4, 3^2=2, 4^2=2, 5^2=4, 6^2=1
    # So squares are {0, 1, 2, 4}
    squares = set((u * u) % p for u in range(p))
    
    points = []
    # Iterate through all 2401 points
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # Check if excluded (in unital and y not special)
                    # Unital condition: 2*xa + ya^2 - yb^2*k == 0 mod p
                    if (2 * xa + ya * ya - yb * yb * k) % p == 0:
                        if ya not in squares:
                            continue  # Excluded
                    points.append([[xa, xb], [ya, yb]])
                    
                    # Time check (unlikely to hit, but safe)
                    if time.time() - start_time > time_limit - margin:
                        break
                if time.time() - start_time > time_limit - margin:
                    break
            if time.time() - start_time > time_limit - margin:
                break
        if time.time() - start_time > time_limit - margin:
            break

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
