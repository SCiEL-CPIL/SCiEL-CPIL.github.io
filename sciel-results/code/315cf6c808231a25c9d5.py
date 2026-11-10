# EVOLVE-BLOCK-START
import numpy as np

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_{49}^2 by exploiting the affine structure.

    The strategy is to start from a full affine plane (N = F_{49}^2) and try to
    remove points from N to make N as small as possible, while preserving the
    Nikodym property. Since the full plane trivially satisfies the Nikodym condition,
    we can greedily attempt to remove points. However, the full plane is not minimal.
    
    Instead, we use a known construction for Nikodym sets in F_q^2 for q=p^2:
    For F_q^2, a Nikodym set can be constructed using the Hermitian curve. 
    Specifically, consider the Hermitian unital U = {(x, y) : Tr(x) + Norm(y) = 0}.
    A set of the form N = F_q^2 \ (U \ Y) where Y is a carefully chosen subset 
    of y-coordinates yields a Nikodym set.
    
    From the seed program, the construction uses:
    - Unital U: {(x, y) : 2*x_a + y_a^2 - k*y_b^2 = 0 mod p} where k is the smallest QNR.
    - Special Y: y_a is a quadratic residue mod p.
    - S = U \ Y (points in U whose y_a is NOT a square).
    - N = F_{49}^2 \ S.
    
    We can try to generalize this by changing which points to exclude. 
    The key insight is that for a point x to be excluded from N (i.e., x in S),
    every line through x that doesn't contain any other point of S (except possibly x itself)
    must have its other points in N. But since we want N to be small, S must be large.
    
    Let's implement the baseline construction and then try to extend S.
    """
    p = 7
    q = p * p  # 49
    
    # Find smallest quadratic non-residue mod 7
    k = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k = c
            break
    if k is None:
        k = 3  # fallback for p=7, 3 is QNR
    
    # Generate all points as flat indices 0..2400
    # Point index = ((xa * p + xb) * p + ya) * p + yb
    total = p * p * p * p
    
    # Function to get point from index
    def idx_to_point(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        return xa, xb, ya, yb
    
    def point_to_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb
    
    # Check if point is in unital U
    def in_unital(xa, xb, ya, yb):
        # Tr(x) = 2*xa, Norm(y) = ya^2 - k*yb^2
        return (2 * xa + ya * ya - k * yb * yb) % p == 0
    
    # Quadratic residues mod p
    qr = set((u * u) % p for u in range(p))
    
    # Baseline S: points in U with ya NOT in qr
    # New Strategy: Try points in U with ya IN qr to maximize S size
    # QR = {0, 1, 2, 4}, Non-QR = {3, 5, 6}
    # Size of S will be ~ 343 * 4/7 = 196 vs 147.
    
    S = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if in_unital(xa, xb, ya, yb) and ya in qr:
                        S.add(point_to_idx(xa, xb, ya, yb))
    
    # N is the complement of S
    N_indices = []
    for idx in range(total):
        if idx not in S:
            N_indices.append(idx)
    
    # Convert to (k, 2, 2) array
    result = np.zeros((len(N_indices), 2, 2), dtype=np.int64)
    for i, idx in enumerate(N_indices):
        xa, xb, ya, yb = idx_to_point(idx)
        result[i] = [[xa, xb], [ya, yb]]
    
    return result

# EVOLVE-BLOCK-END
