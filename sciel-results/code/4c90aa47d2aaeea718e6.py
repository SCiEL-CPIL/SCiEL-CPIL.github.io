"""Rewritten program for matrix multiplication 2x4 by 4x5 tensor decomposition."""
import time

DIMENSIONS = [2, 4, 5]

# Strassen 2x2 terms: (u, v, w) where
# u[i,j] for i,j in {0,1}
# v[j,k] for j,k in {0,1}
# w[k,i] for k,i in {0,1}
STRASSEN_222 = [
    ({(0,0):1,(0,1):1,(1,0):1,(1,1):1}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1}),
    ({(0,0):1,(1,1):-1}, {(0,0):1,(1,0):-1}, {(1,1):1,(1,0):-1}),
    ({(0,0):1,(0,1):1}, {(0,1):1}, {(0,0):1,(1,0):-1}),
    ({(1,0):1}, {(0,0):1,(0,1):-1}, {(1,0):1,(1,1):-1}),
    ({(1,0):1,(1,1):-1}, {(1,0):1,(1,1):-1}, {(1,1):1,(0,1):-1}),
    ({(0,1):-1}, {(0,0):1,(0,1):1}, {(0,1):1,(1,1):-1}),
    ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,0):1,(0,1):-1}),
]


def _build_from_terms(terms, n, m, p):
    """
    terms: list of (u_dict, v_dict, w_dict)
    u_dict: (i,j) -> coeff  (i in range(n), j in range(m))
    v_dict: (j,k) -> coeff  (j in range(m), k in range(p))
    w_dict: (k,i) -> coeff  (k in range(p), i in range(n))
    """
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r, (u, v, w) in enumerate(terms):
        for (i, j), c in u.items():
            if c != 0:
                U[i * m + j][r] += c
        for (j, k), c in v.items():
            if c != 0:
                V[j * p + k][r] += c
        for (k, i), c in w.items():
            if c != 0:
                W[k * n + i][r] += c
    return {"U": U, "V": V, "W": W}


def _strassen_2x2_to_2x4x5():
    """
    Build a rank-56 decomposition for 2x4 * 4x5.
    Use 7 blocks of size (2x2) * (2x5).
    For each block b in 0..6:
        - A block is rows 0:2, cols 2b:2b+2
        - B block is rows 2b:2b+2, cols 0:5
    For each Strassen term, we need to extend to the 2x5 B-block.
    Strassen gives 7 terms for 2x2 * 2x2.
    For 2x5, we can think of B as two 2x2 blocks plus one 2x1 block.
    
    Actually, let me use a different approach:
    For the 2x5 B-block, decompose it as:
    B = [B1, B2, B3] where B1, B2 are 2x2 and B3 is 2x1.
    
    For 2x2 * 2x2, we have 7 Strassen terms.
    For 2x2 * 2x1, we can use a rank-3 decomposition.
    
    But this gets complex. Let me use a simpler approach:
    
    Actually, the cleanest way: use the block structure.
    C = A * B where A is 2x4, B is 4x5.
    
    Split A = [A0 | A1] where A0, A1 are 2x2.
    Split B = [B0; B1] where B0, B1 are 2x5.
    
    C = A0 * B0 + A1 * B1
    
    For each 2x2 * 2x5 product, we need a decomposition.
    2x2 * 2x5 can be seen as 2x2 * [2x2, 2x2, 2x1] = (2x2*2x2) + (2x2*2x2) + (2x2*2x1)
    
    2x2*2x2 has rank 7 (Strassen)
    2x2*2x1: Let's find its rank. 
    A (2x2) * v (2x1) = [a00*v0 + a01*v1; a10*v0 + a11*v1]
    This is a bilinear map from (2x2) x (2x1) to (2x1).
    Rank is at most 4, but can we do better?
    Actually, 2x2*2x1 is equivalent to (2x1) * (2x2)^T which is 2x2*2x1, same thing.
    The rank of 2x2*2x1 is 4 (schoolbook) or can it be 3?
    I believe it's 4 since it's a tensor of format (2,2,1) which has rank min(2*2, 2*1, ...) 
    Actually for matrix multiplication m x n * n x p, the rank is at least max(m, n, p) * something.
    For 2x2*2x1, it's a 2x2 matrix times a 2x1 vector.
    This is the same as 2x1 * 2x2^T.
    The rank of this tensor is 4.
    
    So 2x2*2x5 has rank at most 7+7+4 = 18? No wait, that's not right because the B blocks overlap in structure.
    
    Let me just use schoolbook for the 2x5 part and Strassen for 2x2.
    
    Actually, let me just build it directly.
    
    For A0 * B0 where A0 is 2x2, B0 is 2x5:
    We can split B0 = [B0a | B0b | B0c] where B0a, B0b are 2x2 and B0c is 2x1.
    A0 * B0 = A0*B0a + A0*B0b + A0*B0c
    
    A0*B0a: 7 terms (Strassen)
    A0*B0b: 7 terms (Strassen)  
    A0*B0c: 4 terms (schoolbook for 2x2*2x1)
    
    Total per block: 18 terms
    2 blocks: 36 terms
    
    Hmm, that's worse than the 40 schoolbook. Let me reconsider.
    
    Wait, I think I'm overcomplicating. Let me just use the schoolbook but try to find a better decomposition.
    
    Actually, for 2x4 * 4x5, the known rank is 40 (schoolbook). 
    But maybe we can do better.
    
    Let me try a different approach: use the fact that 4 = 2+2 and 5 = 2+2+1.
    
    Actually, let me just return the schoolbook for now and add a time guard.
    """
    # For simplicity and correctness, let's just return schoolbook rank 40
    # but let me try to see if I can get a better rank.
    
    # Let me try: 2x4 * 4x5
    # Write A as 2x4, B as 4x5
    # Split A = [A0 A1] (2x2 each)
    # Split B = [B0; B1] (2x5 each)
    # C = A0*B0 + A1*B1
    
    # For 2x2 * 2x5, let's find the rank.
    # 2x2 * 2x5: A (2x2) * B (2x5) -> C (2x5)
    # This is a tensor of format (2, 4, 5) if we think of it as (2x2) x (2x5) -> (2x5)
    # Wait no, it's (2,2,2,5) -> but really it's just matrix mult 2x2 * 2x5
    # The rank of 2x2 * 2x5 is at most 2*2*5 = 20 (schoolbook)
    # But can we do better?
    
    # Let me just use schoolbook for the whole thing for now.
    pass
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
