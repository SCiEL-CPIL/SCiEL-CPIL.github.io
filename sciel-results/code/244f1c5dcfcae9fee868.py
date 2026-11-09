"""Matrix multiplication 2x3x3: rank-17 exact tensor decomposition using block decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """Return an exact rank-17 tensor decomposition for 2x3x3 matrix multiplication.
    
    Uses block decomposition:
    - 2x2x2 block (A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]): Strassen rank-7
    - 2x2x1 block (A[0:2,0:2] * B[0:2,2:3] -> C[0:2,2:3]): schoolbook rank-4
    - 2x1x3 block (A[0:2,2:3] * B[2:3,0:3] -> C[0:2,0:3]): schoolbook rank-6
    Total: 7 + 4 + 6 = 17
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=3
    rank = 17
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Helper to add a term
    def add_term(u_vals, v_vals, w_vals):
        nonlocal col
        U[:, col] = u_vals
        V[:, col] = v_vals
        W[:, col] = w_vals
        col += 1
    
    # ===== Block 1: 2x2x2 (Strassen rank 7) =====
    # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # U indices: i*3+j for i in {0,1}, j in {0,1} -> {0,1,3,4}
    # V indices: j*3+k for j in {0,1}, k in {0,1} -> {0,1,3,4}
    # W indices: k*2+i for k in {0,1}, i in {0,1} -> {0,1,2,3}
    
    # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
    # U: U[0]=1, U[4]=1
    # V: V[0]=1, V[4]=1
    # W: C[0,0]=+1, C[1,1]=+1 -> W[0]=1, W[3]=1
    add_term(
        [1,0,0, 0,1,0],
        [1,0,0, 0,1,0, 0,0,0],
        [1,0, 0,0, 0,0, 0,1]
    )
    
    # M2 = (A[1,0]+A[1,1])B[0,0]
    # U: U[3]=1, U[4]=1
    # V: V[0]=1
    # W: C[1,0]=+1, C[1,1]=-1 -> W[2]=1, W[3]=-1
    add_term(
        [0,0,0, 1,1,0],
        [1,0,0, 0,0,0, 0,0,0],
        [0,0, 1,0, 0,0, 0,-1]
    )
    
    # M3 = A[0,0](B[0,1]-B[1,1])
    # U: U[0]=1
    # V: V[1]=1, V[4]=-1
    # W: C[0,1]=+1, C[1,1]=+1 -> W[1]=1, W[3]=1
    add_term(
        [1,0,0, 0,0,0],
        [0,1,0, 0,0,0, 0,0,0],
        [0,1, 0,0, 0,0, 0,1]
    )
    
    # M4 = A[1,1](B[1,0]-B[0,0])
    # U: U[4]=1
    # V: V[3]=1, V[0]=-1
    # W: C[0,0]=+1, C[1,0]=+1 -> W[0]=1, W[2]=1
    add_term(
        [0,0,0, 0,1,0],
        [-1,0,0, 1,0,0, 0,0,0],
        [1,0, 1,0, 0,0, 0,0]
    )
    
    # M5 = (A[0,0]+A[0,1])B[1,1]
    # U: U[0]=1, U[1]=1
    # V: V[4]=1
    # W: C[0,0]=-1, C[0,1]=+1 -> W[0]=-1, W[1]=1
    add_term(
        [1,1,0, 0,0,0],
        [0,0,0, 0,1,0, 0,0,0],
        [-1,1, 0,0, 0,0, 0,0]
    )
    
    # M6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1])
    # U: U[3]=1, U[0]=-1
    # V: V[0]=1, V[1]=1
    # W: C[1,1]=+1 -> W[3]=1
    add_term(
        [-1,0,0, 1,0,0],
        [1,1,0, 0,0,0, 0,0,0],
        [0,0, 0,0, 0,0, 0,1]
    )
    
    # M7 = (A[0,1]-A[1,1])(B[1,0]+B[1,1])
    # U: U[1]=1, U[4]=-1
    # V: V[3]=1, V[4]=1
    # W: C[0,0]=+1 -> W[0]=1
    add_term(
        [0,1,0, 0,-1,0],
        [0,0,0, 1,1,0, 0,0,0],
        [1,0, 0,0, 0,0, 0,0]
    )
    
    # ===== Block 2: 2x2x1 (schoolbook rank 4) =====
    # A[0:2,0:2] * B[0:2,2:3] -> C[0:2,2:3]
    # U indices: i*3+j for i in {0,1}, j in {0,1} -> {0,1,3,4}
    # V indices: j*3+2 for j in {0,1} -> {2,5}
    # W indices: 2*2+i for i in {0,1} -> {4,5}
    
    # Term (i=0,j=0): A[0,0]*B[0,2] -> C[0,2]
    add_term(
        [1,0,0, 0,0,0],
        [0,0,1, 0,0,0, 0,0,0],
        [0,0, 0,0, 1,0]
    )
    
    # Term (i=0,j=1): A[0,1]*B[1,2] -> C[0,2]
    add_term(
        [0,1,0, 0,0,0],
        [0,0,0, 0,0,1, 0,0,0],
        [0,0, 0,0, 1,0]
    )
    
    # Term (i=1,j=0): A[1,0]*B[0,2] -> C[1,2]
    add_term(
        [0,0,0, 1,0,0],
        [0,0,1, 0,0,0, 0,0,0],
        [0,0, 0,0, 0,1]
    )
    
    # Term (i=1,j=1): A[1,1]*B[1,2] -> C[1,2]
    add_term(
        [0,0,0, 0,1,0],
        [0,0,0, 0,0,1, 0,0,0],
        [0,0, 0,0, 0,1]
    )
    
    # ===== Block 3: 2x1x3 (schoolbook rank 6) =====
    # A[0:2,2:3] * B[2:3,0:3] -> C[0:2,0:3]
    # U indices: i*3+2 for i in {0,1} -> {2,5}
    # V indices: 2*3+k for k in {0,1,2} -> {6,7,8}
    # W indices: k*2+i for k in {0,1,2}, i in {0,1} -> {0,1,2,3,4,5}
    
    # Term (i=0,k=0): A[0,2]*B[2,0] -> C[0,0]
    add_term(
        [0,0,1, 0,0,0],
        [0,0,0, 0,0,0, 1,0,0],
        [1,0, 0,0, 0,0, 0,0]
    )
    
    # Term (i=0,k=1): A[0,2]*B[2,1] -> C[0,1]
    add_term(
        [0,0,1, 0,0,0],
        [0,0,0, 0,0,0, 0,1,0],
        [0,1, 0,0, 0,0, 0,0]
    )
    
    # Term (i=0,k=2): A[0,2]*B[2,2] -> C[0,2]
    add_term(
        [0,0,1, 0,0,0],
        [0,0,0, 0,0,0, 0,0,1],
        [0,0, 0,0, 1,0, 0,0]
    )
    
    # Term (i=1,k=0): A[1,2]*B[2,0] -> C[1,0]
    add_term(
        [0,0,0, 0,0,1],
        [0,0,0, 0,0,0, 1,0,0],
        [0,0, 1,0, 0,0, 0,0]
    )
    
    # Term (i=1,k=1): A[1,2]*B[2,1] -> C[1,1]
    add_term(
        [0,0,0, 0,0,1],
        [0,0,0, 0,0,0, 0,1,0],
        [0,0, 0,1, 0,0, 0,0]
    )
    
    # Term (i=1,k=2): A[1,2]*B[2,2] -> C[1,2]
    add_term(
        [0,0,0, 0,0,1],
        [0,0,0, 0,0,0, 0,0,1],
        [0,0, 0,0, 0,1, 0,0]
    )
    
    # Verify: col should be 17
    assert col == 17, f"Expected 17 terms, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
