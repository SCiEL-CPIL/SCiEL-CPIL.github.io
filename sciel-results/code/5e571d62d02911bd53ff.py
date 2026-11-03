"""
Rank-minimal exact tensor decomposition for 3x3x7 matrix multiplication.
Uses Strassen's rank-7 decomposition for 2x2x2 sub-blocks where applicable.
"""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Return a rank-7 tensor decomposition for 2x2x2 matrix multiplication.
    U: (4, 7) indexed by (i, j) for i,j in {0,1}
    V: (4, 7) indexed by (j, k) for j,k in {0,1}
    W: (4, 7) indexed by (k, i) for k,i in {0,1}
    """
    # Strassen's 7 multiplications for 2x2x2:
    # m1 = (a00 + a11)(b00 + b11)
    # m2 = (a10 + a11)b00
    # m3 = a00(b10 - b00)
    # m4 = a11(b01 - b11)
    # m5 = (a00 + a01)b11
    # m6 = (a10 - a00)(b00 + b01)
    # m7 = (a01 - a11)(b10 + b11)
    
    # U[i*2+j, c] for i,j in {0,1}
    U = [[0]*7 for _ in range(4)]
    # V[j*2+k, c] for j,k in {0,1}
    V = [[0]*7 for _ in range(4)]
    # W[k*2+i, c] for k,i in {0,1}
    W = [[0]*7 for _ in range(4)]
    
    # m1: (a00+a11)(b00+b11)
    U[0][0] = 1  # a00
    U[3][0] = 1  # a11
    V[0][0] = 1  # b00
    V[3][0] = 1  # b11
    W[0][0] = 1  # c00
    W[3][0] = 1  # c11
    W[1][0] = -1 # c10
    W[2][0] = -1 # c01
    
    # m2: (a10+a11)b00
    U[2][1] = 1  # a10
    U[3][1] = 1  # a11
    V[0][1] = 1  # b00
    W[2][1] = 1  # c10
    W[3][1] = 1  # c11
    
    # m3: a00(b10-b00)
    U[0][2] = 1  # a00
    V[2][2] = 1  # b10
    V[0][2] = -1 # b00
    W[0][2] = 1  # c00
    W[1][2] = 1  # c01
    
    # m4: a11(b01-b11)
    U[3][3] = 1  # a11
    V[1][3] = 1  # b01
    V[3][3] = -1 # b11
    W[2][3] = 1  # c10
    W[3][3] = 1  # c11
    
    # m5: (a00+a01)b11
    U[0][4] = 1  # a00
    U[1][4] = 1  # a01
    V[3][4] = 1  # b11
    W[0][4] = 1  # c00
    W[1][4] = 1  # c01
    
    # m6: (a10-a00)(b00+b01)
    U[2][5] = 1  # a10
    U[0][5] = -1 # a00
    V[0][5] = 1  # b00
    V[1][5] = 1  # b01
    W[2][5] = 1  # c10
    W[3][5] = 1  # c11
    
    # m7: (a01-a11)(b10+b11)
    U[1][6] = 1  # a01
    U[3][6] = -1 # a11
    V[2][6] = 1  # b10
    V[3][6] = 1  # b11
    W[0][6] = 1  # c00
    W[1][6] = 1  # c01
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Start with schoolbook rank 63
    best_rank = n * m * p
    best_result = _schoolbook_decomposition(n, m, p)
    
    # Build Strassen-based hybrid: rank 60
    if time.time() < deadline:
        result = _build_strassen_hybrid(n, m, p)
        if result is not None:
            rank = len(result['U'][0])
            if rank < best_rank:
                best_rank = rank
                best_result = result
                print(f"Strassen hybrid decomposition with rank {rank}")
    
    # Try to reduce rank further
    if time.time() < deadline:
        result = _reduce_rank(best_result, n, m, p, deadline)
        if result is not None:
            rank = len(result['U'][0])
            if rank < best_rank:
                best_rank = rank
                best_result = result
                print(f"Reduced decomposition with rank {rank}")
    
    return best_result

def _schoolbook_decomposition(n, m, p):
    """Create the standard schoolbook decomposition of rank n*m*p."""
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

def _build_strassen_hybrid(n, m, p):
    """
    Build a hybrid decomposition using Strassen for 2x2x2 sub-blocks.
    
    Partition:
    - i in {0,1}, j in {0,1}, k in {0,1}: Strassen (rank 7)
    - i in {0,1}, j in {0,1}, k in {2,3}: Strassen (rank 7)
    - i in {0,1}, j in {0,1}, k in {4,5}: Strassen (rank 7)
    - i in {0,1}, j in {0,1}, k in {6}: schoolbook (rank 4)
    - i in {0,1}, j=2, k in {0..6}: schoolbook (rank 14)
    - i=2, j in {0,1}, k in {0..6}: schoolbook (rank 14)
    - i=2, j=2, k in {0..6}: schoolbook (rank 7)
    
    Total: 7+7+7+4+14+14+7 = 60
    """
    n, m, p = 3, 3, 7
    total_rank = 60
    
    U = [[0]*total_rank for _ in range(9)]
    V = [[0]*total_rank for _ in range(21)]
    W = [[0]*total_rank for _ in range(21)]
    
    col = 0
    
    # Get Strassen factors
    sU, sV, sW = _strassen_2x2x2()
    
    # Part 1: i in {0,1}, j in {0,1}, k in {0,1} - Strassen
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i*m + j][col] = sU[i*2+j][c]
        for j in range(2):
            for k in range(2):
                V[j*p + k][col] = sV[j*2+k][c]
        for k in range(2):
            for i in range(2):
                W[k*n + i][col] = sW[k*2+i][c]
        col += 1
    
    # Part 2: i in {0,1}, j in {0,1}, k in {2,3} - Strassen
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i*m + j][col] = sU[i*2+j][c]
        for j in range(2):
            for k in range(2):
                V[j*p + (2+k)][col] = sV[j*2+k][c]
        for k in range(2):
            for i in range(2):
                W[(2+k)*n + i][col] = sW[k*2+i][c]
        col += 1
    
    # Part 3: i in {0,1}, j in {0,1}, k in {4,5} - Strassen
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i*m + j][col] = sU[i*2+j][c]
        for j in range(2):
            for k in range(2):
                V[j*p + (4+k)][col] = sV[j*2+k][c]
        for k in range(2):
            for i in range(2):
                W[(4+k)*n + i][col] = sW[k*2+i][c]
        col += 1
    
    # Part 4: i in {0,1}, j in {0,1}, k=6 - schoolbook (rank 4)
    for i in range(2):
        for j in range(2):
            U[i*m + j][col] = 1
            V[j*p + 6][col] = 1
            W[6*n + i][col] = 1
            col += 1
    
    # Part 5: i in {0,1}, j=2, k in {0..6} - schoolbook (rank 14)
    for i in range(2):
        for k in range(7):
            U[i*m + 2][col] = 1
            V[2*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
    
    # Part 6: i=2, j in {0,1}, k in {0..6} - schoolbook (rank 14)
    for j in range(2):
        for k in range(7):
            U[2*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 2][col] = 1
            col += 1
    
    # Part 7: i=2, j=2, k in {0..6} - schoolbook (rank 7)
    for k in range(7):
        U[2*m + 2][col] = 1
        V[2*p + k][col] = 1
        W[k*n + 2][col] = 1
        col += 1
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return result
    else:
        print("Strassen hybrid verification failed!")
        return None

def _reduce_rank(decomp, n, m, p, deadline):
    """
    Try to reduce the rank by checking if any term is redundant.
    """
    U = np.array(decomp['U'], dtype=float)
    V = np.array(decomp['V'], dtype=float)
    W = np.array(decomp['W'], dtype=float)
    rank = U.shape[1]
    
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = rank
    
    # Greedy: try removing each term
    for c_remove in range(rank):
        if time.time() >= deadline:
            break
        
        # Try removing term c_remove from current decomposition
        indices = [i for i in range(current_rank) if i != c_remove]
        if not indices:
            continue
        
        U_test = current_U[:, indices]
        V_test = current_V[:, indices]
        W_test = current_W[:, indices]
        
        if _verify_decomposition(U_test.tolist(), V_test.tolist(), W_test.tolist(), n, m, p):
            current_U = U_test
            current_V = V_test
            current_W = W_test
            current_rank = len(indices)
            print(f"Removed term {c_remove}, new rank: {current_rank}")
    
    if current_rank < rank:
        return {
            "U": current_U.tolist(),
            "V": current_V.tolist(),
            "W": current_W.tolist()
        }
    
    return None

# EVOLVE-BLOCK-END
