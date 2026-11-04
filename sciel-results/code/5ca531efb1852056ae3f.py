"""Matrix multiplication tensor decomposition for (2,3,10) using Strassen block decomposition."""
import time
import numpy as np
import random

DIMENSIONS = [2, 3, 10]

def _verify_decomposition(U, V, W, n, m, p, max_checks=10):
    """Verify if U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    random.seed(42)
    for _ in range(max_checks):
        A = np.random.uniform(-2, 2, (n, m))
        B = np.random.uniform(-2, 2, (m, p))
        C = A @ B
        
        U_reshaped = U.reshape(n, m, rank)
        u_a = np.einsum('ijr,ij->ir', U_reshaped, A)
        
        V_reshaped = V.reshape(m, p, rank)
        v_b = np.einsum('jkr,jk->kr', V_reshaped, B)
        
        W_reshaped = W.reshape(p, n, rank)
        C_computed = np.einsum('kir,ir,kr->ik', W_reshaped, u_a, v_b)
        
        if np.max(np.abs(C - C_computed)) > 1e-9:
            return False
    return True

def _build_strassen_2x2x2():
    """Build rank-7 Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    
    Returns U (4,7), V (4,7), W (4,7) where:
    - U rows indexed by (i,j) for i in {0,1}, j in {0,1}
    - V rows indexed by (j,k) for j in {0,1}, k in {0,1}  
    - W rows indexed by (k,i) for k in {0,1}, i in {0,1}
    """
    # Strassen's 7 products for 2x2 * 2x2
    # C = [[c00, c01], [c10, c11]]
    # where:
    # c00 = M1 - M4 + M5 + M7
    # c01 = M3 + M4
    # c10 = M1 + M2
    # c11 = M3 - M2 + M5 + M6
    #
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1  # a00, a11
    V[0, 0] = 1; V[3, 0] = 1  # b00, b11
    W[0, 0] = 1; W[3, 0] = 1  # c00, c11
    W[1, 0] = -1; W[2, 0] = 1  # c01, c10
    
    # M2: (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1  # a10, a11
    V[0, 1] = 1  # b00
    W[2, 1] = 1  # c10
    W[3, 1] = -1  # c11
    
    # M3: a00(b01-b11)
    U[0, 2] = 1  # a00
    V[1, 2] = 1; V[3, 2] = -1  # b01, b11
    W[1, 2] = 1  # c01
    W[3, 2] = 1  # c11
    
    # M4: a11(b10-b00)
    U[3, 3] = 1  # a11
    V[2, 3] = 1; V[0, 3] = -1  # b10, b00
    W[0, 3] = -1  # c00
    W[1, 3] = 1  # c01
    
    # M5: (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1  # a00, a01
    V[3, 4] = 1  # b11
    W[0, 4] = 1  # c00
    W[3, 4] = 1  # c11
    
    # M6: (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1  # a10, a00
    V[0, 5] = 1; V[1, 5] = 1  # b00, b01
    W[0, 5] = 1  # c00
    W[3, 5] = 1  # c11
    
    # M7: (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1  # a01, a11
    V[2, 6] = 1; V[3, 6] = 1  # b10, b11
    W[0, 6] = 1  # c00
    W[2, 6] = 1  # c10
    
    return U, V, W

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Strategy: Split the 10 columns into 5 pairs of 2 columns.
    # For each pair, we multiply 2x3 * 3x2, which can be done with:
    # - First 2 columns: 2x2 * 2x2 (Strassen rank 7) for rows 0-1 of A, cols 0-1 of B
    # - Plus schoolbook for the remaining row/col combinations
    # 
    # Actually, let's use a cleaner decomposition:
    # A is 2x3, B is 3x10
    # Split B into 5 blocks of 3x2: B = [B1, B2, B3, B4, B5]
    # C = [A*B1, A*B2, A*B3, A*B4, A*B5]
    # Each A*Bi is 2x2, can use Strassen rank 7
    # Total rank: 5 * 7 = 35
    # But we need to handle the 3 rows of A properly.
    
    # Better approach: 
    # For each 3x2 block of B, compute A * B_block (2x3 * 3x2 = 2x2)
    # We can write A*B_block = [A_top * B_top + A_bot * B_bot]
    # where A_top is row 0 of A (1x3), A_bot is row 1 of A (1x3)
    # B_top is top 2 rows of B_block (2x2), B_bot is bottom row (1x2)
    
    # Actually, the simplest correct approach:
    # Use the fact that 2x3 * 3x2 can be decomposed with rank at most 8
    # (schoolbook would be 6, but let's use a known decomposition)
    
    # Let me just build a correct rank-55 solution:
    # Split the 10 columns into 5 pairs. For each pair (k0, k1):
    # We need to compute C[:, k0:k1+1] = A @ B[:, k0:k1+1]
    # This is a 2x3 * 3x2 multiplication.
    # 
    # Using the standard 6-term schoolbook for 2x3*3x2:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # But we can do better. Let's use the known result that
    # rank(2x3 * 3x2) <= 8 (or even less).
    
    # Actually, let me just verify the original fallback code is correct by
    # carefully checking the indices. The issue was in the W indices.
    
    # Let me build a clean rank-55 solution from scratch:
    # For each of 5 column pairs (k0, k1):
    # Use 7 Strassen terms for the 2x2 * 2x2 part (using A rows 0,1 and B rows 0,1)
    # Plus 1 schoolbook term for A row 2 * B rows 0,1
    # That's 8 terms per pair, 5*8 = 40 total.
    
    # Wait, A is 2x3, so it has 2 rows and 3 columns.
    # B is 3x10, so it has 3 rows and 10 columns.
    # 
    # For a 2x3 * 3x2 block:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    # 
    # We can split this as:
    # C[i,k] = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + A[i,2]*B[2,k]
    # 
    # The first part is a 2x2 * 2x2 multiplication (using A cols 0,1 and B rows 0,1)
    # which has rank 7 via Strassen.
    # The second part is a 2x1 * 1x2 multiplication, which has rank 2.
    # Total: 7 + 2 = 9 per block, 5*9 = 45 total.
    
    # Even better: we can combine more cleverly.
    # Let's just use the schoolbook for the 3rd row:
    # For each pair, 7 (Strassen) + 2 (schoolbook for row 2) = 9
    # Total: 5 * 9 = 45
    
    # But we can do even better by using a different decomposition.
    # Let's try to achieve rank 45.
    
    U, V, W = _build_strassen_2x2x2()
    
    rank = 45
    U_full = np.zeros((n * m, rank))  # (6, 45)
    V_full = np.zeros((m * p, rank))  # (30, 45)
    W_full = np.zeros((p * n, rank))  # (20, 45)
    
    col = 0
    for pair in range(5):
        k0 = pair * 2
        k1 = pair * 2 + 1
        
        # Strassen part: uses A cols 0,1 and B rows 0,1
        # U rows: (i,j) where i in {0,1}, j in {0,1} -> indices 0,1,2,3
        # V rows: (j,k) where j in {0,1}, k in {0,1} -> indices 0,1,10+k0,10+k1
        # W rows: (k,i) where k in {0,1}, i in {0,1} -> indices 2*k0, 2*k0+1, 2*k1, 2*k1+1
        
        for r in range(7):
            # Copy U
            U_full[0:4, col+r] = U[0:4, r]  # rows (0,0),(0,1),(1,0),(1,1)
            
            # Copy V: map B rows 0,1 to global rows k0,k1
            V_full[k0, col+r] = V[0, r]      # (j=0, k_local=0) -> global row k0
            V_full[k1, col+r] = V[1, r]      # (j=0, k_local=1) -> global row k1
            V_full[10+k0, col+r] = V[2, r]   # (j=1, k_local=0) -> global row 10+k0
            V_full[10+k1, col+r] = V[3, r]   # (j=1, k_local=1) -> global row 10+k1
            
            # Copy W: map (k_local, i) to global (k, i)
            W_full[2*k0, col+r] = W[0, r]    # (k_local=0, i=0) -> global row 2*k0
            W_full[2*k0+1, col+r] = W[1, r]  # (k_local=0, i=1) -> global row 2*k0+1
            W_full[2*k1, col+r] = W[2, r]    # (k_local=1, i=0) -> global row 2*k1
            W_full[2*k1+1, col+r] = W[3, r]  # (k_local=1, i=1) -> global row 2*k1+1
        
        col += 7
        
        # Schoolbook part for A col 2 * B rows 0,1
        # C[i,k] += A[i,2] * B[2,k]
        # For k=k0: C[i,k0] += A[i,2] * B[2,k0]
        # For k=k1: C[i,k1] += A[i,2] * B[2,k1]
        
        # Term 1: A[i,2] * B[2,k0]
        for i in range(2):
            U_full[i*3+2, col] = 1
            V_full[20+k0, col] = 1
            W_full[2*k0+i, col] = 1
        col += 1
        
        # Term 2: A[i,2] * B[2,k1]
        for i in range(2):
            U_full[i*3+2, col] = 1
            V_full[20+k1, col] = 1
            W_full[2*k1+i, col] = 1
        col += 1
    
    assert col == rank, f"Column count {col} != expected rank {rank}"
    
    if _verify_decomposition(U_full, V_full, W_full, n, m, p):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback to schoolbook if verification fails
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
