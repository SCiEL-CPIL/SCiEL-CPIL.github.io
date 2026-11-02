"""Program to find a rank-minimal exact tensor decomposition for 2x2x5 matrix multiplication."""
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 5]

def construct():
    n, m, p = 2, 2, 5
    # We will use a block decomposition:
    # Block 1: k=0 (2x2x1) -> rank 4
    # Block 2: k=1,2 (2x2x2) -> rank 7 (Strassen)
    # Block 3: k=3,4 (2x2x2) -> rank 7 (Strassen)
    # Total rank = 4 + 7 + 7 = 18
    
    rank = 18
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    # Helper to set entries
    def set_uv_w(i, j, k, r_idx, u_val, v_val, w_val):
        u_row = i * m + j
        v_row = j * p + k
        w_row = k * n + i
        U[u_row][r_idx] += u_val
        V[v_row][r_idx] += v_val
        W[w_row][r_idx] += w_val

    # --- Block 1: k=0 (Schoolbook 2x2x1) ---
    # This is just standard multiplication for the first column
    # C[:,0] = A * B[:,0]
    # C_{i,0} = sum_j A_{i,j} B_{j,0}
    # Tensor entries: T_{i,j,0} = A_{i,j} B_{j,0} C_{i,0}^*
    # Rank 1 terms: (i,j,0) for i in 0,1 and j in 0,1
    r_idx = 0
    for i in range(n):
        for j in range(m):
            set_uv_w(i, j, 0, r_idx, Fraction(1), Fraction(1), Fraction(1))
            r_idx += 1
    # r_idx is now 4

    # --- Block 2: k=1,2 (Strassen 2x2x2) ---
    # This multiplies A (2x2) by B[:,1:3] (2x2) to get C[:,1:3] (2x2)
    # Let A = [[a11, a12], [a21, a22]]
    # Let B' = [[b11, b12], [b21, b22]] where b_{j,k'} = B_{j, k'+1}
    # Let C' = [[c11, c12], [c21, c22]] where c_{i,k'} = C_{i, k'+1}
    
    # Strassen products:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b22)
    # M4 = a22(b12-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # Reconstruction:
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # We define the coefficients for U, V, W for each M_r
    # Indices for Block 2:
    # U rows: i*m+j (i=0,1; j=0,1) -> 0,1,2,3
    # V rows: j*p+k (j=0,1; k=1,2) -> 5,6, 10,11
    # W rows: k*n+i (k=1,2; i=0,1) -> 2,3, 4,5
    
    # Let's map Strassen variables to indices:
    # a11=A[0,0]=U[0], a12=A[0,1]=U[1], a21=A[1,0]=U[2], a22=A[1,1]=U[3]
    # b11=B[0,1]=V[5], b12=B[0,2]=V[6], b21=B[1,1]=V[10], b22=B[1,2]=V[11]
    # c11=C[0,1]=W[2], c12=C[0,2]=W[3], c21=C[1,1]=W[4], c22=C[1,2]=W[5]
    
    strassen_block = [
        # (M_name, u_coeffs, v_coeffs, w_coeffs)
        # u_coeffs: dict {u_row: coeff}
        # v_coeffs: dict {v_row: coeff}
        # w_coeffs: dict {w_row: coeff}
        {
            'u': {0: 1, 3: 1},  # a11 + a22
            'v': {5: 1, 10: 1}, # b11 + b22
            'w': {2: 1, 5: -1}  # c11 - c22 (part of M1) -> Wait, W entries are for C.
        },
        # Let's be precise. The tensor term is sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = (C A B)_{i,k}
        # So for a specific rank-1 term r, we add to W_{k,i,r} the coefficient of C_{i,k} in the expression for M_r?
        # No. The decomposition is T = sum_r u^{(r)} \otimes v^{(r)} \otimes w^{(r)}.
        # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
        # For Block 2, T_{i,j,k} corresponds to C_{i, k} = sum_j A_{i,j} B_{j,k}.
        # So we need sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = C_{i,k} for k in {1,2}.
        
        # Let's re-derive the W coefficients.
        # M1 contributes to c11 (+1) and c22 (-1).
        # So for r=0 (M1): W_{1,0,0} = 1 (c11), W_{2,1,0} = -1 (c22).
        # Note W row index is k*n+i.
        # k=1, i=0 -> row 1*2+0 = 2. So W[2, r] += 1.
        # k=2, i=1 -> row 2*2+1 = 5. So W[5, r] += -1.
    ]
    
    # Let's just hardcode the 7 Strassen terms explicitly for Block 2 (k=1,2)
    # r_base = 4
    def add_strassen_block(k_start, r_base):
        # k_start is 1 or 3
        # k1 = k_start, k2 = k_start+1
        # V rows: j*p+k1, j*p+k2 -> 0*p+k1, 0*p+k2, 1*p+k1, 1*p+k2
        # W rows: k1*n+0, k1*n+1, k2*n+0, k2*n+1
        
        # M1 = (a11+a22)(b11+b22)
        # c11 += M1, c22 += -M1
        U[0][r_base] += 1; U[3][r_base] += 1
        V[0*p+k_start][r_base] += 1; V[1*p+k_start+1][r_base] += 1 # b11 is B[0,k1], b22 is B[1,k2]
        # Wait, b11 is B[0, k_start], b22 is B[1, k_start+1]?
        # B' = [[B[0,k1], B[0,k2]], [B[1,k1], B[1,k2]]]
        # b11 = B[0,k1], b12 = B[0,k2], b21 = B[1,k1], b22 = B[1,k2]
        # M1 = (a11+a22)(b11+b22) -> V[0*p+k_start] + V[1*p+k_start+1]
        # Let's check indices:
        # V row for B[0,k] is 0*p+k = k.
        # V row for B[1,k] is 1*p+k = p+k.
        # So b11 (B[0,k1]) is V[k_start]. b22 (B[1,k2]) is V[p+k_start+1].
        # My previous line: V[0*p+k_start] (ok), V[1*p+k_start+1] (ok).
        
        W[k_start*n+0][r_base] += 1   # c11 (i=0, k=k1)
        W[(k_start+1)*n+1][r_base] += -1 # c22 (i=1, k=k2)
        
        # M2 = (a21+a22)b11
        # c21 += M2
        U[2][r_base+1] += 1; U[3][r_base+1] += 1
        V[0*p+k_start][r_base+1] += 1
        W[k_start*n+1][r_base+1] += 1   # c21 (i=1, k=k1)
        
        # M3 = a11(b21-b22)
        # c12 += M3, c22 += M3
        U[0][r_base+2] += 1
        V[1*p+k_start][r_base+2] += 1   # b21
        V[1*p+k_start+1][r_base+2] += -1 # b22
        W[(k_start+1)*n+0][r_base+2] += 1 # c12 (i=0, k=k2)
        W[(k_start+1)*n+1][r_base+2] += 1 # c22 (i=1, k=k2)
        
        # M4 = a22(b12-b11)
        # c11 += M4, c21 += M4
        U[3][r_base+3] += 1
        V[0*p+k_start+1][r_base+3] += 1 # b12
        V[0*p+k_start][r_base+3] += -1  # b11
        W[k_start*n+0][r_base+3] += 1   # c11
        W[k_start*n+1][r_base+3] += 1   # c21
        
        # M5 = (a11+a12)b22
        # c11 += -M5, c12 += M5
        U[0][r_base+4] += 1; U[1][r_base+4] += 1
        V[1*p+k_start+1][r_base+4] += 1 # b22
        W[k_start*n+0][r_base+4] += -1  # c11
        W[(k_start+1)*n+0][r_base+4] += 1 # c12
        
        # M6 = (a21-a11)(b11+b12)
        # c22 += M6
        U[2][r_base+5] += 1; U[0][r_base+5] += -1
        V[0*p+k_start][r_base+5] += 1   # b11
        V[0*p+k_start+1][r_base+5] += 1 # b12
        W[(k_start+1)*n+1][r_base+5] += 1 # c22
        
        # M7 = (a12-a22)(b21+b22)
        # c11 += M7
        U[1][r_base+6] += 1; U[3][r_base+6] += -1
        V[1*p+k_start][r_base+6] += 1   # b21
        V[1*p+k_start+1][r_base+6] += 1 # b22
        W[k_start*n+0][r_base+6] += 1   # c11

    add_strassen_block(1, 4)
    add_strassen_block(3, 11)
    
    # Convert to lists of floats/ints for output
    # The task allows integer/half-integer. Fractions are exact.
    # We should probably output floats if they are integers, or keep as fractions?
    # The interface says "integer/half-integer entries".
    # Let's convert to float for simplicity, or keep as Fraction?
    # "exactly decompose" suggests exact arithmetic might be checked.
    # However, standard output is usually numeric.
    # Let's return floats. If they are all integers, it's fine.
    
    U_out = [[float(x) for x in row] for row in U]
    V_out = [[float(x) for x in row] for row in V]
    W_out = [[float(x) for x in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
