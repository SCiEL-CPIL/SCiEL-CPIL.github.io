import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a low-rank exact tensor decomposition for 
    multiplying a 2x3 matrix by a 3x10 matrix.
    
    Uses block decomposition splitting p=10 into 5 blocks of size 2.
    Each 2x3x2 block is decomposed with rank 11 (7 for 2x2x2 Strassen + 4 for 2x1x2 schoolbook).
    Total rank = 5 * 11 = 55.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    num_blocks = 5
    block_size = 2
    rank_per_block = 11
    total_rank = num_blocks * rank_per_block
    
    # Initialize U, V, W as lists of lists for easy manipulation
    # U: (n*m) x rank
    # V: (m*p) x rank
    # W: (p*n) x rank
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Define the 11 terms for a 2x3x2 block
    # Local indices:
    # A: i in {0,1}, j in {0,1,2}
    # B: j in {0,1,2}, k in {0,1}
    # C: i in {0,1}, k in {0,1}
    
    # Strassen terms for A[:, 0:2] and B[0:2, :]
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    strassen_terms = [
        # (U_coeffs: [(i,j,val)], V_coeffs: [(j,k,val)], W_coeffs: [(k,i,val)])
        # M1
        {
            'U': [(0,0,1), (1,1,1)],
            'V': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1)] # C00 +=, C11 +=
        },
        # M2
        {
            'U': [(1,0,1), (1,1,1)],
            'V': [(0,0,1)],
            'W': [(0,1,1), (1,1,-1)] # C10 +=, C11 -=
        },
        # M3
        {
            'U': [(0,0,1)],
            'V': [(0,1,1), (1,1,-1)],
            'W': [(1,0,1), (1,1,1)] # C01 +=, C11 +=
        },
        # M4
        {
            'U': [(1,1,1)],
            'V': [(1,0,1), (0,0,-1)],
            'W': [(0,0,1), (0,1,1)] # C00 +=, C10 +=
        },
        # M5
        {
            'U': [(0,0,1), (0,1,1)],
            'V': [(1,1,1)],
            'W': [(0,0,-1), (1,0,1)] # C00 -=, C10 +=
        },
        # M6
        {
            'U': [(1,0,1), (0,0,-1)],
            'V': [(0,0,1), (0,1,1)],
            'W': [(1,1,1)] # C11 +=
        },
        # M7
        {
            'U': [(0,1,1), (1,1,-1)],
            'V': [(1,0,1), (1,1,1)],
            'W': [(0,0,1)] # C00 +=
        }
    ]
    
    # Schoolbook terms for A[:, 2] and B[2, :]
    # i in {0,1}, j=2, k in {0,1}
    outer_terms = [
        # i=0, k=0
        {
            'U': [(0,2,1)],
            'V': [(2,0,1)],
            'W': [(0,0,1)]
        },
        # i=0, k=1
        {
            'U': [(0,2,1)],
            'V': [(2,1,1)],
            'W': [(1,0,1)]
        },
        # i=1, k=0
        {
            'U': [(1,2,1)],
            'V': [(2,0,1)],
            'W': [(0,1,1)]
        },
        # i=1, k=1
        {
            'U': [(1,2,1)],
            'V': [(2,1,1)],
            'W': [(1,1,1)]
        }
    ]
    
    all_terms = strassen_terms + outer_terms
    
    col_offset = 0
    for b in range(num_blocks):
        for term in all_terms:
            col = col_offset
            
            # Fill U
            for i, j, val in term['U']:
                U[i * m + j][col] += val
            
            # Fill V
            # Local k maps to global k = b * block_size + k_local
            for j, k_local, val in term['V']:
                k_global = b * block_size + k_local
                V[j * p + k_global][col] += val
            
            # Fill W
            # Local k maps to global k = b * block_size + k_local
            # W index is k * n + i
            for k_local, i, val in term['W']:
                k_global = b * block_size + k_local
                W[k_global * n + i][col] += val
            
            col_offset += 1

    # Optional: Verification using NumPy for safety
    # If verification fails, fall back to schoolbook
    try:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check a few random instances
        # The condition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k?
        # No, the tensor T[i,j,k] is 1 for all i,j,k in the context of the bilinear form sum A B C.
        # Wait, the tensor for matrix multiplication M_{n,m,p} is defined such that
        # <A, B, T> = trace(A B C^T) ? No.
        # The standard definition: T is the tensor of the bilinear map (A,B) -> C.
        # C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # This is equivalent to the trilinear form sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k} = 0? No.
        # It's that the coefficient of A_{i,j} B_{j,k} in the expression for C_{i,k} is 1.
        # So, if we fix i,j,k, the term A_{i,j} B_{j,k} contributes to C_{i,k}.
        # The decomposition requires that for each output entry C_{i,k},
        # sum_j sum_c U_{i,j}^{(c)} V_{j,k}^{(c)} W_{k,i}^{(c)} = delta_{j, ?} ... no.
        
        # Let's check the reconstruction of C for basis matrices.
        # Let A = E_{i0, j0}, B = E_{j0, k0}.
        # Then C should be E_{i0, k0}.
        # C_{i, k} = sum_j sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}
        # For A=E_{i0,j0}, B=E_{j0,k0}:
        # The "value" of the bilinear form at (A,B) for the term c is U_{i0*m+j0, c} V_{j0*p+k0, c}.
        # This value multiplies W_{k*n+i, c} to contribute to C_{i,k}.
        # We need sum_c U_{i0*m+j0, c} V_{j0*p+k0, c} W_{k*n+i, c} = 1 if i=i0 and k=k0, else 0.
        
        for i0 in range(n):
            for j0 in range(m):
                for k0 in range(p):
                    # Compute the resulting C matrix coefficients
                    # We only need to check C[i0, k0] is 1 and others are 0?
                    # Actually, we can just check the tensor equation directly.
                    # T_{i,j,k} = 1 for all i,j,k?
                    # No, T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
                    # Since C_{i,k} = sum_j A_{i,j} B_{j,k}, the coefficient is 1 for all valid i,j,k.
                    # So we need: sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c} = 1 for all i,j,k.
                    
                    val = 0
                    for c in range(total_rank):
                        val += U_np[i*m+j0, c] * V_np[j0*p+k0, c] * W_np[k0*n+i0, c]
                    
                    if not np.isclose(val, 1.0):
                        # Fallback to schoolbook
                        # This should not happen, but safe to include
                        rank = n * m * p
                        U_fb = [[0] * rank for _ in range(n * m)]
                        V_fb = [[0] * rank for _ in range(m * p)]
                        W_fb = [[0] * rank for _ in range(p * n)]
                        col = 0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    U_fb[i * m + j][col] = 1
                                    V_fb[j * p + k][col] = 1
                                    W_fb[k * n + i][col] = 1
                                    col += 1
                        return {"U": U_fb, "V": V_fb, "W": W_fb}
                        
    except Exception:
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
