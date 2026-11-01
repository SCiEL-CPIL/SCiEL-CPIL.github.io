"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    Uses Strassen's algorithm decomposition, which is the proven minimal rank 
    for 2x2 matrix multiplication over the reals.
    """
    n, m, p = DIMENSIONS  # 2, 2, 2
    start_time = time.time()
    
    rank = 7

    # Strassen's 7 products mapping to columns c=0..6
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U[i*m+j, c]: coefficient of A[i,j] in product c
    # Row order: A[0,0], A[0,1], A[1,0], A[1,1]
    U_data = [
        [1, 0, 1, 0, 1, -1, 0],  # A[0,0]
        [0, 0, 0, 0, 1, 0, 1],   # A[0,1]
        [0, 1, 0, 0, 0, 1, 0],   # A[1,0]
        [1, 1, 0, 1, 0, 0, -1]   # A[1,1]
    ]

    # V[j*p+k, c]: coefficient of B[j,k] in product c
    # Row order: B[0,0], B[0,1], B[1,0], B[1,1]
    V_data = [
        [1, 1, 0, -1, 0, 1, 0],  # B[0,0]
        [0, 0, 1, 0, 0, 1, 0],   # B[0,1]
        [0, 0, 0, 1, 0, 0, 1],   # B[1,0]
        [1, 0, -1, 0, 1, 0, 1]   # B[1,1]
    ]

    # W[k*n+i, c]: coefficient for C[i,k] from product c
    # Row index = k*n + i (official trace convention)
    # Row 0: k=0, i=0 -> C[0,0] = M1 + M4 - M5 + M7
    # Row 1: k=0, i=1 -> C[1,0] = M2 + M4
    # Row 2: k=1, i=0 -> C[0,1] = M3 + M5
    # Row 3: k=1, i=1 -> C[1,1] = M1 - M2 + M3 + M6
    W_data = [
        [1, 0, 0, 1, -1, 0, 1],  # C[0,0]
        [0, 1, 0, 1, 0, 0, 0],   # C[1,0]
        [0, 0, 1, 0, 1, 0, 0],   # C[0,1]
        [1, -1, 1, 0, 0, 1, 0]   # C[1,1]
    ]

    U = np.array(U_data, dtype=float)
    V = np.array(V_data, dtype=float)
    W = np.array(W_data, dtype=float)

    # Quick verification to ensure no typos in static data
    # The condition is sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # This holds for the diagonal components of the multiplication tensor (where i,j,k correspond to the product term)
    # Actually, the tensor T_{ijk} is 1 if (i,j,k) is a valid term in the sum for C, but in the decomposition context:
    # We are decomposing the tensor T such that T_{ijk} = sum_c U_{ic} V_{jc} W_{kc} (using appropriate index mapping)
    # For matrix mult, the tensor is defined on (i, j, k) where C_{ik} = sum_j A_{ij} B_{jk}.
    # The "tensor" here is the set of coefficients 1 for (i, j, k) such that they contribute to C_{ik}.
    # Wait, the standard tensor for matrix mult M_{n,m,p} has entries T_{i,j,k} = 1 if i,j,k are valid indices.
    # The decomposition is T = sum_c u_c \otimes v_c \otimes w_c.
    # T_{i,j,k} = sum_c U_{i,c} V_{j,c} W_{k,c}.
    # In our context:
    # U index is i*m+j (flattened A)
    # V index is j*p+k (flattened B)
    # W index is k*n+i (flattened C, trace convention)
    
    # Let's verify for all i, j, k in 0..1
    valid = True
    if time.time() - start_time < 115.0:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Calculate the reconstructed value for the tensor component corresponding to A_ij, B_jk, C_ik
                    # Note: The tensor component T_{i,j,k} is 1 for all valid i,j,k in the standard definition?
                    # No, the tensor for matrix multiplication is defined such that 
                    # C_{ik} = sum_j A_{ij} B_{jk}.
                    # The tensor T has indices (i, j, k) and T_{i,j,k} = 1.
                    # The decomposition must satisfy: sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = 1?
                    # Let's check the indices carefully.
                    # U is (n*m, rank). Index u_idx = i*m + j.
                    # V is (m*p, rank). Index v_idx = j*p + k.
                    # W is (p*n, rank). Index w_idx = k*n + i.
                    
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    
                    # For the matrix multiplication tensor, every triplet (i,j,k) where i,j,k are valid indices
                    # contributes to the product. The tensor entry is 1.
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
    
    if not valid:
        # Fallback to rank 8 (Schoolbook) just in case, though Strassen is known correct
        rank = 8
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
