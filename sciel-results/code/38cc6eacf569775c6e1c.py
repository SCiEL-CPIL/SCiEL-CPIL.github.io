import numpy as np
import time

def construct():
    """
    Returns a rank-49 tensor decomposition for 2x2 * 2x14 matrix multiplication.
    Uses block-wise Strassen algorithm.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 2, 14
    
    # Number of 2x2 blocks in the 2x14 matrix B
    num_blocks = p // 2
    # Rank of 2x2 matrix multiplication (Strassen)
    strassen_rank = 7
    total_rank = num_blocks * strassen_rank
    
    # Initialize matrices with zeros
    # U: (n*m, rank) = (4, 49)
    # V: (m*p, rank) = (28, 49)
    # W: (p*n, rank) = (28, 49)
    U = np.zeros((n * m, total_rank), dtype=np.int8)
    V = np.zeros((m * p, total_rank), dtype=np.int8)
    W = np.zeros((p * n, total_rank), dtype=np.int8)
    
    # Strassen coefficients for 2x2 * 2x2
    # Indices: A is 2x2 (0,1), B is 2x2 (0,1), C is 2x2 (0,1)
    # We define the 7 products P1..P7
    # Each product is a triple (u_indices, v_indices, w_indices) with coefficients
    # u_indices: list of (row, col) in A
    # v_indices: list of (row, col) in B
    # w_indices: list of (row, col) in C
    
    # P1 = (A00 + A11) * (B00 + B11)
    # C00 += P1, C11 += P1
    # P2 = (A10 + A11) * B00
    # C10 += P2, C11 -= P2
    # P3 = A00 * (B01 - B11)
    # C01 += P3, C11 += P3
    # P4 = A11 * (B10 - B00)
    # C00 += P4, C10 += P4
    # P5 = (A00 + A01) * B11
    # C00 -= P5, C01 += P5
    # P6 = (A10 - A00) * (B00 + B01)
    # C11 += P6
    # P7 = (A01 - A11) * (B10 + B11)
    # C00 += P7
    
    # Define the patterns for each Strassen step s (0 to 6)
    # Pattern: list of (U_idx, U_val, V_idx, V_val, W_idx, W_val)
    # U_idx is i*m + j
    # V_idx is j*p + k (local k 0 or 1)
    # W_idx is k*n + i (local k 0 or 1)
    
    patterns = []
    
    # s=0
    patterns.append([
        (0, 1, 0, 1, 0, 1),   # A00, B00, C00
        (3, 1, 1, 1, 3, 1),   # A11, B11, C11
        (0, 1, 1, 1, 3, 1),   # A00, B11? No, (A00+A11)(B00+B11) expands to 4 terms? 
        # Wait, Strassen uses 7 *scalar* multiplications.
        # The decomposition is:
        # P1 = (A00+A11)(B00+B11)
        # This is ONE scalar multiplication.
        # In the tensor decomposition, this corresponds to ONE rank term?
        # NO. The rank term $r$ has $U_{ij}^r, V_{jk}^r, W_{ki}^r$.
        # For a scalar multiplication $x \cdot y$, we set $U_{ij}^r$ such that $\sum U_{ij}^r A_{ij} = x$,
        # $V_{jk}^r$ such that $\sum V_{jk}^r B_{jk} = y$,
        # and $W_{ki}^r$ such that $C_{ki} \leftarrow W_{ki}^r (x \cdot y)$.
        
        # So for P1 = (A00+A11)(B00+B11):
        # x = A00 + A11 -> U[0,0]=1, U[1,1]=1 (indices 0 and 3)
        # y = B00 + B11 -> V[0,0]=1, V[1,1]=1 (indices 0*p+0=0, 1*p+1=p+1)
        # Contribution to C:
        # C00 gets P1, C11 gets P1.
        # W[0,0] (k=0, i=0) = 1
        # W[1,1] (k=1, i=1) = 1 (Index k*n+i = 1*2+1 = 3)
        
        # So the rank term has multiple non-zero entries in U, V, W.
        # My previous code did this correctly.
    ])
    
    # Let's just hardcode the assignments as in the previous working code,
    # but using NumPy arrays for cleaner distinctness.
    
    for t in range(num_blocks):
        k_offset = t * 2
        
        for s in range(strassen_rank):
            col = t * strassen_rank + s
            
            if s == 0:
                # P1 = (A00 + A11)(B00 + B11)
                U[0, col] += 1   # A00
                U[3, col] += 1   # A11
                V[0 * p + k_offset + 0, col] += 1   # B00
                V[1 * p + k_offset + 1, col] += 1   # B11
                W[(k_offset + 0) * n + 0, col] += 1 # C00
                W[(k_offset + 1) * n + 1, col] += 1 # C11
                
            elif s == 1:
                # P2 = (A10 + A11) B00
                U[2, col] += 1   # A10
                U[3, col] += 1   # A11
                V[0 * p + k_offset + 0, col] += 1   # B00
                W[(k_offset + 0) * n + 1, col] += 1 # C10
                W[(k_offset + 1) * n + 1, col] += -1 # C11
                
            elif s == 2:
                # P3 = A00 (B01 - B11)
                U[0, col] += 1   # A00
                V[0 * p + k_offset + 1, col] += 1   # B01
                V[1 * p + k_offset + 1, col] += -1  # B11
                W[(k_offset + 1) * n + 0, col] += 1 # C01
                W[(k_offset + 1) * n + 1, col] += 1 # C11
                
            elif s == 3:
                # P4 = A11 (B10 - B00)
                U[3, col] += 1   # A11
                V[1 * p + k_offset + 0, col] += 1   # B10
                V[0 * p + k_offset + 0, col] += -1  # B00
                W[(k_offset + 0) * n + 0, col] += 1 # C00
                W[(k_offset + 0) * n + 1, col] += 1 # C10
                
            elif s == 4:
                # P5 = (A00 + A01) B11
                U[0, col] += 1   # A00
                U[1, col] += 1   # A01
                V[1 * p + k_offset + 1, col] += 1   # B11
                W[(k_offset + 0) * n + 0, col] += -1 # C00
                W[(k_offset + 1) * n + 0, col] += 1 # C01
                
            elif s == 5:
                # P6 = (A10 - A00) (B00 + B01)
                U[0, col] += -1  # A00
                U[2, col] += 1   # A10
                V[0 * p + k_offset + 0, col] += 1   # B00
                V[0 * p + k_offset + 1, col] += 1   # B01
                W[(k_offset + 1) * n + 1, col] += 1 # C11
                
            elif s == 6:
                # P7 = (A01 - A11) (B10 + B11)
                U[1, col] += 1   # A01
                U[3, col] += -1  # A11
                V[1 * p + k_offset + 0, col] += 1   # B10
                V[1 * p + k_offset + 1, col] += 1   # B11
                W[(k_offset + 0) * n + 0, col] += 1 # C00

    # Convert to lists for return, ensuring integer types
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Quick verification
    if time.time() < deadline - 5.0:
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U[i * m + j]
                    v_row = V[j * p + k]
                    w_row = W[k * n + i]
                    val = np.dot(u_row, v_row * w_row)
                    if abs(val - 1.0) > max_err:
                        max_err = abs(val - 1.0)
        if max_err > 1e-9:
            # Fallback to schoolbook if verification fails
            rank_fb = n * m * p
            U_fb = np.zeros((n * m, rank_fb), dtype=np.int8)
            V_fb = np.zeros((m * p, rank_fb), dtype=np.int8)
            W_fb = np.zeros((p * n, rank_fb), dtype=np.int8)
            col = 0
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        U_fb[ii * m + jj, col] = 1
                        V_fb[jj * p + kk, col] = 1
                        W_fb[kk * n + ii, col] = 1
                        col += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
            
    return {"U": U_list, "V": V_list, "W": W_list}
