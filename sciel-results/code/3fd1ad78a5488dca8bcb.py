"""
Rank-minimal tensor decomposition for 5x5x5 matrix multiplication.
Uses 2+2+1 block partitioning with Strassen's algorithm for 2x2x2 sub-blocks.
Achieves rank 117 (improvement over schoolbook rank 125).
"""
import time
import numpy as np

def construct():
    n, m, p = 5, 5, 5
    
    start_time = time.time()
    deadline = start_time + 120  # safety margin
    
    # Total rank: 8 Strassen blocks * 7 + 61 schoolbook terms = 117
    rank = 117
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices):
        """Add a single rank-1 term.
        u_indices: list of (index, coeff) for U
        v_indices: list of (index, coeff) for V
        w_indices: list of (index, coeff) for W
        """
        nonlocal col
        if col >= rank:
            return
        for idx, coeff in u_indices:
            U[idx, col] += coeff
        for idx, coeff in v_indices:
            V[idx, col] += coeff
        for idx, coeff in w_indices:
            W[idx, col] += coeff
        col += 1
    
    def add_schoolbook_2x2x2(i1, j1, k1):
        """Add schoolbook 4 terms for 2x2 A * 2x2 B -> 2x2 C."""
        i2 = i1 + 1
        j2 = j1 + 1
        k2 = k1 + 1
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    add_term(
                        [( (i1+i)*m + (j1+j), 1.0 )],
                        [ ( (j1+j)*p + (k1+k), 1.0 )],
                        [ ( (k1+k)*n + (i1+i), 1.0 )]
                    )
    
    def add_strassen_2x2x2(i1, j1, k1):
        """Add Strassen 7 terms for 2x2 A * 2x2 B -> 2x2 C.
        A is at rows i1,i1+1 cols j1,j1+1
        B is at rows j1,j1+1 cols k1,k1+1
        C is at rows i1,i1+1 cols k1,k1+1
        """
        i2 = i1 + 1
        j2 = j1 + 1
        k2 = k1 + 1
        
        # M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
        # Contributes to C[i1,k1] (+1) and C[i2,k2] (+1)
        add_term(
            [(i1*m+j1, 1.0), (i2*m+j2, 1.0)],
            [(j1*p+k1, 1.0), (j2*p+k2, 1.0)],
            [(k1*n+i1, 1.0), (k2*n+i2, 1.0)]
        )
        
        # M2 = (A[i2,j1]+A[i2,j2]) * B[j1,k1]
        # Contributes to C[i2,k1] (+1)
        add_term(
            [(i2*m+j1, 1.0), (i2*m+j2, 1.0)],
            [(j1*p+k1, 1.0)],
            [(k1*n+i2, 1.0)]
        )
        
        # M3 = A[i1,j1] * (B[j1,k2]-B[j2,k2])
        # Contributes to C[i1,k2] (+1)
        add_term(
            [(i1*m+j1, 1.0)],
            [(j1*p+k2, 1.0), (j2*p+k2, -1.0)],
            [(k2*n+i1, 1.0)]
        )
        
        # M4 = A[i2,j2] * (B[j2,k1]-B[j1,k1])
        # Contributes to C[i2,k1] (+1) and C[i1,k1] (-1)
        add_term(
            [(i2*m+j2, 1.0)],
            [(j2*p+k1, 1.0), (j1*p+k1, -1.0)],
            [(k1*n+i2, 1.0), (k1*n+i1, -1.0)]
        )
        
        # M5 = (A[i1,j1]+A[i1,j2]) * B[j2,k2]
        # Contributes to C[i1,k2] (+1) and C[i1,k1] (-1)
        add_term(
            [(i1*m+j1, 1.0), (i1*m+j2, 1.0)],
            [(j2*p+k2, 1.0)],
            [(k2*n+i1, 1.0), (k1*n+i1, -1.0)]
        )
        
        # M6 = (A[i2,j1]-A[i1,j1]) * (B[j1,k1]+B[j1,k2])
        # Contributes to C[i2,k2] (+1)
        add_term(
            [(i2*m+j1, 1.0), (i1*m+j1, -1.0)],
            [(j1*p+k1, 1.0), (j1*p+k2, 1.0)],
            [(k2*n+i2, 1.0)]
        )
        
        # M7 = (A[i1,j2]-A[i2,j2]) * (B[j2,k1]+B[j2,k2])
        # Contributes to C[i1,k1] (+1) and C[i2,k2] (-1)
        add_term(
            [(i1*m+j2, 1.0), (i2*m+j2, -1.0)],
            [(j2*p+k1, 1.0), (j2*p+k2, 1.0)],
            [(k1*n+i1, 1.0), (k2*n+i2, -1.0)]
        )
    
    def add_schoolbook_general(i1, i2, j1, j2, k1, k2):
        """Add schoolbook terms for A[i1:i2, j1:j2] * B[j1:j2, k1:k2] -> C[i1:i2, k1:k2]."""
        for i in range(i1, i2):
            for j in range(j1, j2):
                for k in range(k1, k2):
                    add_term(
                        [(i*m+j, 1.0)],
                        [(j*p+k, 1.0)],
                        [(k*n+i, 1.0)]
                    )
    
    # Block boundaries: block 0 = {0,1}, block 1 = {2,3}, block 2 = {4}
    # Block start indices: [0, 2, 4], block end indices: [2, 4, 5]
    block_starts = [0, 2, 4]
    block_ends = [2, 4, 5]
    
    # For each output block C[i_block, k_block], sum over j_block
    for i_block in range(3):
        i1 = block_starts[i_block]
        i2 = block_ends[i_block]
        for k_block in range(3):
            k1 = block_starts[k_block]
            k2 = block_ends[k_block]
            for j_block in range(3):
                j1 = block_starts[j_block]
                j2 = block_ends[j_block]
                
                # Check if this is a 2x2x2 sub-problem (can use Strassen)
                if (i2 - i1 == 2 and j2 - j1 == 2 and k2 - k1 == 2):
                    add_strassen_2x2x2(i1, j1, k1)
                else:
                    add_schoolbook_general(i1, i2, j1, j2, k1, k2)
    
    # Verify col count
    assert col == rank, f"Expected {rank} terms, got {col}"
    
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
