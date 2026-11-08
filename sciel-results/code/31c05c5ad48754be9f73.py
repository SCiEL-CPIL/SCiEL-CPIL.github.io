"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = 2, 4, 6
    rank = 48
    
    # Build rank-48 decomposition using Kronecker product structure.
    # We decompose the tensor T_{i,j,k} = e_{i,j} * e_{j,k} * e_{k,i} (for A[i,j] * B[j,k] = C[k,i] in W convention W[k*n+i])
    # Actually W[k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c]
    # Wait, the convention is W[k*n+i, r] for output C[i,k] where i is row (0..n-1), k is col (0..p-1)
    # But the index is k*n + i, so W is indexed by (k, i) -> k*n+i
    # U has shape (n*m, rank) -> U[i*m + j, r]
    # V has shape (m*p, rank) -> V[j*p + k, r]
    
    # Standard matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T_{i,j,k} = 1 if this is a valid multiplication term
    # In the rank-r decomposition: T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Schoolbook: r = (i,j,k) triple, U[i*m+j, (i,j,k)] = 1, V[j*p+k, (i,j,k)] = 1, W[k*n+i, (i,j,k)] = 1
    
    # To reduce rank, we can use the fact that we can apply Strassen-like combinations.
    # Let's use a block approach: split j into two groups of 2.
    # For j in {0,1} and j in {2,3}.
    # A is 2x4: A[:, 0:2] and A[:, 2:4]
    # B is 4x6: B[0:2, :] and B[2:4, :]
    # C = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:]
    
    # Each term is a 2x2 @ 2x6 multiplication.
    # For 2x2 @ 2x6: we can use Strassen on the 2x2 part.
    # Let A_block = [[a00, a01], [a10, a11]], B_block = [[b00, b01, b02, b03, b04, b05], [b10, b11, b12, b13, b14, b15]]
    # C_block = A_block @ B_block (2x6)
    # C[0,:] = a00*B[0,:] + a01*B[1,:]
    # C[1,:] = a10*B[0,:] + a11*B[1,:]
    
    # Using Strassen-like:
    # m1 = (a00+a11) * (b00+b11)  -- but B is 2x6, so we need 6 separate products
    # Actually, for 2x2 @ 2xk, we can use 7 multiplications per output column pair? No.
    # For 2x2 @ 2xk: C[0,c] = a00*b0c + a01*b1c, C[1,c] = a10*b0c + a11*b1c
    # We can define:
    # s1 = a00 + a11, s2 = a10, s3 = a01, s4 = a00 - a11
    # t1 = b0c + b1c, t2 = b1c, t3 = b0c - b1c
    # Then:
    # C[0,c] = (s1*t1 + s3*t3) / 2 + s2*t2  -- no, this introduces fractions and doesn't work cleanly
    # 
    # Actually, the standard rank for 2x2 @ 2xk is 7k? No, that's not right either.
    # The tensor rank of 2x2 @ 2xk is at most 4k (schoolbook) and at least 2k+2 (for k>=2).
    # For k=6, the rank is between 14 and 24.
    
    # Let me just use the schoolbook for each 2x2 @ 2x6 block: 4*6 = 24 multiplications per block.
    # Two blocks: 48 total. That's the same as the full schoolbook.
    
    # So I need a better approach. Let me use the known rank 19 decomposition for 4x4 @ 4x4.
    # But our problem is 2x4 @ 4x6.
    
    # Alternative: Use the fact that the tensor for 2x4 @ 4x6 has structure.
    # Let me try to construct a rank 44 decomposition.
    
    # Actually, let me just output the schoolbook decomposition to ensure correctness first.
    # The score for rank 48 is 48/48 = 1.0.
    
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
# EVOLVE-BLOCK-END
