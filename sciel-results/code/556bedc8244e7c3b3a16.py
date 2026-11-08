"""Program for rank-minimal tensor decomposition of 2x3x10 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices
    for the tensor decomposition of 2x3x10 matrix multiplication.
    
    Uses a block structure: 5 blocks of 2 columns each, with rank 55 total.
    The decomposition is constructed by combining small rank-optimal sub-decompositions.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    # The total rank is 55 = 5 * 11
    # Each block handles 2 columns of B (and corresponding 2 columns of C)
    # Rank per block: 11
    
    rank = 55
    # Initialize U, V, W as integer matrices
    U = np.zeros((n * m, rank), dtype=int)  # 6 x 55
    V = np.zeros((m * p, rank), dtype=int)  # 30 x 55
    W = np.zeros((p * n, rank), dtype=int)  # 20 x 55
    
    # For each block of 2 columns (k0, k1 = 2t, 2t+1):
    # We need to compute C[:, k0] and C[:, k1] from A and B[:, k0:2t+2]
    # 
    # The sub-problem is: multiply 2x3 matrix A by 3x2 matrix B_block
    # Output is 2x2 matrix C_block
    #
    # For 2x3x2 matrix multiplication, the rank is 11 (known optimal).
    # We use a specific rank-11 decomposition for each block.
    
    # Rank-11 decomposition for 2x3x2 matrix multiplication:
    # A is 2x3, B is 3x2, C = A * B is 2x2
    # 
    # The 11 rank-1 tensors are (U_vec, V_vec, W_vec) where:
    # C[i,j] = sum_k U[i,k] * V[j,k] * W[k]  (not quite, let me think)
    #
    # Actually, the tensor T[i,j,k,l] for (A * B)[i,l] = sum_j A[i,j] * B[j,l]
    # We decompose T = sum_r U_r ⊗ V_r ⊗ W_r where:
    #   U_r is a vector in R^6 (flattened A, row-major: A[0,0],A[0,1],A[0,2],A[1,0],A[1,1],A[1,2])
    #   V_r is a vector in R^6 (flattened B, row-major: B[0,0],B[0,1],B[1,0],B[1,1],B[2,0],B[2,1])
    #   W_r is a vector in R^4 (flattened C, row-major: C[0,0],C[0,1],C[1,0],C[1,1])
    #
    # The decomposition satisfies: T[i,j,k,l] = sum_r U_r[i] * V_r[j] * W_r[l]
    # Wait, let me re-index. The tensor for matrix multiplication (2x3, 3x2 -> 2x2):
    # T[a, b, c, d] where a=0,1 (row of A/C), b=0,1,2 (col of A / row of B), 
    #   c=0,1,2 (row of B), d=0,1 (col of B/C)
    # T[a,b,c,d] = delta[b,c] (equals 1 if b==c, else 0)
    #
    # We want: T[a,b,c,d] = sum_r U_r[a,b] * V_r[c,d] * ... no that's not right either.
    #
    # Standard tensor decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r ∈ R^{n*m}, v_r ∈ R^{m*p}, w_r ∈ R^{p*n}
    # and T[α, β, γ] = sum_r u_r[α] * v_r[β] * w_r[γ]
    #
    # For 2x3x2: n=2, m=3, p=2
    # u_r has 6 entries (index by (i,j) for i=0,1 and j=0,1,2)
    # v_r has 6 entries (index by (j,k) for j=0,1,2 and k=0,1)
    # w_r has 4 entries (index by (k,i) for k=0,1 and i=0,1)
    #
    # The tensor T[i,j,k,l] (where the full tensor is T[(i,j),(j',k),(k',i')])
    # Actually the tensor is T[α, β, γ] where α=(i,j), β=(j',k), γ=(k',i')
    # T[α,β,γ] = 1 if j==j' and k==k', else 0
    #
    # So T[(i,j),(j',k),(k',i')] = δ(j,j') * δ(k,k')
    #
    # Decomposition: T = sum_r U_r[i,j] * V_r[j',k] * W_r[k',i']
    #
    # For rank 11, here is one known decomposition (I'll construct it):
    
    # I'll use a systematic construction. For 2x3x2, we can use the following:
    # The 11 terms are:
    
    terms = []
    
    # Term 0: (A00 + A11) * (B00 + B11) * (C00 + C11)  -- wait, need to be careful
    # Let me just enumerate a known rank-11 decomposition.
    
    # Based on literature, for 2x3x2 the following 11 rank-1 tensors work:
    
    # I'll define them as (U_entries, V_entries, W_entries) where each is a dict or list
    
    # Let's use the structure from the current program but verify it's correct.
    # The current program uses per-block rank 11, which matches.
    
    # For block t (columns k0=2t, k1=2t+1):
    # The 11 rank-1 terms contribute to U[0:6, 11*t:11*t+11], V[0:6, 11*t:11*t+11], W[0:4, 11*t:11*t+11]
    # but with appropriate column shifts in V and W for the global indexing.
    
    # In the global V (30x55), the entries for block t are at rows:
    #   V[0, 11*t + r] for B[0, k0]
    #   V[1, 11*t + r] for B[1, k0]  -- wait no, V is indexed by (j, k) where j=0,1,2 and k=0,...,9
    # So V[j*10 + k, col] = coefficient for B[j, k] in term col
    
    # Let me just follow the current program's structure but write it more cleanly.
    
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        base_col = 11 * t
        
        # Term 0: (A00 + A11) * (B0,k0 + B1,k1) * (C0,k0 + C1,k1)
        U[0, base_col] += 1
        U[4, base_col] += 1  # A[1,1] is at index 1*3+1 = 4
        V[0*10 + k0, base_col] += 1  # B[0,k0]
        V[1*10 + k1, base_col] += 1  # B[1,k1]
        W[k0*2 + 0, base_col] += 1  # C[0,k0]
        W[k1*2 + 1, base_col] += 1  # C[1,k1]
        
        # Term 1: (A10 + A11) * B0,k0 * (C1,k0 - C1,k1)
        U[3, base_col + 1] += 1   # A[1,0] is at index 1*3+0 = 3
        U[4, base_col + 1] += 1   # A[1,1]
        V[0*10 + k0, base_col + 1] += 1
        W[k0*2 + 1, base_col + 1] += 1  # C[1,k0]
        W[k1*2 + 1, base_col + 1] -= 1  # C[1,k1]
        
        # Term 2: A00 * (B0,k1 - B1,k1) * (C0,k1 + C1,k1)
        U[0, base_col + 2] += 1
        V[0*10 + k1, base_col + 2] += 1
        V[1*10 + k1, base_col + 2] -= 1
        W[k1*2 + 0, base_col + 2] += 1
        W[k1*2 + 1, base_col + 2] += 1
        
        # Term 3: A11 * (B1,k0 - B0,k0) * (C0,k0 + C1,k0)
        U[4, base_col + 3] += 1
        V[1*10 + k0, base_col + 3] += 1
        V[0*10 + k0, base_col + 3] -= 1
        W[k0*2 + 0, base_col + 3] += 1
        W[k0*2 + 1, base_col + 3] += 1
        
        # Term 4: (A00 + A01) * B1,k1 * (C0,k0 - C0,k1)  -- wait, let me check the sign
        # From the current program: W[k0*2+0] -= 1, W[k1*2+0] += 1
        # So C0,k0 gets -1 and C0,k1 gets +1
        U[0, base_col + 4] += 1
        U[1, base_col + 4] += 1  # A[0,1] is at index 0*3+1 = 1
        V[1*10 + k1, base_col + 4] += 1
        W[k0*2 + 0, base_col + 4] -= 1
        W[k1*2 + 0, base_col + 4] += 1
        
        # Term 5: (A10 - A00) * (B0,k0 + B0,k1) * C1,k1
        U[3, base_col + 5] += 1   # A[1,0]
        U[0, base_col + 5] -= 1   # A[0,0]
        V[0*10 + k0, base_col + 5] += 1
        V[0*10 + k1, base_col + 5] += 1
        W[k1*2 + 1, base_col + 5] += 1
        
        # Term 6: (A01 - A11) * (B1,k0 + B1,k1) * C0,k0
        U[1, base_col + 6] += 1   # A[0,1]
        U[4, base_col + 6] -= 1   # A[1,1]
        V[1*10 + k0, base_col + 6] += 1
        V[1*10 + k1, base_col + 6] += 1
        W[k0*2 + 0, base_col + 6] += 1
        
        # Term 7: A02 * B2,k0 * C0,k0
        U[2, base_col + 7] += 1   # A[0,2] is at index 0*3+2 = 2
        V[2*10 + k0, base_col + 7] += 1
        W[k0*2 + 0, base_col + 7] += 1
        
        # Term 8: A02 * B2,k1 * C0,k1
        U[2, base_col + 8] += 1
        V[2*10 + k1, base_col + 8] += 1
        W[k1*2 + 0, base_col + 8] += 1
        
        # Term 9: A12 * B2,k0 * C1,k0
        U[5, base_col + 9] += 1   # A[1,2] is at index 1*3+2 = 5
        V[2*10 + k0, base_col + 9] += 1
        W[k0*2 + 1, base_col + 9] += 1
        
        # Term 10: A12 * B2,k1 * C1,k1
        U[5, base_col + 10] += 1
        V[2*10 + k1, base_col + 10] += 1
        W[k1*2 + 1, base_col + 10] += 1
    
    # Convert to lists as required
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
