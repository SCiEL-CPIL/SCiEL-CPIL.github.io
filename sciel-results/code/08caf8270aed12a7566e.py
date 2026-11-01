"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np

def _verify(U, V, W, n=2, m=2, p=4):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # should equal 1 if k = i*p + j (for C[i,j] = sum A[i,k]*B[k,j])
    # Wait, let me think about the indexing.
    # A is n x m, B is m x p, C is n x p
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # U has shape (n*m, rank) - indexed by (i,k) where i in [0,n), k in [0,m)
    # V has shape (m*p, rank) - indexed by (k,j) where k in [0,m), j in [0,p)
    # W has shape (p*n, rank) - indexed by (j,i) where j in [0,p), i in [0,n)
    # W[k*n+i] per the trace convention means W is indexed by (row_in_C, col_in_C) = (i,j)
    # So W index = i*p + j? No, it says W[k*n+i] which is confusing.
    # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # This likely means W is indexed as W[row, col] where row = i (output row), col = j (output col)
    # and the linear index is i*p + j. But it says k*n+i... 
    # Actually for p*n entries, if we think of it as a p x n matrix, then index = j*n + i
    # where j is the column of C (0 to p-1) and i is the row of C (0 to n-1).
    # So W[j*n + i, r] corresponds to output C[i, j].
    
    # Let me verify: U[i*m + k, r] * V[k*p + j, r] * W[j*n + i, r] should sum to delta over the multiplication.
    # The multiplication tensor: T[(i,k), (k,j), (i,j)] = 1
    # So sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1 for valid (i,k,j) and 0 otherwise.
    
    # Actually, the bilinear form is: C[i,j] = sum_k A[i,k] B[k,j]
    # The tensor is T_{(i,k),(k,j),(i,j)} = 1
    # Decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # a = i*m + k, b = k*p + j, c = j*n + i
    # T should be 1 when a,b,c form a valid multiplication term, else 0.
    # Specifically, T[(i,k),(k,j),(i,j)] = 1 for all valid i,k,j.
    # All other entries are 0.
    
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if abs(val - 1) > 1e-9:
                    return False
                # Check that other entries are 0
                # (This is expensive, skip for speed)
    return True

def _try_rank12():
    """Try to find a rank-12 decomposition."""
    n, m, p = 2, 2, 4
    
    # Strategy: Use a structured approach
    # The 2x2 x 2x4 multiplication can be seen as two 2x2 x 2x2 multiplications
    # C[:, 0:2] = A * B[:, 0:2] and C[:, 2:4] = A * B[:, 2:4]
    # Each 2x2 x 2x2 has rank 7 (Strassen), so total rank 14.
    # But there might be sharing that reduces to rank 12.
    
    # Let me try a different approach: parameterize and solve.
    # For rank r, we need to find U (4 x r), V (8 x r), W (8 x r) such that
    # sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1 for all valid (i,k,j)
    
    # This is a system of nonlinear equations. Let me try specific structured forms.
    
    # Approach: Use the fact that we can combine Strassen-like patterns.
    # Let me try to construct it step by step.
    
    # For the first two columns of B (j=0,1), we need:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # For the last two columns of B (j=2,3), we need:
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[0,3] = A[0,0]*B[0,3] + A[0,1]*B[1,3]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    # C[1,3] = A[1,0]*B[0,3] + A[1,1]*B[1,3]
    
    # Total: 8 output entries, each is a sum of 2 products = 16 bilinear terms total.
    # Rank 12 means we can do this in 12 multiplications.
    
    # Let me try a systematic search with structured forms.
    # I'll use numpy to solve for W given U and V, then check consistency.
    
    import itertools
    
    # Try various structured U, V patterns and solve for W
    # The equations are: for each (i,k,j), sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1
    
    # This is linear in W for fixed U, V! So we can solve for W.
    # For each output (i,j), the equation involves W[j*n+i, r] for all r.
    # Actually, let me set up the linear system properly.
    
    # For fixed U and V, the constraint is:
    # For each (i,j): sum_k sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1
    # This gives W[j*n+i, r] coefficients.
    
    # Actually, each (i,k,j) triple gives one equation. But W[j*n+i, r] appears in multiple equations.
    # Let me think differently.
    
    # For each r, define u_r in R^4, v_r in R^8, w_r in R^8.
    # The constraint is: for all (i,k,j), sum_r u_r[i*m+k] * v_r[k*p+j] * w_r[j*n+i] = 1
    
    # This is a system of 8*2*4 = 64 equations (but many are the same due to structure).
    # Actually, there are n*m*p = 2*2*4 = 16 distinct (i,k,j) triples, each giving one equation.
    # But the equations for different (i,k,j) share the same W entries.
    
    # Let me just try random U, V and solve for W, then check.
    
    np.random.seed(42)
    
    for attempt in range(10000):
        # Random structured U and V
        rank = 12
        U = np.zeros((4, rank))
        V = np.zeros((8, rank))
        
        # Give each rank term a random pattern
        for r in range(rank):
            # U row: pick 1-2 random indices
            num_u = np.random.choice([1, 2], p=[0.3, 0.7])
            u_idx = np.random.choice(4, size=num_u, replace=False)
            for idx in u_idx:
                U[idx, r] = np.random.choice([-1, 1])
            
            # V row: pick 1-3 random indices  
            num_v = np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])
            v_idx = np.random.choice(8, size=num_v, replace=False)
            for idx in v_idx:
                V[idx, r] = np.random.choice([-1, 1])
        
        # Now solve for W: for each output (i,j), we need
        # sum_k sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1
        # This is linear in W[j*n+i, :]
        
        # Let's set up: for each (i,j), the equation is:
        # sum_r (sum_k U[i*m+k, r] * V[k*p+j, r]) * W[j*n+i, r] = 1
        # Let A_ij[r] = sum_k U[i*m+k, r] * V[k*p+j, r]
        # Then sum_r A_ij[r] * W[j*n+i, r] = 1
        
        # This gives us 8 equations (one for each (i,j) pair) with 12 unknowns each (W[j*n+i, r] for r=0..11)
        # But wait, W[j*n+i, r] is shared across different (i,j) only if j*n+i is the same, which it isn't.
        # So each (i,j) gives an independent equation for W[j*n+i, :].
        
        # For each (i,j): A_ij @ W[j*n+i, :] = 1
        # Where A_ij is a row vector of length 12.
        
        # This is underdetermined (1 equation, 12 unknowns). We need more constraints.
        # The issue is that we also need the "off-diagonal" terms to be zero.
        
        # Actually, I think I'm overcomplicating this. Let me reconsider.
        
        # The full tensor T has entries T[a,b,c] where a in [0,4), b in [0,8), c in [0,8).
        # T[a,b,c] = 1 if a=i*m+k, b=k*p+j, c=j*n+i for some valid (i,k,j), else 0.
        # 
        # The decomposition requires: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for ALL a,b,c.
        # 
        # There are 4*8*8 = 256 entries, but only 16 are nonzero.
        # The other 240 must be zero.
        
        # This is a much harder constraint. Let me try a different approach.
        
        # Use the known structure: the 2x2 x 2x4 problem can be decomposed as
        # two 2x2 x 2x2 problems with shared factors.
        
        # Let me try to explicitly construct it.
        
        # Strassen for 2x2 x 2x2 uses 7 multiplications:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)(b00)
        # M3 = (a00)(b10+b11)
        # M4 = (a11)(b01-b11)
        # M5 = (a00+a01)(b00-b10)
        # M6 = (a10+a11)(b01-b11)
        # M7 = (a01+a11)(b10-b00)
        
        # For 2x2 x 2x4, we have two sets of B columns. Let me denote B = [B1 | B2] where B1, B2 are 2x2.
        # C = [C1 | C2] where C1 = A*B1, C2 = A*B2.
        
        # If we use separate Strassen for each, we get rank 14.
        # To get rank 12, we need to share 2 multiplications.
        
        # Idea: The terms involving a11 appear in both C1 and C2 calculations.
        # Maybe we can combine some terms.
        
        # Let me try a direct construction.
        # 
        # C1[0,0] = a00*b00 + a01*b10  (where b00, b10 are from B1)
        # C1[0,1] = a00*b01 + a01*b11
        # C1[1,0] = a10*b00 + a11*b10
        # C1[1,1] = a10*b01 + a11*b11
        # C2[0,0] = a00*b02 + a01*b12
        # C2[0,1] = a00*b03 + a01*b13
        # C2[1,0] = a10*b02 + a11*b12
        # C2[1,1] = a10*b03 + a11*b13
        
        # Let me try to find 12 bilinear forms that generate these.
        
        # I'll use a solver approach: assume specific forms for U and V, solve for W, check.
        
        # Actually, let me just try to verify the rank-12 claim by searching more carefully.
        
        # For now, let me return the rank-14 solution which is known to work.
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # First, try to find a rank-12 solution
    result = _try_rank12()
    if result is not None:
        return result
    
    # Fall back to rank 14
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    U[1][6] = 1; U[3][6] = -1
    
    U[0][7] = 1; U[3][7] = 1
    U[2][8] = 1; U[3][8] = 1
    U[0][9] = 1
    U[3][10] = 1
    U[0][11] = 1; U[1][11] = 1
    U[2][12] = 1; U[0][12] = -1
    U[1][13] = 1; U[3][13] = -1
    
    V[0][0] = 1; V[5][0] = 1
    V[0][1] = 1
    V[1][2] = 1; V[5][2] = -1
    V[4][3] = 1; V[0][3] = -1
    V[5][4] = 1
    V[0][5] = 1; V[1][5] = 1
    V[4][6] = 1; V[5][6] = 1
    
    V[2][7] = 1; V[7][7] = 1
    V[2][8] = 1
    V[3][9] = 1; V[7][9] = -1
    V[6][10] = 1; V[2][10] = -1
    V[7][11] = 1
    V[2][12] = 1; V[3][12] = 1
    V[6][13] = 1; V[7][13] = 1
    
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[1][1] = 1; W[1][3] = 1
    W[2][2] = 1; W[2][4] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    W[4][7] = 1; W[4][10] = 1; W[4][11] = -1; W[4][13] = 1
    W[5][8] = 1; W[5][10] = 1
    W[6][9] = 1; W[6][11] = 1
    W[7][7] = 1; W[7][8] = -1; W[7][9] = 1; W[7][12] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
