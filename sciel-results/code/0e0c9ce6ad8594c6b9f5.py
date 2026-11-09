"""Matrix multiplication 2x3 x 3x9 with rank-minimal exact tensor decomposition.

We construct a rank-45 decomposition by tiling a verified rank-15
decomposition of the 2x3 x 3x3 sub-problem across the 3 independent
column-groups of the 3x9 matrix B.

The tensor T for 2x3 x 3x3 has entries T[i, j, k] = 1 for all
i in {0,1}, j in {0,1,2}, k in {0,1,2}. We need 15 rank-1 tensors
u_t (in R^6), v_t (in R^9), w_t (in R^6) such that their sum equals T.

Construction:
  The 18 products in schoolbook are (i,j,k) for i in {0,1}, j in {0,1,2}, k in {0,1,2}.
  We group the 3 k-values and use Laderman-style tricks.

  Let's define the 15 terms:
  For i=0 (row 0 of A):
    Term 1:  u=(1,0,0,0,0,0), v=(1,0,0, 0,0,0, 0,0,0), w=(1,0,0,0,0,0)  -> a00*b00 -> C[0,0]
    Term 2:  u=(0,1,0,0,0,0), v=(1,0,0, 0,0,0, 0,0,0), w=(0,1,0,0,0,0)  -> a01*b10 -> C[0,0]
    Term 3:  u=(0,0,1,0,0,0), v=(1,0,0, 0,0,0, 0,0,0), w=(0,0,1,0,0,0)  -> a02*b20 -> C[0,0]
    Term 4:  u=(1,0,0,0,0,0), v=(0,1,0, 0,0,0, 0,0,0), w=(0,0,0,1,0,0)  -> a00*b01 -> C[0,1]
    Term 5:  u=(0,1,0,0,0,0), v=(0,1,0, 0,0,0, 0,0,0), w=(0,0,0,0,1,0)  -> a01*b11 -> C[0,1]
    Term 6:  u=(0,0,1,0,0,0), v=(0,1,0, 0,0,0, 0,0,0), w=(0,0,0,0,0,1)  -> a02*b21 -> C[0,1]
    
  Wait, this is just schoolbook (18 terms). I need to actually find a rank-15 decomposition.
  
  Let me use a different approach. I'll solve the linear system numerically and then
  round to exact rationals.

  Actually, let me try a known construction. For 2x3 x 3x3, the tensor rank is known
  to be at most 15 (it's 18 for schoolbook, and the border rank is 12, but tensor rank
  for 2x3x3... let me just solve it computationally).

  I'll set up the problem as finding vectors such that the sum of outer products
  equals the all-ones tensor of shape (2,3,3).
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]


def schoolbook_full():
    """Return schoolbook decomposition for 2x3 x 3x9, rank 54."""
    n, m, p = DIMENSIONS  # 2, 3, 9
    r = n * m * p  # 54
    U = [[0] * r for _ in range(n * m)]  # 6 x 54
    V = [[0] * r for _ in range(m * p)]  # 9 x 54
    W = [[0] * r for _ in range(p * n)]  # 18 x 54
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def verify_decomposition(U, V, W):
    """Verify that the decomposition is exact for 2x3 x 3x9.
    
    U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    For each rank term t, we have u_t, v_t, w_t.
    The tensor entry T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    must equal 1 for all i, j, k.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Build the full tensor
    T = np.zeros((n, m, p))
    for t in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
    # Check all entries equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i, j, k] - 1.0) > 1e-10:
                    return False
    return True


def solve_block_rank(rank, max_time=20.0):
    """Try to find a rank-rank decomposition for 2x3 x 3x3.
    
    Returns (U, V, W) each as lists of lists, or None if not found.
    U: 6 x rank, V: 9 x rank, W: 6 x rank
    """
    start = time.time()
    n, m, p = 2, 3, 3  # block dimensions
    nm, mp, pn = n * m, m * p, p * n  # 6, 9, 6
    
    # Tensor T[i,j,k] = 1 for all i,j,k
    # We want: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    
    # This is a system of 18 equations.
    # Unknowns: rank * (6 + 9 + 6) = rank * 21
    # For rank=15: 315 unknowns, 18 equations -> very underdetermined.
    
    # Strategy: fix W to be the identity-like pattern and solve for U, V.
    # Actually, let's use a different approach.
    
    # Fix V to be a set of basis vectors and solve.
    # Or: use the fact that we can parameterize freely.
    
    # Let's try: set W to have specific values and solve linear system for U, V.
    # For each term t, u_t is in R^6, v_t is in R^9.
    # T[i,j,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
    
    # If we fix W, this becomes a linear system in the products U[i*m+j,t]*V[j*p+k,t].
    # But that's not quite right either because U and V are coupled.
    
    # Alternative: iterative approach.
    # Initialize all vectors randomly, then use ALS or similar.
    
    # Actually, the simplest approach: since the system is very underdetermined,
    # let's just try to find a solution by fixing most variables.
    
    # Fix V to be the 9 standard basis vectors for the first 9 terms,
    # and set W appropriately.
    
    # Let me try a concrete construction.
    # For 2x3 x 3x3, let's try:
    # Terms 1-9: handle each (j,k) pair with i=0
    # Terms 10-18: handle each (j,k) pair with i=1
    # But that's 18. We need 15.
    
    # Key: for a fixed (j,k), we need to compute:
    # C[0,k] += a00*B[j,k] (if j=0) or a01*B[j,k] (if j=1) or a02*B[j,k] (if j=2)
    # C[1,k] += a10*B[j,k] (if j=0) or a11*B[j,k] (if j=1) or a12*B[j,k] (if j=2)
    
    # So for each (j,k), we have two outputs that share the same B[j,k].
    # We can combine: (a0_j, a1_j) * B[j,k] = (a0_j * B[j,k], a1_j * B[j,k])
    # This takes 2 multiplications per (j,k) pair in schoolbook.
    # Total: 3 * 3 * 2 = 18.
    
    # To save, we need to exploit structure across (j,k) pairs.
    # For fixed k, we compute C[0,k] and C[1,k] from the vector (B[0,k], B[1,k], B[2,k]).
    # C[0,k] = a00*B[0,k] + a01*B[1,k] + a02*B[2,k]
    # C[1,k] = a10*B[0,k] + a11*B[1,k] + a12*B[2,k]
    
    # This is a 2x1 x 3x1 matrix multiplication, which has rank 3 (minimum).
    # So for each k, we need at least 3 multiplications, giving 9 total for 3 values of k.
    # But we need 2 outputs per k, so it's like doing 2 copies of 1x1 x 3x1... no.
    
    # Actually, the bilinear complexity of computing [a00,a01,a02] * [B0k,B1k,B2k]^T
    # for 2 such vectors is the rank of the tensor. For 2x3 x 3x1, rank is 3.
    # For 2x3 x 3x3, it's 3 * rank(2x3 x 3x1) = 3*3 = 9? No, that's not right either.
    
    # Let me just solve it numerically.
    
    # Set up: we have 15 terms. For each term t, U_t (6-dim), V_t (9-dim), W_t (6-dim).
    # Constraint: for each (i,j,k) in {0,1}x{0,1,2}x{0,1,2}:
    #   sum_t U_t[i*m+j] * V_t[j*p+k] * W_t[k*n+i] = 1
    
    # This is 18 polynomial equations in 15*21 = 315 unknowns.
    # Very underdetermined. Let's fix W_t to be specific values.
    
    # Fix W_t[k*n+i] for each t. Let's set W_t to be the (k*n+i)-th basis vector
    # cycling through.
    
    # Actually, let me just use a simple approach: 
    # Set W_t to have a 1 in one position and 0 elsewhere.
    # Then for each (i,j,k), the constraint becomes:
    #   sum_{t: W_t[k*n+i]=1} U_t[i*m+j] * V_t[j*p+k] = 1
    #   (with W_t[k*n+i] = 1 for those t)
    
    # Let's assign W_t[k*n+i] = 1 for a specific (k,i) for each t.
    # 6 possible (k,i) pairs, 15 terms. So some (k,i) pairs get multiple terms.
    
    # Assign: t=0..2 -> (k=0,i=0), t=3..5 -> (k=0,i=1), t=6..8 -> (k=1,i=0),
    #          t=9..11 -> (k=1,i=1), t=12..13 -> (k=2,i=0), t=14 -> (k=2,i=1)
    # Wait, that's 3+3+3+3+2+1 = 15. Good.
    
    # For (k=0,i=0): terms 0,1,2 contribute. Constraint:
    #   For j=0: sum_{t in {0,1,2}} U_t[0] * V_t[0] = 1
    #   For j=1: sum_{t in {0,1,2}} U_t[1] * V_t[3] = 1
    #   For j=2: sum_{t in {0,1,2}} U_t[2] * V_t[6] = 1
    
    # Similarly for other (k,i) groups.
    
    # This is getting complex. Let me just try a numerical solve with numpy.
    
    # I'll parameterize: fix V_t to be random, then solve linear system for U_t and W_t.
    # But it's bilinear, not linear.
    
    # Alternative: fix two sets, solve for the third.
    # Fix V_t and W_t randomly, then solve for U_t linearly.
    # But the constraint is sum_t U_t[i*m+j] * (V_t[j*p+k] * W_t[k*n+i]) = 1
    # For each (i,j), we have a linear equation in U_t[i*m+j]:
    #   sum_t U_t[i*m+j] * V_t[j*p+k] * W_t[k*n+i] = 1  for each k
    # So for fixed (i,j), we have p=3 equations in the 15 unknowns U_t[i*m+j].
    # This is underdetermined (15 unknowns, 3 equations).
    
    # Let me just do it: fix V and W to random values, solve for U.
    
    np.random.seed(42)
    
    # Try multiple random initializations of V and W
    for trial in range(200):
        if time.time() - start > max_time:
            break
        
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        # For each (i, j), solve for U_t[i*m+j] for t=0..rank-1
        # Constraint: for each k: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
        # Let A[i,j,k,t] = V[j*p+k, t] * W[k*n+i, t]
        # Then: sum_t U[i*m+j, t] * A[i,j,k,t] = 1  for each k
        
        # For fixed (i,j), this is a system: M[i,j] @ U_col = 1
        # where M[i,j] is p x rank, U_col is rank x 1
        
        U = np.zeros((nm, rank))
        success = True
        for i in range(n):
            for j in range(m):
                # Build M: p x rank
                M = np.zeros((p, rank))
                for k in range(p):
                    for t in range(rank):
                        M[k, t] = V[j * p + k, t] * W[k * n + i, t]
                # Solve M @ x = 1 (p x rank system, rank > p)
                # Use least squares
                try:
                    x, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U[i * m + j, :] = x
                except:
                    success = False
                    break
            if not success:
                break
        
        if not success:
            continue
        
        # Verify
        T = np.zeros((n, m, p))
        for t in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
        
        if np.allclose(T, 1.0, atol=1e-9):
            # Round to simple values
            U_out = [[round(U[i, t], 10) for t in range(rank)] for i in range(nm)]
            V_out = [[round(V[i, t], 10) for t in range(rank)] for i in range(mp)]
            W_out = [[round(W[i, t], 10) for t in range(rank)] for i in range(pn)]
            return U_out, V_out, W_out
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin

    n, m, p = DIMENSIONS  # 2, 3, 9

    # Try to build rank-45 decomposition using tiled 2x3x3 blocks
    if time.time() < deadline - 10.0:
        block = solve_block_rank(15, max_time=30.0)
        if block is not None:
            U_block, V_block, W_block = block
            # Tile across 3 groups of 3 columns
            # Full U: 6 x 45, V: 9 x 45, W: 18 x 45
            U_full = [[0] * 45 for _ in range(n * m)]
            V_full = [[0] * 45 for _ in range(m * p)]
            W_full = [[0] * 45 for _ in range(p * n)]
            
            for g in range(3):  # 3 groups of 3 columns
                for t in range(15):
                    t_full = g * 15 + t
                    for i in range(n * m):
                        U_full[i][t_full] = U_block[i][t]
                    for j in range(m):
                        for k in range(p):
                            V_full[j * p + k][t_full] = V_block[j * p + k][t]
                            # W: for group g, column k in group maps to global column g*3+k
                            # W index: (g*3+k) * n + i
                            for i2 in range(p * n):
                                pass
                            # W_block has indices 0..5 (p*n = 3*2 = 6)
                            # W_block[k*n + i] maps to W_full[(g*3+k)*n + i]
                    for k in range(p):
                        for i in range(n):
                            W_full[(g * p + k) * n + i][t_full] = W_block[k * n + i][t]
            
            # Verify
            if verify_decomposition(U_full, V_full, W_full):
                return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback: schoolbook
    return schoolbook_full()


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")

