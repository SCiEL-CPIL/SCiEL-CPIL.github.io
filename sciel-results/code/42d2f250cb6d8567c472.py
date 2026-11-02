"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    # Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] == W[k*n+i, r]
    # Actually the tensor identity is: M[i,j,k] = delta_{ik} for matmul
    # The decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r where
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix multiplication, T[i,j,k] = 1 if k==i... no wait.
    # Actually for matmul: C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is T[i,j,k] = delta_{i,k} in some convention... 
    # Let me re-read: "exactly decompose the matrix multiplication tensor"
    # The tensor M_{n,m,p}[i,j,k] = 1 if we're computing (A*B)[i,k] from A[i,j]*B[j,k]
    # So the tensor has T[i,j,k] = 1 for all i,j,k (it's the identity tensor for matmul)
    # Wait no. The matmul tensor T satisfies: (A*B)[i,k] = sum_{j} A[i,j] B[j,k]
    # This can be written as sum_{j} T[i,j,k] A[i,j] B[j,k] where T[i,j,k] = 1
    # So the tensor is all-ones? That doesn't make sense for rank.
    # 
    # Re-reading the interface: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # The decomposition must satisfy: for all i,j,k:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r] ... no that's not right either.
    #
    # Standard tensor decomposition of matmul:
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    # where T[i,j,k] = delta_{ik} ... no.
    # 
    # Actually, the matrix multiplication tensor M_{n,m,p} is defined by:
    # M_{n,m,p}[i,j,k] = 1 if i==k (for the output index), but that's not standard.
    #
    # Let me think again. The bilinear map (A,B) -> A*B can be written as:
    # (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # This is a 3-tensor T with T[i,j,k] such that:
    # sum_{j} T[i,j,k] * A[i,j] * B[j,k] = (A*B)[i,k]
    # So T[i,j,k] = 1 for all i,j,k? No, that would give sum_j A[i,j]B[j,k] which is correct.
    # But then the tensor is all-ones, and its rank is n*m*p (schoolbook).
    #
    # Wait, I think the convention is:
    # The tensor M_{n,m,p} has entries M[i,j,k] = 1, and the decomposition is:
    # M[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[?,r]
    # 
    # Looking at the seed code: U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
    # So for each (i,j,k) triple, there's a rank-1 term that is 1 at those positions.
    # The sum over r of U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should equal 1 if (i,j,k) is valid.
    # In the schoolbook case, each (i,j,k) has its own term, so the product is 1 for that term and 0 for others.
    #
    # So the condition is: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for "invalid" combinations (which don't exist in this indexing), it should be 0.
    # But actually, the tensor M_{n,m,p} is defined such that it represents the matmul operation.
    # The standard definition: M_{n,m,p}[i,j,k] = 1 for all i,j,k, and the bilinear form is
    # <M, A ⊗ B> = sum_{i,j,k} M[i,j,k] A[i,j] B[j,k] = sum_{i,j,k} A[i,j] B[j,k]
    # But this equals sum_i (sum_j A[i,j] B[j,k]) summed over k, which is not quite right.
    #
    # I think the correct interpretation is:
    # The tensor T has T[i,j,k] = delta_{i,k} in some sense... 
    #
    # Let me just verify with the seed: it sets up rank n*m*p with each term being a single (i,j,k) fiber.
    # So the decomposition is T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k (in the appropriate spaces)
    # And T[i,j,k] = 1 for all i,j,k.
    #
    # So the condition for a valid decomposition of rank R is:
    # For all i in range(n), j in range(m), k in range(p):
    #   sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    #
    # Let me verify this with a simple case.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _rank_of_decomposition(U, V, W):
    return len(U[0])

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _try_rank(n, m, p, rank, max_iter=1000, tol=1e-8):
    """Try to find a decomposition of given rank using alternating minimization."""
    if time.time() > _DEADLINE - 5.0:
        return None
    
    # Initialize randomly
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Alternating minimization
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5.0:
            return None
        
        # Update U: for each (i,j), solve for U[i*m+j, :] given V, W
        # Constraint: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
        # This is a linear system in U[i*m+j, :]
        # For fixed (i,j): sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
        # Let A_k,r = V[j*p+k,r] * W[k*n+i,r]
        # Then sum_r U[i*m+j,r] * A_k,r = 1 for all k
        # This is A * U_col = 1 where A is p x rank
        for i in range(n):
            for j in range(m):
                # Build matrix A: A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                A = np.zeros((p, rank))
                for k in range(p):
                    for r in range(rank):
                        A[k, r] = V[j * p + k, r] * W[k * n + i, r]
                # Solve A * x = ones(p)
                try:
                    x = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    U[i * m + j, :] = x
                except:
                    pass
        
        # Update V: for each (j,k), solve for V[j*p+k, :] given U, W
        # Constraint: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
        # For fixed (j,k): sum_r V[j*p+k,r] * (U[i*m+j,r] * W[k*n+i,r]) = 1 for all i
        # Let B_i,r = U[i*m+j,r] * W[k*n+i,r]
        # Then sum_r V[j*p+k,r] * B_i,r = 1 for all i
        # This is B * V_col = 1 where B is n x rank
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, rank))
                for i in range(n):
                    for r in range(rank):
                        B[i, r] = U[i * m + j, r] * W[k * n + i, r]
                try:
                    x = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                    V[j * p + k, :] = x
                except:
                    pass
        
        # Update W: for each (k,i), solve for W[k*n+i, :] given U, V
        # Constraint: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j
        # For fixed (k,i): sum_r W[k*n+i,r] * (U[i*m+j,r] * V[j*p+k,r]) = 1 for all j
        # Let C_j,r = U[i*m+j,r] * V[j*p+k,r]
        # Then sum_r W[k*n+i,r] * C_j,r = 1 for all j
        # This is C * W_col = 1 where C is m x rank
        for k in range(p):
            for i in range(n):
                C = np.zeros((m, rank))
                for j in range(m):
                    for r in range(rank):
                        C[j, r] = U[i * m + j, r] * V[j * p + k, r]
                try:
                    x = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                    W[k * n + i, :] = x
                except:
                    pass
        
        # Check convergence
        if it % 10 == 0:
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        err = max(err, abs(s - 1.0))
            if err < tol:
                # Success! Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                # Round to nearest half-integer if close
                for row in U_list:
                    for idx in range(len(row)):
                        if abs(row[idx] * 2 - round(row[idx] * 2)) < 1e-6:
                            row[idx] = round(row[idx] * 2) / 2.0
                for row in V_list:
                    for idx in range(len(row)):
                        if abs(row[idx] * 2 - round(row[idx] * 2)) < 1e-6:
                            row[idx] = round(row[idx] * 2) / 2.0
                for row in W_list:
                    for idx in range(len(row)):
                        if abs(row[idx] * 2 - round(row[idx] * 2)) < 1e-6:
                            row[idx] = round(row[idx] * 2) / 2.0
                return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try ranks from low to high
    for rank in range(15, 30):  # Try ranks 15 to 29
        if time.time() > _DEADLINE - 5.0:
            break
        result = _try_rank(n, m, p, rank, max_iter=500)
        if result is not None:
            U, V, W = result
            # Verify
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
