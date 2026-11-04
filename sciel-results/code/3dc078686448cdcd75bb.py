"""Matrix multiplication 3x4x6: search for low-rank exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

def _validate(U, V, W, n=3, m=4, p=6):
    """Check that U,V,W exactly decompose the matrix multiplication tensor.
    
    U: (n*m, rank)
    V: (m*p, rank)  
    W: (p*n, rank)  with W[k*n+i, c] convention
    
    The tensor T[i,j,k] = delta_{i,i} delta_{j,j} delta_{k,k} (identity in each dim)
    Actually the matmul tensor has T[i,j,k,l] = delta_{j,k} but we're doing rank-1 decomposition.
    
    For matmul: (A*B)[i,l] = sum_j A[i,j]*B[j,l]
    Tensor form: T[i,j,l] where the multiplication tensor has entries 1 at (i,j,l) for the 
    "structure" - actually the tensor is defined by:
    
    For rank-1 term c: U[:,c] * V[:,c] * W[:,c] should reproduce the matmul operation.
    
    The matmul tensor T has T[i,j,l] = 1 if j is the shared index... 
    
    Let me think more carefully. The matrix multiplication tensor for n x m times m x p:
    We want: C[i,l] = sum_j A[i,j] * B[j,l]
    
    As a bilinear map: T: R^{n*m} x R^{m*p} -> R^{n*p}
    T(A,B)[i,l] = sum_j A[i,j] B[j,l]
    
    A rank-1 term is: (u . A) * (v . B) * w^T added to output.
    So for term c: sum_c U[i*m+j, c] * A[i,j] * V[j*p+l, c] * B[j,l] * W[l*n+i, c] = C[i,l]
    
    Wait, the W convention is W[k*n+i, r] where k is the column of B / row of C output.
    
    So the condition is: for all i, l:
    sum_c U[i*m+j1, c] * V[j1*p+l, c] * W[l*n+i, c] ... no that's not right either.
    
    Let me re-derive. The standard formulation:
    
    Matmul tensor T of shape (n, m, p) where T[i,j,k] represents the coefficient such that:
    (A B)[i,k] = sum_j A[i,j] B[j,k]
    
    In the border rank / tensor decomposition framework:
    T = sum_c u_c ⊗ v_c ⊗ w_c
    
    where u_c ∈ R^{n*m} (indexed by (i,j)), v_c ∈ R^{m*p} (indexed by (j,k)), 
    w_c ∈ R^{n*p} (indexed by (i,k)).
    
    The tensor entry T[i,j,k] (as a trilinear form on three copies) doesn't quite work because
    matmul is not a trilinear form on three independent vectors.
    
    Actually, the correct formulation: the matrix multiplication tensor M ∈ R^{n*m} ⊗ R^{m*p} ⊗ R^{n*p}
    is defined by:
    M = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{i*p+k}
    
    Wait, with the W convention W[k*n+i, r], the third index is (k,i) not (i,k).
    
    So M = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    
    A rank-1 decomposition: M = sum_c U_c ⊗ V_c ⊗ W_c
    where U_c ∈ R^{n*m}, V_c ∈ R^{m*p}, W_c ∈ R^{n*p} (with W indexed as k*n+i).
    
    The condition is: for all (i,j,k):
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if this is a valid (i,j,k) triple, else 0.
    
    But every (i,j,k) with i in [0,n), j in [0,m), k in [0,p) is valid, so the RHS is always 1.
    
    Wait no. The tensor M has entries:
    M[(i,j),(j',k),(k',i')] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
    
    In flat indexing:
    M[i*m+j, j'*p+k, k'*n+i'] = 1 iff j==j' AND k==k' AND i==i'
    
    So the condition for a rank-r decomposition U,V,W (each with r columns) is:
    For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    For all other combinations (where the indices don't match up properly): = 0
    
    But actually, since i ranges over [0,n), j over [0,m), k over [0,p), and the tensor only has 
    nonzero entries when the "matching" conditions hold, and in this formulation every triple (i,j,k)
    gives a valid equation... 
    
    Hmm, I think I'm overcomplicating this. Let me use the standard verification:
    
    The multiplication tensor for m_{n,m,p} can be verified by checking that the bilinear 
    algorithm correctly computes matrix multiplication. That is, for random A (n x m) and B (m x p):
    C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    should equal (A @ B)[i,k].
    
    Let me implement verification this way.
    """
    n, m, p = 3, 4, 6
    rank = len(U[0])
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    AB = A @ B
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape
    U_r = U_arr.reshape(n, m, rank)  # U[i,j,c]
    V_r = V_arr.reshape(m, p, rank)  # V[j,k,c]
    W_r = W_arr.reshape(p, n, rank)  # W[k,i,c] with index k*n+i
    
    # Compute: C[i,k] = sum_c (sum_j U[i,j,c]*A[i,j]) * (sum_j V[j,k,c]*B[j,k]) * W[k,i,c]
    C = np.zeros((n, p))
    for c in range(rank):
        left = np.einsum('ij,ic->ic', A, U_r[:, :, c])   # (n)  : sum_j U[i,j,c]*A[i,j] for each i
        # Actually: left[i] = sum_j A[i,j] * U_r[i,j,c]
        left = np.sum(A * U_r[:, :, c], axis=1)  # shape (n,)
        right = np.sum(B * V_r[:, :, c], axis=0)  # shape (p,) : right[k] = sum_j B[j,k]*V_r[j,k,c]
        C += np.outer(left, right) * W_r[:, :, c].T  # W_r[k,i,c], so W_r[:,:,c] is (p,n), .T is (n,p)
    
    if not np.allclose(C, AB, atol=1e-9):
        return False
    return True


def _build_tensor_matrix(n=3, m=4, p=6):
    """Build the matrix representation of the matmul tensor for linear algebra approaches.
    
    Returns a sparse-like representation: list of (row, col, val) for the tensor as a 
    linear operator. For our purposes, we'll work with the verification condition directly.
    """
    pass


def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    n, m, p = DIMENSIONS  # 3, 4, 6
    schoolbook_rank = n * m * p  # 72
    
    # Start with schoolbook
    best_U = None
    best_V = None
    best_W = None
    best_rank = schoolbook_rank
    
    # Initialize schoolbook
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_U, best_V, best_W = [row[:] for row in U_sb], [row[:] for row in V_sb], [row[:] for row in W_sb]
    
    # Strategy: Greedy elimination / merging of rank-1 terms
    # The idea: start with schoolbook (72 terms), try to merge pairs of terms into one,
    # or use algebraic structure to find lower rank decompositions.
    
    # Better approach: use the fact that for m_{3,4,6}, we can exploit the rectangular structure.
    # 
    # Known results: 
    # - m_{2,2,2} has rank 7 (Strassen)
    # - m_{3,3,3} has rank <= 23 (Strassen), best known 23
    # - For rectangular: m_{3,4,6} 
    #
    # A simple improvement: use Strassen-like ideas on blocks.
    # 
    # Alternative: random search with linear algebra to find exact decompositions.
    
    # Let's try a structured approach:
    # For 3x4x6, think of it as: (3x2)@(2x6) combined with (3x2)@(2x6) shifted,
    # or use the fact that 4 = 2+2, 6 = 3+3.
    
    # Actually, let's try a direct algebraic approach:
    # The tensor M = sum_{i=0}^{2} sum_{j=0}^{3} sum_{k=0}^{5} e_{3i+j} ⊗ e_{4j+k} ⊗ e_{6k+i}
    # 
    # We want to write this as sum_{c=0}^{r-1} u_c ⊗ v_c ⊗ w_c with minimum r.
    #
    # Lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # So rank >= 24.
    #
    # Upper bound from schoolbook: 72.
    #
    # Can we do better? Let's try to use block structure.
    #
    # View A as [A1; A2] where A1 is 3x2, A2 is 3x2.
    # View B as [B1 B2] where B1 is 2x6, B2 is 2x6.
    # C = A1*B1 + A2*B2
    #
    # This is a sum of two 3x2x6 multiplications.
    # Each 3x2x6 has schoolbook rank 36.
    # Total: 72 (same as before).
    #
    # Alternative: use Strassen on a 2x2 subblock.
    # 
    # Let me try a different decomposition:
    # A (3x4) = [A_left | A_right] where A_left is 3x2, A_right is 3x2
    # B (4x6) = [B_top; B_bottom] where B_top is 2x6, B_bottom is 2x6
    # C = A_left @ B_top + A_right @ B_bottom
    #
    # For each 3x2x6 block multiplication, schoolbook is 36.
    # But maybe we can share computation between the two blocks.
    
    # Let's try a random algebraic approach:
    # Pick random rank-1 terms and solve linear equations.
    
    np.random.seed(12345)
    
    def try_random_decomposition(target_rank, max_iters=200):
        """Try to find a rank-`target_rank` decomposition by random initialization and solving."""
        if target_rank < 24:
            return None  # Below lower bound
        
        for iteration in range(max_iters):
            # Random initialization
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # The condition: for each (i,j,k):
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # 
            # This is a system of n*m*p = 72 equations in r unknowns (the entries of U,V,W).
            # But it's trilinear, not linear. 
            #
            # However, if we fix two of the three, it's linear in the third.
            # 
            # Approach: iterative linearization.
            # 1. Fix V and W, solve for U (linear).
            # 2. Fix U and W, solve for V (linear).
            # 3. Fix U and V, solve for W (linear).
            # Repeat until convergence.
            
            U_cur = U.copy()
            V_cur = V.copy()
            W_cur = W.copy()
            
            converged = False
            for inner_iter in range(50):
                if time.time() > deadline:
                    return None
                
                # Solve for U given V, W:
                # For each (i,j): sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c] summed over k... no)
                # 
                # Actually the equation is:
                # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                #
                # If we fix V and W, for a fixed (i,j), the equations over k are:
                # sum_k [sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]] ... no, each (i,j,k) is a separate equation.
                #
                # For fixed i,j: the equations are for k=0,...,p-1:
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1  for each k
                #
                # This is linear in U[i*m+j,:]: 
                # Let a_k[c] = V[j*p+k,c] * W[k*n+i,c]
                # Then: sum_c U[i*m+j,c] * a_k[c] = 1 for k=0,...,p-1
                # This is a system of p equations in r unknowns (the r components of U[i*m+j,:]).
                # If r >= p, this is generally solvable.
                
                # Build the linear system for U:
                # For each (i,j): p equations, r unknowns
                # Total: n*m*p equations, n*m*r unknowns
                # We can solve block by block.
                
                U_new = np.zeros((n * m, target_rank))
                for i in range(n):
                    for j in range(m):
                        # System: A_u @ U[i*m+j, :] = b
                        # A_u[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        # b[k] = 1
                        A_u = np.zeros((p, target_rank))
                        for k in range(p):
                            A_u[k, :] = V_cur[j * p + k, :] * W_cur[k * n + i, :]
                        b_u = np.ones(p)
                        # Solve least squares
                        try:
                            U_new[i * m + j, :], residuals, rank_u, sv = np.linalg.lstsq(A_u, b_u, rcond=None)
                        except:
                            return None
                
                # Similarly for V:
                V_new = np.zeros((m * p, target_rank))
                for j in range(m):
                    for k in range(p):
                        # Equations over i: sum_c V[j*p+k,c] * U[i*m+j,c] * W[k*n+i,c] = 1 for i=0,...,n-1
                        A_v = np.zeros((n, target_rank))
                        for i in range(n):
                            A_v[i, :] = U_cur[i * m + j, :] * W_cur[k * n + i, :]
                        b_v = np.ones(n)
                        try:
                            V_new[j * p + k, :], _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                        except:
                            return None
                
                # Similarly for W:
                W_new = np.zeros((p * n, target_rank))
                for k in range(p):
                    for i in range(n):
                        # Equations over j: sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1 for j=0,...,m-1
                        A_w = np.zeros((m, target_rank))
                        for j in range(m):
                            A_w[j, :] = U_cur[i * m + j, :] * V_cur[j * p + k, :]
                        b_w = np.ones(m)
                        try:
                            W_new[k * n + i, :], _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
                        except:
                            return None
                
                # Check convergence
                change = (np.linalg.norm(U_new - U_cur) + 
                          np.linalg.norm(V_new - V_cur) + 
                          np.linalg.norm(W_new - W_cur))
                U_cur, V_cur, W_cur = U_new, V_new, W_new
                
                if change < 1e-8:
                    converged = True
                    break
            
            # Verify the solution
            if converged:
                U_list = U_cur.tolist()
                V_list = V_cur.tolist()
                W_list = W_cur.tolist()
                if _validate(U_list, V_list, W_list):
                    # Round to nearest half-integer if close
                    return U_list, V_list, W_list
            elif time.time() > deadline:
                return None
        
        return None
    
    # Try to find a decomposition with rank less than 72
    # Start from just above the lower bound and go up
    
    for target_rank in range(24, schoolbook_rank):
        if time.time() > deadline - 10:
            break
        
        result = try_random_decomposition(target_rank, max_iters=50)
        if result is not None:
            U_list, V_list, W_list = result
            # Verify once more
            if _validate(U_list, V_list, V_list and W_list, n, m, p) if False else _validate(U_list, V_list, W_list):
                best_U, best_V, best_W = U_list, V_list, W_list
                best_rank = target_rank
                break
    
    # If we found something better, return it; otherwise return schoolbook
    if best_rank < schoolbook_rank:
        # Convert to proper format
        U_out = [[float(x) for x in row] for row in best_U]
        V_out = [[float(x) for x in row] for row in best_V]
        W_out = [[float(x) for x in row] for row in best_W]
        return {"U": U_out, "V": V_out, "W": W_out}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
