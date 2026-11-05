"""Matrix multiplication 4x5x6 - rank-minimal tensor decomposition."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # A is n x m, B is m x p, C is n x p
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Build the multiplication tensor T[i,j,k] = 1 if k = i*m+j (standard)
    # Actually, T[i,j,k] where i indexes A (n*m), j indexes B (m*p), k indexes C (p*n)
    # C[k*n+i] = sum over j of A[i*m+j] * B[j*p+k]... wait, let me be careful.
    
    # A is n x m: A[row, col], row in 0..n-1, col in 0..m-1
    # B is m x p: B[row, col], row in 0..m-1, col in 0..p-1
    # C is n x p: C[row, col], row in 0..n-1, col in 0..p-1
    # C[row, col] = sum_{j=0}^{m-1} A[row, j] * B[j, col]
    
    # Flattened: A_flat[i*m+j] = A[i,j] for i in 0..n-1, j in 0..m-1
    # B_flat[j*p+k] = B[j,k] for j in 0..m-1, k in 0..p-1
    # C_flat[k*n+i] = C[i,k] for i in 0..n-1, k in 0..p-1 (trace convention W[k*n+i, r])
    
    # So C_flat[k*n+i] = sum_j A_flat[i*m+j] * B_flat[j*p+k]
    
    # Tensor T[a, b, c] where:
    # a = i*m + j (index into A_flat)
    # b = j*p + k (index into B_flat)  
    # c = k*n + i (index into C_flat)
    # T[a,b,c] = 1 if the indices are consistent (same j, same i, same k)
    
    # For the decomposition: C_flat[c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # We need: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all a,b,c
    
    # Schoolbook: for each (i,j,k), one term with U[i*m+j, c_idx]=1, V[j*p+k, c_idx]=1, W[k*n+i, c_idx]=1
    
    def build_schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    def verify(U, V, W, tol=1e-9):
        """Check if U,V,W exactly decompose the tensor."""
        rank = U.shape[1]
        # T[a,b,c] should equal sum_r U[a,r]*V[b,r]*W[c,r]
        # Check on a random set of (a,b,c) triples
        # Actually, check all: for each c, C[c] = sum_r W[c,r] * (U[:,r] * V[:,r])
        # (U[:,r] * V[:,r]) is the outer product flattened... no.
        # sum_r U[a,r]*V[b,r]*W[c,r] = sum_r W[c,r] * (U[:,r] .* V[:,r])[a,b]... 
        # No: U[a,r]*V[b,r] for fixed r gives a matrix M_r[a,b] = U[a,r]*V[b,r]
        # Then sum_r W[c,r]*M_r[a,b] should equal T[a,b,c]
        
        # Efficient check: for each r, M_r = U[:,r] outer V[:,r]
        # Then for each c, sum_r W[c,r] * M_r should equal T[:,:,c]
        
        # Build T
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    T[a, b, c] = 1
        
        # Check
        for c in range(p*n):
            reconstructed = np.zeros((n*m, m*p))
            for r in range(rank):
                if W[c, r] != 0:
                    reconstructed += W[c, r] * np.outer(U[:, r], V[:, r])
            if not np.allclose(reconstructed, T[:, :, c], atol=tol):
                return False
        return True
    
    # Start with schoolbook
    U, V, W = build_schoolbook()
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try Strassen-like reductions
    # The key idea: if we have terms that can be combined using (a+b)(c+d) = ac + ad + bc + bd
    # we can replace 4 multiplications with 7 (Strassen) or fewer in some cases
    
    # For 4x5x6, let's try a specific block decomposition
    # Split: A rows (2,2), A cols (3,2); B rows (3,2), B cols (3,3)
    # This gives blocks:
    # A11(2x3), A12(2x2), A21(2x3), A22(2x2)
    # B11(3x3), B12(3x3), B21(2x3), B22(2x3)
    # C11(2x3), C12(2x3), C21(2x3), C22(2x3)
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # We can use Strassen on the 2x2 block structure:
    # M1 = (A11+A22)*(B11+B22)  -- but dimensions don't match!
    # A11 is 2x3, A22 is 2x2 -- can't add directly
    
    # Alternative: use rank-revealing combinations
    # Let's try: define intermediate quantities that allow sharing
    
    # Actually, let me try a different partition that makes Strassen work better
    # Split A: rows (2,2), cols (2,3)
    # Split B: rows (2,3), cols (3,3)
    # A11(2x2), A12(2x3), A21(2x2), A22(2x3)
    # B11(2x3), B12(2x3), B21(3x3), B22(3x3)
    # C11(2x3), C12(2x3), C21(2x3), C22(2x3)
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Still dimension mismatches for Strassen.
    
    # Let me try yet another approach: use the fact that we can do rank reduction
    # by finding linear dependencies in the tensor.
    
    # For now, let's try a randomized approach: perturb the schoolbook decomposition
    # and try to reduce rank by finding zero columns or dependent columns.
    
    # Actually, the most promising approach: use known results.
    # The rank of 4x5x6 matrix multiplication is known to be at most 108.
    # Let me construct it explicitly.
    
    # One known construction: use the 2x2x2 Strassen (rank 7) recursively.
    # 4x5x6 can be embedded in 4x6x6 (rank <= 7*7*... no)
    
    # Let me try: 4x5x6 <= 4x6x6. 
    # 4x6x6: split as 2x2 blocks of 2x3 matrices.
    # Each 2x3x3 has rank 18 (schoolbook). 4 such blocks = 72.
    # But with Strassen: 7 * (something) ...
    
    # Actually, let me just do a numerical optimization.
    # Initialize with schoolbook, then try to reduce rank by 
    # removing columns that are linear combinations of others.
    
    # Better: use the fact that the tensor has symmetries.
    
    # Let me try a direct construction for rank 108.
    # Idea: use 12 "Strassen-like" reductions, each saving 4 terms.
    # 120 - 12*1 = 108 (each reduction saves 1 term on average)
    
    # Actually, each Strassen reduction of 4 terms to 3 saves 1.
    # 120 - 12 = 108.
    
    # Let me identify 12 independent Strassen reductions.
    
    # For the 4x5x6 case, group the 120 terms into groups of 4 where
    # Strassen applies.
    
    # A Strassen reduction applies when we have terms:
    # A[i,j]*B[j,k] for (i,j,k) in {(i1,j1,k1), (i1,j2,k2), (i2,j1,k3), (i2,j2,k4)}
    # where i1!=i2, j1!=j2, k1!=k2, k3!=k4, and the "cross" terms can be combined.
    
    # Specifically, for fixed i1,i2 and j1,j2, and k1,k2:
    # C[i1,k1] += A[i1,j1]*B[j1,k1]
    # C[i1,k2] += A[i1,j2]*B[j2,k2]
    # C[i2,k1] += A[i2,j1]*B[j1,k1]
    # C[i2,k2] += A[i2,j2]*B[j2,k2]
    
    # Strassen: 
    # M1 = (A[i1,j1]+A[i2,j1])*(B[j1,k1]+B[j1,k2]) -- wait, dimensions
    # Actually for scalar entries:
    # a11*b11, a12*b22, a21*b11, a22*b22
    # Strassen: M1=(a11+a21)(b11+b22), M2=(a11+a21)b22, M3=a11(b11-b22), M4=a22(b11+b22)
    # Then: a11*b11 = M1 - M2 - M3 + ... this is getting complicated.
    
    # Let me just use a simpler approach: try all possible pairs of terms and see
    # if they can be combined.
    
    # Given time constraints, let me implement a pragmatic solution:
    # 1. Start with schoolbook (rank 120)
    # 2. Try to find linear dependencies to reduce rank
    # 3. Return the best found within deadline
    
    # For linear dependency detection:
    # If column r of [U;V;W] (stacked) is a linear combination of other columns,
    # we can eliminate it. But this is not straightforward for tensor decompositions.
    
    # Alternative: use the fact that if two terms differ only in one index,
    # we might be able to combine them.
    
    # Let me try a different tactic: use a known rank-108 construction.
    # After research, one construction uses the following idea:
    # 
    # For 4x5x6, we can write:
    # 4x5x6 = 2x2x2 (Strassen, rank 7) * 2x2x3 (rank 12) ... no, that's not right.
    #
    # Let me just do the numerical approach with a time limit.
    
    # Approach: Start with schoolbook, then use iterative rank reduction.
    # At each step, try to express one column as a combination of others.
    
    # Actually, the cleanest approach: use the tensor's structure directly.
    
    # Let me build the full tensor and use SVD-like methods to find low-rank structure.
    
    # The tensor T has shape (20, 30, 24). We want to find U(20,r), V(30,r), W(24,r)
    # such that T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r].
    
    # This is a 3-mode tensor decomposition. We can use ALS (alternating least squares)
    # to find a rank-r decomposition, then check for exactness.
    
    def try_rank_reduction(rank_target, max_iters=100, seed=42):
        """Try to find an exact rank-r decomposition using ALS."""
        np.random.seed(seed)
        
        # Build tensor T
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    T[a, b, c] = 1
        
        # Initialize
        U = np.random.randn(n*m, rank_target) * 0.1
        V = np.random.randn(m*p, rank_target) * 0.1
        W = np.random.randn(p*n, rank_target) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each a, U[a,:] should satisfy
            # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all b,c
            # This is: U[a,:] @ (V * W) = T[a,:,:] where (V*W)[b,c] = sum_r V[b,r]*W[c,r]
            # Actually: for fixed a, we need U[a,r] such that
            # sum_r U[a,r] * (V[:,r] outer W[:,r]) = T[a,:,:]
            # Let M_r = V[:,r] outer W[:,r], then U[a,:] @ M = T[a,:,:]
            # where M is rank_target x (m*p)*(p*n)... this is large.
            
            # Better: for fixed a, define B_a = T[a,:,:] (shape (m*p, p*n))
            # We need: sum_r U[a,r] * np.outer(V[:,r], W[:,r]) = B_a
            # This is: U[a,:] is a coefficient vector for the basis {np.outer(V[:,r], W[:,r])}
            
            # Form the Gram matrix: G[r1,r2] = sum_{b,c} V[b,r1]*W[c,r1]*V[b,r2]*W[c,r2]
            # = (V^T V) * (W^T W) element-wise... no.
            # G[r1,r2] = sum_b V[b,r1]*V[b,r2] * sum_c W[c,r1]*W[c,r2]
            # = (V^T V)[r1,r2] * (W^T W)[r1,r2]
            
            VtV = V.T @ V  # rank_target x rank_target
            WtW = W.T @ W
            G = VtV * WtW  # element-wise product
            
            # RHS: rhs[r] = sum_{b,c} T[a,b,c] * V[b,r] * W[c,r]
            # = sum_b V[b,r] * sum_c T[a,b,c] * W[c,r]
            # Let's compute: for each r, rhs[r] = <T[a,:,:], V[:,r] outer W[:,r]>
            
            # Efficient: rhs = np.einsum('abc,br,cr->r', T, V, W) for each a
            # But T is sparse (only 120 nonzeros), so let's use that.
            
            # For each a, the nonzeros of T[a,:,:] are at specific (b,c) positions.
            # T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
            # Given a = i*m+j, then b must be j*p+k for some k, and c = k*n+i
            
            # Precompute: for each a, list of (b,c) where T[a,b,c]=1
            a_to_bc = {}
            for i in range(n):
                for j in range(m):
                    a = i*m + j
                    for k in range(p):
                        b = j*p + k
                        c = k*n + i
                        if a not in a_to_bc:
                            a_to_bc[a] = []
                        a_to_bc[a].append((b, c))
            
            # Update U
            U_new = np.zeros_like(U)
            for a in range(n*m):
                if a not in a_to_bc:
                    continue
                # rhs[r] = sum_{(b,c) in a_to_bc[a]} V[b,r] * W[c,r]
                rhs = np.zeros(rank_target)
                for b, c in a_to_bc[a]:
                    rhs += V[b, :] * W[c, :]
                # Solve G @ U_new[a,:] = rhs
                try:
                    U_new[a, :] = np.linalg.solve(G, rhs)
                except np.linalg.LinAlgError:
                    U_new[a, :] = np.linalg.lstsq(G, rhs, rcond=None)[0]
            U = U_new
            
            # Update V similarly
            UtU = U.T @ U
            WtW = W.T @ W
            G = UtU * WtW
            
            V_new = np.zeros_like(V)
            # For each b, find (a,c) where T[a,b,c]=1
            b_to_ac = {}
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i*m + j
                        b = j*p + k
                        c = k*n + i
                        if b not in b_to_ac:
                            b_to_ac[b] = []
                        b_to_ac[b].append((a, c))
            
            for b in range(m*p):
                if b not in b_to_ac:
                    continue
                rhs = np.zeros(rank_target)
                for a, c in b_to_ac[b]:
                    rhs += U[a, :] * W[c, :]
                try:
                    V_new[b, :] = np.linalg.solve(G, rhs)
                except np.linalg.LinAlgError:
                    V_new[b, :] = np.linalg.lstsq(G, rhs, rcond=None)[0]
            V = V_new
            
            # Update W
            UtU = U.T @ U
            VtV = V.T @ V
            G = UtU * VtV
            
            W_new = np.zeros_like(W)
            c_to_ab = {}
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i*m + j
                        b = j*p + k
                        c = k*n + i
                        if c not in c_to_ab:
                            c_to_ab[c] = []
                        c_to_ab[c].append((a, b))
            
            for c in range(p*n):
                if c not in c_to_ab:
                    continue
                rhs = np.zeros(rank_target)
                for a, b in c_to_ab[c]:
                    rhs += U[a, :] * V[b, :]
                try:
                    W_new[c, :] = np.linalg.solve(G, rhs)
                except np.linalg.LinAlgError:
                    W_new[c, :] = np.linalg.lstsq(G, rhs, rcond=None)[0]
            W = W_new
            
            # Check convergence and exactness
            if it % 10 == 0:
                # Quick check on a few entries
                err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i*m + j
                            b = j*p + k
                            c = k*n + i
                            val = np.sum(U[a, :] * V[b, :] * W[c, :])
                            err += abs(val - 1)
                if err < 1e-6:
                    # Verify fully
                    if verify(U, V, W, tol=1e-8):
                        return U, V, W, True
                if time.time() > deadline - 10:
                    break
        
        # Final verification
        if verify(U, V, W, tol=1e-8):
            return U, V, W, True
        return U, V, W, False
    
    # Try to find lower-rank decompositions
    for target_rank in range(108, schoolbook_rank + 1):
        if time.time() > deadline - 20:
            break
        
        U_try, V_try, W_try, success = try_rank_reduction(target_rank, max_iters=50)
        if success:
            best_rank = target_rank
            best_U, best_V, best_W = U_try, V_try, W_try
            break
    
    # If no lower rank found, use schoolbook
    if best_rank == schoolbook_rank:
        U, V, W = build_schoolbook()
    
    # Convert to lists (the interface expects lists)
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    # Round to nearest integer or half-integer if close
    def round_to_half(x):
        return round(2*x) / 2.0
    
    U_list = [[round_to_half(x) for x in row] for row in U_list]
    V_list = [[round_to_half(x) for x in row] for row in V_list]
    W_list = [[round_to_half(x) for x in row] for row in W_list]
    
    # Final verification after rounding
    U_final = np.array(U_list)
    V_final = np.array(V_list)
    W_final = np.array(W_list)
    
    if verify(U_final, V_final, W_final, tol=1e-9):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # If rounding broke it, use unrounded
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    return {"U": U_list, "V": V_list, "W": W_list}
