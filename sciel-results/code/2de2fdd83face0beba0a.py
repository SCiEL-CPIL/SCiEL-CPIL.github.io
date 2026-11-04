"""Matrix multiplication 2x4x6: rank-minimal exact tensor decomposition search."""
import time
import itertools
import random
from fractions import Fraction

import numpy as np
import scipy.linalg as sla

DIMENSIONS = [2, 4, 6]
TIME_LIMIT = 110.0  # safety margin before 120s wall-clock kill


def _tensor_entries(n, m, p):
    """Return the list of (i, j, k) entries of the matrix multiplication tensor."""
    return [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]


def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W with shapes (n*m, rank), (m*p, rank), (p*n, rank)
    exactly decompose the matrix multiplication tensor.
    
    The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    should equal 1 if (i,j,k) is a valid entry, 0 otherwise.
    """
    rank = U.shape[1]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need this to be 1 for all (i,j,k) in the tensor support.
    
    # Efficient check: for each r, the outer product U[:,r] * V[:,r] * W[:,r]
    # U[:,r] is a vector over (i,j) indices, V[:,r] over (j,k), W[:,r] over (k,i)
    # The product at (i,j,k) is U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # Build the full tensor as a 3D array
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    T = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        u_r = U[:, r].reshape(n, m)
        v_r = V[:, r].reshape(m, p)
        w_r = W[:, r].reshape(p, n)
        # For each (i,j,k): u_r[i,j] * v_r[j,k] * w_r[k,i]
        # This is a 3D tensor: for fixed r, it's the elementwise product
        # T_r[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[k,i]
        # We can compute this as: for each i,j: u_r[i,j] * (v_r[j,:] * w_r[:,i])
        for i in range(n):
            for j in range(m):
                if u_r[i, j] != 0:
                    T[i, j, :] += u_r[i, j] * (v_r[j, :] * w_r[:, i])
    
    # Check: T should be all ones (since all (i,j,k) are valid entries for 2x4x6)
    # Actually for matrix multiplication, ALL (i,j,k) combinations are valid.
    # The tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True


def _check_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using matrix operations."""
    rank = U.shape[1]
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # Reshape
    U3 = U.reshape(n, m, rank)  # U3[i,j,r]
    V3 = V.reshape(m, p, rank)  # V3[j,k,r]
    W3 = W.reshape(p, n, rank)  # W3[k,i,r]
    
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    
    # For fixed i: T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # = sum_r (U3[i,j,r] * W3[k,i,r]) * V3[j,k,r]
    
    # Let's compute using einsum-like approach
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    
    T = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        u_r = U3[:, :, r]  # (n, m)
        v_r = V3[:, :, r]  # (m, p)
        w_r = W3[:, :, r]  # (p, n)
        # T[i,j,k] += u_r[i,j] * v_r[j,k] * w_r[k,i]
        # For fixed (i,j): u_r[i,j] * sum over k... no, k is free.
        # T[i,j,k] += u_r[i,j] * v_r[j,k] * w_r[k,i]
        for i in range(n):
            for j in range(m):
                if u_r[i, j] != 0:
                    # T[i,j,:] += u_r[i,j] * (v_r[j,:] * w_r[:,i])
                    T[i, j, :] += u_r[i, j] * (v_r[j, :] * w_r[:, i])
    
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True


def _try_strassen_pattern(n, m, p, max_time):
    """
    Try to find a rank-r decomposition based on Strassen-like patterns.
    For 2x4x6, the schoolbook rank is 48.
    Known lower bound: max(n*m, m*p, p*n) = max(8, 24, 12) = 24.
    
    Strategy: Start from schoolbook and try to merge terms.
    Use a randomized/local search approach to find lower rank.
    """
    start_time = time.time()
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Try to find lower rank via iterative merging
    # Each "term" r contributes: U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # We want to replace multiple terms with fewer terms.
    
    # Approach: Use the fact that for matrix multiplication,
    # we can sometimes combine terms that share structure.
    
    # Let's try a direct approach: assume rank r, and solve for the factors.
    # The constraint is: for all (i,j,k), sum_{r=1}^{R} U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # This is a system of n*m*p equations in R*(n*m + m*p + p*n) unknowns.
    # For n=2, m=4, p=6: 48 equations, R*(8+24+12) = 44*R unknowns.
    # For R=4: 176 unknowns, 48 equations -> very underdetermined.
    
    # But the equations are trilinear, so we can't just solve linearly.
    # However, if we fix two of the three matrices, the equations become linear in the third.
    
    # Strategy: Fix U and V randomly, then solve for W.
    # T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # For fixed (k,i): W[k*n+i, r] appears in equations for all j.
    # T[i,j,k] = sum_r (U[i*m+j,r]*V[j*p+k,r]) * W[k*n+i,r]
    # Let A_{(i,j,k), r} = U[i*m+j,r] * V[j*p+k,r]
    # Then T[i,j,k] = sum_r A_{(i,j,k), r} * W[k*n+i, r]
    
    # But W[k*n+i, r] is shared across different (i,j,k) with same (k,i).
    # For fixed (k,i), the unknowns are W[k*n+i, 0..R-1].
    # The equations involving W[k*n+i, :] are: for all j, T[i,j,k] = sum_r A_{(i,j,k),r} * W[k*n+i,r]
    # That's m equations in R unknowns.
    
    # Total: for each (k,i) pair, we have m equations in R unknowns.
    # Number of (k,i) pairs: p*n = 12.
    # Total equations: p*n*m = 48. Total unknowns: p*n*R = 12*R.
    
    # If R > m = 4, we have more unknowns than equations per (k,i) group, so solutions exist.
    # But we need the SAME U and V to work for ALL (k,i) groups simultaneously.
    
    # Let's try: fix U and V, then check if W can be solved.
    # For each (k,i): we need W[k*n+i, :] such that for all j:
    #   sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # This is a linear system: M_{(k,i)} * w_{(k,i)} = 1
    # where M_{(k,i)}[j, r] = U[i*m+j, r] * V[j*p+k, r]
    # and w_{(k,i)} = W[k*n+i, :]
    
    # So we need M_{(k,i)} to have rank >= 1 (which it will if U,V are nonzero)
    # and the system to be consistent.
    
    # Since we have m equations in R unknowns, and m < R for R > 4,
    # the system is underdetermined and always has a solution IF the rows
    # of M_{(k,i)} are such that 1 is in the column space... actually no.
    # We need: there exists w such that M*w = 1_vector.
    # This is possible iff 1_vector is in the column space of M.
    # Since M has R columns and m rows, column space is a subspace of R^m.
    # We need the all-ones vector to be in this subspace.
    
    # This is a strong constraint. Let's try random U, V and see if W exists.
    
    # Actually, let's think differently. Let's parameterize more cleverly.
    
    # For matrix multiplication C = A*B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # A rank-R decomposition means: C[i,k] = sum_{r=1}^R f_r(A) * g_r(B)
    # where f_r and g_r are bilinear (or linear) forms.
    
    # More precisely: C[i,k] = sum_r U[i,*,r] dot something... 
    
    # Let me use the standard tensor decomposition form:
    # T_{ijk} = sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)}
    # where U_{ij}^{(r)} is a scalar (not a vector), same for V and W.
    
    # So U is n*m*R, V is m*p*R, W is p*n*R, each entry is a scalar.
    # T_{ijk} = sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)}
    
    # This is what we have. Let's try a direct optimization approach.
    
    # For small ranks, we can try to solve the system.
    # Let's try rank R from low to high.
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Start with schoolbook
    best_result = (U_sb, V_sb, W_sb)
    
    # Try to find lower rank
    # For each candidate rank R, we have 48 equations and 44*R unknowns.
    # The system is trilinear, so we use alternating minimization.
    
    for R in range(max(24, n*m), schoolbook_rank):  # Start from lower bound
        if time.time() - start_time > max_time * 0.7:
            break
        
        # Try multiple random initializations
        for trial in range(50):
            if time.time() - start_time > max_time * 0.9:
                break
            
            # Random initialization
            U_r = np.random.randn(n * m, R)
            V_r = np.random.randn(m * p, R)
            W_r = np.random.randn(p * n, R)
            
            # Alternating minimization: fix two, solve for the third
            converged = False
            for it in range(200):
                if time.time() - start_time > max_time * 0.95:
                    break
                
                # Fix U and V, solve for W
                # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # For fixed (k,i): sum_r (U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1 for all j
                # M_{(k,i)}[j, r] = U[i*m+j, r] * V[j*p+k, r]
                # Need M_{(k,i)} * W[k*n+i, :] = ones(m)
                
                W_new = np.zeros((p * n, R))
                feasible = True
                for k in range(p):
                    for i in range(n):
                        # Build M matrix: (m, R)
                        M = np.zeros((m, R))
                        for j in range(m):
                            for r in range(R):
                                M[j, r] = U_r[i * m + j, r] * V_r[j * p + k, r]
                        # Solve M * w = ones(m)
                        try:
                            # Use least squares
                            w, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                            W_new[k * n + i, :] = w
                        except:
                            feasible = False
                            break
                    if not feasible:
                        break
                if not feasible:
                    break
                W_r = W_new
                
                # Fix V and W, solve for U
                # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # For fixed (i,j): sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
                # M_{(i,j)}[k, r] = V[j*p+k, r] * W[k*n+i, r]
                # Need M_{(i,j)} * U[i*m+j, :] = ones(p)
                
                U_new = np.zeros((n * m, R))
                feasible = True
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, R))
                        for k in range(p):
                            for r in range(R):
                                M[k, r] = V_r[j * p + k, r] * W_r[k * n + i, r]
                        try:
                            u, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                            U_new[i * m + j, :] = u
                        except:
                            feasible = False
                            break
                    if not feasible:
                        break
                if not feasible:
                    break
                U_r = U_new
                
                # Fix U and W, solve for V
                # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # For fixed (j,k): sum_r V[j*p+k,r] * (U[i*m+j,r] * W[k*n+i,r]) = 1 for all i
                # M_{(j,k)}[i, r] = U[i*m+j, r] * W[k*n+i, r]
                # Need M_{(j,k)} * V[j*p+k, :] = ones(n)
                
                V_new = np.zeros((m * p, R))
                feasible = True
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, R))
                        for i in range(n):
                            for r in range(R):
                                M[i, r] = U_r[i * m + j, r] * W_r[k * n + i, r]
                        try:
                            v, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                            V_new[j * p + k, :] = v
                        except:
                            feasible = False
                            break
                    if not feasible:
                        break
                if not feasible:
                    break
                V_r = V_new
                
                # Check convergence
                if _check_decomposition_fast(U_r, V_r, W_r, n, m, p, tol=1e-6):
                    converged = True
                    break
            
            if converged:
                # Verify with tighter tolerance
                if _check_decomposition_fast(U_r, V_r, W_r, n, m, p, tol=1e-9):
                    best_result = (U_r.copy(), V_r.copy(), W_r.copy())
                    best_rank = R
                    break  # Found valid decomposition at this rank, try lower? No, we're going up.
                elif _check_decomposition_fast(U_r, V_r, W_r, n, m, p, tol=1e-7):
                    # Close but not exact - try to refine
                    # Use the current solution as starting point for more iterations
                    for it in range(500):
                        if time.time() - start_time > max_time * 0.95:
                            break
                        # Same alternating minimization
                        W_new = np.zeros((p * n, R))
                        feasible = True
                        for k in range(p):
                            for i in range(n):
                                M = np.zeros((m, R))
                                for j in range(m):
                                    for r in range(R):
                                        M[j, r] = U_r[i * m + j, r] * V_r[j * p + k, r]
                                try:
                                    w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                                    W_new[k * n + i, :] = w
                                except:
                                    feasible = False
                                    break
                            if not feasible:
                                break
                        if not feasible:
                            break
                        W_r = W_new
                        
                        U_new = np.zeros((n * m, R))
                        feasible = True
                        for i in range(n):
                            for j in range(m):
                                M = np.zeros((p, R))
                                for k in range(p):
                                    for r in range(R):
                                        M[k, r] = V_r[j * p + k, r] * W_r[k * n + i, r]
                                try:
                                    u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                                    U_new[i * m + j, :] = u
                                except:
                                    feasible = False
                                    break
                            if not feasible:
                                break
                        if not feasible:
                            break
                        U_r = U_new
                        
                        V_new = np.zeros((m * p, R))
                        feasible = True
                        for j in range(m):
                            for k in range(p):
                                M = np.zeros((n, R))
                                for i in range(n):
                                    for r in range(R):
                                        M[i, r] = U_r[i * m + j, r] * W_r[k * n + i, r]
                                try:
                                    v, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                                    V_new[j * p + k, :] = v
                                except:
                                    feasible = False
                                    break
                            if not feasible:
                                break
                        if not feasible:
                            break
                        V_r = V_new
                        
                        if _check_decomposition_fast(U_r, V_r, W_r, n, m, p, tol=1e-9):
                            best_result = (U_r.copy(), V_r.copy(), W_r.copy())
                            best_rank = R
                            break
                    if best_rank == R:
                        break
    
    # Convert to lists
    U_list = best_result[0].tolist()
    V_list = best_result[1].tolist()
    W_list = best_result[2].tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}


def _try_direct_linear_solve(n, m, p, R, max_time):
    """
    Try to find a rank-R decomposition by fixing U and V with a structured form
    and solving linearly for W, then checking.
    
    Alternative: use the fact that for fixed U, V, the problem in W is linear.
    Let's try structured U, V that exploit the matrix multiplication structure.
    """
    start_time = time.time()
    
    # For matrix multiplication, a natural structure is:
    # U[i*m+j, r] = a_r * b_{i,j}^{(r)}
    # V[j*p+k, r] = c_r * d_{j,k}^{(r)}
    # W[k*n+i, r] = e_r * f_{k,i}^{(r)}
    
    # Actually, let's try a simpler approach: 
    # Use the "block" structure. For 2x4x6:
    # A is 2x4, B is 4x6, C is 2x6.
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # A rank-R decomposition: C[i,k] = sum_r F_r(A) G_r(B)
    # where F_r is linear in A and G_r is linear in B.
    # F_r(A) = sum_{i,j} U_{ij}^{(r)} A[i,j]
    # G_r(B) = sum_{j,k} V_{jk}^{(r)} B[j,k]
    # C[i,k] = sum_r (sum_{i',j'} U_{i'j'}^{(r)} A[i',j']) (sum_{j'',k''} V_{j''k''}^{(r)} B[j'',k''])
    
    # This doesn't directly give the right form. Let me reconsider.
    
    # The standard form is:
    # C[i,k] = sum_r sum_j U_{ij}^{(r)} A[i,j] * V_{jk}^{(r)} B[j,k] * W_{ki}^{(r)}
    # No, that's not right either.
    
    # The correct form for matrix multiplication tensor decomposition:
    # T_{ijk} = sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)}
    # where T_{ijk} = 1 for all valid (i,j,k).
    
    # And the multiplication is:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_j T_{ijk} A[i,j] B[j,k]
    # = sum_r sum_j U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)} A[i,j] B[j,k]
    
    # For this to equal sum_j A[i,j] B[j,k] for all A, B:
    # We need: sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)} = 1 for all (i,j,k).
    
    # This is exactly our constraint. Let's try a structured approach.
    
    # For the 2x2x2 case (Strassen), we have 7 terms.
    # For 2x4x6, let's try to find a good rank.
    
    # Lower bound: max(n*m, m*p, p*n) = max(8, 24, 12) = 24
    # Upper bound (schoolbook): 48
    # Best known for similar problems suggests maybe around 30-40.
    
    # Let's try a more systematic approach:
    # Fix U to have a specific structure, then solve.
    
    # Structure: U[i*m+j, r] depends on (i,j) and r.
    # Let's try U[i*m+j, r] = delta_{i, r mod n} * alpha_{j, r} for some pattern.
    
    # Actually, let's just do the alternating minimization but with better initialization.
    
    # Initialize U, V, W with the schoolbook pattern and then randomly perturb.
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Initialize with a subset of schoolbook terms
    # If R < n*m*p, we can't use all schoolbook terms.
    # Let's use a random subset or random initialization.
    
    for r in range(R):
        # Random initialization
        U[:, r] = np.random.randn(n * m)
        V[:, r] = np.random.randn(m * p)
        W[:, r] = np.random.randn(p * n)
    
    # Alternating minimization
    for it in range(1000):
        if time.time() - start_time > max_time:
            break
        
        # Solve for W (fix U, V)
        W_new = np.zeros((p * n, R))
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, R))
                for j in range(m):
                    for r in range(R):
                        M[j, r] = U[i * m + j, r] * V[j * p + k, r]
                try:
                    w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W_new[k * n + i, :] = w
                except:
                    pass
        W = W_new
        
        # Solve for U (fix V, W)
        U_new = np.zeros((n * m, R))
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, R))
                for k in range(p):
                    for r in range(R):
                        M[k, r] = V[j * p + k, r] * W[k * n + i, r]
                try:
                    u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U_new[i * m + j, :] = u
                except:
                    pass
        U = U_new
        
        # Solve for V (fix U, W)
        V_new = np.zeros((m * p, R))
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, R))
                for i in range(n):
                    for r in range(R):
                        M[i, r] = U[i * m + j, r] * W[k * n + i, r]
                try:
                    v, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    V_new[j * p + k, :] = v
                except:
                    pass
        V = V_new
        
        # Check
        if _check_decomposition_fast(U, V, W, n, m, p, tol=1e-8):
            return (U, V, W)
    
    return None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4x6 matrix multiplication.
    Returns {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Default: schoolbook
    rank = n * m * p
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
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try to find lower rank
    # Search from lower bound up to schoolbook
    lower_bound = max(n * m, m * p, p * n)
    
    for R in range(lower_bound, best_rank):
        if time.time() - start_time > TIME_LIMIT * 0.8:
            break
        
        # Try multiple random starts
        found = False
        for trial in range(30):
            if time.time() - start_time > TIME_LIMIT * 0.9:
                break
            
            result = _try_direct_linear_solve(n, m, p, R, max_time=TIME_LIMIT * 0.1)
            if result is not None:
                best_U, best_V, best_W = result
                best_rank = R
                found = True
                break
        
        if found:
            # Try even lower
            continue
    
    # Final verification
    if not _check_decomposition_fast(best_U, best_V, best_W, n, m, p, tol=1e-9):
        # Fall back to schoolbook
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
        best_U, best_V, best_W = U, V, W
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
