"""Matrix multiplication tensor decomposition for 4x5 * 5x5."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    n, m, p = DIMENSIONS

    # Schoolbook decomposition: rank = n*m*p = 100 (guaranteed valid)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Try to find lower rank decompositions using ALS with rational rounding
    # Known lower bounds: for 4x5*5x5, rank is at least 4*5 = 20 (trivial)
    # Better lower bound: max(n*p, m*p, n*m) = max(20, 25, 20) = 25
    # Try ranks from a reasonable lower bound up to schoolbook

    # Use the matrix multiplication tensor T[i,j,k] = delta[i,k] (for C[i,k] = sum_j A[i,j] B[j,k])
    # Tensor shape: (n, m, p) where T[i,j,k] = 1 if i==k else 0... 
    # Actually for matrix mult: T[i,j,k] where output is C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor has T[i,j,k] = delta_{i,k} ... no.
    # Standard form: the multiplication tensor has entries T[i,j,k] for the term A[i,j]*B[j,k] contributing to C[i,k]
    # So T is indexed by (i, j, k) with value 1 if the (i,j) entry of A and (j,k) entry of B multiply to give (i,k) entry of C
    # This means T[i,j,k] = 1 for all i,j,k (each triple (i,j,k) corresponds to one product term)
    # Wait, no. The tensor for matrix multiplication M_{n,m,p} has:
    # T[i, j, k] = 1 if and only if... actually it's always 1 for valid indices.
    # The tensor is: for each (i,j) in A and (j,k) in B, the product goes to (i,k) in C.
    # So the tensor entries are T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # That's just an all-ones tensor, which doesn't seem right.
    
    # Let me reconsider. The matrix multiplication tensor is defined as:
    # T in R^{n x m} x R^{m x p} x R^{n x p}
    # T(A, B, C) = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C_{i,k}
    # where T_{i,j,k} = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (i,k) entry of C
    # So T_{i,j,k} = 1 for all valid i,j,k (since A_{i,j} * B_{j,k} does contribute to C_{i,k})
    # 
    # Actually no. The standard definition: the multiplication tensor has
    # T_{i,j,k} = 1 if i is the row of C, j is the column of A / row of B, k is the column of C
    # and the contribution is A_{i,j} * B_{j,k} -> C_{i,k}
    # So T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Hmm, but that would make it an all-ones tensor which has rank 1, which is wrong.
    # 
    # Let me look at this more carefully. The tensor decomposition of matrix multiplication:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # 
    # In tensor form, we want: C_{i,k} = sum_r U_{i,j,r} V_{j,k,r} W_{i,k,r} ... no
    # 
    # The factorization is: C_{i,k} = sum_r (sum_j U_{i,j,r} A_{i,j}) (sum_k' V_{j,k',r} B_{j,k'}) ... 
    # 
    # Standard form: We want to write the multiplication as:
    # C_{i,k} = sum_{r=1}^{R} U_{i,k,r} * (sum_j U_{i,j,r} A_{i,j}) * (sum_k V_{j,k,r} B_{j,k})
    # 
    # No, the correct form is:
    # C_{i,k} = sum_r W_{i,k,r} * (sum_j U_{i,j,r} A_{i,j}) * (sum_k' V_{j,k',r} B_{j,k'})
    # 
    # Wait, I'm confusing myself. Let me use the standard formulation.
    # 
    # The matrix multiplication tensor T has entries T_{i,j,k} where:
    # - i indexes rows of A and C (i = 0..n-1)
    # - j indexes columns of A and rows of B (j = 0..m-1)  
    # - k indexes columns of B and C (k = 0..p-1)
    # 
    # T_{i,j,k} = 1 if the (i,j) entry of A and (j,k) entry of B multiply to contribute to (i,k) entry of C
    # This is always true, so T_{i,j,k} = 1 for all i,j,k.
    # 
    # But that can't be right because the rank of an all-ones tensor is 1.
    # 
    # I think I'm misunderstanding the tensor structure. Let me reconsider.
    #
    # The correct formulation: The tensor T is in the space of trilinear forms on 
    # (R^{n x m})^* x (R^{m x p})^* x (R^{n x p})^*
    # 
    # T(A, B, C) = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k} * T_{i,j,k}
    # 
    # For matrix multiplication, we want T such that:
    # sum_j A_{i,j} B_{j,k} = C_{i,k}
    # 
    # So T_{i,j,k} should be 1 for all i,j,k? No...
    # 
    # Actually, I think the issue is that the tensor is not all ones. The tensor
    # T_{i,j,k} = 1 only when the indices are "compatible" in the right way.
    # 
    # Let me just use the standard result: the multiplication tensor for M_{n,m,p}
    # has T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # 
    # But then the rank would be 1, which contradicts known results.
    #
    # I think I need to be more careful. Let me define the tensor properly.
    #
    # The bilinear map (A, B) -> C = AB can be written as a sum of rank-1 tensors:
    # C_{i,k} = sum_r alpha_{i,k,r} * (sum_j U_{i,j,r} A_{i,j}) * (sum_j V_{j,k,r} B_{j,k})
    #
    # No, that's not right either. The correct form is:
    # C_{i,k} = sum_r W_{i,k,r} * (sum_j U_{i,j,r} A_{i,j}) * (sum_j V_{j,k,r} B_{j,k})
    #
    # Hmm, but that has three sums over j which doesn't make sense.
    #
    # Let me look at this from the factorization perspective:
    # We want: C_{i,k} = sum_{r=1}^R (sum_j U_{i,j,r} A_{i,j}) * (sum_j V_{j,k,r} B_{j,k}) * W_{i,k,r}
    #
    # No, that's still not right. The standard factorization is:
    # C_{i,k} = sum_{r=1}^R U_{i,k,r} * (sum_j A_{i,j} U_{i,j,r}) * (sum_j B_{j,k} V_{j,k,r})
    #
    # I think I'm overcomplicating this. Let me just use the known structure.
    #
    # The multiplication tensor T for M_{n,m,p} is a tensor in R^{n*m} x R^{m*p} x R^{n*p}
    # where T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = i*p+k for some i,j,k
    # (using the flattening conventions from the task)
    #
    # Wait, the task says: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # And the output factor follows W[k*n+i, r] convention.
    #
    # So the tensor T is in R^{n*m} x R^{m*p} x R^{p*n}
    # T[a, b, c] where:
    # - a indexes (i,j) pairs: a = i*m + j (i in [0,n), j in [0,m))
    # - b indexes (j,k) pairs: b = j*p + k (j in [0,m), k in [0,p))
    # - c indexes (k,i) pairs: c = k*n + i (k in [0,p), i in [0,n))
    #
    # T[a, b, c] = 1 if there exist i,j,k such that:
    #   a = i*m + j, b = j*p + k, c = k*n + i
    #
    # So for a given (i,j,k): a = i*m+j, b = j*p+k, c = k*n+i
    # T[i*m+j, j*p+k, k*n+i] = 1
    #
    # This is the correct tensor! It's not all ones; it's a specific pattern.
    #
    # Now I can try to find a low-rank decomposition of this tensor.

    def build_tensor(n, m, p):
        """Build the matrix multiplication tensor as a 3D array."""
        T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    def verify_decomposition(U, V, W, T, tol=1e-6):
        """Verify that the factorization U, V, W reconstructs T."""
        # Reconstruct T from factors
        # T_reconstructed[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Efficient reconstruction using einsum
        T_rec = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        
        error = np.max(np.abs(T_rec - T))
        return error < tol, error

    def round_to_half_integer(arr):
        """Round array entries to nearest half-integer (multiple of 0.5)."""
        return np.round(arr * 2) / 2

    def try_rank(R, T, max_iter=500, n_restarts=3, tol=1e-8):
        """Try to find a rank-R decomposition using ALS."""
        n, m, p = DIMENSIONS
        nm, mp, pn = n * m, m * p, p * n
        
        best_U = None
        best_V = None
        best_W = None
        best_err = np.inf
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
                
            # Initialize with random factors
            rng = np.random.RandomState(42 + restart)
            U = rng.randn(nm, R)
            V = rng.randn(mp, R)
            W = rng.randn(pn, R)
            
            # Scale to have reasonable magnitude
            U /= np.sqrt(R)
            V /= np.sqrt(R)
            W /= np.sqrt(R)
            
            prev_err = np.inf
            converged = False
            
            for iteration in range(max_iter):
                if time.time() > deadline - 5:
                    break
                    
                # Update U: fix V, W
                # For each a: U[a, :] = argmin sum_{b,c} T[a,b,c]^2 - 2*U[a,:] * sum_{b,c} T[a,b,c]*V[b,:]*W[c,:]
                # This is a linear system for each a
                # sum_r U[a,r] * (sum_{b,c} T[a,b,c] V[b,r] W[c,r]) = sum_{b,c} T[a,b,c] * ...
                
                # More efficiently:
                # Let S_U[a, r] = sum_{b,c} T[a,b,c] * V[b,r] * W[c,r]
                # Then U[a, :] = S_U[a, :] / (V^T W element-wise sum)... 
                
                # Actually, the ALS update for U is:
                # U[:, r] = (T * (V[:, r] * W[:, r])) / (V[:, r]^T W[:, r])
                # where T * vec means tensor-matrix product
                
                # Let's do it more carefully.
                # The objective is: min_U sum_{a,b,c} (T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r])^2
                # Taking derivative w.r.t. U[a, r]:
                # -2 sum_{b,c} T[a,b,c] V[b,r] W[c,r] + 2 U[a,r] sum_{b,c} V[b,r]^2 W[c,r]^2 = 0
                # So U[a,r] = sum_{b,c} T[a,b,c] V[b,r] W[c,r] / sum_{b,c} V[b,r]^2 W[c,r]^2
                
                # Compute for all a, r efficiently
                # V[b,r]^2 W[c,r]^2 summed over b,c: this is (sum_b V[b,r]^2) * (sum_c W[c,r]^2)
                denom_U = np.sum(V**2, axis=0) * np.sum(W**2, axis=0)  # shape (R,)
                
                # Numerator: for each a, r: sum_{b,c} T[a,b,c] V[b,r] W[c,r]
                # = sum_c W[c,r] * (sum_b T[a,b,c] V[b,r])
                # Let's compute M[a,c,r] = sum_b T[a,b,c] V[b,r]
                M = np.einsum('abc,br->acr', T, V)  # shape (nm, pn, R)
                numer_U = np.einsum('acr,cr->ar', M, W)  # shape (nm, R)
                
                U_new = numer_U / denom_U[np.newaxis, :]
                
                # Update V: fix U, W
                denom_V = np.sum(U**2, axis=0) * np.sum(W**2, axis=0)
                M2 = np.einsum('abc,ar->bcr', T, U)  # shape (mp, pn, R)
                numer_V = np.einsum('bcr,cr->br', M2, W)  # shape (mp, R)
                V_new = numer_V / denom_V[np.newaxis, :]
                
                # Update W: fix U, V
                denom_W = np.sum(U**2, axis=0) * np.sum(V**2, axis=0)
                M3 = np.einsum('abc,br->acr', T, V)  # shape (nm, pn, R)
                numer_W = np.einsum('acr,ar->cr', M3, U)  # shape (pn, R)
                W_new = numer_W / denom_W[np.newaxis, :]
                
                # Compute error
                T_rec = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
                err = np.max(np.abs(T_rec - T))
                
                U, V, W = U_new, V_new, W_new
                
                if err < prev_err * (1 - 1e-4) and iteration > 10:
                    # Converging slowly
                    pass
                if abs(prev_err - err) < 1e-10:
                    converged = True
                    break
                prev_err = err
            
            # Round to half-integers and verify
            U_rounded = round_to_half_integer(U)
            V_rounded = round_to_half_integer(V)
            W_rounded = round_to_half_integer(W)
            
            # Check if rounded version is still valid
            T_rec_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
            err_rounded = np.max(np.abs(T_rec_rounded - T))
            
            if err_rounded < 1e-6:
                # Convert to lists of Fractions for exactness
                U_frac = [[Fraction(int(U_rounded[i, r] * 2)) / 2 for r in range(R)] for i in range(nm)]
                V_frac = [[Fraction(int(V_rounded[i, r] * 2)) / 2 for r in range(R)] for i in range(mp)]
                W_frac = [[Fraction(int(W_rounded[i, r] * 2)) / 2 for r in range(R)] for i in range(pn)]
                
                if err < best_err:
                    best_U, best_V, best_W = U_frac, V_frac, W_frac
                    best_err = err
        
        if best_err < 1e-6:
            return best_U, best_V, best_W, best_err
        return None, None, None, np.inf

    # Try ranks from lower bound up to schoolbook
    # Lower bound: max(n*p, m*p, n*m) = max(20, 25, 20) = 25
    # But we can try a few values
    T = build_tensor(n, m, p)
    
    # Try a few candidate ranks
    candidate_ranks = [25, 26, 28, 30, 32, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    
    for R in candidate_ranks:
        if time.time() > deadline - 30:
            break
        if R >= best_rank:
            break
            
        print(f"Trying rank {R}...")
        U_try, V_try, W_try, err = try_rank(R, T, max_iter=300, n_restarts=2)
        
        if U_try is not None:
            print(f"  Found valid rank-{R} decomposition with error {err:.2e}")
            best_rank = R
            best_result = {"U": U_try, "V": V_try, "W": W_try}
            break  # Found a valid one, this is our best
    
    # If no better decomposition found, return schoolbook
    if best_rank == rank_school:
        return best_result
    
    # Convert Fractions to simple numbers (int or half-int)
    def frac_to_num(f):
        if f.denominator == 1:
            return f.numerator
        elif f.denominator == 2:
            return f.numerator / 2.0
        else:
            return float(f)
    
    U_final = [[frac_to_num(f) for f in row] for row in best_result["U"]]
    V_final = [[frac_to_num(f) for f in row] for row in best_result["V"]]
    W_final = [[frac_to_num(f) for f in row] for row in best_result["W"]]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
