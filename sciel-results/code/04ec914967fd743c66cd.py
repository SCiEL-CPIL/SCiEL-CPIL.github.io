"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
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

def _verify(U, V, W, n, m, p):
    """Check if U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # For each (i, j, k), sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1) > 1e-9:
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, max_time=120.0):
    """Try to reduce the rank by finding linear dependencies."""
    start_time = time.time()
    deadline = start_time + max_time
    
    n_rows_U = n * m
    n_rows_V = m * p
    n_rows_W = p * n
    current_rank = len(U[0])
    
    # Convert to numpy for speed
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The multiplication tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1
    # A rank-r decomposition means: T[a,b,c] = sum_l U[a,l] V[b,l] W[c,l]
    # If we have rank r+1, we can try to find if one column is a linear combination of others
    # by checking if there exist vectors u, v, w such that the (r+1)th term can be absorbed
    
    # Strategy: Try to find a linear dependency among the rank-r terms
    # For each subset of r columns out of r+1, check if the remaining column can be
    # expressed as a linear combination
    
    # A more efficient approach: use the fact that if the tensor has rank r,
    # then for any (r+1) terms, there must be a linear dependency
    # We can try to find a non-trivial solution to:
    # sum_l alpha_l * (u_l ⊗ v_l ⊗ w_l) = 0
    
    # Let's try a randomized approach:
    # 1. Pick r+1 random columns
    # 2. Check if they are linearly dependent in the tensor product space
    # 3. If yes, we can eliminate one column
    
    # Actually, let's use a simpler approach:
    # The rank of the tensor is the minimum r such that T = sum_{l=1}^r u_l ⊗ v_l ⊗ w_l
    # We can try to find a lower rank by solving for linear dependencies
    
    # Let's try: for each column c, check if it can be written as a linear combination
    # of the other columns in the tensor product space
    
    # A column c contributes: U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # This is a vector in R^(n*m * m*p * p*n)
    # But that's too large. Instead, we can use the flattening trick.
    
    # For a fixed index in one mode, the flattening gives a matrix.
    # T_{a,b,c} = sum_l U[a,l] V[b,l] W[c,l]
    # Fix c: T_{a,b,c} = sum_l (U[a,l] W[c,l]) V[b,l]
    # So for fixed c, the matrix M_c[a,b] = T_{a,b,c} has rank at most 1
    # and M_c = (U[:,c] * W[c,:]) ⊗ V[:,c] (outer product with scaling)
    
    # Actually, M_c[a,b] = sum_l U[a,l] V[b,l] W[c,l]
    # This is a sum of rank-1 matrices.
    
    # Let's try a different approach: use the fact that if we have r terms,
    # we can try to find if one is redundant by checking if the tensor can be
    # decomposed with r-1 terms.
    
    # For small dimensions, we can try brute force with random projections.
    
    # Let's try: reduce by 1 each time, checking if the new rank works.
    # We'll use a randomized search:
    # 1. Start with the schoolbook decomposition
    # 2. Try to find a linear dependency among the terms
    # 3. If found, eliminate the redundant term and continue
    
    # A practical approach: use the fact that for matrix multiplication,
    # the tensor has a specific structure. The (i,j,k) entry is 1 if the
    # indices are consistent.
    
    # Let's try to find a lower rank by solving a system of equations.
    # Suppose we want rank r. We need to find U, V, W of rank r such that
    # sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all valid (i,j,k)
    
    # This is a system of n*m*p equations with 3*r*(n*m+m*p+p*n) unknowns.
    # For n=3, m=4, p=4: 48 equations, 3*r*(12+16+12) = 3*r*40 = 120r unknowns.
    # For r=48: 5760 unknowns, 48 equations. Very underdetermined.
    
    # The key is that the equations are nonlinear (products of three variables).
    # We can try to solve this using a randomized approach.
    
    # Let's try: fix U and V randomly, then solve for W.
    # For each (i,j,k): sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
    # This is linear in W! For each k*n+i, we have:
    # sum_c (U[i*m+j,c] V[j*p+k,c]) W[k*n+i,c] = 1
    # But this must hold for all j, so we have a system:
    # For fixed (i,k): for all j, sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
    # This is m equations in p*n*rank unknowns for W.
    
    # Let's try a simpler approach: use known results about matrix multiplication rank.
    # For 3x4 * 4x4, the rank is not known to be less than 48, but let's try.
    
    # Actually, let's just try to reduce the rank by finding linear dependencies
    # in the tensor product space using random projections.
    
    # For each pair of columns (c1, c2), check if u1⊗v1⊗w1 - u2⊗v2⊗w2 is
    # in the span of the other terms.
    
    # This is getting complex. Let's try a simpler heuristic:
    # 1. Try to eliminate one column at a time
    # 2. For each column c, check if the tensor without column c can be
    #    represented with the remaining columns (with adjusted weights)
    
    # A practical approach: use the fact that the tensor is sparse.
    # Many entries are 0 (only the "diagonal" entries are 1).
    
    # Let's try: for each column c, try to find coefficients alpha_d (d != c)
    # such that U[:,c]⊗V[:,c]⊗W[:,c] = sum_{d!=c} alpha_d * U[:,d]⊗V[:,d]⊗W[:,d]
    
    # This is a system of n*m*p equations in (rank-1) unknowns.
    # For rank=48, that's 47 unknowns and 48 equations. Might be solvable!
    
    best_rank = current_rank
    best_U, best_V, best_W = U, V, W
    
    while time.time() < deadline - 5.0:
        # Try to eliminate a random column
        c_elim = random.randint(0, current_rank - 1)
        
        # Build the system: for all (i,j,k), 
        # sum_{d != c_elim} alpha_d * U[i*m+j,d] * V[j*p+k,d] * W[k*n+i,d] 
        # = U[i*m+j,c_elim] * V[j*p+k,c_elim] * W[k*n+i,c_elim]
        
        # This is linear in alpha_d!
        # Number of equations: n*m*p = 48
        # Number of unknowns: current_rank - 1
        
        if current_rank - 1 >= n * m * p:
            # Underdetermined or square system, try to solve
            # Build the matrix A and vector b
            num_eqs = n * m * p
            num_unk = current_rank - 1
            
            A = np.zeros((num_eqs, num_unk))
            b = np.zeros(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        target = U_np[i*m+j, c_elim] * V_np[j*p+k, c_elim] * W_np[k*n+i, c_elim]
                        b[eq_idx] = target
                        
                        unk_idx = 0
                        for d in range(current_rank):
                            if d != c_elim:
                                A[eq_idx, unk_idx] = U_np[i*m+j, d] * V_np[j*p+k, d] * W_np[k*n+i, d]
                                unk_idx += 1
                        
                        eq_idx += 1
            
            # Solve the linear system
            try:
                # Use least squares if overdetermined, or direct solve if square
                if num_eqs >= num_unk:
                    alpha, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    # Check if the solution is good
                    if np.linalg.norm(A @ alpha - b) < 1e-8:
                        # Great! We can eliminate column c_elim
                        # Update the remaining columns
                        new_U = np.zeros((n_rows_U, current_rank - 1))
                        new_V = np.zeros((n_rows_V, current_rank - 1))
                        new_W = np.zeros((n_rows_W, current_rank - 1))
                        
                        unk_idx = 0
                        for d in range(current_rank):
                            if d != c_elim:
                                new_U[:, unk_idx] = U_np[:, d] * alpha[unk_idx]
                                new_V[:, unk_idx] = V_np[:, d]
                                new_W[:, unk_idx] = W_np[:, d]
                                unk_idx += 1
                        
                        # Verify the new decomposition
                        # Check that sum_c new_U[i*m+j,c] * new_V[j*p+k,c] * new_W[k*n+i,c] = 1
                        valid = True
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    total = 0
                                    for c in range(current_rank - 1):
                                        total += new_U[i*m+j, c] * new_V[j*p+k, c] * new_W[k*n+i, c]
                                    if abs(total - 1) > 1e-8:
                                        valid = False
                                        break
                            if not valid:
                                break
                        if not valid:
                            break
                        
                        if valid:
                            U_np = new_U
                            V_np = new_V
                            W_np = new_W
                            current_rank -= 1
                            best_rank = current_rank
                            # Convert back to lists for best result
                            best_U = U_np.tolist()
                            best_V = V_np.tolist()
                            best_W = W_np.tolist()
                            continue
                else:
                    # Underdetermined, use lstsq to find minimum norm solution
                    alpha, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    if np.linalg.norm(A @ alpha - b) < 1e-8:
                        new_U = np.zeros((n_rows_U, current_rank - 1))
                        new_V = np.zeros((n_rows_V, current_rank - 1))
                        new_W = np.zeros((n_rows_W, current_rank - 1))
                        
                        unk_idx = 0
                        for d in range(current_rank):
                            if d != c_elim:
                                new_U[:, unk_idx] = U_np[:, d] * alpha[unk_idx]
                                new_V[:, unk_idx] = V_np[:, d]
                                new_W[:, unk_idx] = W_np[:, d]
                                unk_idx += 1
                        
                        valid = True
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    total = 0
                                    for c in range(current_rank - 1):
                                        total += new_U[i*m+j, c] * new_V[j*p+k, c] * new_W[k*n+i, c]
                                    if abs(total - 1) > 1e-8:
                                        valid = False
                                        break
                            if not valid:
                                break
                        if not valid:
                            break
                        
                        if valid:
                            U_np = new_U
                            V_np = new_V
                            W_np = new_W
                            current_rank -= 1
                            best_rank = current_rank
                            best_U = U_np.tolist()
                            best_V = V_np.tolist()
                            best_W = W_np.tolist()
                            continue
            except np.linalg.LinAlgError:
                pass
        
        # If we couldn't eliminate, try a different column
        # Also try random perturbations to escape local minima
    
    # Convert back to lists with exact fractions if possible
    # For now, return the best found
    return best_U, best_V, best_W, best_rank

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank
    U_red, V_red, W_red, rank_red = _try_reduce_rank(U, V, W, n, m, p, max_time=110.0)
    
    if rank_red < best_rank:
        best_rank = rank_red
        best_U, best_V, best_W = U_red, V_red, W_red
    
    # Verify the best result
    if not _verify(best_U, best_V, best_W, n, m, p):
        # Fall back to schoolbook
        best_U, best_V, best_W = U, V, W
        best_rank = n * m * p
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
