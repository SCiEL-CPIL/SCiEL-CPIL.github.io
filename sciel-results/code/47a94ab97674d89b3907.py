import time
import numpy as np
import random
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact using NumPy."""
    if not U or not V or not W:
        return False
        
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reshape to 3D
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    
    # Check full tensor:
    # Sum_c U[i,j,c] * V[j',k,c] * W[k',i',c] == delta(j,j') * delta(k,k') * delta(i,i')
    
    for i in range(n):
        for i_prime in range(n):
            for j in range(m):
                for j_prime in range(m):
                    for k in range(p):
                        for k_prime in range(p):
                            target = 1.0 if (i == i_prime and j == j_prime and k == k_prime) else 0.0
                            val = np.dot(U_3d[i, j, :], V_3d[j_prime, k, :] * W_3d[k_prime, i_prime, :])
                            if abs(val - target) > 1e-9:
                                return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
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
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(n, m, p, U, V, W, max_time):
    """
    Try to reduce the rank by finding linear dependencies.
    Returns reduced (U, V, W) or None if no reduction found.
    """
    start_time = time.time()
    rank = len(U[0])
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the tensor as a matrix where each column is one term
    # Term c contributes: U[:,c] ⊗ V[:,c] ⊗ W[:,c] flattened
    # The full tensor T[i,j,k,i',j',k'] = sum_c U[i*m+j,c] * V[j'*p+k,c] * W[k'*n+i',c]
    
    # We want to check if any term is a linear combination of others.
    # This is equivalent to checking if the matrix of term vectors has rank < number of terms.
    
    # Each term c is a vector in R^(n*m*p*n*m*p) but that's too large.
    # Instead, use the fact that we only need to check the 50 diagonal entries plus off-diagonal constraints.
    
    # Better approach: For each term c, check if it can be expressed as linear combo of others
    # by solving a linear system.
    
    # The constraint is: for all (i,i',j,j',k,k'):
    #   sum_{c' != c} alpha_{c'} * U[i*m+j,c'] * V[j'*p+k,c'] * W[k'*n+i',c'] = U[i*m+j,c] * V[j'*p+k,c] * W[k'*n+i',c]
    
    # This is a linear system in alpha_{c'}.
    # Number of equations: n*n*m*m*p*p = 2*2*5*5*5*5 = 2500
    # Number of unknowns: rank - 1
    
    # Use a subset of equations for speed. Use all equations where the target is non-zero.
    # Non-zero target when i=i', j=j', k=k': n*m*p = 50 equations.
    # Also add some off-diagonal equations to ensure correctness.
    
    for c in range(rank):
        if time.time() - start_time > max_time:
            break
        
        # Build linear system
        A_rows = []
        b_vals = []
        
        # Diagonal equations (i=i', j=j', k=k')
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = []
                    for c_prime in range(rank):
                        if c_prime == c:
                            continue
                        row.append(U[i*m+j][c_prime] * V[j*p+k][c_prime] * W[k*n+i][c_prime])
                    A_rows.append(row)
                    b_vals.append(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c])
        
        # Off-diagonal equations: sample some to keep system size manageable
        # Add equations where exactly one index pair differs
        num_offdiag = 0
        for i in range(n):
            for i_prime in range(n):
                if i == i_prime:
                    continue
                for j in range(m):
                    for j_prime in range(m):
                        for k in range(p):
                            for k_prime in range(p):
                                if num_offdiag >= 100:  # Limit off-diagonal equations
                                    break
                                row = []
                                for c_prime in range(rank):
                                    if c_prime == c:
                                        continue
                                    row.append(U[i*m+j][c_prime] * V[j_prime*p+k][c_prime] * W[k_prime*n+i_prime][c_prime])
                                A_rows.append(row)
                                b_vals.append(0.0)  # Target is 0 for off-diagonal
                                num_offdiag += 1
                            if num_offdiag >= 100:
                                break
                        if num_offdiag >= 100:
                            break
                    if num_offdiag >= 100:
                        break
                if num_offdiag >= 100:
                    break
            if num_offdiag >= 100:
                break
        
        A = np.array(A_rows, dtype=float)
        b = np.array(b_vals, dtype=float)
        
        try:
            alpha, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
        
        # Check if solution is exact
        if not np.allclose(A @ alpha, b, atol=1e-10):
            continue
        
        # Verify the full decomposition with this term removed and others adjusted
        # The new decomposition: sum_{c' != c} (1 + alpha_{c'}) * term_{c'}
        # Wait, that's not right. If term_c = sum_{c' != c} alpha_{c'} * term_{c'},
        # then sum_{all c} term_c = sum_{c' != c} term_{c'} + sum_{c' != c} alpha_{c'} * term_{c'}
        # = sum_{c' != c} (1 + alpha_{c'}) * term_{c'}
        # So we need to scale the remaining terms.
        
        # But scaling U, V, W by different factors for each term is not straightforward
        # because each term has its own U, V, W columns.
        
        # Actually, we can just remove term c and add alpha_{c'} to the coefficient of term c'.
        # But the decomposition is sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c], so we can't just add scalars.
        
        # The correct approach: if term_c = sum_{c' != c} alpha_{c'} * term_{c'},
        # then we can replace the sum by sum_{c' != c} (1 + alpha_{c'}) * term_{c'},
        # which means we scale U[:,c'], V[:,c'], W[:,c'] by (1 + alpha_{c'})^{1/3} for each c' != c.
        # But this only works if we can take cube roots, and the entries must remain integer/half-integer.
        
        # This is problematic. Let me think differently.
        
        # Actually, the tensor equation is:
        # T = sum_c u_c ⊗ v_c ⊗ w_c
        # If u_c ⊗ v_c ⊗ w_c = sum_{c' != c} alpha_{c'} (u_{c'} ⊗ v_{c'} ⊗ w_{c'}),
        # then T = sum_{c' != c} (1 + alpha_{c'}) (u_{c'} ⊗ v_{c'} ⊗ w_{c'})
        # So we can just remove term c and the remaining terms still sum to T.
        # Wait, that's only true if the equality holds for the tensor, not just for the sampled equations.
        
        # Let me verify with the full tensor check.
        U_new = [U[i][:c] + U[i][c+1:] for i in range(nm)]
        V_new = [V[i][:c] + V[i][c+1:] for i in range(mp)]
        W_new = [W[i][:c] + W[i][c+1:] for i in range(pn)]
        
        if _verify_decomposition(U_new, V_new, W_new, n, m, p):
            return U_new, V_new, W_new
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Schoolbook is a valid baseline
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p  # 50
    
    # Try to reduce rank
    U, V, W = best_result["U"], best_result["V"], best_result["W"]
    
    # Attempt rank reduction
    reduced = _try_reduce_rank(n, m, p, U, V, W, deadline - time.time())
    if reduced is not None:
        U, V, W = reduced
        best_rank = len(U[0])
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
