"""Find a rank-minimal exact tensor decomposition for 2x3x3 matrix multiplication."""
DIMENSIONS = [2, 3, 3]

import time
import sys
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def _to_frac_list(mat):
    """Convert a numpy array of floats/ints to a list of lists of ints or halves."""
    result = []
    for row in mat:
        r = []
        for x in row:
            # Check if close to integer
            xi = int(round(x))
            if abs(x - xi) < 1e-6:
                r.append(xi)
            else:
                # Check if close to half-integer (x.5)
                xh = int(round(x * 2)) / 2.0
                if abs(x - xh) < 1e-6:
                    # Return as float 0.5 or int if .0 (but .0 caught above)
                    r.append(xh)
                else:
                    # If not integer or half-integer, round to nearest half?
                    # The task allows integer/half-integer entries.
                    # If the solver found something else, it might be invalid or need rounding.
                    # For safety, if it's not close to .0 or .5, we might have to reject or round.
                    # Let's round to nearest half.
                    r.append(xh)
        result.append(r)
    return result

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin for 120s limit
    
    # 1. Schoolbook (rank 18)
    rank_sb = n * m * p
    U = np.zeros((n * m, rank_sb))
    V = np.zeros((m * p, rank_sb))
    W = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    best_U, best_V, best_W = U, V, W
    best_rank = rank_sb
    
    # 2. Try to find rank 17 via random linear combination
    # The tensor T_{i,j,k} = sum_c U_{i,m,j,c} V_{j,p,k,c} W_{k,n,i,c}
    # We fix W to be 0/1 (binary) and solve for U, V?
    # Actually, it's easier to fix U and V and W as binary and see if we can satisfy the equations.
    # But there are many constraints.
    
    # Alternative: Random Search on the secant variety.
    # If rank 17 exists, it is a limit of rank 18 tensors.
    # We can try to find a dependency in the rank 18 solution.
    
    # Let's try a structured search for rank 17.
    # If we set one column of U, V, W to be a linear combination of others, we might reduce rank.
    
    # More robust: Use scipy's least squares to find a low-rank approximation.
    # But we need EXACT.
    
    # Let's try to find a rank 17 solution by assuming a specific structure.
    # For 2x3x3, the Strassen-like decomposition might be:
    # We need 17 equations to be satisfied.
    
    # Let's use a random restart search for rank 17.
    # We will fix W to be 0/1 (since it often is in these decompositions) and solve for U and V?
    # No, U and V are also 0/1 or simple rationals.
    
    # Strategy:
    # 1. Start with random U, V, W with entries in {0, 1, -1}.
    # 2. Compute the resulting tensor.
    # 3. Check if it matches the matrix multiplication tensor.
    # This is unlikely to hit exactly.
    
    # Better Strategy:
    # The matrix multiplication tensor has 18 non-zero entries (all 1s).
    # We want sum_{c=1}^{17} u_c(i) v_c(j) w_c(k) = delta_{i,j,k} (where delta is 1 if indices valid, else 0? No, it's 1 for all i,j,k in 2x3x3).
    # Wait, the tensor for matrix multiplication C = AB where A is 2x3, B is 3x3, C is 2x3.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # So T_{i, j, k} = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    # There are 2*3*3 = 18 equations.
    # We have 17 * (3 + 9 + 6) = 17 * 18 = 306 variables.
    # This is an underdetermined system.
    
    # We can set up the linear equations for rank 17.
    # Let U be 6x17, V be 9x17, W be 6x17.
    # For each (i,j,k), sum_c U_{i,m,j,c} V_{j,p,k,c} W_{k,n,i,c} = 1.
    
    # This is a system of quadratic equations in the variables.
    # We can linearize by fixing W.
    # If we fix W to be 0/1, then for each c, we are adding a term.
    
    # Let's try a specific heuristic:
    # Pick random 0/1 vectors for U and V.
    # Solve for W?
    # The equations are linear in W if U and V are fixed.
    # Let A_c be the 18x18 matrix formed by outer product of u_c and v_c?
    # No. The tensor is T = sum_c u_c \otimes v_c \otimes w_c.
    # Flattened: T_flat = sum_c (u_c \otimes v_c) \otimes w_c.
    # This is not linear in w_c if we sum over c.
    # Actually, T_{ijk} = sum_c U_{i,m,j,c} V_{j,p,k,c} W_{k,n,i,c}.
    # For a fixed c, the contribution to T_{ijk} is U_{i,m,j,c} V_{j,p,k,c} W_{k,n,i,c}.
    # This is trilinear.
    
    # However, if we fix U and V, the equation for T_{ijk} is linear in the components of W?
    # No, W_{k,n,i,c} is a specific entry.
    # Let's flatten W. W is 6x17.
    # For each c, we have a vector w_c of length 6.
    # The term is sum_c U_{i,m,j,c} V_{j,p,k,c} W_{k,n,i,c}.
    # Let's group by c.
    # T_{ijk} = sum_c M_{ijk,c} W_{k,n,i,c} where M_{ijk,c} = U_{i,m,j,c} V_{j,p,k,c}.
    # This is linear in the entries of W if U and V are fixed.
    # But W_{k,n,i,c} is the entry of W at row (k*n+i) and column c.
    # So we have 18 equations (for i,j,k) and 17*6 = 102 unknowns (entries of W).
    # We can solve for W given U and V.
    # But we also need W to be 0/1 or simple.
    # And we need the solution to be consistent.
    
    # Let's try random U and V with 0/1 entries, solve for W, and check if W is "nice".
    
    np.random.seed(42)
    
    attempts = 0
    while time.time() < deadline and attempts < 5000:
        attempts += 1
        # Random U and V with entries in {0, 1}
        U_rand = np.random.randint(0, 2, size=(n*m, 17))
        V_rand = np.random.randint(0, 2, size=(m*p, 17))
        
        # Construct the system A W_flat = b
        # b is 18x1 (all 1s)
        # A is 18x102
        # A_{(i,j,k), (row_w, c)} = U_{i,m,j,c} * V_{j,p,k,c} if row_w == k*n+i, else 0.
        
        A = np.zeros((18, 17*6))
        b = np.ones(18)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(17):
                        row_w = k * n + i
                        col_w = c
                        A[idx, row_w * 17 + col_w] = U_rand[i*m+j, c] * V_rand[j*p+k, c]
                    idx += 1
        
        # Solve A W_flat = b
        # Least squares solution
        try:
            W_flat, residuals, rank_a, s = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
            
        W_sol = W_flat.reshape(17, 6).T # W is 6x17
        
        # Check if solution is exact
        # Reconstruct tensor
        T_calc = np.zeros((n, m, p))
        for c in range(17):
            T_calc += U_rand[:, c][:, None] * V_rand[:, c][None, :] * W_sol[:, c][None, :, None] # Wait, dimensions.
            # U_rand[i*m+j, c] -> shape (n*m)
            # V_rand[j*p+k, c] -> shape (m*p)
            # W_sol[k*n+i, c] -> shape (p*n)
            
            # We need to check T_calc[i,j,k] == 1
            # Let's just check the residual of the linear system
            res = np.linalg.norm(A @ W_flat - b)
            if res < 1e-6:
                # Check if W_sol entries are close to integers or half-integers
                ok = True
                for row in W_sol:
                    for x in row:
                        xi = int(round(x))
                        xh = int(round(x * 2)) / 2.0
                        if not (abs(x - xi) < 1e-6 or abs(x - xh) < 1e-6):
                            ok = False
                            break
                    if not ok: break
                if ok:
                    # Also check U and V are 0/1 (they are)
                    # Check if this is a valid decomposition (non-degenerate)
                    # Return it
                    best_U, best_V, best_W = U_rand, V_rand, W_sol
                    best_rank = 17
                    break
    
    if best_rank > 17:
        # If rank 17 not found, try rank 16?
        # Unlikely to find quickly.
        pass
        
    # Convert to lists
    U_out = _to_frac_list(best_U)
    V_out = _to_frac_list(best_V)
    W_out = _to_frac_list(best_W)
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
