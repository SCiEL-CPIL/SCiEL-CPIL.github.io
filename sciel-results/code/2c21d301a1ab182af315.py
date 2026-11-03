"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 63
    
    def schoolbook_decomposition():
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
    
    def verify_decomp(U, V, W, tol=1e-8):
        """Verify that the decomposition is correct."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        # Build the tensor T from the decomposition
        # T[i*m+j, j'*p+k, k'*n+i'] should match the multiplication tensor
        # Actually, let's check: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
        # should equal 1 if (i,j) matches (i',j') and k=k', else 0
        
        # The multiplication tensor: T[(i,j),(j2,k2),(k3,i3)] = 1 if j==j2 and i==i3 and k==k2
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # For each (i,j,k), check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # And for other combinations, it should be 0
        # Actually the tensor is: T[a,b,c] where a=(i,j), b=(j2,k2), c=(k3,i3)
        # T[a,b,c] = 1 if j==j2, i==i3, k==k2, else 0
        
        # Let's just check the 63 "on" entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U_np[a], V_np[b]) @ W_np[c]  # This is wrong, let me fix
                    # Actually: sum_r U[a,r] * V[b,r] * W[c,r]
                    val = np.sum(U_np[a] * V_np[b] * W_np[c])
                    if abs(val - 1) > tol:
                        return False
        return True
    
    # Try to find a lower rank decomposition using structured approach
    # For 3x3 * 3x7, try rank 49 (Strassen per column: 7 cols * 7 rank each)
    # But first let's try a simple search
    
    # Build the target tensor
    # T has shape (n*m, m*p, p*n) = (9, 21, 21)
    # T[i*m+j, j*p+k, k*n+i] = 1, else 0
    
    # Try to find decomposition of rank r
    # Use random initialization and alternating least squares
    
    def try_rank(r, max_iter=100, n_restarts=5):
        """Try to find a rank-r decomposition."""
        if r >= rank_schoolbook:
            return schoolbook_decomposition()
        
        np.random.seed(42)
        n, m, p = DIMENSIONS
        
        # Target tensor entries
        target = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    target[(a, b, c)] = 1.0
        
        best_rank = None
        best_decomp = None
        
        for restart in range(n_restarts):
            # Initialize
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # Alternating least squares
            for it in range(max_iter):
                # Update U: for each a, solve sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
                # This is a linear system for U[a,:] given V and W
                for a in range(n * m):
                    # Build system: for each (b,c) where T[a,b,c] != 0
                    # We have: sum_r U[a,r] * (V[b,r] * W[c,r]) = T[a,b,c]
                    # Let M[a,b,c,r] = V[b,r]*W[c,r]
                    # Then sum_r U[a,r] * M[a,b,c,r] = T[a,b,c]
                    
                    # Collect equations
                    eqs = []
                    rhs = []
                    for (ba, bc, bc_val) in target.items():
                        if ba == a:
                            b, c = bc[0], bc[1]
                            row = V[b] * W[c]  # shape (r,)
                            eqs.append(row)
                            rhs.append(bc_val)
                    
                    if not eqs:
                        continue
                    
                    eqs_arr = np.array(eqs)  # shape (num_eqs, r)
                    rhs_arr = np.array(rhs)
                    
                    # Solve least squares
                    if len(eqs_arr) >= r:
                        U[a], residuals, rank_ls, sv = np.linalg.lstsq(eqs_arr, rhs_arr, rcond=None)
                    
                # Update V
                for b in range(m * p):
                    eqs = []
                    rhs = []
                    for (ba, bc, bc_val) in target.items():
                        a, c = ba, bc[1]
                        if bc[0] == b:
                            row = U[a] * W[c]
                            eqs.append(row)
                            rhs.append(bc_val)
                    
                    if not eqs:
                        continue
                    
                    eqs_arr = np.array(eqs)
                    rhs_arr = np.array(rhs)
                    
                    if len(eqs_arr) >= r:
                        V[b], residuals, rank_ls, sv = np.linalg.lstsq(eqs_arr, rhs_arr, rcond=None)
                
                # Update W
                for c in range(p * n):
                    eqs = []
                    rhs = []
                    for (ba, bc, bc_val) in target.items():
                        a, b = ba, bc[0]
                        if bc[1] == c:
                            row = U[a] * V[b]
                            eqs.append(row)
                            rhs.append(bc_val)
                    
                    if not eqs:
                        continue
                    
                    eqs_arr = np.array(eqs)
                    rhs_arr = np.array(rhs)
                    
                    if len(eqs_arr) >= r:
                        W[c], residuals, rank_ls, sv = np.linalg.lstsq(eqs_arr, rhs_arr, rcond=None)
                
                # Check convergence
                if it % 10 == 0 or it == max_iter - 1:
                    if verify_decomp(U.tolist(), V.tolist(), W.tolist(), tol=1e-4):
                        # Found a valid decomposition
                        if best_decomp is None or r < best_rank:
                            best_rank = r
                            best_decomp = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        return best_decomp
            
            if time.time() > deadline - 10:
                break
        
        return best_decomp
    
    # Try ranks from 49 down to 40
    for r in range(49, 39, -1):
        if time.time() > deadline - 20:
            break
        decomp = try_rank(r, max_iter=50, n_restarts=3)
        if decomp is not None:
            # Verify more carefully
            if verify_decomp(decomp["U"], decomp["V"], decomp["W"], tol=1e-6):
                return decomp
    
    # Fallback to schoolbook
    return schoolbook_decomposition()
# EVOLVE-BLOCK-END
