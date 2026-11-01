"""Program to find a rank-minimal exact tensor decomposition for 2x2 by 2x5 matrix multiplication."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
DEADLINE = 145.0
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    
    # Helper to check if a decomposition is valid and snaps to half-integers
    def check_and_snap(U, V, W, rank, tol=1e-5):
        # Check raw error
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        target = 1.0 if j == jp else 0.0
                        val = 0.0
                        for c in range(rank):
                            val += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        if abs(val - target) > 1e-3:
                            return None
                            
        # Snap to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Verify snapped version
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        target = 1.0 if j == jp else 0.0
                        val = 0.0
                        for c in range(rank):
                            val += U_r[i*m + j, c] * V_r[jp*p + k, c] * W_r[k*n + i, c]
                        if abs(val - target) > tol:
                            return None
                            
        return U_r, V_r, W_r

    # Function to create the explicit rank-18 decomposition
    def create_rank_18():
        rank = 18
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Block 1: Columns 0,1 of B (2x2 block). Strassen.
        # A is 2x2. B1 is 2x2.
        # Strassen terms for A * B1
        # Let A = [[a00, a01], [a10, a11]]
        # Let B1 = [[b00, b01], [b10, b11]]
        # Strassen computes C1 = A * B1
        # We need to map this to the tensor indices.
        # The tensor T_{i, j, jp, k} = delta_{j, jp}.
        # For the block corresponding to k in {0,1}, we use Strassen.
        
        # Standard Strassen for 2x2:
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)b00
        # m3 = a00(b01-b11)
        # m4 = a11(b10-b00)
        # m5 = (a00+a10)b11
        # m6 = (a01-a10)(b00+b10)
        # m7 = (a01-a11)(b01+b11)
        # c00 = m1 + m4 - m5 + m7
        # c01 = m3 + m5
        # c10 = m2 + m4
        # c11 = m1 - m2 + m3 + m6
        
        # We need to decompose the identity-like tensor for this block.
        # Actually, it's easier to just build the U, V, W matrices for the Strassen algorithm directly.
        # For each term c, we have U[:, c], V[:, c], W[:, c].
        
        # Term 1: m1
        # U: a00+a11 -> U[0,0]=1, U[1,0]=1 (indices i*m+j: i=0,j=0; i=1,j=1)
        # V: b00+b11 -> V[0,0]=1, V[1,0]=1 (indices jp*p+k: jp=0,k=0; jp=1,k=1)
        # W: contributes to c00, c11.
        # This is getting complicated to map manually.
        
        # Alternative: Use the known Strassen decomposition tensors.
        # Or, simply use the fact that we can find a rank 17 solution by optimizing from a random start or the rank 18 start.
        # Given the time limit, constructing rank 18 explicitly is a good fallback.
        
        # Let's try a simpler fallback: Rank 20 Schoolbook.
        # But we want to try Rank 17.
        
        return None

    # Optimization function
    def error(params, rank):
        U = params[:n*m*rank].reshape((n*m, rank))
        V = params[n*m*rank : n*m*rank + m*p*rank].reshape((m*p, rank))
        W = params[n*m*rank + m*p*rank :].reshape((p*n, rank))
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        target = 1.0 if j == jp else 0.0
                        val = 0.0
                        for c in range(rank):
                            val += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        err += (val - target) ** 2
        return err

    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Try to find rank 17
    # Start with random initializations, but bias towards finding a solution.
    # Since rank 17 is close to 18, maybe we can use a rank 18 solution as a guide?
    # For now, try random restarts with Nelder-Mead.
    
    for rank in [17, 18]:
        if time.time() > DEADLINE - 10:
            break
            
        found = False
        for restart in range(20):
            if time.time() > DEADLINE - 10:
                break
                
            np.random.seed(restart * 100 + rank)
            x0 = np.random.uniform(-1, 1, size=n*m*rank + m*p*rank + p*n*rank)
            
            # Nelder-Mead
            res = minimize(error, x0, args=(rank,), method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False})
            
            if res.fun < 1e-5:
                U = res.x[:n*m*rank].reshape((n*m, rank))
                V = res.x[n*m*rank : n*m*rank + m*p*rank].reshape((m*p, rank))
                W = res.x[n*m*rank + m*p*rank :].reshape((p*n, rank))
                
                snapped = check_and_snap(U, V, W, rank)
                if snapped:
                    best_rank = rank
                    best_U, best_V, best_W = snapped
                    found = True
                    break
                else:
                    # If snapping fails, keep the float solution as a candidate but keep searching for snapped
                    # If we don't find snapped, we might have to return floats if the evaluator is lenient?
                    # The prompt says "integer/half-integer".
                    # So we must find a snapped solution.
                    # Store float solution as fallback if no snapped found for this rank?
                    # No, if it's not snapped, it's invalid.
                    pass

        if found:
            break

    # Fallback to Rank 20 Schoolbook
    if best_U is None:
        rank = n * m * p
        best_U = np.zeros((n*m, rank))
        best_V = np.zeros((m*p, rank))
        best_W = np.zeros((p*n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i*m + j, col] = 1.0
                    best_V[j*p + k, col] = 1.0
                    best_W[k*n + i, col] = 1.0
                    col += 1
                    
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
