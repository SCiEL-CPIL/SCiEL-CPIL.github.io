import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145  # Leave 5s margin for 150s limit
    
    # Target tensor T of shape (n, m, p)
    # T[i, j, k] = 1 if i,j,k are valid indices? No.
    # The matrix multiplication tensor is defined by:
    # (A * B)[i, k] = sum_j A[i, j] B[j, k]
    # The tensor T has components T[i, j, k] such that
    # C[i, k] = sum_{j} A[i, j] B[j, k]
    # In the decomposition:
    # C[i, k] = sum_{r} U[i, r] * (sum_{j} V[j, r] * W[k, r]?) 
    # Wait, the standard decomposition is:
    # C[i, k] = sum_{r} (sum_{j} U[i, j, r] A[i, j]?) 
    # No, the factors are:
    # U: input A factors. Shape (n*m, R). U[i*m+j, r] multiplies A[i, j].
    # V: input B factors. Shape (m*p, R). V[j*p+k, r] multiplies B[j, k].
    # W: output C factors. Shape (p*n, R). W[k*n+i, r] multiplies C[k, i]?
    # The convention is W[k*n+i, r].
    # The identity is:
    # sum_{j} A[i, j] B[j, k] = sum_{r} (sum_{j} U[i*m+j, r] A[i, j]) * (sum_{j} V[j*p+k, r] B[j, k])?
    # No, that's not how it works.
    # The tensor identity is:
    # T[i, j, k] = sum_{r} U[i, j, r] V[j, k, r] W[i, k, r] ?
    # No.
    # The matrix multiplication tensor $T$ is in $A^* \otimes B^* \otimes C$.
    # $T = \sum_{i,j,k} e_i \otimes e_j \otimes e_k$ ? No.
    # $T$ acts on $A$ and $B$ to produce $C$.
    # $C = T(A, B)$.
    # The decomposition is $T = \sum_{r=1}^R u_r \otimes v_r \otimes w_r$.
    # Here $u_r \in A^*$, $v_r \in B^*$, $w_r \in C$.
    # $u_r$ is a linear functional on $A$. $u_r(A) = \sum_{i,j} U[i*m+j, r] A[i, j]$.
    # $v_r$ is a linear functional on $B$. $v_r(B) = \sum_{j,k} V[j*p+k, r] B[j, k]$.
    # $w_r$ is a vector in $C$. $w_r = \sum_{i,k} W[k*n+i, r] E_{i,k}$.
    # The product is $C = \sum_{r} u_r(A) v_r(B) w_r$.
    # So $C[i, k] = \sum_{r} u_r(A) v_r(B) W[k*n+i, r]$.
    # We need this to equal $\sum_{j} A[i, j] B[j, k]$.
    # So the tensor $T$ in the space $A^* \otimes B^* \otimes C$ has components:
    # $T[i, j, k]$? No, the indices are mixed.
    # Let's flatten the tensor to a vector of length $n \cdot m \cdot p$.
    # The basis for $A^*$ is indexed by $(i, j)$.
    # The basis for $B^*$ is indexed by $(j, k)$.
    # The basis for $C$ is indexed by $(i, k)$.
    # The tensor $T$ is defined by the identity:
    # $\sum_{j} A[i, j] B[j, k] = \sum_{r} (\sum_{i', j'} U[i' m + j', r] A[i', j']) (\sum_{j', k'} V[j' p + k', r] B[j', k']) W[k n + i, r]$
    # This must hold for all $A, B$.
    # This implies that the coefficient of $A[i, j] B[j, k]$ in the RHS must be 1 for all $i, j, k$?
    # No, the LHS is $\sum_{j} A[i, j] B[j, k]$.
    # The term $A[i, j] B[j, k]$ appears in the LHS with coefficient 1.
    # In the RHS, the term $A[i, j] B[j, k]$ appears in the product $u_r(A) v_r(B)$ with coefficient $U[i m + j, r] V[j p + k, r]$.
    # This term is then multiplied by $W[k n + i, r]$ and summed over $r$.
    # So we require:
    # $\sum_{r} U[i m + j, r] V[j p + k, r] W[k n + i, r] = 1$ for all $i, j, k$.
    # This is the condition.
    # Note: The indices $i, j, k$ are independent in the LHS sum?
    # In $\sum_{j} A[i, j] B[j, k]$, $j$ is summed.
    # So for a fixed $i, k$, the LHS is $\sum_{j} A[i, j] B[j, k]$.
    # The RHS is $\sum_{r} W[k n + i, r] (\sum_{i', j'} U[i' m + j', r] A[i', j']) (\sum_{j', k'} V[j' p + k', r] B[j', k'])$.
    # Expanding the RHS, the coefficient of $A[i, j] B[j, k]$ is $\sum_{r} U[i m + j, r] V[j p + k, r] W[k n + i, r]$.
    # This must equal 1.
    # So we have $n \cdot m \cdot p = 36$ equations:
    # For each $i \in [0, n)$, $j \in [0, m)$, $k \in [0, p)$:
    # $\sum_{r=1}^R U[i m + j, r] V[j p + k, r] W[k n + i, r] = 1$.
    
    # This is a system of bilinear (actually trilinear) equations in U, V, W.
    
    # We can flatten U, V, W into a single vector x.
    # Size of x: 9*R + 12*R + 12*R = 33*R.
    
    def residuals(x, R):
        U = x[0:9*R].reshape(9, R)
        V = x[9*R:9*R+12*R].reshape(12, R)
        W = x[9*R+12*R:33*R].reshape(12, R)
        
        res = np.zeros(36)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over r
                    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    u_val = U[i*m+j]
                    v_val = V[j*p+k]
                    w_val = W[k*n+i]
                    val = np.sum(u_val * v_val * w_val)
                    res[idx] = val - 1.0
                    idx += 1
        return res

    best_result = None
    best_rank = n * m * p # 36
    
    # Try ranks from 12 to 25
    for R in range(12, 26):
        if time.time() > deadline:
            break
            
        # Try multiple restarts
        for restart in range(50):
            if time.time() > deadline:
                break
                
            # Initialize x
            # Random initialization
            np.random.seed(42 + restart + R)
            x0 = np.random.normal(0, 1, 33*R)
            
            try:
                sol = least_squares(residuals, x0, args=(R,), method='lm', max_nfev=1000)
                if sol.cost < 1e-12:
                    # Found a solution
                    # Check if it can be made integer/half-integer
                    U, V, W = sol.x[0:9*R].reshape(9, R), sol.x[9*R:9*R+12*R].reshape(12, R), sol.x[9*R+12*R:33*R].reshape(12, R)
                    
                    # Try to round to half-integers
                    U_int = np.round(U * 2) / 2
                    V_int = np.round(V * 2) / 2
                    W_int = np.round(W * 2) / 2
                    
                    # Verify exactness
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U_int[i*m+j] * V_int[j*p+k] * W_int[k*n+i])
                                if abs(val - 1.0) > 1e-9:
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                    if valid:
                        # Convert to lists of floats/ints
                        U_list = [[float(x) for x in row] for row in U_int]
                        V_list = [[float(x) for x in row] for row in V_int]
                        W_list = [[float(x) for x in row] for row in W_int]
                        return {"U": U_list, "V": V_list, "W": W_list}
                        
            except Exception as e:
                pass
                
    # Fallback to schoolbook
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
