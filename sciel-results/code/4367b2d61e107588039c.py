import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    start_time = time.time()
    deadline = start_time + 140
    
    # Winograd-style structured decomposition
    # For (2,3,3): We can use the Winograd scheme for (n=2, m=3, p=3)
    # The Winograd algorithm for (n, m, p) has rank:
    #   For (2,3,3): We can use the factorization from Winograd's paper
    #   The rank for (2,3,3) using Winograd is: (n+m-1)*(m+p-1) - something
    #   Actually, let me think about this more carefully.
    
    # Winograd's algorithm for (n, m, p) matrix multiplication:
    # It groups the computation into blocks and uses FFT-like transforms.
    # For (2,3,3):
    #   - The "input" transform has size related to n+m-1 = 4
    #   - The "output" transform has size related to m+p-1 = 5
    #   - But the actual rank depends on the specific factorization
    
    # Let me try a different approach: use the known minimal rank results.
    # For (2,3,3), the schoolbook rank is 18.
    # Winograd's algorithm can reduce this.
    
    # Actually, let me try a structured random search with the constraint
    # that entries are in {-1, -0.5, 0, 0.5, 1} and use a smarter search.
    
    # First, try to find a rank-16 or rank-15 solution using structured patterns.
    
    # The Winograd scheme for (2,3,3):
    # We can think of it as:
    #   A (2x3) * B (3x3) = C (2x3)
    #   
    # Using Winograd's F(2,3) transform:
    #   The transform for the 2x3 input has 4 rows (n+m-1 = 4)
    #   The transform for the 3x3 input has 5 rows (m+p-1 = 5)
    #   But we need to be careful about the exact scheme.
    
    # Let me try a direct approach: enumerate possible support patterns
    # for small ranks and solve the linear system.
    
    # For rank r, we need to find U (6xr), V (9xr), W (6xr) such that
    # sum_s U[a,s]*V[b,s]*W[c,s] = T[a,b,c]
    
    # This is equivalent to finding r rank-1 tensors that sum to T.
    
    # Let me try rank 16 first with a structured approach.
    
    # Actually, let me try a different strategy: use the fact that the
    # matrix multiplication tensor has a specific structure and try
    # to find a decomposition with entries in {-1, 0, 1} or {-1, -0.5, 0, 0.5, 1}.
    
    # For rank 16:
    # Let's try to construct it using the Winograd idea.
    # 
    # Winograd's F(2,3) for (2,3,3):
    # The key insight is that we can write:
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Using the Winograd transform, we can reduce the number of multiplications.
    # 
    # For (2,3,3), one possible Winograd decomposition has rank 15 or 16.
    # 
    # Let me try to construct it explicitly.
    
    # Try rank 15 first
    for rank in [15, 16]:
        if time.time() > deadline - 20:
            break
        
        # Use a structured random search with constraints
        # Try to find U, V, W with entries in {-1, -0.5, 0, 0.5, 1}
        
        max_trials = 50000
        found = False
        
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                break
            
            # Generate sparse random matrices with entries in {-1, -0.5, 0, 0.5, 1}
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            
            for c in range(rank):
                # U column: place 2-4 non-zeros
                nz_u = np.random.randint(2, 5)
                idx_u = np.random.choice(nm, nz_u, replace=False)
                U[idx_u, c] = np.random.choice([-1, -0.5, 0.5, 1], nz_u)
                
                # V column: place 2-4 non-zeros
                nz_v = np.random.randint(2, 5)
                idx_v = np.random.choice(mp, nz_v, replace=False)
                V[idx_v, c] = np.random.choice([-1, -0.5, 0.5, 1], nz_v)
                
                # W column: place 2-4 non-zeros
                nz_w = np.random.randint(2, 5)
                idx_w = np.random.choice(pn, nz_w, replace=False)
                W[idx_w, c] = np.random.choice([-1, -0.5, 0.5, 1], nz_w)
            
            # Compute R
            R = np.einsum('as,bs,cs->abc', U, V, W)
            
            # Check error
            err = np.linalg.norm(R - T)
            if err < 1e-9:
                U_out = U.tolist()
                V_out = V.tolist()
                W_out = W.tolist()
                return {"U": U_out, "V": V_out, "W": W_out}
            
            # Local refinement if close
            if err < 0.5:
                for _ in range(30):
                    if time.time() > deadline - 10:
                        break
                    mat_idx = np.random.randint(3)
                    if mat_idx == 0:
                        r_idx, c_idx = np.random.randint(nm), np.random.randint(rank)
                        U[r_idx, c_idx] += np.random.choice([-0.5, 0.5])
                    elif mat_idx == 1:
                        r_idx, c_idx = np.random.randint(mp), np.random.randint(rank)
                        V[r_idx, c_idx] += np.random.choice([-0.5, 0.5])
                    else:
                        r_idx, c_idx = np.random.randint(pn), np.random.randint(rank)
                        W[r_idx, c_idx] += np.random.choice([-0.5, 0.5])
                    
                    R = np.einsum('as,bs,cs->abc', U, V, W)
                    err = np.linalg.norm(R - T)
                    if err < 1e-9:
                        U_out = U.tolist()
                        V_out = V.tolist()
                        W_out = W.tolist()
                        return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback: Schoolbook decomposition of rank n*m*p = 18
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
