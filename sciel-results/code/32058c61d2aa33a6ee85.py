import numpy as np
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill

    n, m, p = 2, 3, 3
    num_inputs_a = n * m  # 6
    num_inputs_b = m * p  # 9
    num_outputs = p * n   # 6

    # Precompute which (ia, ib) pairs contribute to each output index
    # T[ia, ib, out_idx] = 1 when out_idx = ib's j * n + ia's i
    # where ia = i*m+j, ib = j'*p+k, out_idx = k*n+i
    # So T[ia, ib, out] = 1 iff: j == j' AND out = k*n + i where ia = i*m+j, ib = j*p+k

    # For each output index out = k*n + i:
    #   contributing (ia, ib) pairs: ia = i*m + j, ib = j*p + k for all j in range(m)
    output_contribs = {}
    for k in range(p):
        for i in range(n):
            out = k * n + i
            contribs = []
            for j in range(m):
                ia = i * m + j
                ib = j * p + k
                contribs.append((ia, ib))
            output_contribs[out] = contribs

    def target_tensor():
        T = np.zeros((num_inputs_a, num_inputs_b, num_outputs))
        for out, contribs in output_contribs.items():
            for (ia, ib) in contribs:
                T[ia, ib, out] = 1.0
        return T

    target = target_tensor()

    def make_tensor(U, V, W):
        return np.einsum('ir,jr,kr->ijk', U, V, W)

    def check_decomposition(U, V, W, tol=1e-9):
        computed = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.max(np.abs(computed - target)) < tol

    # Rank 17 Strassen-based construction (verified working)
    def build_rank17():
        rank = 17
        U = np.zeros((num_inputs_a, rank))
        V = np.zeros((num_inputs_b, rank))
        W = np.zeros((num_outputs, rank))

        # Index mapping:
        # A rows: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
        # B rows: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)
        # C out: out = k*n+i -> 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1), 4:(0,2), 5:(1,2)

        # Strassen 7 products for 2x2 x 2x2 (cols 0,1 of A; rows 0,1 of B; outputs cols 0,1)
        U[0,0]=1; U[4,0]=1;  V[0,0]=1; V[4,0]=1;  W[0,0]=1; W[3,0]=1
        U[3,1]=1; U[4,1]=1;  V[0,1]=1;             W[1,1]=1; W[3,1]=-1
        U[0,2]=1;                V[1,2]=1; V[4,2]=-1; W[2,2]=1; W[3,2]=1
        U[4,3]=1;                V[3,3]=1; V[0,3]=-1; W[0,3]=1; W[1,3]=1
        U[0,4]=1; U[1,4]=1;    V[4,4]=1;           W[0,4]=-1; W[2,4]=1
        U[3,5]=1; U[0,5]=-1;   V[0,5]=1; V[1,5]=1; W[3,5]=1
        U[1,6]=1; U[4,6]=-1;   V[3,6]=1; V[4,6]=1; W[0,6]=1

        # A[:,:2]*B[:2,2] -> C[:,2] (4 products)
        U[0,7]=1;  V[2,7]=1;  W[4,7]=1
        U[1,8]=1;  V[5,8]=1;  W[4,8]=1
        U[3,9]=1;  V[2,9]=1;  W[5,9]=1
        U[4,10]=1; V[5,10]=1; W[5,10]=1

        # A[:,2]*B[2,:2] -> C[:,:2] (4 products)
        U[2,11]=1; V[6,11]=1; W[0,11]=1
        U[2,12]=1; V[7,12]=1; W[2,12]=1
        U[5,13]=1; V[6,13]=1; W[1,13]=1
        U[5,14]=1; V[7,14]=1; W[3,14]=1

        # A[:,2]*B[2,2] -> C[:,2] (2 products)
        U[2,15]=1; V[8,15]=1; W[4,15]=1
        U[5,16]=1; V[8,16]=1; W[5,16]=1

        return U, V, W, rank

    # Verify rank 17
    U17, V17, W17, r17 = build_rank17()
    computed17 = np.einsum('ir,jr,kr->ijk', U17, V17, W17)
    max_err17 = np.max(np.abs(computed17 - target))

    if max_err17 < 1e-9:
        return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}

    # Attempt rank 16 search via local optimization
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        # Try to find rank 16 via structured search
        # Strategy: use the rank-17 as a starting point, try to merge/eliminate one rank term
        # by allowing half-integer values and searching over perturbed factorizations

        rank = 16
        # Start from rank 17 and try to find a rank-16 approximation
        # Use the known structure: the 7 Strassen terms are irreducible for 2x2x2
        # The remaining 10 terms handle the 3rd column/row
        
        # Try: can we combine some of the 10 non-Strassen terms into fewer?
        # The 10 terms are all rank-1 and involve distinct input/output pairs.
        # Let's check if any pair can be merged.
        
        # Actually, let's try a more systematic approach:
        # Initialize rank-16 as rank-17 with one column set to zero, then optimize
        # all entries to satisfy the tensor equation.
        
        best_score = None
        best_U, best_V, best_W = None, None, None
        
        for zero_col in range(rank17):
            if time.time() - start_time > deadline - 10:
                break
            U_init = U17.copy()
            V_init = V17.copy()
            W_init = W17.copy()
            U_init[:, zero_col] = 0
            V_init[:, zero_col] = 0
            W_init[:, zero_col] = 0
            # Now we have rank-16 with one zero column; try to optimize
            # Use scipy or manual gradient descent if available
            try:
                from scipy.optimize import minimize
            except ImportError:
                break
            
            params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            # Free the previously zero column
            U_init[:, zero_col] = 0.5  # small nonzero start
            V_init[:, zero_col] = 0.5
            W_init[:, zero_col] = 0.5
            params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            size_U = num_inputs_a * rank
            size_V = num_inputs_b * rank
            
            def objective(x):
                Ux = x[:size_U].reshape(num_inputs_a, rank)
                Vx = x[size_U:size_U+size_V].reshape(num_inputs_b, rank)
                Wx = x[size_U+size_V:].reshape(num_outputs, rank)
                T_comp = np.einsum('ir,jr,kr->ijk', Ux, Vx, Wx)
                return np.sum((T_comp - target)**2)
            
            try:
                res = minimize(objective, params, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.success and res.fun < 1e-8:
                    Ux = res.x[:size_U].reshape(num_inputs_a, rank)
                    Vx = res.x[size_U:size_U+size_V].reshape(num_inputs_b, rank)
                    Wx = res.x[size_U+size_V:].reshape(num_outputs, rank)
                    # Check if values are approximately half-integers
                    vals = np.concatenate([Ux.flatten(), Vx.flatten(), Wx.flatten()])
                    vals2 = vals * 2
                    if np.all(np.abs(vals2 - np.round(vals2)) < 0.1):
                        if best_score is None or res.fun < best_score:
                            best_score = res.fun
                            best_U, best_V, best_W = Ux, Vx, Wx
            except Exception:
                pass
        
        if best_U is not None:
            # Round to half-integers
            best_U = np.round(best_U * 2) / 2
            best_V = np.round(best_V * 2) / 2
            best_W = np.round(best_W * 2) / 2
            if check_decomposition(best_U, best_V, best_W):
                return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Fallback: rank 17 if not found rank 16, else naive 18
    if max_err17 < 1e-9:
        return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
    
    rank = 18
    U = np.zeros((num_inputs_a, rank))
    V = np.zeros((num_inputs_b, rank))
    W = np.zeros((num_outputs, rank))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ia = i * m + j
                ib = j * p + k
                out_idx = k * n + i
                U[ia, r] = 1
                V[ib, r] = 1
                W[out_idx, r] = 1
                r += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
