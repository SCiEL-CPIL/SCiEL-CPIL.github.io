import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Nonzero entries of the tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    nonzero = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                nonzero.append((u_idx, v_idx, w_idx))
    
    # Function to verify a solution
    def verify(U, V, W, rank):
        for (ui, vi, wi) in nonzero:
            val = np.sum(U[ui, :] * V[vi, :] * W[wi, :])
            if abs(val - 1) > 1e-6:
                return False
        # Check zeros (sample)
        for _ in range(200):
            ui = np.random.randint(0, n * m)
            vi = np.random.randint(0, m * p)
            wi = np.random.randint(0, p * n)
            is_nonzero = (ui, vi, wi) in nonzero
            if not is_nonzero:
                val = np.sum(U[ui, :] * V[vi, :] * W[wi, :])
                if abs(val) > 1e-6:
                    return False
        return True

    # Try to find rank 16
    for rank in [16, 17, 18]:
        if time.time() > deadline - 20:
            break
            
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(attempt * 100 + rank)
            
            # Initialize
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Gradient Descent
            lr = 0.01
            for it in range(500):
                if time.time() > deadline - 2:
                    break
                    
                # Compute gradients
                # Error E = sum (T - U V W)^2
                # dE/dU[ui, c] = -2 * sum_{(ui, vi, wi) in S} (T - val) * V[vi, c] * W[wi, c]
                # Simplified: just do alternating least squares which is faster for small problems
                
                # ALS for U
                for ui in range(n * m):
                    eqs = []
                    for (u, v, w) in nonzero:
                        if u == ui:
                            eqs.append(V[v, :] * W[w, :])
                    if eqs:
                        A_mat = np.array(eqs)
                        b_vec = np.ones(len(eqs))
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        U[ui, :] = sol
                    else:
                        U[ui, :] = 0
                
                # ALS for V
                for vi in range(m * p):
                    eqs = []
                    for (u, v, w) in nonzero:
                        if v == vi:
                            eqs.append(U[u, :] * W[w, :])
                    if eqs:
                        A_mat = np.array(eqs)
                        b_vec = np.ones(len(eqs))
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        V[vi, :] = sol
                    else:
                        V[vi, :] = 0
                        
                # ALS for W
                for wi in range(p * n):
                    eqs = []
                    for (u, v, w) in nonzero:
                        if w == wi:
                            eqs.append(U[u, :] * V[v, :])
                    if eqs:
                        A_mat = np.array(eqs)
                        b_vec = np.ones(len(eqs))
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        W[wi, :] = sol
                    else:
                        W[wi, :] = 0
                        
                # Check convergence
                max_err = 0
                for (u, v, w) in nonzero:
                    val = np.sum(U[u, :] * V[v, :] * W[w, :])
                    max_err = max(max_err, abs(val - 1))
                
                if max_err < 1e-8:
                    if verify(U, V, W, rank):
                        return {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }

    # Fallback: Construct Rank 28 solution using 4 Strassen blocks
    # Strassen 2x2x2
    # U factors (7)
    # M1 = (a+d)(e+h) -> U: a,d ; V: e,h ; W: 1
    # M2 = (c+d)e -> U: c,d ; V: e ; W: 1
    # M3 = a(f-h) -> U: a ; V: f,h ; W: 1, -1
    # M4 = d(g-e) -> U: d ; V: g,e ; W: 1, -1
    # M5 = (a+b)h -> U: a,b ; V: h ; W: 1
    # M6 = (c-a)(e+f) -> U: c,a ; V: e,f ; W: 1, -1
    # M7 = (b-d)(g+h) -> U: b,d ; V: g,h ; W: 1, -1
    
    # Indices for 2x2: i in {0,1}, j in {0,1}, k in {0,1}
    # U indices: i*2+j
    # V indices: j*2+k
    # W indices: k*2+i
    
    # We have 4 blocks:
    # 1. A[:, 0:2] * B[0:2, 0:2] -> j in {0,1}, k in {0,1}
    # 2. A[:, 0:2] * B[0:2, 2:4] -> j in {0,1}, k in {2,3}
    # 3. A[:, 2:4] * B[2:4, 0:2] -> j in {2,3}, k in {0,1}
    # 4. A[:, 2:4] * B[2:4, 2:4] -> j in {2,3}, k in {2,3}
    
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def add_strassen_block(U, V, W, offset, i_offset, j_offset, k_offset):
        # i_offset: 0 or 2 (rows of A)
        # j_offset: 0 or 2 (cols of A / rows of B)
        # k_offset: 0 or 2 (cols of B / rows of C)
        
        # Local indices for 2x2
        # i in {0,1}, j in {0,1}, k in {0,1}
        # Global U idx: (i_offset+i)*m + (j_offset+j)
        # Global V idx: (j_offset+j)*p + (k_offset+k)
        # Global W idx: (k_offset+k)*n + (i_offset+i)
        
        # Strassen Factors
        # c=0: M1
        U[(i_offset+0)*m + (j_offset+0), offset+0] += 1
        U[(i_offset+1)*m + (j_offset+0), offset+0] += 1
        U[(i_offset+0)*m + (j_offset+1), offset+0] += 1
        U[(i_offset+1)*m + (j_offset+1), offset+0] += 1
        
        V[(j_offset+0)*p + (k_offset+0), offset+0] += 1
        V[(j_offset+1)*p + (k_offset+0), offset+0] += 1
        V[(j_offset+0)*p + (k_offset+1), offset+0] += 1
        V[(j_offset+1)*p + (k_offset+1), offset+0] += 1
        
        W[(k_offset+0)*n + (i_offset+0), offset+0] += 1
        W[(k_offset+1)*n + (i_offset+0), offset+0] += 1
        W[(k_offset+0)*n + (i_offset+1), offset+0] += 1
        W[(k_offset+1)*n + (i_offset+1), offset+0] += 1
        
        # c=1: M2
        U[(i_offset+1)*m + (j_offset+0), offset+1] += 1
        U[(i_offset+1)*m + (j_offset+1), offset+1] += 1
        V[(j_offset+0)*p + (k_offset+0), offset+1] += 1
        W[(k_offset+0)*n + (i_offset+1), offset+1] += 1
        W[(k_offset+1)*n + (i_offset+1), offset+1] += 1
        
        # c=2: M3
        U[(i_offset+0)*m + (j_offset+0), offset+2] += 1
        V[(j_offset+0)*p + (k_offset+1), offset+2] += 1
        V[(j_offset+1)*p + (k_offset+1), offset+2] += -1
        W[(k_offset+1)*n + (i_offset+0), offset+2] += 1
        
        # c=3: M4
        U[(i_offset+1)*m + (j_offset+1), offset+3] += 1
        V[(j_offset+1)*p + (k_offset+0), offset+3] += 1
        V[(j_offset+0)*p + (k_offset+0), offset+3] += -1
        W[(k_offset+0)*n + (i_offset+1), offset+3] += 1
        
        # c=4: M5
        U[(i_offset+0)*m + (j_offset+0), offset+4] += 1
        U[(i_offset+0)*m + (j_offset+1), offset+4] += 1
        V[(j_offset+1)*p + (k_offset+1), offset+4] += 1
        W[(k_offset+1)*n + (i_offset+0), offset+4] += 1
        
        # c=5: M6
        U[(i_offset+1)*m + (j_offset+0), offset+5] += 1
        U[(i_offset+0)*m + (j_offset+0), offset+5] += -1
        V[(j_offset+0)*p + (k_offset+0), offset+5] += 1
        V[(j_offset+0)*p + (k_offset+1), offset+5] += 1
        W[(k_offset+0)*n + (i_offset+1), offset+5] += 1
        W[(k_offset+1)*n + (i_offset+1), offset+5] += 1
        
        # c=6: M7
        U[(i_offset+0)*m + (j_offset+1), offset+6] += 1
        U[(i_offset+1)*m + (j_offset+1), offset+6] += -1
        V[(j_offset+1)*p + (k_offset+0), offset+6] += 1
        V[(j_offset+1)*p + (k_offset+1), offset+6] += 1
        W[(k_offset+0)*n + (i_offset+0), offset+6] += 1
        W[(k_offset+1)*n + (i_offset+0), offset+6] += 1

    # Block 1: i_off=0, j_off=0, k_off=0
    add_strassen_block(U, V, W, 0, 0, 0, 0)
    # Block 2: i_off=0, j_off=0, k_off=2
    add_strassen_block(U, V, W, 7, 0, 0, 2)
    # Block 3: i_off=0, j_off=2, k_off=0
    add_strassen_block(U, V, W, 14, 0, 2, 0)
    # Block 4: i_off=0, j_off=2, k_off=2
    add_strassen_block(U, V, W, 21, 0, 2, 2)
    
    # Verify
    if verify(U, V, W, rank):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
        
    # Fallback to Rank 32
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
