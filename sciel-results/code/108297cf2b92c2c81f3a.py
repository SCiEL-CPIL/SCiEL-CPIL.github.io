import numpy as np

DIMENSIONS = [2, 2, 2]

def construct():
    n, m, p = DIMENSIONS
    rank = 7
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    
    def ui(i,j): return i*m+j
    def vi(j,k): return j*p+k
    
    U[ui(0,0),0]=1; U[ui(1,1),0]=1
    V[vi(0,0),0]=1; V[vi(1,1),0]=1
    
    U[ui(1,0),1]=1; U[ui(1,1),1]=1
    V[vi(0,0),1]=1
    
    U[ui(0,0),2]=1
    V[vi(0,1),2]=1; V[vi(1,1),2]=-1
    
    U[ui(1,1),3]=1
    V[vi(1,0),3]=1; V[vi(0,0),3]=-1
    
    U[ui(0,0),4]=1; U[ui(0,1),4]=1
    V[vi(1,1),4]=1
    
    U[ui(1,0),5]=1; U[ui(0,0),5]=-1
    V[vi(0,0),5]=1; V[vi(0,1),5]=1
    
    U[ui(0,1),6]=1; U[ui(1,1),6]=-1
    V[vi(1,0),6]=1; V[vi(1,1),6]=1
    
    # Solve for W: for each output (i,k), W[k*n+i, c] such that
    # sum_c W[k*n+i,c] * M_c = c_{i,k} for all A,B
    # M_c = (U[:,c] · a_vec) * (V[:,c] · b_vec)
    # This must hold for all a,b, so it's a bilinear identity.
    # We can solve by generating enough random (A,B) pairs and solving least squares.
    
    W = np.zeros((p*n, rank))
    
    # Generate random test cases
    rng = np.random.RandomState(42)
    N = 50
    rows = []
    cols = []
    for _ in range(N):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C = A @ B
        for i in range(n):
            for k in range(p):
                w_row = k*n + i
                # M_c values
                Mc = np.zeros(rank)
                for c in range(rank):
                    a_flat = A.flatten()  # a[i,j] at index i*m+j
                    b_flat = B.flatten()  # b[j,k] at index j*p+k
                    Mc[c] = np.dot(U[:,c], a_flat) * np.dot(V[:,c], b_flat)
                rows.append(Mc)
                cols.append(C[i,k])
    
    rows = np.array(rows)
    cols = np.array(cols)
    # rows shape (N*n*p, rank), cols shape (N*n*p,)
    # We want W such that for each (i,k), rows corresponding to that (i,k) dot W[k*n+i,:] = cols
    # Actually rows is interleaved. Let me restructure.
    
    # Better: for each (i,k), collect the Mc vectors and C[i,k] values
    W_solved = np.zeros((p*n, rank))
    for i in range(n):
        for k in range(p):
            w_row = k*n + i
            Mc_list = []
            C_list = []
            for _ in range(N):
                A = rng.randn(n, m)
                B = rng.randn(m, p)
                C = A @ B
                Mc = np.zeros(rank)
                for c in range(rank):
                    a_flat = A.flatten()
                    b_flat = B.flatten()
                    Mc[c] = np.dot(U[:,c], a_flat) * np.dot(V[:,c], b_flat)
                Mc_list.append(Mc)
                C_list.append(C[i,k])
            Mc_arr = np.array(Mc_list)
            C_arr = np.array(C_list)
            # Solve Mc_arr @ W[w_row,:] = C_arr
            W[w_row], residuals, rank_r, sv = np.linalg.lstsq(Mc_arr, C_arr, rcond=None)
    
    # Round to nearest integer
    W = np.round(W).astype(int)
    
    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()
    
    # Verify
    Uv = np.array(U_out)
    Vv = np.array(V_out)
    Wv = np.array(W_out)
    for _ in range(20):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C = A @ B
        C_hat = np.zeros((n, p))
        for c in range(rank):
            a_val = np.dot(Uv[:,c], A.flatten())
            b_val = np.dot(Vv[:,c], B.flatten())
            for i in range(n):
                for k in range(p):
                    C_hat[i,k] += Wv[k*n+i, c] * a_val * b_val
        assert np.allclose(C, C_hat, atol=1e-9), f"Verification failed: {C} vs {C_hat}"
    
    return {"U": U_out, "V": V_out, "W": W_out}
