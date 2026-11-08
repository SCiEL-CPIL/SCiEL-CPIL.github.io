import numpy as np
import time

_START_TIME = time.time()
DEADLINE_SECONDS = 130

def _elapsed():
    return time.time() - _START_TIME

def _time_remaining():
    return DEADLINE_SECONDS - _elapsed()

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x4 * 4x7 matrix multiplication."""
    n, m, p = 2, 4, 7
    nA = n * m   # 8
    nB = m * p   # 28
    nW = p * n   # 14

    # Strategy: Use Strassen 2x2x2 for k=0..5 (3 blocks of 2x2 in j dimension),
    # then solve for the k=6 terms using linear algebra.
    # This should give rank 42 if the linear system has a solution with nice entries.

    def build_strassen_blocks():
        """Build rank-42 decomposition using Strassen for k=0..5."""
        rank = 42
        U = [[0] * rank for _ in range(nA)]
        V = [[0] * rank for _ in range(nB)]
        W = [[0] * rank for _ in range(nW)]
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)
            e_idx = j0 * p + k0
            f_idx = j0 * p + (k0 + 1)
            g_idx = (j0 + 1) * p + k0
            h_idx = (j0 + 1) * p + (k0 + 1)
            c00 = k0 * n + i0
            c01 = (k0 + 1) * n + i0
            c10 = k0 * n + (i0 + 1)
            c11 = (k0 + 1) * n + (i0 + 1)

            # M1 = (a+d)(e+h)
            U[a_idx][col] += 1; U[d_idx][col] += 1
            V[e_idx][col] += 1; V[h_idx][col] += 1
            W[c00][col] += 1; W[c11][col] += 1
            col += 1

            # M2 = (c+d)e
            U[c_idx][col] += 1; U[d_idx][col] += 1
            V[e_idx][col] += 1
            W[c10][col] += 1; W[c11][col] -= 1
            col += 1

            # M3 = a(f-h)
            U[a_idx][col] += 1
            V[f_idx][col] += 1; V[h_idx][col] -= 1
            W[c01][col] += 1; W[c11][col] += 1
            col += 1

            # M4 = d(g-e)
            U[d_idx][col] += 1
            V[g_idx][col] += 1; V[e_idx][col] -= 1
            W[c00][col] += 1; W[c10][col] += 1
            col += 1

            # M5 = (a+b)h
            U[a_idx][col] += 1; U[b_idx][col] += 1
            V[h_idx][col] += 1
            W[c00][col] -= 1; W[c01][col] += 1
            col += 1

            # M6 = (c-a)(e+f)
            U[c_idx][col] += 1; U[a_idx][col] -= 1
            V[e_idx][col] += 1; V[f_idx][col] += 1
            W[c11][col] += 1
            col += 1

            # M7 = (b-d)(g+h)
            U[b_idx][col] += 1; U[d_idx][col] -= 1
            V[g_idx][col] += 1; V[h_idx][col] += 1
            W[c00][col] += 1
            col += 1

        for j_start in [0, 2]:
            for k_start in [0, 2, 4]:
                add_strassen(0, j_start, k_start)

        return U, V, W, col

    def verify(U, V, W):
        """Verify exact decomposition."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    ai = i * m + j
                    bi = j * p + k
                    wi = k * n + i
                    for l in range(r):
                        val += U[ai][l] * V[bi][l] * W[wi][l]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def to_numpy(U, V, W):
        return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

    # Start with the Strassen blocks (rank 42)
    U, V, W, rank = build_strassen_blocks()
    
    # Now we need to add the k=6 terms.
    # For each j, we need to satisfy:
    #   sum_{l=0}^{41} U[i*m+j, l] * V[j*p+6, l] * W[6*n+i, l] = 1 for i=0,1
    # 
    # Let's denote u_{ij} = U[i*m+j, :] (length 42)
    # Let v_j = V[j*p+6, :] (length 42)
    # Let w_i = W[6*n+i, :] (length 42)
    # 
    # We need: <u_{0j}, v_j * w_0> = 1 and <u_{1j}, v_j * w_1> = 1 for each j.
    # 
    # If we set w_0 = w_1 = 1 (all ones), then we need:
    #   <u_{0j}, v_j> = 1 and <u_{1j}, v_j> = 1
    # 
    # This is a linear system for v_j:
    #   [u_{0j}] v_j = 1
    #   [u_{1j}] v_j = 1
    # 
    # Let's try this approach.

    U_np, V_np, W_np = to_numpy(U, V, W)
    r = U_np.shape[1]
    
    # Check if the current decomposition already works (it shouldn't, since k=6 is missing)
    # But let's verify to be safe
    if verify(U, V, W):
        # Already valid, return it
        return {"U": U, "V": V, "W": W}
    
    # Try to solve for the k=6 terms
    success = False
    if _time_remaining() > 30:
        # For each j, solve for v_j
        new_U = [row[:] for row in U]
        new_V = [row[:] for row in V]
        new_W = [row[:] for row in W]
        
        # Extend to rank r+4 (one new column per j for k=6)
        new_rank = r + 4
        for i in range(nA):
            new_U[i].extend([0] * 4)
        for i in range(nB):
            new_V[i].extend([0] * 4)
        for i in range(nW):
            new_W[i].extend([0] * 4)
        
        # For each j, solve for v_j
        for j in range(m):
            if _time_remaining() < 10:
                break
            
            # Extract u_{0j} and u_{1j}
            u0j = U_np[j, :]  # U[0*m+j, :]
            u1j = U_np[m+j, :]  # U[1*m+j, :]
            
            # Set w_0 and w_1 to 1 for the new column
            col_idx = r + j
            new_W[6*n+0][col_idx] = 1.0
            new_W[6*n+1][col_idx] = 1.0
            
            # Solve for v_j: <u0j, v_j> = 1 and <u1j, v_j> = 1
            # This is underdetermined, so we can use least squares or find a particular solution
            A = np.vstack([u0j, u1j])
            b = np.array([1.0, 1.0])
            
            try:
                # Find minimum norm solution
                v_j = np.linalg.lstsq(A, b, rcond=None)[0]
                
                # Check if the solution is "nice" (integer or half-integer)
                # For now, just use it as is
                for i in range(nB):
                    new_V[i][col_idx] = v_j[i] if i == j*p+6 else 0.0
                
                # Actually, we only need V[j*p+6, col_idx] = v_j[j*p+6]
                # But wait, v_j is the entire vector V[:, col_idx]
                # So we need to set V[i, col_idx] = v_j[i] for all i
                for i in range(nB):
                    new_V[i][col_idx] = v_j[i]
                    
            except Exception:
                continue
        
        # Verify the new decomposition
        new_U_np, new_V_np, new_W_np = to_numpy(new_U, new_V, new_W)
        
        # Check if it works
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    ai = i * m + j
                    bi = j * p + k
                    wi = k * n + i
                    for l in range(new_rank):
                        val += new_U_np[ai][l] * new_V_np[bi][l] * new_W_np[wi][l]
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Check if entries are nice (integer or half-integer)
            # For now, accept any solution
            return {"U": new_U, "V": new_V, "W": new_W}
    
    # Fallback: pure schoolbook
    U = [[0] * (nA * nW) for _ in range(nA)]
    V = [[0] * (nA * nW) for _ in range(nB)]
    W = [[0] * (nA * nW) for _ in range(nW)]
    col = 0
    for j in range(m):
        for i in range(n):
            for k in range(p):
                ai = i * m + j
                bi = j * p + k
                wi = k * n + i
                U[ai][col] = 1
                V[bi][col] = 1
                W[wi][col] = 1
                col += 1
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}
