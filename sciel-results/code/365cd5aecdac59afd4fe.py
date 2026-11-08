"""Rank-minimal tensor decomposition for 4x5 * 5x6 matrix multiplication.

Strategy: Use alternating linear solves (ALS) to find a low-rank decomposition.
Start from a known-good decomposition (block-Strassen rank 108) and attempt to
project it into lower-rank spaces using ALS iterations with integer/half-integer
rounding. This is a fundamentally different optimization approach from the
combinatorial block construction + greedy removal used previously.
"""
import time
import random

try:
    import numpy as np
except Exception:
    np = None

try:
    import sympy
except Exception:
    sympy = None

DIMENSIONS = [4, 5, 6]
SEED = 42


def _strassen_2x2():
    """Return the 7 rank-1 terms for 2x2 matrix multiplication."""
    terms = []
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
    terms.append(([0,0,0,1], [-1,0,1,0], [1,0,1,0]))
    terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    return terms


def _build_block_strassen(n=4, m=5, p=6):
    """Build a block-Strassen decomposition for n x m * m x p (rank 108)."""
    R = 108
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    strassen = _strassen_2x2()
    col = 0
    
    i_blocks = [(0,1), (2,3)]
    j_blocks = [(0,1), (2,3), (4,)]
    k_blocks = [(0,1), (2,3), (4,5)]
    
    for ib in i_blocks:
        for jb in j_blocks:
            for kb in k_blocks:
                si = len(ib)
                sj = len(jb)
                sk = len(kb)
                
                if si == 2 and sj == 2 and sk == 2:
                    for t in range(7):
                        u_vec, v_vec, w_vec = strassen[t]
                        for di in range(2):
                            for dj in range(2):
                                i = ib[0] + di
                                j = jb[0] + dj
                                U[i * m + j][col] += u_vec[di * 2 + dj]
                        for dj in range(2):
                            for dk in range(2):
                                j = jb[0] + dj
                                k = kb[0] + dk
                                V[j * p + k][col] += v_vec[dj * 2 + dk]
                        for dk in range(2):
                            for di in range(2):
                                k = kb[0] + dk
                                i = ib[0] + di
                                W[k * n + i][col] += w_vec[dk * 2 + di]
                        col += 1
                elif si == 2 and sj == 1 and sk == 2:
                    for di in range(2):
                        for dk in range(2):
                            i = ib[0] + di
                            j = jb[0]
                            k = kb[0] + dk
                            U[i * m + j][col] = 1
                            V[j * p + k][col] = 1
                            W[k * n + i][col] = 1
                            col += 1
    
    return U, V, W, R


def _build_schoolbook(n=4, m=5, p=6):
    """Build the schoolbook decomposition: rank = n*m*p = 120."""
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return U, V, W, R


def _verify_numpy(U, V, W, n, m, p):
    """Verify the decomposition using numpy."""
    if np is None:
        return False
    R = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    for i in range(n):
        for k in range(p):
            M = np.zeros((m, m), dtype=np.float64)
            for c in range(R):
                wc = Wa[k * n + i, c]
                if wc == 0:
                    continue
                uc = Ua[i * m: i * m + m, c]
                vc = Va[k * p: k * p + m, c]
                M += np.outer(uc, vc) * wc
            target = np.eye(m, dtype=np.float64)
            if not np.allclose(M, target, atol=1e-9):
                return False
    return True


def _verify_sympy(U, V, W, n, m, p):
    """Verify the decomposition using sympy for exactness."""
    if sympy is None:
        return False
    R = len(U[0])
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for j2 in range(m):
                    s = sympy.Integer(0)
                    for c in range(R):
                        wv = sympy.Integer(W[k * n + i][c])
                        uu = sympy.Integer(U[i * m + j][c])
                        vv = sympy.Integer(V[j2 * p + k][c])
                        if wv == 0 or uu == 0 or vv == 0:
                            continue
                        s += wv * uu * vv
                    target = sympy.Integer(1) if j == j2 else sympy.Integer(0)
                    if s != target:
                        return False
    return True


def _round_to_half_int(arr):
    """Round array entries to nearest half-integer (0, 0.5, 1, 1.5, ...)."""
    if np is None:
        return arr
    return np.round(arr * 2.0) / 2.0


def _als_iterate(U, V, W, n, m, p, R, max_iters=50, deadline=None):
    """Alternating linear solve to refine a rank-R decomposition.
    
    For each (i,k) pair, we need:
    sum_t W[k*n+i, t] * U[i*m+j, t] * V[j*p+k, t] = delta(j)
    
    This is equivalent to: for each (i,k), the matrix sum_t W[k*n+i,t] * outer(U[:,t], V[:,t]) = I_m
    
    ALS: Fix V,W -> solve for U (linear system per (i,k))
         Fix U,W -> solve for V (linear system per (j,k)) 
         Fix U,V -> solve for W (linear system per (i,k))
    """
    if np is None:
        return U, V, W
    
    Ua = np.array(U, dtype=np.float64).copy()
    Va = np.array(V, dtype=np.float64).copy()
    Wa = np.array(W, dtype=np.float64).copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 10.0:
            break
        
        # Solve for U: fix V, W
        # For each (i,k): sum_t Wa[k*n+i,t] * outer(Ua[i*m:(i+1)*m, t], Va[t]) = I_m
        # This is a linear system in Ua[i*m:(i+1)*m, :] for each (i,k)
        # Reshape: for fixed i, we have m*p equations (from all k) and R*m unknowns (U[i*m:(i+1)*m, :])
        # Actually, let's think of it differently.
        # For fixed i: sum_k sum_t Wa[k*n+i,t] * Ua[i*m+j,t] * Va[j*p+k,t] = delta(j)
        # For fixed i, j: sum_k sum_t Ua[i*m+j,t] * (Wa[k*n+i,t] * Va[j*p+k,t]) = delta(j)
        # Let A_i_j[t] = sum_k Wa[k*n+i,t] * Va[j*p+k,t]  (length R vector)
        # Then sum_t Ua[i*m+j,t] * A_i_j[t] = delta(j)
        # This is underdetermined (R unknowns, 1 equation per j)
        # So we can't directly solve. Instead, use least-squares projection.
        
        # Better approach: treat the whole thing as a tensor optimization.
        # Use a simple gradient-free approach: for each factor, project onto 
        # the subspace that best fits the target.
        
        # Alternative: use the fact that if we have a valid decomposition of rank R,
        # we can try to find a lower-rank one by finding linear dependencies.
        
        # Let's try a different ALS formulation:
        # Fix V and W. For each i, we want U[i*m:(i+1)*m, :] such that
        # for all k: sum_t Wa[k*n+i,t] * outer(U[i*m:(i+1)*m,t], Va[t]) = I_m
        # Let B_i = Va  (m x R)
        # Let C_ik[t] = Wa[k*n+i, t]
        # We need: sum_t C_ik[t] * U_i[:,t] * B_i[:,t].T = I_m
        # => (U_i * diag(C_ik)) @ B_i.T = I_m  where U_i is m x R, C_ik is R-vector
        # => U_i @ (B_i.T * C_ik) = I_m  (element-wise scaling of columns)
        # Let D_ik = B_i.T * C_ik  (R x m matrix, column t scaled by C_ik[t])
        # Then U_i @ D_ik = I_m
        # This is m x R @ R x m = m x m
        # If R >= m, we can solve: U_i = I_m @ pinv(D_ik)
        
        for i in range(n):
            # Build system for U_i
            # For each k, D_ik = Va.T * Wa[k*n+i, :]  (R x m)
            # U_i @ D_ik = I_m
            # Stack all k: [U_i @ D_i0; U_i @ D_i1; ...] = [I; I; ...]
            # This is (p*m) x (R*m) system
            # Let's solve via least squares
            # U_i is m x R. Let u_i be vectorized (R*m unknowns)
            # For each k: vec(U_i @ D_ik) = vec(I_m)
            # vec(U_i @ D_ik) = (D_ik.T ⊗ I_m) vec(U_i)
            
            # Build big system
            A_big = np.zeros((p * m, R * m))
            b_big = np.zeros(p * m)
            for k in range(p):
                D_ik = Va.T * Wa[k * n + i, :]  # R x m
                # vec(U_i @ D_ik) = (D_ik.T ⊗ I_m) vec(U_i)
                # D_ik.T is m x R, I_m is m x m
                # Kronecker product: (D_ik.T ⊗ I_m) is (m*m) x (R*m)
                kron = np.kron(D_ik.T, np.eye(m))
                start = k * m
                end = (k + 1) * m
                A_big[start:end, :] = kron
                b_big[start:end] = 1.0  # vec(I_m) has 1s at positions 0, m+1, 2*(m+1), ...
            
            # Actually vec(I_m) for m=5: [1,0,0,0,0, 0,1,0,0,0, 0,0,1,0,0, 0,0,0,1,0, 0,0,0,0,1]
            b_big = np.zeros(p * m)
            for k in range(p):
                for j in range(m):
                    b_big[k * m + j] = 1.0 if j == 0 else 0.0
                # Wait, vec(I_m) in row-major: I_m[j,j] = 1, so position j*m+j
                # Let me redo
            b_big = np.zeros(p * m)
            for k in range(p):
                for jj in range(m):
                    b_big[k * m + jj] = 1.0 if jj == 0 else 0.0
            # That's wrong. vec(I_m) in column-major (Fortran) or row-major?
            # np.kron uses row-major by default. Let's be careful.
            # D_ik is R x m. U_i is m x R. U_i @ D_ik is m x m.
            # vec(U_i @ D_ik) in row-major: element (a,b) is at position a*m+b
            # (U_i @ D_ik)[a,b] = sum_t U_i[a,t] * D_ik[t,b]
            # = sum_t U_i[a,t] * Va.T[t,b] * Wa[k*n+i,t]
            # = sum_t U_i[a,t] * Va[b,t] * Wa[k*n+i,t]
            
            # So the system is: for each (k, a, b):
            # sum_t U_i[a,t] * Va[b,t] * Wa[k*n+i,t] = delta(a,b)
            
            # Let's build it properly
            A_big = np.zeros((p * m * m, R * m))
            b_big = np.zeros(p * m * m)
            for k in range(p):
                for a in range(m):
                    for b in range(m):
                        eq_idx = (k * m + a) * m + b
                        # sum_t U_i[a,t] * (Va[b,t] * Wa[k*n+i,t]) = delta(a,b)
                        for t in range(R):
                            A_big[eq_idx, a * R + t] = Va[b, t] * Wa[k * n + i, t]
                        b_big[eq_idx] = 1.0 if a == b else 0.0
            
            # Solve least squares
            try:
                sol, _, _, _ = np.linalg.lstsq(A_big, b_big, rcond=None)
                Ua[i * m:(i+1) * m, :] = sol.reshape(m, R)
            except Exception:
                pass
        
        # Solve for V: fix U, W
        # For each j: sum_t U_i[a,t] * V_j[b,t] * Wa[k*n+i,t] = delta(a,b) for all (i,k,a,b)
        # For fixed j: sum_i sum_k sum_a U_i[a,t] * Wa[k*n+i,t] * V_j[b,t] = delta(a,b)
        # Let E_j[t] = sum_i sum_k sum_a ... hmm this is getting complex.
        # For fixed j and b: sum_t V_j[b,t] * (sum_i sum_k sum_a U_i[a,t] * Wa[k*n+i,t] * delta(a,b))
        # = sum_t V_j[b,t] * (sum_i sum_k U_i[b,t] * Wa[k*n+i,t])
        # Let F_j[b,t] = sum_i sum_k U_i[b,t] * Wa[k*n+i,t]  -- no, t is the index
        # Actually: for fixed j, the equation is:
        # sum_t V_j[b,t] * G_j[b,t] = delta for all (i,k,a) where a=b
        # G_j[b,t] = sum_i sum_k U_i[b,t] * Wa[k*n+i,t]  -- but this sums over i,k
        # Wait, the constraint is: for ALL (i,k,a,b): sum_t U_i[a,t] * V_j[b,t] * Wa[k*n+i,t] = delta(a,b)
        # For fixed j, this gives n*p*m*m equations in m*R unknowns (V_j is m x R)
        
        for j in range(m):
            A_big = np.zeros((n * p * m * m, R * m))
            b_big = np.zeros(n * p * m * m)
            for i in range(n):
                for k in range(p):
                    for a in range(m):
                        for b in range(m):
                            eq_idx = ((i * p + k) * m + a) * m + b
                            for t in range(R):
                                A_big[eq_idx, b * R + t] = Ua[i * m + a, t] * Wa[k * n + i, t]
                            b_big[eq_idx] = 1.0 if a == b else 0.0
            
            try:
                sol, _, _, _ = np.linalg.lstsq(A_big, b_big, rcond=None)
                Va[j * p:(j+1) * p, :] = sol.reshape(p, R)
            except Exception:
                pass
        
        # Solve for W: fix U, V
        # For each (i,k): sum_t U_i[a,t] * V_j[b,t] * W_ik[t] = delta(a,b)
        # For fixed (i,k): n*p... no, m*m equations in R unknowns
        for i in range(n):
            for k in range(p):
                A_big = np.zeros(m * m)
                b_big = np.zeros(m * m)
                for a in range(m):
                    for b in range(m):
                        eq_idx = a * m + b
                        for t in range(R):
                            A_big[eq_idx] += Ua[i * m + a, t] * Va[j * p + k, t] * 0  # placeholder
                        # Actually: sum_t W_ik[t] * U_i[a,t] * V_j[b,t] = delta(a,b)
                        # So A_ik[t] = sum over... no, for each t:
                        # The equation is: sum_t W_ik[t] * U_i[a,t] * V_j[b,t] = delta(a,b)
                        # This is R unknowns (W_ik[0..R-1]) and m*m equations
                # Build system
                A_sys = np.zeros((m * m, R))
                b_sys = np.zeros(m * m)
                for a in range(m):
                    for b in range(m):
                        eq_idx = a * m + b
                        for t in range(R):
                            A_sys[eq_idx, t] = Ua[i * m + a, t] * Va[j * p + k, t]
                        b_sys[eq_idx] = 1.0 if a == b else 0.0
                
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                    Wa[k * n + i, :] = sol
                except Exception:
                    pass
    
    # Round to half-integers
    Ua = _round_to_half_int(Ua)
    Va = _round_to_half_int(Va)
    Wa = _round_to_half_int(Wa)
    
    # Convert back to lists
    U_out = [[Ua[r, c] for c in range(R)] for r in range(len(Ua))]
    V_out = [[Va[r, c] for c in range(R)] for r in range(len(Va))]
    W_out = [[Wa[r, c] for c in range(R)] for r in range(len(Wa))]
    
    return U_out, V_out, W_out


def _als_reduce_rank(U, V, W, n, m, p, target_rank, deadline):
    """Try to find a decomposition of rank target_rank using ALS from a random init."""
    if np is None:
        return None
    
    random.seed(SEED + target_rank)
    np.random.seed(SEED + target_rank)
    
    R = target_rank
    # Random initialization
    Ua = np.random.randint(-2, 3, size=(n * m, R)) * 0.5
    Va = np.random.randint(-2, 3, size=(m * p, R)) * 0.5
    Wa = np.random.randint(-2, 3, size=(p * n, R)) * 0.5
    
    # Run ALS
    Ua, Va, Wa = _als_iterate(Ua.tolist(), Va.tolist(), Wa.tolist(), n, m, p, R, max_iters=30, deadline=deadline)
    
    # Verify
    if _verify_numpy(Ua, Va, Wa, n, m, p):
        return Ua, Va, Wa
    
    return None


def _try_reduce_rank_greedy(U, V, W, n, m, p, deadline):
    """Try to reduce rank by removing columns that are redundant."""
    R = len(U[0])
    changed = True
    while changed and time.time() < deadline - 15.0:
        changed = False
        R = len(U[0])
        for c in range(R):
            if time.time() > deadline - 15.0:
                break
            keep = [x for x in range(R) if x != c]
            U2 = [[U[r][x] for x in keep] for r in range(n * m)]
            V2 = [[V[r][x] for x in keep] for r in range(m * p)]
            W2 = [[W[r][x] for x in keep] for r in range(p * n)]
            if _verify_numpy(U2, V2, W2, n, m, p):
                U, V, W = U2, V2, W2
                changed = True
                break
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    n, m, p = DIMENSIONS
    
    try:
        # Start with block-Strassen decomposition (rank 108)
        U, V, W, R = _build_block_strassen(n, m, p)
        
        # Verify
        if sympy is not None and R <= 130:
            ok = _verify_sympy(U, V, W, n, m, p)
        else:
            ok = _verify_numpy(U, V, W, n, m, p)
        
        if not ok:
            U, V, W, R = _build_schoolbook(n, m, p)
            return {"U": U, "V": V, "W": W}
        
        best_rank = R
        best_U, best_V, best_W = U, V, W
        
        # Try ALS-based rank reduction
        for target_rank in range(best_rank - 1, max(best_rank - 20, 80) - 1, -1):
            if time.time() > deadline - 20.0:
                break
            result = _als_reduce_rank(U, V, W, n, m, p, target_rank, deadline)
            if result is not None:
                U, V, W = result
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
        
        # Greedy column removal on the best found
        U, V, W = _try_reduce_rank_greedy(best_U, best_V, best_W, n, m, p, deadline)
        
        return {"U": U, "V": V, "W": W}
    
    except Exception:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
