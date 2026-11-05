import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def _build_2x2_rank6():
    """
    Build a rank-6 decomposition of the 2x2 matrix multiplication tensor.
    Returns (U6, V6, W6) each of shape (4, 6) in the 2x2 local index convention:
      A indices: 0=A00, 1=A01, 2=A10, 3=A11
      B indices: 0=B00, 1=B01, 2=B10, 3=B11
      C indices: 0=C00, 1=C01, 2=C10, 3=C11
    such that sum_r U6[a,r]*V6[b,r]*W6[c,r] = 1 iff (a,b,c) form a valid term, else 0.
    Returns None if no valid half-integer solution found.
    """
    # The 2x2 matrix multiplication tensor has 8 nonzero entries out of 64.
    # We need rank-6 factorization. Known: Strassen gives rank 7.
    # Rank 6 is the border rank but may not be achievable with exact decomposition.
    # However, let's try a general approach: fix U to have a specific structure
    # and solve for V and W.
    
    # Strategy: Use the known fact that the 2x2x2 tensor can be written with
    # 6 bilinear forms. Try various 0/1 patterns for U and solve for V, W.
    
    # Approach: enumerate small sets of "sums" for U, then solve linear system for V and W.
    # For each rank-1 term r, U[:,r] defines a linear functional on A.
    # We need: for all (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    # This is a 3-tensor decomposition. Let's try fixing U and solving for V*W.
    
    # Key insight: if we fix U, then for each r, the matrix M_r = V[:,r] * W[:,r]^T
    # and we need sum_r U[a,r] * M_r[b,c] = T[a,b,c]
    # i.e., sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    # This is linear in V if W is fixed, and vice versa.
    
    # Try: fix U to be a 4x6 matrix with entries in {0, 1/2, 1}, 
    # then solve the resulting system.
    
    # The 2x2x2 tensor T has T[a,b,c]=1 when a=i*2+j, b=j*2+k, c=i*2+k.
    # So T[0,0,0]=1, T[0,1,1]=1, T[1,2,0]=1, T[1,3,1]=1,
    #    T[2,0,2]=1, T[2,1,3]=1, T[3,2,2]=1, T[3,3,3]=1
    
    # Let's try the "sum" approach more carefully.
    # For rank 6, we can use the following 6 products (from literature on border rank):
    # Actually, let me try a direct numerical approach.
    
    # Fix U to have a specific structure, then solve for V and W.
    # Try U where each column is a characteristic vector of a subset.
    
    # Let's try all combinations of 6 subsets of {0,1,2,3} for U columns
    # where each subset has size 1 or 2 (to keep it sparse).
    # This is too many. Instead, let's try specific known patterns.
    
    # Pattern 1: The 6 "edge" sums
    patterns_U = [
        # Each column is a 4-vector with 0/1 entries
        [1,0,1,0],  # {0,2}
        [1,0,0,1],  # {0,3}
        [0,1,0,1],  # {1,3}
        [1,1,0,0],  # {0,1}
        [0,0,1,1],  # {2,3}
        [1,0,0,1],  # {0,3} - duplicate, let's use different
    ]
    # Let me use 6 distinct patterns:
    patterns_U = [
        [1,0,1,0],  # {0,2}
        [1,0,0,1],  # {0,3}
        [0,1,0,1],  # {1,3}
        [1,1,0,0],  # {0,1}
        [0,0,1,1],  # {2,3}
        [0,1,1,0],  # {1,2}
    ]
    
    U6 = np.array(patterns_U, dtype=float).T  # shape (4, 6)
    
    # Now solve for V and W.
    # For each (a,b,c): sum_r U6[a,r]*V6[b,r]*W6[c,r] = T[a,b,c]
    # This is a system of 64 equations in 48 unknowns (V6 and W6 each 4x6).
    # But it's bilinear in V and W.
    
    # Strategy: fix V, solve for W (linear), then fix W, solve for V, iterate.
    # Or: use the fact that for fixed U, we can write it as:
    # For each a: sum_r U6[a,r] * (V6[:,r] * W6[:,r]^T) = T[a,:,:]
    # This is a linear system in the 36 entries of the 6 rank-1 matrices V6[:,r]*W6[:,r]^T
    # but with the constraint that each is rank 1.
    
    # Alternative: use alternating least squares.
    # Initialize V and W randomly, then alternate:
    #   Fix V, solve for W (linear system)
    #   Fix W, solve for V (linear system)
    
    # Initialize
    np.random.seed(42)
    V6 = np.random.randn(4, 6) * 0.5
    W6 = np.random.randn(4, 6) * 0.5
    
    # Target tensor
    T = np.zeros((4, 4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                a = i * 2 + j
                b = j * 2 + k
                c = i * 2 + k
                T[a, b, c] = 1.0
    
    # Alternating least squares
    for it in range(200):
        # Fix V, solve for W
        # For each (a,c): sum_r U6[a,r] * (sum_b T[a,b,c] * V6[b,r]) = ... 
        # Actually: sum_r U6[a,r] * V6[b,r] * W6[c,r] = T[a,b,c]
        # For fixed V: this is linear in W6[c,r]
        # Unknowns: W6[c,r] for c=0..3, r=0..5 → 24 unknowns
        # Equations: 64 (one per (a,b,c))
        A_mat = np.zeros((64, 24))
        b_vec = np.zeros(64)
        eq = 0
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    for r in range(6):
                        A_mat[eq, c * 6 + r] = U6[a, r] * V6[b, r]
                    b_vec[eq] = T[a, b, c]
                    eq += 1
        sol, res, rk, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        W6 = sol.reshape(4, 6)
        
        # Fix W, solve for V
        A_mat = np.zeros((64, 24))
        b_vec = np.zeros(64)
        eq = 0
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    for r in range(6):
                        A_mat[eq, b * 6 + r] = U6[a, r] * W6[c, r]
                    b_vec[eq] = T[a, b, c]
                    eq += 1
        sol, res, rk, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        V6 = sol.reshape(4, 6)
        
        # Check convergence
        err = 0.0
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    val = sum(U6[a, r] * V6[b, r] * W6[c, r] for r in range(6))
                    err += abs(val - T[a, b, c])
        if err < 1e-10:
            break
    
    # Check if solution is valid
    err = 0.0
    for a in range(4):
        for b in range(4):
            for c in range(4):
                val = sum(U6[a, r] * V6[b, r] * W6[c, r] for r in range(6))
                err += abs(val - T[a, b, c])
    if err > 1e-8:
        return None
    
    # Try to round to half-integers
    U6_r = np.round(U6 * 2) / 2
    V6_r = np.round(V6 * 2) / 2
    W6_r = np.round(W6 * 2) / 2
    
    # Verify rounded solution
    err = 0.0
    for a in range(4):
        for b in range(4):
            for c in range(4):
                val = sum(U6_r[a, r] * V6_r[b, r] * W6_r[c, r] for r in range(6))
                err += abs(val - T[a, b, c])
    if err > 1e-9:
        return None
    
    return U6_r, V6_r, W6_r

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill
    start_time = time.time()
    
    # Try to build rank-6 2x2 decomposition
    rank6 = _build_2x2_rank6()
    
    # Check deadline
    if time.time() > deadline - 5:
        return _strassen_rank28(n, m, p)
    
    if rank6 is not None:
        U6, V6, W6 = rank6
        # Total rank: Part1 (2x5 direct) = 10, Part2 (2x1 direct) = 2, Part3 (2x2 rank6) = 6, Part4 (2x2 rank6) = 6
        rank = 10 + 2 + 6 + 6
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        
        # Part 1: A2 B2 (j=2, k=0..4) - 10 terms
        for i in range(n):
            for k in range(p):
                U[i * m + 2][r] = 1
                V[2 * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
        
        # Part 2: A1 B13 (j=0,1, k=4) - 2 terms
        for i in range(n):
            for j in range(2):
                U[i * m + j][r] = 1
                V[j * p + 4][r] = 1
                W[4 * n + i][r] = 1
                r += 1
        
        # Part 3: 2x2 block (j=0,1, k=0,1) using rank-6
        # A local: 0→U[0], 1→U[1], 2→U[3], 3→U[4]
        # B local: 0→V[0], 1→V[1], 2→V[5], 3→V[6]
        # C local: 0→W[0], 1→W[1], 2→W[2], 3→W[3]
        A_map3 = [0, 1, 3, 4]   # local A idx → global U row
        B_map3 = [0, 1, 5, 6]   # local B idx → global V row
        C_map3 = [0, 1, 2, 3]   # local C idx → global W row
        for rr in range(6):
            for a_local in range(4):
                if U6[a_local, rr] != 0:
                    U[A_map3[a_local]][r] += U6[a_local, rr]
            for b_local in range(4):
                if V6[b_local, rr] != 0:
                    V[B_map3[b_local]][r] += V6[b_local, rr]
            for c_local in range(4):
                if W6[c_local, rr] != 0:
                    W[C_map3[c_local]][r] += W6[c_local, rr]
            r += 1
        
        # Part 4: 2x2 block (j=0,1, k=2,3) using rank-6
        # A local: same as Part 3
        # B local: 0→V[2], 1→V[3], 2→V[7], 3→V[8]
        # C local: 0→W[4], 1→W[5], 2→W[6], 3→W[7]
        A_map4 = [0, 1, 3, 4]
        B_map4 = [2, 3, 7, 8]
        C_map4 = [4, 5, 6, 7]
        for rr in range(6):
            for a_local in range(4):
                if U6[a_local, rr] != 0:
                    U[A_map4[a_local]][r] += U6[a_local, rr]
            for b_local in range(4):
                if V6[b_local, rr] != 0:
                    V[B_map4[b_local]][r] += V6[b_local, rr]
            for c_local in range(4):
                if W6[c_local, rr] != 0:
                    W[C_map4[c_local]][r] += W6[c_local, rr]
            r += 1
        
        # Verify full decomposition
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = 0.0
                    for r_idx in range(rank):
                        s += U_arr[idx_u, r_idx] * V_arr[idx_v, r_idx] * W_arr[idx_w, r_idx]
                    if abs(s - 1.0) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            return {"U": U, "V": V, "W": W}
    
    # Fallback: original rank-28 Strassen-based decomposition
    if time.time() > deadline - 2:
        return _strassen_rank28(n, m, p)
    return _strassen_rank28(n, m, p)

def _strassen_rank28(n, m, p):
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    # Part 1: A2 B2 (j=2, k=0..4) - 10 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    # Part 2: A1 B13 (j=0,1, k=4) - 2 terms
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
    # Part 3: Strassen for (j=0,1, k=0,1) - 7 terms
    U[0][r] = 1; U[4][r] = 1; V[0][r] = 1; V[6][r] = 1; W[0][r] = 1; W[3][r] = 1; r += 1
    U[3][r] = 1; U[4][r] = 1; V[0][r] = 1; W[1][r] = 1; W[3][r] = -1; r += 1
    U[0][r] = 1; V[1][r] = 1; V[6][r] = -1; W[2][r] = 1; W[3][r] = 1; r += 1
    U[4][r] = 1; V[5][r] = 1; V[0][r] = -1; W[0][r] = 1; W[1][r] = 1; r += 1
    U[0][r] = 1; U[1][r] = 1; V[6][r] = 1; W[0][r] = -1; W[2][r] = 1; r += 1
    U[3][r] = 1; U[0][r] = -1; V[0][r] = 1; V[1][r] = 1; W[3][r] = 1; r += 1
    U[1][r] = 1; U[4][r] = -1; V[5][r] = 1; V[6][r] = 1; W[0][r] = 1; r += 1
    # Part 4: Strassen for (j=0,1, k=2,3) - 7 terms
    U[0][r] = 1; U[4][r] = 1; V[2][r] = 1; V[8][r] = 1; W[4][r] = 1; W[7][r] = 1; r += 1
    U[3][r] = 1; U[4][r] = 1; V[2][r] = 1; W[5][r] = 1; W[7][r] = -1; r += 1
    U[0][r] = 1; V[3][r] = 1; V[8][r] = -1; W[6][r] = 1; W[7][r] = 1; r += 1
    U[4][r] = 1; V[7][r] = 1; V[2][r] = -1; W[4][r] = 1; W[5][r] = 1; r += 1
    U[0][r] = 1; U[1][r] = 1; V[8][r] = 1; W[4][r] = -1; W[6][r] = 1; r += 1
    U[3][r] = 1; U[0][r] = -1; V[2][r] = 1; V[3][r] = 1; W[7][r] = 1; r += 1
    U[1][r] = 1; U[4][r] = -1; V[7][r] = 1; V[8][r] = 1; W[4][r] = 1; r += 1
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U_arr[i*m+j, ri] * V_arr[j*p+k, ri] * W_arr[k*n+i, ri] for ri in range(rank))
                if abs(s - 1.0) > 1e-9:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    if ok:
        return {"U": U, "V": V, "W": W}
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
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
