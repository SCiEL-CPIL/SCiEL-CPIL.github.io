import time
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [2, 2, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0

    # Schoolbook fallback (rank 28)
    def schoolbook():
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i*m + k][col] = 1
                    V[k*p + j][col] = 1
                    W[j*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    fallback = schoolbook()

    def verify(U_mat, V_mat, W_mat):
        r = U_mat.shape[1]
        u_r = U_mat.T.reshape(n, m, r)
        v_r = V_mat.T.reshape(m, p, r)
        w_r = W_mat.T.reshape(p, n, r)
        T = np.einsum('ikc,kjc,jic->ijk', u_r, v_r, w_r)
        return np.allclose(T, 1.0, atol=1e-8)

    def is_half_integer(mat):
        return np.allclose(mat * 2, np.round(mat * 2), atol=1e-8)

    def to_half_int_list(mat):
        doubled = np.round(mat * 2)
        return doubled.astype(int).tolist()

    # Strategy: Build via Kronecker product structure
    # 2x2x7 = 2x2x(2+2+2+1)
    # Use: A⊗B decomposition where we split the 7 columns into blocks
    # Key insight: Try to find a decomposition where U, V, W are built from
    # Kronecker products of 2x2 structured matrices, reducing the search space.

    # Known: Laderman for 2x2x2 has rank 7
    # We want to exploit the structure: (2x2x2) ⊗ (1x1x1) type constructions
    
    # Alternative: Direct structured search
    # Represent each factor column as a linear combination of basis Kronecker products
    # For 2x2x7, try rank-20, 21, 22, 23 with structured patterns

    # Build Laderman block
    def laderman_block():
        U_b = np.zeros((4, 7))
        V_b = np.zeros((4, 7))
        W_b = np.zeros((4, 7))
        U_b[0,0]=1; U_b[1,0]=-1; U_b[2,0]=-1; U_b[3,0]=1
        V_b[0,0]=1; V_b[1,0]=-1; V_b[2,0]=-1; V_b[3,0]=1
        W_b[0,0]=1; W_b[1,0]=1; W_b[2,0]=1; W_b[3,0]=1
        U_b[0,1]=1; U_b[1,1]=-1
        V_b[1,1]=1; V_b[3,1]=-1
        W_b[1,1]=1; W_b[3,1]=1
        U_b[0,2]=1; U_b[2,2]=-1
        V_b[2,2]=1; V_b[0,2]=-1
        W_b[2,2]=1; W_b[3,2]=1
        U_b[0,3]=1; V_b[0,3]=1; W_b[0,3]=1
        U_b[1,4]=1; V_b[1,4]=1; W_b[1,4]=1
        U_b[2,5]=1; V_b[2,5]=1; W_b[2,5]=1
        U_b[3,6]=1; V_b[3,6]=1; W_b[3,6]=1
        return U_b, V_b, W_b

    def schoolbook_2x2x1():
        U_b = np.zeros((4, 2))
        V_b = np.zeros((2, 2))
        W_b = np.zeros((2, 2))
        U_b[0,0]=1; V_b[0,0]=1; W_b[0,0]=1
        U_b[1,1]=1; V_b[1,1]=1; W_b[1,1]=1
        return U_b, V_b, W_b

    def build_laderman_decomposition():
        rank = 23
        U_full = np.zeros((n*m, rank))
        V_full = np.zeros((m*p, rank))
        W_full = np.zeros((p*n, rank))
        U_l, V_l, W_l = laderman_block()
        U_s, V_s, W_s = schoolbook_2x2x1()
        for block_idx in range(3):
            col_offset = block_idx * 7
            j_start = block_idx * 2
            for c_local in range(7):
                c_global = col_offset + c_local
                for idx in range(4):
                    U_full[idx, c_global] = U_l[idx, c_local]
                for k in range(2):
                    for j_local in range(2):
                        j_global = j_start + j_local
                        V_full[k*p + j_global, c_global] = V_l[k*2 + j_local, c_local]
                for j_local in range(2):
                    for i in range(2):
                        j_global = j_start + j_local
                        W_full[j_global*n + i, c_global] = W_l[j_local*2 + i, c_local]
        col_offset = 21
        for c_local in range(2):
            c_global = col_offset + c_local
            for idx in range(4):
                U_full[idx, c_global] = U_s[idx, c_local]
            for k in range(2):
                V_full[k*p + 6, c_global] = V_s[k, c_local]
            for i in range(2):
                W_full[6*n + i, c_global] = W_s[i, c_local]
        return U_full, V_full, W_full

    U0, V0, W0 = build_laderman_decomposition()
    best_rank = 23
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()

    if not verify(U0, V0, W0):
        return fallback

    # NEW STRATEGY: Structured Kronecker-product-based search
    # For 2x2x7, decompose as:
    #   M(2,2,7) = M(2,2,2) ⊗ M(1,1,1) for the first 2 j-indices
    #   + M(2,2,2) ⊗ M(1,1,1) for next 2
    #   + M(2,2,2) ⊗ M(1,1,1) for next 2  
    #   + M(2,2,1) for last 1
    # But we can try to MERGE columns across blocks using shared structure.
    
    # Key idea: Try rank-20 by finding a 20-column structured decomposition
    # where some columns serve multiple block purposes via coefficient sharing.
    
    # Approach: Parametrize U, V, W as linear combinations of a small set of
    # Kronecker product basis tensors, then solve for coefficients.
    
    # For 2x2x7, the tensor has 28 components. We need rank r such that
    # sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] = T (the multiplication tensor)
    
    # Try: Build from 4x4x4 block structure (rank 18 for 4x4x4 is known? No, min is 18? Actually 4x4x4 min rank is 18 or 20)
    # Actually, let's try a different structural approach:
    # Use the fact that 2x2x7 can be embedded in 2x2x8 = 2x2x(2x4)
    # 2x2x8 has known decompositions.
    
    # Most promising: Exhaustive search over structured patterns for rank 20-22
    # Pattern: U, V, W entries are from {-1, -0.5, 0, 0.5, 1}
    # but with Kronecker product structure: U[i,k,c] = a[i,c] * b[k,c] type
    
    # Let's try: U[i*m+k, c] = U1[i,c] * U2[k,c] (Kronecker in input)
    # V[k*p+j, c] = V1[k,c] * V2[j,c]
    # W[j*n+i, c] = W1[j,c] * W2[i,c]
    
    # This gives: sum_c U1[i,c]*U2[k,c]*V1[k,c]*V2[j,c]*W1[j,c]*W2[i,c]
    # = sum_c (U1[i,c]*W2[i,c]) * (U2[k,c]*V1[k,c]) * (V2[j,c]*W1[j,c])
    # This is a rank-r decomposition where the effective factors are:
    #   U'[i,c] = U1[i,c]*W2[i,c]  (size n x r)
    #   V'[k,c] = U2[k,c]*V1[k,c]  (size m x r)  
    #   W'[j,c] = V2[j,c]*W1[j,c]  (size p x r)
    # But this doesn't directly give us the 2x2x7 structure we need.
    
    # Better: Direct structured search with time budget
    def compute_error(U_mat, V_mat, W_mat):
        r = U_mat.shape[1]
        u_r = U_mat.T.reshape(n, m, r)
        v_r = V_mat.T.reshape(m, p, r)
        w_r = W_mat.T.reshape(p, n, r)
        T = np.einsum('ikc,kjc,jic->ijk', u_r, v_r, w_r)
        return np.max(np.abs(T - 1.0))

    # Try structured rank reduction: start from rank-23, try to find rank-20,21,22
    # by solving linear systems with half-integer constraints
    # Use: For each target rank, try random structured initializations and refine
    
    rng = np.random.default_rng(12345)
    
    # More efficient: Use the fact that we only need to search a smaller space
    # by exploiting symmetry and structure
    
    # Try rank 20, 21, 22 with structured random search
    for target_rank in [20, 21, 22]:
        if time.time() > deadline - 10:
            break
            
        # Structured initialization: use Kronecker-like patterns
        best_err = float('inf')
        best_U_t = None
        best_V_t = None
        best_W_t = None
        
        n_tries = 200
        for attempt in range(n_tries):
            if time.time() > deadline - 5:
                break
                
            # Initialize with structured pattern
            # U: 4 x target_rank, V: 14 x target_rank, W: 14 x target_rank
            # Use block structure: first 3 blocks of 7, last block of 2 (for rank 23)
            # For lower rank, mix the blocks
            
            U_t = np.zeros((n*m, target_rank))
            V_t = np.zeros((m*p, target_rank))
            W_t = np.zeros((p*n, target_rank))
            
            # Random structured init: each column is a random combination
            for c in range(target_rank):
                # U column: random half-integers with structure
                U_t[:, c] = rng.choice([-1, -0.5, 0, 0.5, 1], size=(n*m,))
                V_t[:, c] = rng.choice([-1, -0.5, 0, 0.5, 1], size=(m*p,))
                W_t[:, c] = rng.choice([-1, -0.5, 0, 0.5, 1], size=(p*n,))
            
            err = compute_error(U_t, V_t, W_t)
            if err < best_err:
                best_err = err
                best_U_t = U_t.copy()
                best_V_t = V_t.copy()
                best_W_t = W_t.copy()
            
            if err < 1e-8:
                break
        
        if best_U_t is not None:
            # Local refinement with structured moves
            U_t = best_U_t.copy()
            V_t = best_V_t.copy()
            W_t = best_W_t.copy()
            err = best_err
            
            max_iters = 5000
            for it in range(max_iters):
                if time.time() > deadline - 3:
                    break
                    
                # Random structured move: perturb a whole row or column
                move_type = rng.randint(3)
                if move_type == 0:
                    row = rng.randint(U_t.shape[0])
                    col = rng.randint(U_t.shape[1])
                    old = U_t[row, col]
                    U_t[row, col] = old + rng.choice([-1, -0.5, 0.5, 1])
                    new_err = compute_error(U_t, V_t, W_t)
                    if new_err < err:
                        err = new_err
                        if err < 1e-8:
                            break
                    else:
                        U_t[row, col] = old
                elif move_type == 1:
                    row = rng.randint(V_t.shape[0])
                    col = rng.randint(V_t.shape[1])
                    old = V_t[row, col]
                    V_t[row, col] = old + rng.choice([-1, -0.5, 0.5, 1])
                    new_err = compute_error(U_t, V_t, W_t)
                    if new_err < err:
                        err = new_err
                        if err < 1e-8:
                            break
                    else:
                        V_t[row, col] = old
                else:
                    row = rng.randint(W_t.shape[0])
                    col = rng.randint(W_t.shape[1])
                    old = W_t[row, col]
                    W_t[row, col] = old + rng.choice([-1, -0.5, 0.5, 1])
                    new_err = compute_error(U_t, V_t, W_t)
                    if new_err < err:
                        err = new_err
                        if err < 1e-8:
                            break
                    else:
                        W_t[row, col] = old
            
            if err < 1e-8 and verify(U_t, V_t, W_t) and is_half_integer(U_t) and is_half_integer(V_t) and is_half_integer(W_t):
                best_rank = target_rank
                best_U, best_V, best_W = U_t.copy(), V_t.copy(), W_t.copy()
                print(f"Found rank {target_rank}")
                break  # Found lowest possible in this range
    
    # Final check
    if verify(best_U, best_V, best_W) and is_half_integer(best_U) and is_half_integer(best_V) and is_half_integer(best_W):
        return {
            "U": to_half_int_list(best_U),
            "V": to_half_int_list(best_V),
            "W": to_half_int_list(best_W)
        }
    
    return fallback

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
