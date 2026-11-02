import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    
    # We will solve for M_{n,m,p} directly or split it?
    # Splitting is safer for finding reductions in small blocks.
    # But the tensor is coupled in the index k? No, k is the output index.
    # C_{ik} depends on A_{ij} and B_{jk}.
    # If we fix k, it's a separate problem? No, A is shared.
    # So we cannot simply sum ranks of M_{n,m,k} for each k.
    # Wait.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor T_{i,j,k} = 1 if the indices are valid.
    # The support is all (i,j,k).
    # The block decomposition argument:
    # B = [B_1, B_2].
    # C = A B_1 + A B_2.
    # The term A B_1 produces C_{i, 1..3}.
    # The term A B_2 produces C_{i, 4..7}.
    # The output indices i are shared, but k is disjoint.
    # The tensor for A B_1 has non-zero entries only for k in 1..3.
    # The tensor for A B_2 has non-zero entries only for k in 4..7.
    # Since the supports in the k-dimension are disjoint, the rank is additive.
    # Yes, this is correct.
    
    # So we can solve M_{3,3,3} and M_{3,3,4} separately.
    
    def solve_block(n, m, p_block, time_limit):
        # Trivial rank
        rank_trivial = n * m * p_block
        # Initialize U, V, W with trivial decomposition
        U = np.zeros((n * m, rank_trivial), dtype=np.float64)
        V = np.zeros((m * p_block, rank_trivial), dtype=np.float64)
        W = np.zeros((p_block * n, rank_trivial), dtype=np.float64)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p_block):
                    U[i * m + j, col] = 1
                    V[j * p_block + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        # Target tensor
        T = np.zeros((n, m, p_block))
        for i in range(n):
            for j in range(m):
                for k in range(p_block):
                    T[i, j, k] = 1
        
        best_rank = rank_trivial
        best_U = U.copy()
        best_V = V.copy()
        best_W = W.copy()
        
        # Try to find lower rank
        # We will try ranks from rank_trivial - 1 down to some min
        # Using a simple local search on the factors
        
        # Due to time constraints, we might not find a lower rank.
        # But for 3x3x3, rank 23 is known.
        # We will try to optimize for rank 26, 25, 24, 23.
        
        min_rank_to_try = 20 if p_block == 3 else 30
        if p_block == 4:
            min_rank_to_try = 30
            
        for target_rank in range(rank_trivial - 1, min_rank_to_try - 1, -1):
            if time.time() > deadline - 10:
                break
                
            # Initialize with random projection of the trivial solution?
            # Or just random?
            # Random is better for escaping local minima.
            
            # We use a solver that minimizes || T - sum(U_r V_r W_r) ||^2
            # subject to U,V,W having integer/half-integer entries?
            # No, optimize over reals, then round.
            
            def objective(params):
                # params shape: (rank * (nm + mp + pn))
                r = target_rank
                nm = n * m
                mp = m * p_block
                pn = p_block * n
                size = r * (nm + mp + pn)
                
                if len(params) != size:
                    return 1e10
                    
                U_flat = params[:r * nm]
                V_flat = params[r * nm : r * nm + r * mp]
                W_flat = params[r * nm + r * mp : ]
                
                U_mat = U_flat.reshape(r, nm).T # (nm, r)
                V_mat = V_flat.reshape(r, mp).T # (mp, r)
                W_mat = W_flat.reshape(r, pn).T # (pn, r)
                
                # Compute sum of rank-1 tensors
                # T_approx[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                # This is a tensor contraction.
                # We can compute it efficiently?
                # For small sizes, loop is fine.
                
                T_approx = np.zeros((n, m, p_block))
                for r_idx in range(r):
                    u = U_mat[:, r_idx].reshape(n, m)
                    v = V_mat[:, r_idx].reshape(m, p_block)
                    w = W_mat[:, r_idx].reshape(p_block, n)
                    
                    # Outer product structure
                    # term[i,j,k] = u[i,j] * v[j,k] * w[k,i]
                    for i in range(n):
                        for j in range(m):
                            for k in range(p_block):
                                T_approx[i,j,k] += u[i,j] * v[j,k] * w[k,i]
                                
                diff = T - T_approx
                return np.sum(diff**2)
            
            # Try a few random restarts
            num_restarts = 5
            for _ in range(num_restarts):
                if time.time() > deadline - 10:
                    break
                    
                # Initialize with small noise around 0?
                # Or initialize with the trivial solution projected?
                # Projecting is hard. Random is okay.
                params0 = np.random.randn(target_rank * (n*m + m*p_block + p_block*n)) * 0.1
                
                res = minimize(objective, params0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-4})
                
                if res.fun < 1e-6:
                    # Check if it can be rounded to integer/half-integer
                    r = target_rank
                    nm = n * m
                    mp = m * p_block
                    pn = p_block * n
                    
                    U_flat = res.x[:r * nm]
                    V_flat = res.x[r * nm : r * nm + r * mp]
                    W_flat = res.x[r * nm + r * mp : ]
                    
                    U_mat = U_flat.reshape(r, nm).T
                    V_mat = V_flat.reshape(r, mp).T
                    W_mat = W_flat.reshape(r, pn).T
                    
                    # Round to nearest 0.5
                    U_rounded = np.round(U_mat * 2) / 2
                    V_rounded = np.round(V_mat * 2) / 2
                    W_rounded = np.round(W_mat * 2) / 2
                    
                    # Verify exactness
                    T_check = np.zeros((n, m, p_block))
                    for r_idx in range(r):
                        u = U_rounded[:, r_idx].reshape(n, m)
                        v = V_rounded[:, r_idx].reshape(m, p_block)
                        w = W_rounded[:, r_idx].reshape(p_block, n)
                        for i in range(n):
                            for j in range(m):
                                for k in range(p_block):
                                    T_check[i,j,k] += u[i,j] * v[j,k] * w[k,i]
                                    
                    if np.allclose(T_check, T, atol=1e-9):
                        best_rank = r
                        best_U = U_rounded
                        best_V = V_rounded
                        best_W = W_rounded
                        break # Found a valid lower rank, break restart loop
            
            if best_rank == target_rank:
                break # Found the lowest rank we tried, stop searching lower
        
        # Return the best found
        # If best_rank is still trivial, return trivial
        if best_rank == rank_trivial:
             # Use the initial trivial matrices
             return U, V, W, rank_trivial
        else:
             return best_U, best_V, best_W, best_rank

    # Solve M_{3,3,3}
    n1, m1, p1 = 3, 3, 3
    U1, V1, W1, rank1 = solve_block(n1, m1, p1, 50)
    
    # Solve M_{3,3,4}
    n2, m2, p2 = 3, 3, 4
    U2, V2, W2, rank2 = solve_block(n2, m2, p2, 50)
    
    # Combine
    total_rank = rank1 + rank2
    
    U_final = np.zeros((n1 * m1, total_rank))
    V_final = np.zeros((m1 * (p1 + p2), total_rank))
    W_final = np.zeros(((p1 + p2) * n1, total_rank))
    
    # Copy block 1
    U_final[:, :rank1] = U1
    V_final[:m1*p1, :rank1] = V1
    W_final[:p1*n1, :rank1] = W1
    
    # Copy block 2
    # Note: V and W indices need to be shifted for the second block
    # V2 is for B_2 (cols 4..7). In the full B, these are cols 3..6 (0-indexed).
    # V_final index for B_2: j*p + (k+3)
    # V2 index: j*p2 + k
    # So V_final[j*7 + 3 + k, rank1:r] = V2[j*4 + k, r]
    for j in range(m2):
        for k in range(p2):
            for r in range(rank2):
                V_final[j * (p1 + p2) + p1 + k, rank1 + r] = V2[j * p2 + k, r]
                
    # W2 is for C_2 (rows 4..7). In full C, these are rows 3..6.
    # W_final index: (k+3)*n + i
    # W2 index: k*n2 + i
    for k in range(p2):
        for i in range(n2):
            for r in range(rank2):
                W_final[(k + p1) * n1 + i, rank1 + r] = W2[k * n2 + i, r]
                
    # U is shared? No, U is for A. A is the same.
    # But U1 and U2 are for the same A.
    # Wait, the decomposition is a sum of rank-1 tensors.
    # The first rank1 tensors use U1.
    # The next rank2 tensors use U2.
    # But U1 and U2 are both factors for A.
    # In the sum, we have sum_{r=1}^{R1} U1_r ... + sum_{r=1}^{R2} U2_r ...
    # So U_final should just be [U1, U2] horizontally?
    # Yes, because the rank-1 tensors are added.
    # U_final[:, rank1:rank1+rank2] = U2
    U_final[:, rank1:rank1+rank2] = U2
    
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }

# EVOLVE-BLOCK-END
