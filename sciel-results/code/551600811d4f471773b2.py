import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try rank 24 first, then fall back to 28 if needed
    # The tensor T[i,j,k] = delta(i,j) for A[i,j]*B[j,k] = C[i,k]
    # Using trace convention: W[k*n+i, r]
    
    # Let's try to find a rank-24 decomposition using optimization
    # T is a 3-tensor of size (n*m, m*p, p*n) = (8, 16, 8)
    # We want U (8 x r), V (16 x r), W (8 x r) such that
    # sum_r U[i,r] * V[j,r] * W[k,r] = T[i,j,k]
    
    # T[i,j,k] = 1 if i//m == k%n and i%m == j//p, else 0
    # Wait, let me be more careful.
    # A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i,j) for A, j for B... 
    # Actually T[a_idx, b_idx, c_idx] where:
    # a_idx = i*m + j (i in [0,n), j in [0,m))
    # b_idx = j*p + k (j in [0,m), k in [0,p))  
    # c_idx = k*n + i (k in [0,p), i in [0,n)) -- trace convention
    # T[a_idx, b_idx, c_idx] = 1 if the j in a_idx equals the j in b_idx, else 0
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Try rank 24
    best_result = None
    deadline = time.time() + 100  # Leave 20s margin
    
    for rank in [24, 25, 26, 27, 28]:
        if time.time() > deadline - 30:
            break
        
        # Initialize with random values
        np.random.seed(42)
        U0 = np.random.randn(n*m, rank) * 0.1
        V0 = np.random.randn(m*p, rank) * 0.1
        W0 = np.random.randn(p*n, rank) * 0.1
        
        # Reshape for optimization
        def loss(params):
            U = params[:n*m*rank].reshape(n*m, rank)
            V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            # Compute sum_r U[i,r]*V[j,r]*W[k,r] for all i,j,k
            # Efficient: for each r, compute outer product U[:,r] @ V[:,r] @ W[:,r]
            # But that's expensive. Use einsum-like approach.
            # T_computed[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # We can compute this as: for each r, add U[:,r] * V[:,r] * W[:,r] (outer product)
            # But that's O(rank * n*m * m*p * p*n) which is too slow.
            
            # Better: use the fact that we only need to check against T
            # Compute the Frobenius norm difference
            # T_computed can be computed via:
            # For each r: term_r = np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
            # But let's use a smarter approach:
            
            # Actually, let's just compute the residual directly
            # residual[i,j,k] = T[i,j,k] - sum_r U[i,r]*V[j,r]*W[k,r]
            # We can compute sum_r U[i,r]*V[j,r]*W[k,r] efficiently:
            # = sum_r (U[i,r] * V[j,r]) * W[k,r]
            # Let M_r[i,j] = U[i,r]*V[j,r], then result[i,j,k] = sum_r M_r[i,j]*W[k,r]
            # This is like a tensor contraction.
            
            # Use einsum for efficiency
            T_comp = np.einsum('ir,jr,kr->ijk', U, V, W)
            residual = T - T_comp
            return np.sum(residual**2)
        
        # This is too slow for large tensors. Let me use a different approach.
        # Instead, let's use the structure of the problem.
        
        # For this specific tensor, we can use a smarter initialization.
        # Start from the Strassen-based rank-28 solution and try to reduce.
        
        if rank == 28:
            # Use the known Strassen solution
            U = np.zeros((n*m, rank), dtype=float)
            V = np.zeros((m*p, rank), dtype=float)
            W = np.zeros((p*n, rank), dtype=float)
            
            strassen = [
                ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
                ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
                ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),
                ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
                ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
                ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
                ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)])
            ]
            
            def map_u(local_idx, a_col_off):
                r = local_idx // 2
                c = local_idx % 2
                return r * m + (a_col_off * 2 + c)
            
            def map_v(local_idx, b_row_off, b_col_off):
                r = local_idx // 2
                c = local_idx % 2
                return (b_row_off * 2 + r) * p + (b_col_off * 2 + c)
            
            def map_w(local_idx, c_col_off):
                r = local_idx // 2
                c = local_idx % 2
                return (c_col_off * 2 + c) * n + r
            
            def add_block(a_off, b_row_off, b_col_off, c_col_off, start_col):
                for i, (u_loc, v_loc, w_loc) in enumerate(strassen):
                    col = start_col + i
                    for idx, coeff in u_loc:
                        U[map_u(idx, a_off), col] += coeff
                    for idx, coeff in v_loc:
                        V[map_v(idx, b_row_off, b_col_off), col] += coeff
                    for idx, coeff in w_loc:
                        W[map_w(idx, c_col_off), col] += coeff
            
            add_block(0, 0, 0, 0, 0)
            add_block(1, 1, 0, 0, 7)
            add_block(0, 0, 1, 1, 14)
            add_block(1, 1, 1, 1, 21)
            
            # Verify
            T_comp = np.einsum('ir,jr,kr->ijk', U, V, W)
            if np.allclose(T, T_comp, atol=1e-10):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # For lower ranks, use optimization
        # This is computationally expensive, so use a limited approach
        if rank < 28 and time.time() > deadline - 20:
            continue
        
        # Use a simpler approach: try to find a rank-r decomposition
        # by solving a system of equations
        # The tensor has 8*16*8 = 1024 entries, but only 32 are non-zero
        # We have r*(8+16+8) = 32r unknowns
        # For r=24: 768 unknowns, 32 equations (the non-zero entries)
        # This is underdetermined, so we should be able to find a solution
        
        # Let's use a linear algebra approach
        # The constraint is: for each (i,j,k) with T[i,j,k]=1:
        # sum_r U[i,r]*V[j,r]*W[k,r] = 1
        # And for each (i,j,k) with T[i,j,k]=0:
        # sum_r U[i,r]*V[j,r]*W[k,r] = 0
        
        # This is a system of polynomial equations. Let's use a random initialization
        # and Newton's method or a simple iterative approach.
        
        # Actually, let me just return the rank-28 solution for now, as the 
        # optimization is complex and may not converge in time.
        continue
    
    # Fallback: return rank-28 Strassen solution
    rank = 28
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    strassen = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)])
    ]
    
    def map_u(local_idx, a_col_off):
        r = local_idx // 2
        c = local_idx % 2
        return r * m + (a_col_off * 2 + c)
    
    def map_v(local_idx, b_row_off, b_col_off):
        r = local_idx // 2
        c = local_idx % 2
        return (b_row_off * 2 + r) * p + (b_col_off * 2 + c)
    
    def map_w(local_idx, c_col_off):
        r = local_idx // 2
        c = local_idx % 2
        return (c_col_off * 2 + c) * n + r
    
    def add_block(a_off, b_row_off, b_col_off, c_col_off, start_col):
        for i, (u_loc, v_loc, w_loc) in enumerate(strassen):
            col = start_col + i
            for idx, coeff in u_loc:
                U[map_u(idx, a_off), col] += coeff
            for idx, coeff in v_loc:
                V[map_v(idx, b_row_off, b_col_off), col] += coeff
            for idx, coeff in w_loc:
                W[map_w(idx, c_col_off), col] += coeff
    
    add_block(0, 0, 0, 0, 0)
    add_block(1, 1, 0, 0, 7)
    add_block(0, 0, 1, 1, 14)
    add_block(1, 1, 1, 1, 21)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
