"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target tensor for 2x3x6
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W, T, tol=1e-9):
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_rec, T, atol=tol)

    def to_lists(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try rank-21 construction: use 2x3 rank-7 algorithm for each pair of output columns
    # The 2x3x2 rank-7 algorithm applied to 3 column-pairs
    # For 2x3 x 3x2, one rank-7 decomposition:
    # We'll use a known construction
    
    # Known 2x3x2 rank-7 algorithm (Strassen-style):
    # A = [[a0,a1,a2],[a3,a4,a5]], B = [[b0,b1],[b2,b3],[b4,b5]]
    # C = A*B is 2x2
    # 
    # w1 = a0*b0,  w2 = a1*b1,  w3 = a2*b2
    # w4 = a3*b3,  w5 = a4*b4,  w6 = a5*b5
    # w7 = (a0+a1+a2+a3+a4+a5)*(b0+b1+b2+b3+b4+b5) - w1-w2-w3-w4-w5-w6
    #
    # But we need to express C entries as sums of w's
    # C[0,0] = a0*b0 + a1*b2 + a2*b4
    # C[0,1] = a0*b1 + a1*b3 + a2*b5
    # C[1,0] = a3*b0 + a4*b2 + a5*b4
    # C[1,1] = a3*b1 + a4*b3 + a5*b5
    #
    # The above w's don't directly give C entries. Let me use a different known algorithm.
    #
    # Actually, let me just try to find a valid rank-21 decomposition by block construction.
    # Treat B as [B1, B2, B3] where each Bj is 3x2.
    # C = [A*B1, A*B2, A*B3]
    # Each A*Bj uses rank 7, total rank 21.
    
    # For the 2x3x2 rank-7, here's a valid construction:
    # Using the algorithm from "Algorithms for Matrix Multiplication" 
    # Let me use a direct rank-7 for 2x3x2:
    #
    # The 7 bilinear forms:
    # m1 = a0*b0
    # m2 = a1*b2  
    # m3 = a2*b4
    # m4 = a3*b1
    # m5 = a4*b3
    # m6 = a5*b5
    # m7 = (a0+a1+a2+a3+a4+a5)*(b0+b1+b2+b3+b4+b5) - m1-m2-m3-m4-m5-m6
    #
    # This doesn't work cleanly. Let me try a numerical approach instead.
    
    # Numerical search for low-rank decomposition
    best_rank = n * m * p
    best_result = None
    
    # Fallback: Schoolbook rank 36
    rank = best_rank
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    if verify(U, V, W, T):
        best_result = to_lists(U, V, W)
    
    # Try to find lower rank via random initialization + optimization
    # For each candidate rank, try to find U,V,W
    for target_rank in [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(50):
            if time.time() > deadline - 15:
                break
            
            # Random initialization
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5
            
            # Gradient-free local search: try to reduce error
            # Use a simple iterative approach
            for it in range(200):
                if time.time() > deadline - 10:
                    break
                
                # Compute current tensor
                T_rec = np.einsum('ar,br,cr->abc', U, V, W)
                err = T_rec - T
                err_norm = np.linalg.norm(err)
                
                if err_norm < 1e-8:
                    # Found valid decomposition!
                    # Try to clean up to integers/half-integers
                    found = True
                    break
                
                # Gradient descent on U, V, W
                # dL/dU[a,r] = sum_{b,c} err[a,b,c] * V[b,r] * W[c,r]
                # dL/dV[b,r] = sum_{a,c} err[a,b,c] * U[a,r] * W[c,r]
                # dL/dW[c,r] = sum_{a,b} err[a,b,c] * U[a,r] * V[b,r]
                
                lr = 0.01
                dU = np.einsum('abc,br,cr->ar', err, V, W)
                dV = np.einsum('abc,ar,cr->br', err, U, W)
                dW = np.einsum('abc,ar,br->cr', err, U, V)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            if found:
                break
        
        if found:
            # Clean up: try to snap to nearest 0.5 values
            U_clean = np.round(U * 2) / 2
            V_clean = np.round(V * 2) / 2
            W_clean = np.round(W * 2) / 2
            
            if verify(U_clean, V_clean, W_clean, T, tol=1e-6):
                best_rank = target_rank
                best_result = to_lists(U_clean, V_clean, W_clean)
                break
            elif verify(U, V, W, T, tol=1e-6):
                best_rank = target_rank
                best_result = to_lists(U, V, W)
                break
    
    if best_result is not None:
        return best_result
    
    # Final fallback
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
