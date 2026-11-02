"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 2, 4
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 iff j_a == j_b, i_a == i_c, k_b == k_c
    # a = i*m + j, b = j*p + k, c = k*n + i
    nm = n * m  # 4
    mp = m * p  # 8
    pn = p * n  # 8
    
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try to find a low-rank decomposition
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # We'll try ranks from 13 downward... actually let's try to find the minimum
    # Start with a known good rank and try to reduce
    
    # Let's try rank 13 first, as it's a known result for similar problems
    # Actually, let's try to solve this computationally
    
    # For a given rank r, we have variables U (nm x r), V (mp x r), W (pn x r)
    # and constraints T[a,b,c] = sum_{r'} U[a,r'] V[b,r'] W[c,r']
    
    # This is a system of nonlinear equations. We can try to solve it.
    
    # One approach: fix U and V, solve for W, or use iterative methods.
    
    # Let's try a simpler approach: use the fact that for small dimensions,
    # we can try random initializations and use least squares.
    
    best_rank = None
    best_decomp = None
    
    # Try ranks from 13 to 16
    for rank in range(13, 17):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        found = False
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + trial)
            
            # Initialize U, V, W randomly
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Iterative refinement
            for it in range(200):
                if time.time() > deadline - 10:
                    break
                
                # For fixed U and V, solve for W:
                # T[a,b,c] = sum_r' U[a,r'] V[b,r'] W[c,r']
                # For each c, this is a linear system in W[:,c]... 
                # Actually, let's reshape: for each c, T[:, :, c] = sum_r' (U[:, r'] * V[:, r']) * W[c, r']
                # So T[:, :, c] = (U * V^T) @ W[c, :]^T ... not quite.
                
                # Let me think differently. For each c:
                # T[a,b,c] for all a,b = sum_r' U[a,r'] V[b,r'] W[c,r']
                # This is: T[:, :, c] = sum_r' W[c,r'] * (U[:,r'] outer V[:,r'])
                # So T[:, :, c] is a linear combination of the matrices U[:,r'] outer V[:,r']
                # with coefficients W[c,r']
                
                # For each c, we have a linear system:
                # vec(T[:, :, c]) = sum_r' W[c,r'] * vec(U[:,r'] outer V[:,r'])
                # Let M_c be the matrix with columns vec(U[:,r'] outer V[:,r']) for r'=0..rank-1
                # Then vec(T[:, :, c]) = M_c @ W[c, :]
                
                M_c = np.zeros((nm * mp, rank))
                for r_prime in range(rank):
                    M_c[:, r_prime] = np.outer(U[:, r_prime], V[:, r_prime]).flatten()
                
                # Solve for W
                T_flat = T.reshape(nm * mp, pn)  # T[a,b,c] -> T_flat[a*mp+b, c]
                
                # For each c, solve M_c @ W[c,:] = T_flat[:,c]
                W_new = np.zeros((pn, rank))
                for c in range(pn):
                    # Least squares solve
                    try:
                        W_new[c, :] = np.linalg.lstsq(M_c, T_flat[:, c], rcond=None)[0]
                    except:
                        W_new[c, :] = W[c, :]
                
                W = W_new
                
                # Similarly update U and V
                # For fixed V and W: T[a,b,c] = sum_r' U[a,r'] * (V[b,r'] W[c,r'])
                # For each a: T[a,:,:] = sum_r' U[a,r'] * (V[:,r'] outer W[:,r'])
                # vec(T[a,:,:]) = sum_r' U[a,r'] * vec(V[:,r'] outer W[:,r'])
                
                M_a = np.zeros((mp * pn, rank))
                for r_prime in range(rank):
                    M_a[:, r_prime] = np.outer(V[:, r_prime], W[:, r_prime]).flatten()
                
                T_flat_a = T.reshape(nm, mp * pn)  # T[a,b,c] -> T_flat_a[a, b*pn+c]
                
                U_new = np.zeros((nm, rank))
                for a in range(nm):
                    try:
                        U_new[a, :] = np.linalg.lstsq(M_a, T_flat_a[a, :], rcond=None)[0]
                    except:
                        U_new[a, :] = U[a, :]
                
                U = U_new
                
                # For fixed U and W: T[a,b,c] = sum_r' V[b,r'] * (U[a,r'] W[c,r'])
                # For each b: T[:,b,:] = sum_r' V[b,r'] * (U[:,r'] outer W[:,r'])
                # vec(T[:,b,:]) = sum_r' V[b,r'] * vec(U[:,r'] outer W[:,r'])
                
                M_b = np.zeros((nm * pn, rank))
                for r_prime in range(rank):
                    M_b[:, r_prime] = np.outer(U[:, r_prime], W[:, r_prime]).flatten()
                
                T_flat_b = T.reshape(nm, mp, pn).transpose(0, 2, 1).reshape(nm * pn, mp)
                # T_flat_b[a*pn+c, b] = T[a,b,c]
                
                V_new = np.zeros((mp, rank))
                for b in range(mp):
                    try:
                        V_new[b, :] = np.linalg.lstsq(M_b, T_flat_b[:, b], rcond=None)[0]
                    except:
                        V_new[b, :] = V[b, :]
                
                V = V_new
                
                # Check convergence
                error = 0.0
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            error += (val - T[a, b, c]) ** 2
                
                if error < 1e-10:
                    found = True
                    break
            
            if found:
                best_rank = rank
                best_decomp = (U, V, W)
                break
        
        if best_rank is not None:
            break
    
    # If we found a decomposition, return it
    if best_rank is not None:
        U, V, W = best_decomp
        # Convert to lists
        U_list = [[U[a, r] for r in range(best_rank)] for a in range(nm)]
        V_list = [[V[b, r] for r in range(best_rank)] for b in range(mp)]
        W_list = [[W[c, r] for r in range(best_rank)] for c in range(pn)]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
