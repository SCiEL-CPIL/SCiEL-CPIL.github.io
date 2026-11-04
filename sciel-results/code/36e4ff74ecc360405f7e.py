import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Rank-28 baseline
    def get_rank_28():
        rank = 28
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def strassen_block(i1, i2, j1, j2, k1, k2):
            nonlocal col
            # M1 = (A00+A11)(B00+B11)
            U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
            V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
            col += 1
            # M2 = (A10+A11)B00
            U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
            V[j1*p+k1, col] = 1
            W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
            col += 1
            # M3 = A00(B01-B11)
            U[i1*m+j1, col] = 1
            V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
            W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
            col += 1
            # M4 = A11(B10-B00)
            U[i2*m+j2, col] = 1
            V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
            W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
            col += 1
            # M5 = (A00+A01)B11
            U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
            V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
            col += 1
            # M6 = (A10-A00)(B00+B01)
            U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
            V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
            W[k2*n+i2, col] = 1
            col += 1
            # M7 = (A01-A11)(B10+B11)
            U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
            V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = 1
            col += 1

        def schoolbook_block(i_list, j_list, k_list):
            nonlocal col
            for i in i_list:
                for j in j_list:
                    for k in k_list:
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1

        strassen_block(0, 1, 0, 1, 0, 1)
        strassen_block(0, 1, 0, 1, 2, 3)
        schoolbook_block([0, 1], [0, 1], [4])
        schoolbook_block([0, 1], [2], [0, 1])
        schoolbook_block([0, 1], [2], [2, 3])
        schoolbook_block([0, 1], [2], [4])
        
        return U, V, W, rank

    U28, V28, W28, rank28 = get_rank_28()

    # Try to find rank 27
    def try_rank_27():
        if time.time() > deadline - 10:
            return None
        
        rank = 27
        # Initialize with random values or perturbed 28
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Normalize to prevent overflow
        for r in range(rank):
            U[:, r] /= np.linalg.norm(U[:, r]) + 1e-9
            V[:, r] /= np.linalg.norm(V[:, r]) + 1e-9
            W[:, r] /= np.linalg.norm(W[:, r]) + 1e-9

        # Target tensor T[i,j,k] = 1
        T = np.ones((n, m, p))
        
        best_error = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Simple local search / gradient descent steps
        # We minimize E = || T - sum_r U_r V_r W_r ||^2
        # This is non-convex.
        
        # Use a few restarts
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            if restart == 0:
                U = np.random.randn(n*m, rank)
                V = np.random.randn(m*p, rank)
                W = np.random.randn(p*n, rank)
            else:
                # Perturb best solution
                U = best_U + np.random.randn(n*m, rank) * 0.1
                V = best_V + np.random.randn(m*p, rank) * 0.1
                W = best_W + np.random.randn(p*n, rank) * 0.1
            
            for r in range(rank):
                U[:, r] /= np.linalg.norm(U[:, r]) + 1e-9
                V[:, r] /= np.linalg.norm(V[:, r]) + 1e-9
                W[:, r] /= np.linalg.norm(W[:, r]) + 1e-9

            # Gradient descent
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Compute reconstruction
                # Reshape U, V, W to (n, m, rank), (m, p, rank), (p, n, rank)
                U_r = U.reshape(n, m, rank)
                V_r = V.reshape(m, p, rank)
                W_r = W.reshape(p, n, rank)
                
                # C[i,k] = sum_j sum_r U_r[i,j] V_r[j,k] W_r[k,i]
                # This is efficient to compute?
                # C = sum_r (U_r[:,:,r] * W_r[:,:,r]) @ V_r[:,:,r] ? No.
                # C[i,k] = sum_j sum_r U_r[i,j,r] V_r[j,k,r] W_r[k,i,r]
                
                # Let's compute error and gradients numerically or analytically?
                # Analytical gradient:
                # dE/du[i,j,r] = -2 * sum_k T[i,j,k] V[j,k,r] W[k,i,r] + 2 * sum_k C[i,k] V[j,k,r] W[k,i,r]
                
                # Compute C
                C = np.zeros((n, p))
                for r in range(rank):
                    # C[i,k] += sum_j U_r[i,j] V_r[j,k] W_r[k,i]
                    # This loop is slow in Python. Use numpy.
                    # U_r[:,:,r] is (n,m). V_r[:,:,r] is (m,p). W_r[:,:,r] is (p,n).
                    # Term_r = (U_r[:,:,r] * W_r[:,:,r].T) @ V_r[:,:,r] ?
                    # Let's check indices.
                    # (U_r[:,:,r] * W_r[:,:,r].T) is (n, p). Element [i,k] is U_r[i,k] * W_r[k,i].
                    # Then multiply by V_r[:,:,r] (m, p)? No, dimensions don't match for element-wise.
                    # We need sum_j U_r[i,j] V_r[j,k] W_r[k,i].
                    # Let M_r = W_r[:,:,r] (p, n).
                    # Let A_r = U_r[:,:,r] (n, m).
                    # Let B_r = V_r[:,:,r] (m, p).
                    # C_r[i,k] = sum_j A_r[i,j] B_r[j,k] * M_r[k,i].
                    # This is not a standard matmul.
                    # We can compute it as:
                    # C_r = (A_r @ B_r) * M_r.T
                    # (A_r @ B_r) is (n, p). Element [i,k] is sum_j A_r[i,j] B_r[j,k].
                    # M_r.T is (n, p). Element [i,k] is M_r[k,i].
                    # So yes, C_r = (A_r @ B_r) * M_r.T
                    C += (U_r[:,:,r] @ V_r[:,:,r]) * W_r[:,:,r].T
                
                Error = C - T[:,:,0] # T is constant 1, shape (n,m,p) but here we just need the sum over j?
                # Wait, T[i,j,k] = 1. The tensor decomposition is for the multiplication tensor.
                # The "tensor" T has entries T[i,j,k] = 1.
                # The rank-1 term is U_r[i,j] V_r[j,k] W_r[k,i].
                # So we are decomposing the tensor T[i,j,k] = 1.
                # My C calculation above computes C[i,k] = sum_j sum_r ...
                # This is the matrix multiplication result, not the tensor decomposition error.
                # The task is to decompose the TENSOR T[i,j,k] = 1.
                # So Error[i,j,k] = 1 - sum_r U_r[i,j] V_r[j,k] W_r[k,i].
                
                # This requires storing the full (n, m, p) tensor error.
                # n=2, m=3, p=5 -> 30 elements.
                # rank=27.
                # Loop over r:
                #   Update Error[i,j,k] -= U_r[i,j] V_r[j,k] W_r[k,i]
                
                Error = np.ones((n, m, p))
                for r in range(rank):
                    # U_r[i,j] V_r[j,k] W_r[k,i]
                    # This is an outer product with a twist.
                    # We can compute it as:
                    # Temp[i,k] = sum_j U_r[i,j] V_r[j,k]
                    # Then Error[i,j,k] -= U_r[i,j] * W_r[k,i] * (V_r[j,k] / U_r[i,j] ?? No)
                    
                    # Let's just loop i, j, k since 30 is small.
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                Error[i,j,k] -= U_r[i,j] * V_r[j,k] * W_r[k,i]
                
                err_val = np.sum(Error**2)
                if err_val < best_error:
                    best_error = err_val
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_error < 1e-6:
                        return best_U, best_V, best_W, rank
                
                # Gradient calculation
                # dE/du[i,j,r] = -2 * sum_k Error[i,j,k] * V_r[j,k] * W_r[k,i]
                # dE/dv[j,k,r] = -2 * sum_i Error[i,j,k] * U_r[i,j] * W_r[k,i]
                # dE/dw[k,i,r] = -2 * sum_j Error[i,j,k] * U_r[i,j] * V_r[j,k]
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for r in range(rank):
                    for i in range(n):
                        for j in range(m):
                            dU[i,j,r] = -2 * np.sum(Error[i,j,:] * V_r[j,:,r] * W_r[:,i,r])
                            for k in range(p):
                                dV[j,k,r] -= 2 * Error[i,j,k] * U_r[i,j,r] * W_r[k,i,r]
                                dW[k,i,r] -= 2 * Error[i,j,k] * U_r[i,j,r] * V_r[j,k,r]
                    
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Renormalize
                for r in range(rank):
                    U[:, r] /= np.linalg.norm(U[:, r]) + 1e-9
                    V[:, r] /= np.linalg.norm(V[:, r]) + 1e-9
                    W[:, r] /= np.linalg.norm(W[:, r]) + 1e-9

        if best_error < 1e-4:
            return best_U, best_V, best_W, rank
        return None

    result_27 = try_rank_27()
    if result_27:
        U, V, W, rank = result_27
        # Verify
        T = np.ones((n, m, p))
        Recon = np.zeros((n, m, p))
        for r in range(rank):
            U_r = U[:, r].reshape(n, m)
            V_r = V[:, r].reshape(m, p)
            W_r = W[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        Recon[i,j,k] += U_r[i,j] * V_r[j,k] * W_r[k,i]
        if np.allclose(Recon, T, atol=1e-5):
            # Snap to integers if close
            for mat in [U, V, W]:
                for i in range(mat.shape[0]):
                    for j in range(mat.shape[1]):
                        if abs(mat[i,j] - round(mat[i,j])) < 1e-5:
                            mat[i,j] = round(mat[i,j])
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to rank 28
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
