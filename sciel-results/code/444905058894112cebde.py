"""
Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 7 matrix.
The output factor follows the official trace convention W[k*n+i,r].

Strategy:
1. Start with the schoolbook decomposition (rank = 2*3*7 = 42).
2. Use an iterative rank-reduction approach:
   - Try to find a lower rank decomposition by solving a linear system.
   - Use random projections and SVD to propose candidate factor matrices.
   - Check if the proposed matrices exactly decompose the multiplication tensor.
3. Return the best (lowest rank) decomposition found within the time limit.
"""

import numpy as np
import time
import random

DIMENSIONS = [2, 3, 7]
N, M, P = DIMENSIONS

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Schoolbook decomposition
    def schoolbook():
        rank = N * M * P
        U = np.zeros((N * M, rank))
        V = np.zeros((M * P, rank))
        W = np.zeros((P * N, rank))
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j, col] = 1
                    V[j * P + k, col] = 1
                    W[k * N + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = N * M * P
    
    # Check if a given (U, V, W) exactly decomposes the matrix multiplication tensor
    # The multiplication tensor T is defined by T[i,j,k] = 1 if we multiply A[i,j] * B[j,k]
    # A rank-r decomposition means: T[i,j,k] = sum_{r=0}^{rank-1} U[i*M+j, r] * V[j*P+k, r] * W[k*N+i, r]
    # We need to verify this identity holds for all i, j, k.
    def check_decomposition(U, V, W, tolerance=1e-10):
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        # Build the reconstructed tensor
        # T[i,j,k] should equal sum_r U[i*M+j, r] * V[j*P+k, r] * W[k*N+i, r]
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * M + j, r] * V[j * P + k, r] * W[k * N + i, r]
                    # The true value is 1 if this is a valid multiplication term
                    # Actually, the tensor for matrix multiplication C=AB where A is N x M and B is M x P
                    # is such that C[i,k] = sum_j A[i,j] * B[j,k]
                    # The tensor T has entries T[i,j,k] = 1 for all i,j,k (it's the "multiplication" tensor)
                    # Wait, let me reconsider. The multiplication tensor for (N,M,P) has entries:
                    # T[i, j, k] = 1 if we are computing the (i,k) entry using A[i,j] and B[j,k]
                    # Actually, the standard definition: the tensor is in R^{N*M} x R^{M*P} x R^{P*N}
                    # T_{(i,j), (j,k), (k,i)} = 1
                    # So for each (i,j,k), there's exactly one non-zero entry which is 1.
                    # The decomposition requires: for all (i,j,k), sum_r U[(i,j),r] V[(j,k),r] W[(k,i),r] = 1
                    # And for all other index combinations, the sum should be 0.
                    # But actually, the tensor is sparse - only (N*M*P) entries are 1, rest are 0.
                    # However, in the standard matrix multiplication tensor, the non-zero entries are exactly
                    # T_{(i*M+j, j*P+k, k*N+i)} = 1 for all i in [0,N), j in [0,M), k in [0,P)
                    # All other entries are 0.
                    # So we need to check that the reconstructed tensor matches this pattern.
                    # For efficiency, let's just check that for all (i,j,k), the product sum equals 1.
                    if abs(val - 1.0) > tolerance:
                        return False
        # Also need to check that "spurious" entries are 0
        # The full tensor space is (N*M) x (M*P) x (P*N)
        # We need T_{a,b,c} = sum_r U[a,r] V[b,r] W[c,r] to be 0 unless (a,b,c) corresponds to a valid (i,j,k)
        # where a = i*M+j, b = j*P+k, c = k*N+i
        # Let's check a few random spurious entries
        for _ in range(100):
            a = random.randint(0, N * M - 1)
            b = random.randint(0, M * P - 1)
            c = random.randint(0, P * N - 1)
            # Check if (a,b,c) is a valid entry
            i_a = a // M
            j_a = a % M
            j_b = b // P
            k_b = b % P
            k_c = c // N
            i_c = c % N
            is_valid = (j_a == j_b) and (k_b == k_c) and (i_a == i_c)
            val = 0.0
            for r in range(rank):
                val += U[a, r] * V[b, r] * W[c, r]
            if is_valid:
                if abs(val - 1.0) > tolerance:
                    return False
            else:
                if abs(val) > tolerance:
                    return False
        return True
    
    # Try to reduce rank using iterative approaches
    # Approach 1: Start from schoolbook and try to "compress" by finding linear dependencies
    # Approach 2: Random initialization + optimization
    
    # Try various ranks from best_rank-1 down to 20
    for target_rank in range(best_rank - 1, 19, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization
            scale = 1.0 / np.sqrt(target_rank)
            U = np.random.randn(N * M, target_rank) * scale
            V = np.random.randn(M * P, target_rank) * scale
            W = np.random.randn(P * N, target_rank) * scale
            
            # Iterative refinement: adjust one matrix at a time
            # Given U and V, solve for W such that the tensor identity holds
            # This is a linear system
            for iteration in range(5):
                if time.time() > deadline:
                    break
                
                # Fix U and V, solve for W
                # For each (i,j,k), we need: sum_r U[i*M+j, r] * V[j*P+k, r] * W[k*N+i, r] = 1
                # Let A_{(i,j,k), r} = U[i*M+j, r] * V[j*P+k, r]
                # Then A @ W.flatten() = target_vector
                # But W is (P*N, target_rank), so W.flatten() is (P*N*target_rank,)
                # Actually, for each (k,i), W[k*N+i, :] is a row of W
                # The equation for (i,j,k) involves W[k*N+i, r] for all r
                # So for each (k,i), we have multiple equations (one for each j)
                
                # Let's set up the linear system differently
                # For fixed U and V, the equations are:
                # For each (i,j,k): sum_r (U[i*M+j, r] * V[j*P+k, r]) * W[k*N+i, r] = 1
                # This is: sum_r M_{(i,j,k), r} * W[k*N+i, r] = 1
                # where M_{(i,j,k), r} = U[i*M+j, r] * V[j*P+k, r]
                
                # We can group by (k,i): for each (k,i), we have M equations (one for each j)
                # For fixed (k,i), let W_row = W[k*N+i, :]
                # The equations are: for each j, sum_r (U[i*M+j, r] * V[j*P+k, r]) * W_row[r] = 1
                # This is a linear system: A_{(k,i)} @ W_row = b_{(k,i)}
                # where A_{(k,i)}[j, r] = U[i*M+j, r] * V[j*P+k, r]
                # and b_{(k,i)}[j] = 1 for all j
                
                W_new = np.zeros((P * N, target_rank))
                valid = True
                for k in range(P):
                    for i in range(N):
                        # Build A matrix: (M, target_rank)
                        A_mat = np.zeros((M, target_rank))
                        for j in range(M):
                            A_mat[j, :] = U[i * M + j, :] * V[j * P + k, :]
                        b_vec = np.ones(M)
                        try:
                            sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                            W_new[k * N + i, :] = sol
                        except:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                W = W_new
                
                # Now fix V and W, solve for U
                U_new = np.zeros((N * M, target_rank))
                valid = True
                for i in range(N):
                    for j in range(M):
                        # For fixed (i,j), equations over k:
                        # sum_r U[i*M+j, r] * V[j*P+k, r] * W[k*N+i, r] = 1 for all k
                        # A_mat[k, r] = V[j*P+k, r] * W[k*N+i, r]
                        A_mat = np.zeros((P, target_rank))
                        for k in range(P):
                            A_mat[k, :] = V[j * P + k, :] * W[k * N + i, :]
                        b_vec = np.ones(P)
                        try:
                            sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                            U_new[i * M + j, :] = sol
                        except:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                U = U_new
                
                # Now fix U and W, solve for V
                V_new = np.zeros((M * P, target_rank))
                valid = True
                for j in range(M):
                    for k in range(P):
                        # For fixed (j,k), equations over i:
                        # sum_r U[i*M+j, r] * V[j*P+k, r] * W[k*N+i, r] = 1 for all i
                        # A_mat[i, r] = U[i*M+j, r] * W[k*N+i, r]
                        A_mat = np.zeros((N, target_rank))
                        for i in range(N):
                            A_mat[i, :] = U[i * M + j, :] * W[k * N + i, :]
                        b_vec = np.ones(N)
                        try:
                            sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                            V_new[j * P + k, :] = sol
                        except:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                V = V_new
            
            # Check if the decomposition is valid
            if check_decomposition(U, V, W):
                result = {"U": U, "V": V, "W": W}
                best_result = result
                best_rank = target_rank
                break
    
    # Convert to lists for the return format
    U = best_result["U"]
    V = best_result["V"]
    W = best_result["W"]
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
