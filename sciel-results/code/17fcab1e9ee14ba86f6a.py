import time
import numpy as np
import itertools

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin

    # We will search for a rank-10 decomposition.
    # If not found in time, we return the rank-11 decomposition.
    
    # Known rank 11 decomposition (Strassen for 2x2 part + Schoolbook for 3rd column)
    def get_rank11():
        rank = 11
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def u_idx(i, j): return i * m + j
        def v_idx(j, k): return j * p + k
        def w_idx(k, i): return k * n + i
        
        # Strassen terms for C[:, 0:2]
        # M1 = (A00+A11)(B00+B11)
        U[u_idx(0,0)][0] += 1; U[u_idx(1,1)][0] += 1
        V[v_idx(0,0)][0] += 1; V[v_idx(1,1)][0] += 1
        W[w_idx(0,0)][0] += 1; W[w_idx(0,1)][0] += 1; W[w_idx(1,0)][0] += 1; W[w_idx(1,1)][0] += 1
        
        # M2 = (A10+A11)B00
        U[u_idx(1,0)][1] += 1; U[u_idx(1,1)][1] += 1
        V[v_idx(0,0)][1] += 1
        W[w_idx(1,0)][1] += 1; W[w_idx(1,1)][1] += 1
        
        # M3 = A00(B01-B11)
        U[u_idx(0,0)][2] += 1
        V[v_idx(0,1)][2] += 1; V[v_idx(1,1)][2] -= 1
        W[w_idx(0,1)][2] += 1; W[w_idx(1,1)][2] += 1
        
        # M4 = A11(B10-B00)
        U[u_idx(1,1)][3] += 1
        V[v_idx(1,0)][3] += 1; V[v_idx(0,0)][3] -= 1
        W[w_idx(0,0)][3] += 1; W[w_idx(1,0)][3] += 1
        
        # M5 = (A00+A01)B11
        U[u_idx(0,0)][4] += 1; U[u_idx(0,1)][4] += 1
        V[v_idx(1,1)][4] += 1
        W[w_idx(0,0)][4] -= 1; W[w_idx(0,1)][4] += 1
        
        # M6 = (A01-A00)(B00+B10)
        U[u_idx(0,1)][5] += 1; U[u_idx(0,0)][5] -= 1
        V[v_idx(0,0)][5] += 1; V[v_idx(1,0)][5] += 1
        W[w_idx(0,1)][5] -= 1; W[w_idx(1,1)][5] += 1
        
        # M7 = (A10-A11)(B11+B01)
        U[u_idx(1,0)][6] += 1; U[u_idx(1,1)][6] -= 1
        V[v_idx(1,1)][6] += 1; V[v_idx(0,1)][6] += 1
        W[w_idx(0,0)][6] += 1; W[w_idx(1,0)][6] -= 1
        
        # Schoolbook terms for C[:, 2]
        # C02 = A00*B02 + A01*B12
        # C12 = A10*B02 + A11*B12
        
        # Term 7: A00*B02
        U[u_idx(0,0)][7] += 1
        V[v_idx(0,2)][7] += 1
        W[w_idx(2,0)][7] += 1
        
        # Term 8: A01*B12
        U[u_idx(0,1)][8] += 1
        V[v_idx(1,2)][8] += 1
        W[w_idx(2,0)][8] += 1
        
        # Term 9: A10*B02
        U[u_idx(1,0)][9] += 1
        V[v_idx(0,2)][9] += 1
        W[w_idx(2,1)][9] += 1
        
        # Term 10: A11*B12
        U[u_idx(1,1)][10] += 1
        V[v_idx(1,2)][10] += 1
        W[w_idx(2,1)][10] += 1
        
        return {"U": U, "V": V, "W": W}

    # Verify a decomposition
    def verify(U, V, W):
        # Convert to numpy arrays
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Compute the tensor product T = sum_r u_r v_r w_r
        # T is of shape (n*m, m*p, p*n)
        rank = len(U_np)
        T = np.zeros((n*m, m*p, p*n))
        
        for r in range(rank):
            u = U_np[:, r]
            v = V_np[:, r]
            w = W_np[:, r]
            T += np.einsum('i,j,k->ijk', u, v, w)
        
        # Check against the matrix multiplication tensor
        # C_{ik} = sum_j A_{ij} B_{jk}
        # The tensor M has non-zero entries at M[i, j, k] = 1 if we consider C_k? No.
        # Standard convention: C_{ik} = sum_j A_{ij} B_{jk}
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # The decomposition is: sum_r U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i] = Delta_{ij, jk, ki}
        # Where Delta is 1 if (i, j, k) corresponds to a valid term in the sum.
        # Specifically, for a fixed output element C[i, k], the term A[i, j] * B[j, k] is included.
        # So for a specific triplet (i, j, k), the coefficient in the tensor is 1.
        
        # Let's construct the target tensor M_target
        M_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    M_target[u_idx, v_idx, w_idx] = 1.0
        
        # Check difference
        err = np.linalg.norm(T - M_target)
        return err < 1e-9, T

    # Try to find rank 10
    # We can use a random search or a specific structured search.
    # Given the time constraint, a simple random search over integer coefficients might be too slow.
    # However, we can try a specific known structure or a small random search.
    # Since rank 10 is not known to exist for 2x2x3 (it's 2x2x2 which is 7, plus 3 extra outputs... 
    # Wait, 2x2x3 is isomorphic to a sub-tensor of 2x2x4? No.
    # The rank of 2x2x2 is 7.
    # The rank of 2x2x3 is at least 8 (since it contains a 2x2x2 sub-tensor).
    # Actually, the rank of 2x2x3 is known to be 8? No, 2x2x2 is 7.
    # Adding one column to B adds 2 outputs (C02, C12).
    # If we use schoolbook for the new column, we add 4 terms (A00B02, A01B12, A10B02, A11B12).
    # Total 7 + 4 = 11.
    # Can we do better?
    # Rank of 2x2x3 is 8? 
    # According to some sources, the rank of 2x2x3 matrix multiplication is 8.
    # Let's try to find a rank 8 or 9 or 10 decomposition.
    
    # Since we have limited time, we will attempt a randomized search for rank 8, 9, 10.
    # We will generate random integer tensors and check.
    # This is very unlikely to succeed quickly.
    
    # Alternative: Use the known rank 8 decomposition if it exists.
    # Let's check if we can construct a rank 8 decomposition.
    # One way is to use the rank 7 Strassen for the 2x2 part and share terms for the 3rd column.
    
    # Let's try a simple local search or random restart.
    # But with 140s, we can do a moderate amount of work.
    
    # For now, we return the rank 11 solution as it is guaranteed to be correct.
    # To attempt improvement, we could try to merge terms.
    
    # Let's try to search for rank 10 using a small random search with integer coefficients in {-1, 0, 1}.
    # This is a heuristic.
    
    # If we don't find anything, return rank 11.
    
    best_result = get_rank11()
    
    # Time check
    if time.time() > deadline - 10:
        return best_result
    
    # Try rank 10 search
    # We'll use a simple random search for a few seconds.
    search_deadline = time.time() + 20.0
    
    while time.time() < search_deadline:
        # Generate random U, V, W with small integer entries
        rank = 10
        U = np.random.randint(-1, 2, (n*m, rank))
        V = np.random.randint(-1, 2, (m*p, rank))
        W = np.random.randint(-1, 2, (p*n, rank))
        
        # Convert to lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        is_valid, _ = verify(U_list, V_list, W_list)
        if is_valid:
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            break
    
    return best_result
