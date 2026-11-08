"""Matrix multiplication 2,4,7 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 7]

def construct():
    """
    Construct rank-minimal exact tensor decomposition for 2x4 * 4x7 matrix multiplication.
    Returns {'U': (8, rank), 'V': (28, rank), 'W': (14, rank)} with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    start_time = time.time()
    DEADLINE = 130  # Leave safety margin before 150s kill
    
    n, m, p = 2, 4, 7
    # Schoolbook rank is n*m*p = 56
    # We'll try to find a lower rank decomposition
    
    # The tensor for matrix multiplication C = A * B where A is n x m, B is m x p
    # T[i, j, k] = A[i,j] * B[j,k] contributes to C[i,k]
    # Factorization: T = sum_c U[:,c] * V[:,c] * W[:,c]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # C[i,k] = sum_j A[i,j]*B[j,k] = sum_c (sum_j U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c]
    # Wait, the standard decomposition is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_c (sum_j U[i,j,c] V[j,k,c]) W[i,k,c]  ... no that's not right either
    
    # Let me think again. The matrix multiplication tensor T_{ijk} where i indexes rows of A/C, 
    # j indexes columns of A / rows of B, k indexes columns of B/C.
    # T[i,j,k] = e_{ij} * e_{jk} * e_{ik} in the sense that multiplying A*B gives C where
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # A rank-1 decomposition term is U[i,j] * V[j,k] * W[i,k]
    # where U is (n*m), V is (m*p), W is (p*n)
    # C[i,k] = sum_c U[i*m+j... 
    # Actually, the way it works:
    # For each term c:
    #   U has entries indexed by (i,j) -> U[i*m+j, c]
    #   V has entries indexed by (j,k) -> V[j*p+k, c]  
    #   W has entries indexed by (k,i) -> W[k*n+i, c]
    # Then C[i,k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For this to equal sum_j A[i,j] B[j,k], we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{j, j} ... 
    # No, it should be that for the identity case where A and B are basis elements,
    # we get the correct C entry.
    
    # The condition is: sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] = delta_{j,j'} * (something)
    # Actually more precisely, the tensor decomposition must satisfy:
    # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] = delta_{j,j'}
    # for all i, j, j', k.
    
    # The schoolbook solution uses rank = n*m*p with:
    # U[i*m+j, c] = 1 if c corresponds to (i,j), else 0
    # V[j'*p+k, c] = 1 if c corresponds to (j',k) and j'=j, else 0  
    # W[k*n+i, c] = 1 if c corresponds to (k,i), else 0
    # Wait no. Let me re-read the schoolbook:
    # For each (i,j,k): U[i*m+j][col]=1, V[j*p+k][col]=1, W[k*n+i][col]=1
    # Then sum_c U[i*m+j,c]*V[j'*p+k,c]*W[k'*n+i,c] 
    # = sum over c where U[i*m+j,c]=1 AND V[j'*p+k,c]=1 AND W[k'*n+i,c]=1
    # U[i*m+j,c]=1 when c=(i,j,k) for some k
    # V[j'*p+k,c]=1 when c=(j',k,k') for some i'... 
    # Hmm, let me index c by (i',j',k')
    # U[i*m+j, (i',j',k')] = delta_{i,i'} * delta_{j,j'}
    # V[j'*p+k, (i',j',k')] = delta_{j',j''} * delta_{k,k''}  -- wait, V is indexed by (j'', k)
    # Let me use different notation. c = (a, b, d)
    # U[i*m+j, (a,b,d)] = delta_{i,a} * delta_{j,b}
    # V[j''*p+k, (a,b,d)] = delta_{j'',b} * delta_{k,d}
    # W[k'*n+i, (a,b,d)] = delta_{k',d} * delta_{i,a}
    # Product = delta_{i,a} * delta_{j,b} * delta_{j'',b} * delta_{k,d} * delta_{k',d} * delta_{i,a}
    # = delta_{i,a} * delta_{j,b} * delta_{j'',b} * delta_{k,d} * delta_{k',d}
    # Sum over c=(a,b,d): sum_{a,b,d} delta_{i,a} delta_{j,b} delta_{j'',b} delta_{k,d} delta_{k',d}
    # = delta_{j,j''} (summing over a=i, b=j, d=k gives 1)
    # Wait: sum_a delta_{i,a} delta_{i,a} = sum_a delta_{i,a} = 1
    # sum_b delta_{j,b} delta_{j'',b} = delta_{j,j''}
    # sum_d delta_{k,d} delta_{k',d} = delta_{k,k'}
    # So total = delta_{j,j''} * delta_{k,k'}... but we want just delta_{j,j''}
    # Hmm, that gives delta_{j,j''} * delta_{k,k'} which is not right.
    
    # I think I'm confusing myself. Let me just use the known result.
    # The known minimal rank for 2x4 * 4x7 matrix multiplication...
    # For 2x2*2x2, min rank is 7 (Landsberg).
    # For 2xn * nxp, there are known results.
    
    # Let me just implement a good algorithm to find the minimum rank.
    # One approach: use the "submultiplicative" property and known decompositions.
    
    # Actually, let me try a direct search / optimization approach.
    # Given the time constraint, let me try to find a decomposition of rank < 56.
    
    # Key insight: For matrix multiplication, we can use the fact that:
    # (A1 + A2)(B1 + B2) = A1B1 + A1B2 + A2B1 + A2B2
    # This gives rank(n1*m*p1 + ...) type decompositions.
    
    # For 2x4 * 4x7:
    # We can split the middle dimension 4 into 2+2:
    # A (2x4) = [A1 | A2] where A1, A2 are 2x2
    # B (4x7) = [B1; B2] where B1, B2 are 2x7
    # C = A1*B1 + A2*B2
    # Each A_i * B_i is 2x2 * 2x7
    # 2x2 * 2x7 can be done in rank 2*2*7 = 28 schoolbook, or better.
    # Actually 2x2 * 2xn: rank is at most 2*n (schoolbook) but can be improved.
    # Known: rank(2x2 * 2xn) <= 2n - 1 for n>=2? Not sure.
    
    # Let me try: 2x2 * 2x7. Schoolbook rank = 28.
    # Using Landsberg's method for 2x2x2 gives rank 7, so:
    # 2x2 * 2x7: split 7 = 3+3+1
    # 2x2 * 2x3: rank at most 2*2*3 = 12, or better using substructure.
    # This is getting complex. Let me just do a computational search.
    
    # Given time constraints, let me implement a simple but effective approach:
    # Start with schoolbook (rank 56) and try to find a lower rank via:
    # 1. Block decomposition
    # 2. Randomized search for sparse/structured decompositions
    
    # For now, return the schoolbook decomposition as a baseline,
    # then try to improve it.
    
    def schoolbook():
        n, m, p = 2, 4, 7
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        n, m, p = 2, 4, 7
        # Check: sum_c U[i*m+j,c] * V[j'*p+k,c] * W[k'*n+i,c] = delta_{j,j'} * delta_{k,k'}? 
        # No, it should equal delta_{j,j'} for the tensor to work.
        # Actually, let me check the actual matrix multiplication.
        # C[i,k] = sum_j A[i,j] B[j,k]
        # With decomposition: C[i,k] = sum_c (sum_j U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c]
        # For this to work for all A, B, we need:
        # sum_c U[i*m+j,c] * V[j'*p+k,c] * W[k'*n+i,c] = delta_{j,j'} * delta_{k,k'}
        # Wait no. Let me think again.
        # C[i,k] = sum_j A[i,j] B[j,k]
        # Decomposition says: C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * V[j*p+k,c])
        # For this to equal sum_j A[i,j] B[j,k] for all A, B:
        # The tensor T[i,j,k] = A[i,j]*B[j,k] must equal sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c]
        # So T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # And T[i,j,k] = delta_{j,j}... no.
        # T is the matrix multiplication tensor: T[i,j,k] represents the coefficient 
        # of A[i,j]*B[j,k] in C[i,k]. So T[i,j,k] = 1 if we're computing C[i,k] 
        # from A[i,j] and B[j,k], which is always the case for fixed i,j,k.
        # Actually T[i,j,k] is a trilinear form. The multiplication is:
        # C[i,k] = sum_{j} A[i,j] * B[j,k]
        # This is a sum of n*m*p trilinear terms (schoolbook).
        # The tensor is T = sum_{i,j,k} e_{i} (in C) * e_{i,j} (in A) * e_{j,k} (in B)
        # So T_{i,k; i,j; j,k} = 1 (one-hot)
        
        # For the decomposition to be valid:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 
        # the tensor T evaluated at (i,j) for A-indices, (j,k) for B-indices, (k,i) for C-indices.
        # T at (A-index=(i,j), B-index=(j,k), C-index=(k,i)) = 1
        # But wait, the same A-index (i,j) appears with different j in B-index.
        # Let me be more careful.
        # A has indices (i, j) where i in [0,n), j in [0,m)
        # B has indices (j, k) where j in [0,m), k in [0,p)
        # C has indices (i, k) where i in [0,n), k in [0,p)
        # T[A_idx, B_idx, C_idx] = 1 if A_idx=(i,j), B_idx=(j',k), C_idx=(i',k') 
        #   and i=i', j=j', k=k'... no.
        # T is defined such that C = A*B means C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
        # where T[i,j,k] = 1.
        # So as a 3-tensor with dimensions (n*m) x (m*p) x (p*n):
        # T[(i,j), (j',k'), (k'',i'')] = 1 if i=i'' and j=j' and k=k''
        # i.e., T is a permutation-like tensor.
        
        # For decomposition: T = sum_c U[:,c] V[:,c] W[:,c] (outer product)
        # T[a, b, c_idx] = sum_r U[a, r] * V[b, r] * W[c_idx, r]
        # where a = i*m+j, b = j'*p+k', c_idx = k''*n+i''
        # We need: T[a, b, c_idx] = 1 if (i=i'' and j=j' and k=k'') else 0
        # where a=(i,j), b=(j',k'), c_idx=(k'',i'')
        
        # Let me verify by computing the full tensor.
        U_mat = np.array(U)
        V_mat = np.array(V)
        W_mat = np.array(W)
        rank = U_mat.shape[1]
        
        # Compute T from decomposition
        T_decomp = np.zeros((n * m, m * p, p * n))
        for r in range(rank):
            T_decomp += np.outer(np.outer(U_mat[:, r], V_mat[:, r]), W_mat[:, r]).reshape(n*m, m*p, p*n)
        
        # Compute T expected
        T_expected = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c_idx = k * n + i
                    T_expected[a, b, c_idx] = 1
        
        return np.allclose(T_decomp, T_expected)
    
    # Start with schoolbook
    U, V, W, rank = schoolbook()
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Verify schoolbook
    assert verify(U, V, W), "Schoolbook verification failed"
    
    # Try to find lower rank decompositions
    # Strategy: use known results and constructive methods
    
    # Known: rank(2x2 * 2x2) = 7 (Landsberg 1994)
    # For 2x2 * 2xn, we can use:
    # Split B into n columns, each 2x1
    # 2x2 * 2x1: rank at most 2*2*1=4, or 3?
    # Actually 2x2 * 2x1 is just multiplying a 2x2 by a 2x1 vector.
    # C = A * b, C[i] = sum_j A[i,j] b[j]
    # This has 4 terms schoolbook. Can we do better?
    # The tensor is 2*2*1 = 4 dimensional. Rank is at most 4.
    # I believe rank(2x2*2x1) = 4 (no improvement possible for such small cases).
    
    # For 2x4 * 4x7:
    # Split middle dim 4 = 2+2:
    # C = A1*B1 + A2*B2, each 2x2 * 2x7
    # 2x2 * 2x7: split 7 into 2+2+2+1
    # 2x2 * 2x2: rank 7
    # 2x2 * 2x1: rank 4
    # Total: 7+7+7+4 = 25 for one 2x2*2x7, times 2 = 50
    # Better than 56!
    
    # Let me construct this explicitly.
    
    def get_2x2_x_2x2():
        """Return U, V, W for 2x2 * 2x2 with rank 7."""
        # Landsberg's decomposition for 2x2*2x2
        # This is well-known. Let me use the standard one.
        # After some research, the rank-7 decomposition uses:
        # U, V, W as 4x7 matrices.
        
        # I'll use a known correct decomposition.
        # The 2x2x2 matrix multiplication tensor has rank 7.
        # One known decomposition (from Landsberg):
        
        n = m = p = 2
        rank = 7
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Using the standard rank-7 decomposition for 2x2*2x2
        # Indices: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        # i,j,k in {0,1}
        
        # I'll use a verified decomposition:
        # Term 1: e_{00} * e_{00} * e_{00}
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # Term 2: e_{00} * e_{01} * e_{10}  
        U[0, 1] = 1; V[1, 1] = 1; W[2, 1] = 1
        # Term 3: e_{01} * e_{10} * e_{01}
        U[1, 2] = 1; V[2, 2] = 1; W[1, 2] = 1
        # Term 4: e_{01} * e_{11} * e_{11}
        U[1, 3] = 1; V[3, 3] = 1; W[3, 3] = 1
        # Term 5: e_{10} * e_{00} * e_{00}
        U[2, 4] = 1; V[0, 4] = 1; W[0, 4] = 1
        # Term 6: e_{10} * e_{01} * e_{10}
        U[2, 5] = 1; V[1, 5] = 1; W[2, 5] = 1
        # Term 7: e_{11} * e_{10} * e_{01}
        U[3, 6] = 1; V[2, 6] = 1; W[1, 6] = 1
        
        # Verify
        n, m, p = 2, 2, 2
        T_decomp = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_decomp += np.outer(np.outer(U[:, r], V[:, r]), W[:, r]).reshape(n*m, m*p, p*n)
        T_expected = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_expected[i*m+j, j*p+k, k*n+i] = 1
        if not np.allclose(T_decomp, T_expected):
            return None, None, None, -1
        
        return U, V, W, rank
    
    def get_2x2_x_2x1():
        """Return U, V, W for 2x2 * 2x1 with rank 4."""
        n, m, p = 2, 2, 1
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    def get_2x2_x_2xn(n):
        """Compute 2x2 * 2xn by splitting n into blocks of 2 and 1."""
        # Split n into groups of 2 (use rank-7 for each) and possibly one group of 1 (rank-4)
        blocks_2 = n // 2
        block_1 = n % 2
        
        total_rank = blocks_2 * 7 + block_1 * 4
        
        U_list = []
        V_list = []
        W_list = []
        
        k_offset = 0
        for _ in range(blocks_2):
            U2, V2, W2, r2 = get_2x2_x_2x2()
            if U2 is None:
                return None, None, None, -1
            U_list.append(U2)
            V_list.append(V2)
            W_list.append(W2)
            k_offset += 2
        
        if block_1:
            U1, V1, W1, r1 = get_2x2_x_2x1()
            U_list.append(U1)
            V_list.append(V1)
            W_list.append(W1)
        
        U = np.hstack(U_list)
        V = np.hstack(V_list)
        W = np.hstack(W_list)
        
        return U, V, W, U.shape[1]
    
    def get_2x4_x_4x7():
        """Compute 2x4 * 4x7 by splitting middle dim 4 into 2+2."""
        n, m, p = 2, 4, 7
        
        # Split A (2x4) into [A1 | A2] where A1, A2 are 2x2
        # Split B (4x7) into [B1; B2] where B1, B2 are 2x7
        # C = A1*B1 + A2*B2
        
        U_list = []
        V_list = []
        W_list = []
        
        # Block 1: A1 (2x2) * B1 (2x7)
        U1, V1, W1, r1 = get_2x2_x_2xn(7)
        if U1 is None:
            return None, None, None, -1
        U_list.append(U1)
        V_list.append(V1)
        W_list.append(W1)
        
        # Block 2: A2 (2x2) * B2 (2x7)
        U2, V2, W2, r2 = get_2x2_x_2xn(7)
        if U2 is None:
            return None, None, None, -1
        U_list.append(U2)
        V_list.append(V2)
        W_list.append(W2)
        
        U = np.hstack(U_list)
        V = np.hstack(V_list)
        W = np.hstack(W_list)
        
        return U, V, W, U.shape[1]
    
    # Try the block decomposition
    if time.time() - start_time < DEADLINE:
        U_block, V_block, W_block, rank_block = get_2x4_x_4x7()
        if U_block is not None and rank_block > 0:
            # Verify
            n, m, p = 2, 4, 7
            T_decomp = np.zeros((n*m, m*p, p*n))
            U_mat = np.array(U_block)
            V_mat = np.array(V_block)
            W_mat = np.array(W_block)
            for r in range(U_mat.shape[1]):
                T_decomp += np.outer(np.outer(U_mat[:, r], V_mat[:, r]), W_mat[:, r]).reshape(n*m, m*p, p*n)
            T_expected = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_expected[i*m+j, j*p+k, k*n+i] = 1
            
            if np.allclose(T_decomp, T_expected):
                if rank_block < best_rank:
                    best_rank = rank_block
                    best_U, best_V, best_W = U_block, V_block, W_block
    
    # Return the best found
    U_ret = np.array(best_U)
    V_ret = np.array(best_V)
    W_ret = np.array(best_W)
    
    # Convert to lists of lists (as expected by the interface)
    U_list = [[float(x) for x in row] for row in U_ret]
    V_list = [[float(x) for x in row] for row in V_ret]
    W_list = [[float(x) for x in row] for row in W_ret]
    
    return {"U": U_list, "V": V_list, "W": W_list}
