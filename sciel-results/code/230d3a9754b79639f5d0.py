"""Matrix multiplication 2x3 by 3x9 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    num_a = n * m  # 6
    num_b = m * p  # 27
    num_c = p * n  # 18
    
    # Try to find a low-rank decomposition using Kronecker product structure
    # Decompose 3x9 into three 3x3 blocks
    # The tensor for 2x3x3 multiplication has rank 6
    # So 3 parallel 2x3x3 multiplications give rank 18
    
    deadline = time.time() + 140
    
    # First, try the structured Kronecker decomposition with rank 18
    def try_kronecker_rank18():
        """
        Use the fact that 2x3x3 matrix multiplication has rank 6.
        The 2x3 by 3x9 multiplication can be seen as three parallel 2x3 by 3x3 multiplications.
        Each 2x3x3 multiplication has a rank-6 decomposition.
        Total rank: 3 * 6 = 18
        """
        # For 2x3x3 matrix multiplication:
        # A is 2x3, B is 3x3, C is 2x3
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # Tensor indices: (i,j) for A, (j,k) for B, (k,i) for C
        # We need rank-6 decomposition of this tensor
        
        # The standard schoolbook for 2x3x3 has rank 18, but we know it's actually 6
        # Let's construct the rank-6 decomposition for 2x3x3
        
        # For 2x3x3:
        # A has indices (i,j) where i in {0,1}, j in {0,1,2} -> 6 entries
        # B has indices (j,k) where j in {0,1,2}, k in {0,1,2} -> 9 entries  
        # C has indices (k,i) where k in {0,1,2}, i in {0,1} -> 6 entries
        
        # The tensor T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
        # For 2x3x3: n=2, m=3, p=3
        # a = i*3+j, b = j*3+k, c = k*2+i
        
        # Let's find the rank-6 decomposition
        # One known approach: use the identity that rank of MatMul(2,3,3) = 6
        
        # Constructing the rank-6 decomposition for 2x3x3:
        # We can use the following factorization based on the structure
        
        # For each of the 3 blocks in the 3x9 matrix, we have a 2x3x3 multiplication
        # Let's define the rank-6 factors for a single 2x3x3 multiplication
        
        # The 6 basis vectors for A (2x3):
        # A_{i,j} for i in {0,1}, j in {0,1,2}
        # Index: a = i*3 + j
        
        # The 9 basis vectors for B (3x3):
        # B_{j,k} for j in {0,1,2}, k in {0,1,2}
        # Index: b = j*3 + k
        
        # The 6 basis vectors for C (2x3):
        # C_{k,i} for k in {0,1,2}, i in {0,1}
        # Index: c = k*2 + i
        
        # Rank-6 decomposition:
        # We need U (6x6), V (9x6), W (6x6) such that
        # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
        
        # Let's use the following construction:
        # For r in 0..5, let r = i_r * 3 + j_r where i_r in {0,1}, j_r in {0,1,2}
        # But we need 6 terms, and there are 6 (i,j) pairs
        
        # Actually, let's use a different approach. The rank of MatMul(2,3,3) is 6.
        # One decomposition:
        
        # Let's index the 6 rank-1 terms by (i, j) where i in {0,1}, j in {0,1,2}
        # For term (i,j):
        #   U contribution: U[i*3+j, r(i,j)] = 1
        #   V contribution: V[j*3+k, r(i,j)] = 1 for all k
        #   W contribution: W[k*2+i, r(i,j)] = 1 for all k
        # But this gives rank 6, and we need to verify it's correct
        
        # Wait, this doesn't work because V[j*3+k] would be 1 for all k, 
        # and W[k*2+i] would be 1 for all k, which overcounts
        
        # Let me think again. For 2x3x3:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor is T[i,j; j',k; k',i'] = delta(j,j') * delta(k,k') * delta(i,i')
        # where a=(i,j), b=(j',k), c=(k',i')
        
        # So T[a,b,c] = 1 iff a=(i,j), b=(j,k), c=(k,i) for some i,j,k
        
        # Rank-6 decomposition:
        # We can write this as a sum of 6 rank-1 tensors
        # For each (i,j) pair, define a rank-1 tensor
        
        # Actually, the correct rank-6 decomposition for 2x3x3 is:
        # For r = 0..5, let i_r = r // 3, j_r = r % 3
        # U[i_r*3 + j_r, r] = 1
        # V[j_r*3 + k, r] = 1 for k in {0,1,2}
        # W[k*2 + i_r, r] = 1 for k in {0,1,2}
        
        # But this gives V[j*3+k, r] = 1 for all k, which means
        # sum_r U[a,r] * V[b,r] * W[c,r] = sum_{r: i_r=i, j_r=j} V[j*3+k_b, r] * W[k_c*2+i, r]
        # = V[j*3+k_b, r(i,j)] * W[k_c*2+i, r(i,j)]
        # = 1 * 1 = 1 if j_b = j and i_c = i, regardless of k_b, k_c
        
        # But we need it to be 1 only if k_b = k_c as well
        
        # So this simple construction doesn't work. We need a different approach.
        
        # Let's use the fact that rank(MatMul(2,3,3)) = 6 and find the actual decomposition.
        # One way is to use the identity:
        # MatMul(2,3,3) can be decomposed using 6 terms
        
        # Actually, I recall that for small matrix multiplication, the rank is known.
        # For 2x3x3, the rank is 6.
        
        # Let me try a different construction. We can use:
        # For r in 0..5, define:
        #   U[:, r] has a 1 at position (i_r, j_r) where i_r = r % 2, j_r = r // 2
        #   Wait, that gives only 2*2=4 positions, not 6
        
        # Let me just use numerical optimization to find the rank-6 decomposition
        
        # For now, let's fall back to the schoolbook decomposition with rank 54
        # and try to improve it numerically
        
        return None
    
    # Try the Kronecker approach
    kronecker_result = try_kronecker_rank18()
    if kronecker_result:
        return kronecker_result
    
    # Fall back to the original approach
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i*m+j, j*p+k, k*n+i))
    valid_set = set(triples)
    
    # Precompute index lists for efficiency
    a_to_triples = {}
    b_to_triples = {}
    c_to_triples = {}
    for idx, (a, b, c) in enumerate(triples):
        a_to_triples.setdefault(a, []).append(idx)
        b_to_triples.setdefault(b, []).append(idx)
        c_to_triples.setdefault(c, []).append(idx)
    
    def compute_error(U, V, W):
        err = 0.0
        for (a, b, c) in triples:
            val = np.dot(U[a], V[b] * W[c])
            err += (val - 1.0)**2
        for a in range(num_a):
            for b in range(num_b):
                for c in range(num_c):
                    if (a, b, c) not in valid_set:
                        val = np.dot(U[a], V[b] * W[c])
                        err += val**2
        return err
    
    def try_rank(R, n_restarts=8, max_iters=150):
        best_err = float('inf')
        best = None
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            U = np.random.randn(num_a, R) * 0.5
            V = np.random.randn(num_b, R) * 0.5
            W = np.random.randn(num_c, R) * 0.5
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                for a in range(num_a):
                    idxs = a_to_triples[a]
                    A = np.array([V[triples[i][1]] * W[triples[i][2]] for i in idxs])
                    bvec = np.ones(len(idxs))
                    U[a] = np.linalg.lstsq(A, bvec, rcond=None)[0]
                for b in range(num_b):
                    idxs = b_to_triples[b]
                    A = np.array([U[triples[i][0]] * W[triples[i][2]] for i in idxs])
                    bvec = np.ones(len(idxs))
                    V[b] = np.linalg.lstsq(A, bvec, rcond=None)[0]
                for c in range(num_c):
                    idxs = c_to_triples[c]
                    A = np.array([U[triples[i][0]] * V[triples[i][1]] for i in idxs])
                    bvec = np.ones(len(idxs))
                    W[c] = np.linalg.lstsq(A, bvec, rcond=None)[0]
                err = compute_error(U, V, W)
                if err < best_err:
                    best_err = err
                    best = (U.copy(), V.copy(), W.copy())
                if err < 1e-18:
                    break
        if best and best_err < 1e-12:
            return best
        return None
    
    for R in [18, 27, 36, 45, 54]:
        if time.time() > deadline - 5:
            break
        res = try_rank(R)
        if res:
            U, V, W = res
            if compute_error(U, V, W) < 1e-12:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    rank = n*m*p
    U = [[0]*rank for _ in range(num_a)]
    V = [[0]*rank for _ in range(num_b)]
    W = [[0]*rank for _ in range(num_c)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
