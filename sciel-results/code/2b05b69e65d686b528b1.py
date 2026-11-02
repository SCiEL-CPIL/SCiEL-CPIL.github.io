import numpy as np
import time
import itertools
import random

DIMENSIONS = [2, 3, 6]


def build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array T[i, b, k] = 1 if valid, 0 otherwise.
    Actually, the tensor has indices (a, b, c) where:
      a = row of A (n values)
      b = column of A / row of B (m values)  
      c = column of B (p values)
    And the output is C[a, c] = sum_b A[a,b] * B[b,c]
    
    In the tensor decomposition framework:
    T[a*m+b, c*n+a] ... wait, let me re-read the convention.
    
    The decomposition is:
    sum_l U[a*m+b, l] * V[c*m+b, l] * W[k*n+i, l] = delta(a,i) * delta(b,c) * delta(d,k)
    
    Wait, the task says:
    U: (n*m, rank) -> index by (a, b) where a in [0,n), b in [0,m)
    V: (m*p, rank) -> index by (b, d) where b in [0,m), d in [0,p)
    W: (p*n, rank) -> index by (k, i) where k in [0,p), i in [0,n)
    
    And the target is: delta(a==i) * delta(b==c) * delta(d==k)
    
    So the tensor T has shape (n*m, m*p, p*n) with:
    T[a*m+b, c*p+d, k*n+i] = 1 if a==i and b==c and d==k else 0
    
    Let me verify: sum over l of U[a*m+b, l] * V[c*p+d, l] * W[k*n+i, l] should equal
    1 if (a==i and b==c and d==k) else 0.
    """
    tensor = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for k in range(p):
                        for i in range(n):
                            if a == i and b == c and d == k:
                                tensor[a * m + b, c * p + d, k * n + i] = 1.0
    return tensor


def try_decomposition(tensor, n, m, p, rank, num_trials=50, seed=42):
    """Try to find a rank-rank decomposition by randomly sampling supports and solving linear systems.
    
    For each term l, we choose a support pattern:
    - A subset of U indices (rows of U that are nonzero in column l)
    - A subset of V indices (rows of V that are nonzero in column l)  
    - A subset of W indices (rows of W that are nonzero in column l)
    
    Then we set up a linear system where the unknowns are the coefficients.
    """
    rng = np.random.RandomState(seed)
    
    nU = n * m
    nV = m * p
    nW = p * n
    
    # Total number of tensor entries
    total_entries = nU * nV * nW
    
    # For a given rank, we have rank * (support sizes) unknowns
    # But the tensor has n*m*p nonzero entries (each = 1)
    
    # Strategy: For each term, pick small random supports
    # Then solve the linear system
    
    for trial in range(num_trials):
        # Choose random supports for each term
        supports = []
        for l in range(rank):
            # For U: pick 1-3 random rows
            su = rng.choice(nU, size=rng.randint(1, 4), replace=False)
            # For V: pick 1-3 random rows
            sv = rng.choice(nV, size=rng.randint(1, 4), replace=False)
            # For W: pick 1-3 random rows
            sw = rng.choice(nW, size=rng.randint(1, 4), replace=False)
            supports.append((su, sv, sw))
        
        # Build the linear system
        # Unknowns: for each term l and each (u,v,w) in support, a coefficient
        # But actually, the coefficient is shared: U[u,l] * V[v,l] * W[w,l]
        # So for each term, the unknowns are the products U[u,l]*V[v,l]*W[w,l]
        # which factorize.
        
        # Better approach: use the fact that the tensor equation is:
        # T[i,j,k] = sum_l U[i,l] * V[j,l] * W[k,l]
        # 
        # If we fix the supports, we can set up a bilinear/quadratic system.
        # But for integer solutions, let's try a simpler approach:
        # Assume all coefficients are in {-1, 0, 1, 2} and use a greedy/ILP approach.
        
        # Actually, let's use a different strategy:
        # Fix the W matrix to be a permutation-like matrix (each W[k,l] is 0 or +/-1)
        # Then the problem becomes: for each k, the vector W[:,l] selects which output
        # entries get contribution from term l.
        
        # Simpler: try to find a decomposition where each term has exactly one nonzero
        # in U, one in V, and one in W (i.e., rank-1 terms that are "atomic").
        # This is the schoolbook approach and gives rank = n*m*p.
        # To do better, we need terms with multiple nonzeros.
        
        # Let's try a structured approach:
        # For the 2x3x6 case, use the block structure:
        # Split into three 2x3x2 blocks.
        # For each 2x3x2 block, use a rank-11 decomposition (if we can find one).
        # Total rank = 3 * 11 = 33.
        
        pass  # This trial-based approach is too slow; use the block approach below
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Strategy: Use a rank-11 decomposition for 2x3x2 blocks.
    # The 2x3x2 multiplication: A (2x3) * B (3x2) = C (2x2)
    # Tensor: T[a*3+b, c*2+b, k*2+a] = 1 if a==a', b==b', c==k'
    # 
    # Known: mu(2,2,2) = 7 (Strassen)
    # For 2x3x2: we can use Strassen for the 2x2x2 part (b in {0,1})
    # plus 2 terms for b=2. Total: 7 + 2 = 9? Or 7 + 3 = 10?
    # Actually, let's think more carefully.
    #
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    #
    # The 2x2x2 part (b in {0,1}) can be done in 7 terms (Strassen).
    # The b=2 terms: A[0,2]*B[2,0], A[0,2]*B[2,1], A[1,2]*B[2,0], A[1,2]*B[2,1]
    # These are 4 entries. Can we do them in 2 terms?
    # Term: A[0,2] * (B[2,0] + B[2,1]) -> contributes to C[0,0] and C[0,1]
    # Term: A[1,2] * (B[2,0] + B[2,1]) -> contributes to C[1,0] and C[1,1]
    # Yes! 2 terms suffice. Total: 7 + 2 = 9 terms for 2x3x2.
    #
    # Wait, but Strassen's 7 terms already involve specific linear combinations.
    # Let me verify that the 9-term decomposition works.
    
    # Build the 2x3x2 tensor
    n2, m2, p2 = 2, 3, 2
    tensor_232 = np.zeros((n2*m2, m2*p2, p2*n2), dtype=np.float64)
    for a in range(n2):
        for b in range(m2):
            for c in range(m2):
                for d in range(p2):
                    for k in range(p2):
                        for i in range(n2):
                            if a == i and b == c and d == k:
                                tensor_232[a*m2+b, c*p2+d, k*n2+i] = 1.0
    
    # Strassen for 2x2x2 (b in {0,1}, k in {0,1}):
    # U indices: a*3+b -> 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    # V indices: b*2+k -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1), 4:(2,0), 5:(2,1)
    # W indices: k*2+i -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    U_232 = np.zeros((6, 9))
    V_232 = np.zeros((6, 9))
    W_232 = np.zeros((4, 9))
    
    # Strassen terms for 2x2x2 (using b in {0,1}, k in {0,1}):
    # M1 = (A00+A11)(B00+B11)
    U_232[0, 0] = 1; U_232[4, 0] = 1
    V_232[0, 0] = 1; V_232[3, 0] = 1
    W_232[0, 0] = 1; W_232[1, 0] = 1; W_232[2, 0] = 1; W_232[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U_232[3, 1] = 1; U_232[4, 1] = 1
    V_232[0, 1] = 1
    W_232[1, 1] = 1; W_232[2, 1] = -1
    
    # M3 = A00(B10-B11)
    U_232[0, 2] = 1
    V_232[2, 2] = 1; V_232[3, 2] = -1
    W_232[0, 2] = -1; W_232[2, 2] = 1
    
    # M4 = A11(B01-B00)
    U_232[4, 3] = 1
    V_232[1, 3] = 1; V_232[0, 3] = -1
    W_232[0, 3] = -1; W_232[3, 3] = 1
    
    # M5 = (A00+A01)B11
    U_232[0, 4] = 1; U_232[1, 4] = 1
    V_232[3, 4] = 1
    W_232[1, 4] = -1; W_232[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B10)
    U_232[3, 5] = 1; U_232[0, 5] = -1
    V_232[0, 5] = 1; V_232[2, 5] = 1
    W_232[0, 5] = 1; W_232[3, 5] = -1
    
    # M7 = (A01-A11)(B01+B11)
    U_232[1, 6] = 1; U_232[4, 6] = -1
    V_232[1, 6] = 1; V_232[3, 6] = 1
    W_232[1, 6] = -1; W_232[2, 6] = 1
    
    # Extension terms for b=2:
    # M8 = A02 * (B20 + B21)
    U_232[2, 7] = 1
    V_232[4, 7] = 1; V_232[5, 7] = 1
    W_232[0, 7] = 1; W_232[1, 7] = 1
    
    # M9 = A12 * (B20 + B21)
    U_232[5, 8] = 1
    V_232[4, 8] = 1; V_232[5, 8] = 1
    W_232[2, 8] = 1; W_232[3, 8] = 1
    
    # Verify the 2x3x2 decomposition
    def verify_232(U, V, W, n, m, p):
        rank = U.shape[1]
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    for d in range(p):
                        for k in range(p):
                            for i in range(n):
                                target = 1.0 if (a == i and b == c and d == k) else 0.0
                                s = 0.0
                                for l in range(rank):
                                    s += U[a * m + b, l] * V[c * p + d, l] * W[k * n + i, l]
                                if abs(s - target) > 1e-9:
                                    return False
        return True
    
    valid_232 = verify_232(U_232, V_232, W_232, 2, 3, 2)
    
    if valid_232:
        # Great! We have a rank-9 decomposition for 2x3x2.
        # Now assemble for 2x3x6: split into three 2x3x2 blocks.
        # Total rank = 3 * 9 = 27.
        
        U_full = np.zeros((6, 27))
        V_full = np.zeros((18, 27))
        W_full = np.zeros((12, 27))
        
        for block in range(3):
            offset = block * 9
            U_full[:, offset:offset+9] = U_232
            
            # V: for block j, rows j*6 to j*6+5 correspond to B rows j*2 to j*2+1... 
            # Wait, V has shape (m*p, rank) = (3*6, rank) = (18, rank)
            # V index: b*p + d where b in [0,3), d in [0,6)
            # For block j (columns 2j and 2j+1 of B), we need V[b*6 + 2j + k, l]
            # where k in {0,1} maps to the 2x3x2 V index b*2+k
            for b in range(3):
                for k in range(2):
                    V_full[b * 6 + block * 2 + k, offset:offset+9] = V_232[b * 2 + k]
            
            # W: shape (p*n, rank) = (6*2, rank) = (12, rank)
            # W index: k*n + i where k in [0,6), i in [0,2)
            # For block j, output columns 2j and 2j+1
            for k in range(2):
                for i in range(2):
                    W_full[(block * 2 + k) * 2 + i, offset:offset+9] = W_232[k * 2 + i]
        
        # Verify the full 2x3x6 decomposition
        def verify_full(U, V, W, n, m, p):
            rank = U.shape[1]
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        for d in range(p):
                            for k in range(p):
                                for i in range(n):
                                    target = 1.0 if (a == i and b == c and d == k) else 0.0
                                    s = 0.0
                                    for l in range(rank):
                                        s += U[a * m + b, l] * V[c * p + d, l] * W[k * n + i, l]
                                    if abs(s - target) > 1e-9:
                                        return False
            return True
        
        if verify_full(U_full, V_full, W_full, 2, 3, 6):
            return {
                "U": U_full.tolist(),
                "V": V_full.tolist(),
                "W": W_full.tolist()
            }
    
    # Fallback: use the rank-34 solution from the seed
    U_seed = np.zeros((6, 34))
    V_seed = np.zeros((18, 34))
    W_seed = np.zeros((12, 34))
    
    U_block = np.zeros((6, 17))
    V_block = np.zeros((9, 17))
    W_block = np.zeros((6, 17))
    
    U_block[0, 0] = 1; U_block[4, 0] = 1
    V_block[0, 0] = 1; V_block[4, 0] = 1
    W_block[0, 0] = 1; W_block[3, 0] = 1
    
    U_block[3, 1] = 1; U_block[4, 1] = 1
    V_block[0, 1] = 1
    W_block[1, 1] = 1; W_block[3, 1] = -1
    
    U_block[0, 2] = 1
    V_block[1, 2] = 1; V_block[4, 2] = -1
    W_block[2, 2] = 1; W_block[3, 2] = 1
    
    U_block[4, 3] = 1
    V_block[3, 3] = 1; V_block[0, 3] = -1
    W_block[0, 3] = 1; W_block[1, 3] = 1
    
    U_block[0, 4] = 1; U_block[1, 4] = 1
    V_block[4, 4] = 1
    W_block[0, 4] = -1; W_block[2, 4] = 1
    
    U_block[3, 5] = 1; U_block[0, 5] = -1
    V_block[0, 5] = 1; V_block[1, 5] = 1
    W_block[3, 5] = 1
    
    U_block[1, 6] = 1; U_block[4, 6] = -1
    V_block[3, 6] = 1; V_block[4, 6] = 1
    W_block[0, 6] = 1
    
    U_block[0, 7] = 1
    V_block[2, 7] = 1
    W_block[4, 7] = 1
    
    U_block[1, 8] = 1
    V_block[5, 8] = 1
    W_block[4, 8] = 1
    
    U_block[3, 9] = 1
    V_block[2, 9] = 1
    W_block[5, 9] = 1
    
    U_block[4, 10] = 1
    V_block[5, 10] = 1
    W_block[5, 10] = 1
    
    U_block[2, 11] = 1
    V_block[6, 11] = 1
    W_block[0, 11] = 1
    
    U_block[2, 12] = 1
    V_block[7, 12] = 1
    W_block[2, 12] = 1
    
    U_block[2, 13] = 1
    V_block[8, 13] = 1
    W_block[4, 13] = 1
    
    U_block[5, 14] = 1
    V_block[6, 14] = 1
    W_block[1, 14] = 1
    
    U_block[5, 15] = 1
    V_block[7, 15] = 1
    W_block[3, 15] = 1
    
    U_block[5, 16] = 1
    V_block[8, 16] = 1
    W_block[5, 16] = 1
    
    U_seed[:, :17] = U_block
    U_seed[:, 17:] = U_block
    
    for j in range(3):
        for k in range(3):
            V_seed[j * 6 + k, :17] = V_block[j * 3 + k]
            V_seed[j * 6 + 3 + k, 17:] = V_block[j * 3 + k]
    
    for k in range(3):
        for i in range(2):
            W_seed[k * 2 + i, :17] = W_block[k * 2 + i]
            W_seed[(k + 3) * 2 + i, 17:] = W_block[k * 2 + i]
    
    return {
        "U": U_seed.tolist(),
        "V": V_seed.tolist(),
        "W": W_seed.tolist()
    }


if __name__ == "__main__":
    construct()
