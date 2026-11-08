"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 24
    
    # Build the multiplication tensor T[i,j,k] = 1 for all (i,j,k)
    # We need: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This is a 3-way tensor factorization (CP decomposition)
    
    # Try known low-rank decompositions
    # For 2x3x4, the bilinear complexity is known to be 9
    # Let's try to construct a rank-9 decomposition
    
    # The tensor T has shape (n*m, m*p, p*n) = (6, 12, 8)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # We need to find U (6 x r), V (12 x r), W (8 x r) such that
    # sum_c U[a,c]*V[b,c]*W[c_idx,c] = T[a,b,c_idx]
    
    # Strategy: Use the known structure of small matrix multiplication
    # For 2x3x4, we can use a combination of:
    # - Direct terms
    # - Shared intermediate computations
    
    # Let's try a systematic search with small integer/half-integer coefficients
    # Coefficients from {-1, -0.5, 0, 0.5, 1}
    
    coeffs = [-1, -0.5, 0, 0.5, 1]
    
    def build_tensor():
        """Build the multiplication tensor."""
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1
        return T
    
    T = build_tensor()
    
    def check_decomposition(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        r = U.shape[1]
        # Compute sum_c U[:,c] * V[:,c] * W[:,c] and compare with T
        # This is an outer product sum
        result = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.allclose(result, T, atol=1e-10)
    
    # Try rank 9 first (known minimal for 2x3x4)
    target_ranks = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    
    for rank in target_ranks:
        if time.time() > deadline - 5:
            break
        
        # Try to find a decomposition of this rank
        # Use a greedy/structured approach
        
        # For small ranks, we can try to exploit the structure
        # The key is that many entries of T are 1, and we need to cover them
        
        # Approach: Try different patterns of coefficient assignments
        # Since we have 6*12*8 = 576 constraints and rank*6 + rank*12 + rank*8 = 26*rank unknowns
        
        if rank <= 12:
            # Try a more structured search
            # Initialize with a heuristic pattern
            
            # For 2x3x4, one known approach uses the fact that:
            # (a,b) * (c,d,e) * (f,g,h,i) where:
            # A = [[a,b],[c,d]] (2x3... wait, let me reconsider dimensions)
            
            # Actually: A is 2x3, B is 3x4, C = A*B is 2x4
            # A[i,j] for i in {0,1}, j in {0,1,2}
            # B[j,k] for j in {0,1,2}, k in {0,1,2,3}
            # C[i,k] = sum_j A[i,j]*B[j,k]
            
            # The tensor T[i*m+j, j*p+k, k*n+i] = T[i*3+j, j*4+k, k*2+i] = 1
            
            # Let me try a direct construction for rank 9
            # This is based on the known result that 2x3x4 has bilinear complexity 9
            
            # One known decomposition uses these "slices":
            # We need 9 terms of the form U_vec * V_vec * W_vec
            
            # Let's try a systematic search with limited coefficient set
            # Focus on the structure: each (i,j,k) triple needs to be covered
            
            # There are 2*3*4 = 24 triples (i,j,k)
            # Each term covers some subset of these triples
            
            # Try to find 9 vectors that cover all 24 triples with appropriate signs
            
            # Represent each term by which (i,j,k) it affects
            # Term c affects (i,j,k) if U[i*3+j,c]*V[j*4+k,c]*W[k*2+i,c] != 0
            
            # For simplicity, try terms where each factor has a single nonzero entry
            # or a small number of nonzero entries
            
            # This is getting complex; let's try a randomized search with structured initialization
            
            np.random.seed(42)
            
            for attempt in range(500):
                if time.time() > deadline - 5:
                    break
                
                # Try different initialization strategies
                if attempt % 3 == 0:
                    # Sparse random initialization
                    U = np.zeros((n*m, rank))
                    V = np.zeros((m*p, rank))
                    W = np.zeros((p*n, rank))
                    for c in range(rank):
                        # Each term covers a random subset of (i,j,k)
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    if np.random.random() < 0.3:
                                        U[i*m+j, c] = np.random.choice(coeffs)
                                        V[j*p+k, c] = np.random.choice(coeffs)
                                        W[k*n+i, c] = np.random.choice(coeffs)
                elif attempt % 3 == 1:
                    # Structured: each term covers a block
                    U = np.zeros((n*m, rank))
                    V = np.zeros((m*p, rank))
                    W = np.zeros((p*n, rank))
                    # Divide the 24 triples into 9 groups
                    triples = [(i,j,k) for i in range(n) for j in range(m) for k in range(p)]
                    group_size = len(triples) // rank
                    for c in range(rank):
                        start = c * group_size
                        end = start + group_size if c < rank - 1 else len(triples)
                        for i,j,k in triples[start:end]:
                            U[i*m+j, c] = 1
                            V[j*p+k, c] = 1
                            W[k*n+i, c] = 1
                else:
                    # Hybrid
                    U = np.zeros((n*m, rank))
                    V = np.zeros((m*p, rank))
                    W = np.zeros((p*n, rank))
                    for c in range(rank):
                        # Assign a main (i,j,k) and some neighbors
                        i = np.random.randint(n)
                        j = np.random.randint(m)
                        k = np.random.randint(p)
                        U[i*m+j, c] = 1
                        V[j*p+k, c] = 1
                        W[k*n+i, c] = 1
                        # Add some cross terms
                        for _ in range(3):
                            i2 = np.random.randint(n)
                            j2 = np.random.randint(m)
                            k2 = np.random.randint(p)
                            U[i2*m+j2, c] += np.random.choice(coeffs)
                            V[j2*p+k2, c] += np.random.choice(coeffs)
                            W[k2*n+i2, c] += np.random.choice(coeffs)
                
                if check_decomposition(U, V, W):
                    # Found a valid decomposition!
                    # Convert to list format with integer/half-integer entries
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        # If random search didn't work, try a more deterministic approach
        # For rank >= 12, use a structured decomposition
        
        if rank >= 12 and rank <= 15:
            # Try a block decomposition
            # Split the computation into overlapping blocks
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Use the fact that we can share computations across k values
            # For each (i,j), we need to compute sum_k A[i,j]*B[j,k]
            # This is just A[i,j] * B[j,0], A[i,j]*B[j,1], etc.
            
            # A rank-12 decomposition:
            # For each j (3 values), use 4 terms for the 4 k values
            # But that's 12 terms, and we can potentially merge some
            
            for j in range(m):
                for k in range(p):
                    c = j * p + k
                    if c < rank:
                        U[0*m+j, c] = 1  # i=0
                        V[j*p+k, c] = 1
                        W[k*n+0, c] = 1  # i=0
                        
                        U[1*m+j, c] = 1  # i=1
                        W[k*n+1, c] = 1  # i=1
            
            if rank == 12:
                if check_decomposition(U, V, W):
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    return {"U": U_list, "V": V_list, "W": W_list}
            
            # Try to reduce rank by merging terms
            # For rank 11, try to merge two of the k terms for some j
            if rank == 11:
                # Merge k=0 and k=1 for j=2 (the last column of B)
                # This requires that the contributions can be combined
                # Not straightforward; try a different approach
                
                # Use a different decomposition strategy
                # Based on the known result that 2x3x4 has complexity 9
                # Let's try a specific known decomposition
                
                # One known rank-9 decomposition for 2x3x4:
                # (This is based on the work of Strassen and others)
                
                # Let me try to construct it manually
                # A = [[a, b, c], [d, e, f]]
                # B = [[g, h, i, j], [k, l, m, n], [o, p, q, r]]
                # C = A*B = [[ag+bk+co, ah+bl+cp, ai+bm+cq, aj+bn+cr],
                #            [dg+ek+fo, dh+el+fp, di+em+fq, dj+en+fr]]
                
                # We need 9 bilinear forms that produce all 8 entries of C
                
                # Known decomposition (from literature):
                # t1 = a*g
                # t2 = b*k  
                # t3 = c*o
                # t4 = (a+b)*(g+k)
                # t5 = (b+c)*(k+o)
                # t6 = d*g
                # t7 = e*k
                # t8 = f*o
                # t9 = (d+e+... )
                
                # This is getting complicated. Let's just return the schoolbook
                # if we can't find a better one in time.
                pass
    
    # Fallback: return schoolbook decomposition
    rank = schoolbook_rank
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
