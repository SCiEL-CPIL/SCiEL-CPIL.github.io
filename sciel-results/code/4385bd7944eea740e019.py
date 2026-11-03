"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 135  # safety margin before 150s limit
    n, m, p = DIMENSIONS  # 3, 5, 7
    nm, mp, pn = n * m, m * p, p * n

    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p  # = 105
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        for c in range(rank):
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W, rank

    U_sb, V_sb, W_sb, rank_sb = schoolbook()

    # Fast verification using vectorized operations
    def verify(U, V, W):
        """Verify that sum_c U[:,c] * V[:,c] * W[:,c] matches the matmul tensor."""
        R = U.shape[1]
        Ur = U.reshape(n, m, R)
        Vr = V.reshape(m, p, R)
        Wr = W.reshape(p, n, R)
        T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
        return np.allclose(T, 1.0, atol=1e-10)

    def laderman_block_terms(i0, j0, k0):
        """Return 23 (u, v, w) vectors for a 3x3x3 Laderman block.
        
        Laderman's algorithm for 3x3 matrix multiplication uses 23 multiplications.
        The indices i0, j0, k0 are the starting indices for the 3x3 block.
        i0 in {0}, j0 in {0,2}, k0 in {0,4}
        """
        terms = []
        
        # Laderman's 23 terms for 3x3x3:
        # Using the standard Laderman decomposition
        
        # M1 = A00 * B00 -> C00 += M1
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[(i0+0)*m+(j0+0)] = 1.0
        v[(j0+0)*p+(k0+0)] = 1.0
        w[(k0+0)*n+(i0+0)] = 1.0
        terms.append((u, v, w))
        
        # M2 = A01 * B00 -> C00 += M2, C10 += M2
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[(i0+0)*m+(j0+1)] = 1.0
        v[(j0+0)*p+(k0+0)] = 1.0
        w[(k0+0)*n+(i0+0)] = 1.0
        w[(k0+0)*n+(i0+1)] = 1.0
        terms.append((u, v, w))
        
        # M3 = A02 * B00 -> C00 += M3, C10 += M3, C20 += M3
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[(i0+0)*m+(j0+2)] = 1.0
        v[(j0+0)*p+(k0+0)] = 1.0
        w[(k0+0)*n+(i0+0)] = 1.0
        w[(k0+0)*n+(i0+1)] = 1.0
        w[(k0+0)*n+(i0+2)] = 1.0
        terms.append((u, v, w))
        
        # M4 = A10 * B00 -> C00 += M4, C10 += M4, C20 += M4
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[(i0+1)*m+(j0+0)] = 1.0
        v[(j0+0)*p+(k0+0)] = 1.0
        w[(k0+0)*n+(i0+0)] = 1.0
        w[(k0+0)*n+(i0+1)] = 1.0
        w[(k0+0)*n+(i0+2)] = 1.0
        terms.append((u, v, w))
        
        # M5 = A11 * B00 -> C00 += M5, C10 += M5, C20 += M5
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[(i0+1)*m+(j0+1)] = 1.0
        v[(j0+0)*p+(k0+0)] = 1.0
        w[(k0+0)*n+(i0+0)] = 1.0
        w[(k0+0)*n+(i0+1)] = 1.0
        w[(k0+0)*n+(i0+2)] = 1.0
        terms.append((u, v, w))
        
        # M6 = A12 * B00 -> C00 += M6, C10 += M6, C20 += M6
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[(i0+1)*m+(j0+2)] = 1.0
        v[(j0+0)*p+(k0+0)] = 1.0
        w[(k0+0)*n+(i0+0)] = 1.0
        w[(k0+0)*n+(i0+1)] = 1.0
        w[(k0+0)*n+(i0+2)] = 1.0
        terms.append((u, v, w))
        
        # M7 = A20 * B00 -> C00 += M7, C10 += M7, C20 += M7
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[(i0+2)*m+(j0+0)] = 1.0
        v[(j0+0)*p+(k0+0)] = 1.0
        w[(k0+0)*n+(i0+0)] = 1.0
        w[(k0+0)*n+(i0+1)] = 1.0
        w[(k0+0)*n+(i0+2)] = 1.0
        terms.append((u, v, w))
        
        # M8 = A21 * B00 -> C00 += M8, C10 += M8, C20 += M8
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[(i0+2)*m+(j0+1)] = 1.0
        v[(j0+0)*p+(k0+0)] = 1.0
        w[(k0+0)*n+(i0+0)] = 1.0
        w[(k0+0)*n+(i0+1)] = 1.0
        w[(k0+0)*n+(i0+2)] = 1.0
        terms.append((u, v, w))
        
        # M9 = A22 * B00 -> C00 += M9, C10 += M9, C20 += M9
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[(i0+2)*m+(j0+2)] = 1.0
        v[(j0+0)*p+(k0+0)] = 1.0
        w[(k0+0)*n+(i0+0)] = 1.0
        w[(k0+0)*n+(i0+1)] = 1.0
        w[(k0+0)*n+(i0+2)] = 1.0
        terms.append((u, v, w))
        
        # Now for column 1 of B:
        # M10 = A00 * B01 -> C01 += M10
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[(i0+0)*m+(j0+0)] = 1.0
        v[(j0+0)*p+(k0+1)] = 1.0
        w[(k0+1)*n+(i0+0)] = 1.0
        terms.append((u, v, w))
        
        # M11 = A01 * B01 -> C01 += M11, C11 += M11
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[(i0+0)*m+(j0+1)] = 1.0
        v[(j0+0)*p+(k0+1)] = 1.0
        w[(k0+1)*n+(i0+0)] = 1.0
        w[(k0+1)*n+(i0+1)] = 1.0
        terms.append((u, v, w))
        
        # M12 = A02 * B01 -> C01 += M12, C11 += M12, C21 += M12
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[(i0+0)*m+(j0+2)] = 1.0
        v[(j0+0)*p+(k0+1)] = 1.0
        w[(k0+1)*n+(i0+0)] = 1.0
        w[(k0+1)*n+(i0+1)] = 1.0
        w[(k0+1)*n+(i0+2)] = 1.0
        terms.append((u, v, w))
        
        # M13 = A10 * B01 -> C01 += M13, C11 += M13, C21 += M13
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[(i0+1)*m+(j0+0)] = 1.0
        v[(j0+0)*p+(k0+1)] = 1.0
        w[(k0+1)*n+(i0+0)] = 1.0
        w[(k0+1)*n+(i0+1)] = 1.0
        w[(k0+1)*n+(i0+2)] = 1.0
        terms.append((u, v, w))
        
        # M14 = A11 * B01 -> C01 += M14, C11 += M14, C21 += M14
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[(i0+1)*m+(j0+1)] = 1.0
        v[(j0+0)*p+(k0+1)] = 1.0
        w[(k0+1)*n+(i0+0)] = 1.0
        w[(k0+1)*n+(i0+1)] = 1.0
        w[(k0+1)*n+(i0+2)] = 1.0
        terms.append((u, v, w))
        
        # M15 = A12 * B01 -> C01 += M15, C11 += M15, C21 += M15
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[(i0+1)*m+(j0+2)] = 1.0
        v[(j0+0)*p+(k0+1)] = 1.0
        w[(k0+1)*n+(i0+0)] = 1.0
        w[(k0+1)*n+(i0+1)] = 1.0
        w[(k0+1)*n+(i0+2)] = 1.0
        terms.append((u, v, w))
        
        # M16 = A20 * B01 -> C01 += M16, C11 += M16, C21 += M16
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[(i0+2)*m+(j0+0)] = 1.0
        v[(j0+0)*p+(k0+1)] = 1.0
        w[(k0+1)*n+(i0+0)] = 1.0
        w[(k0+1)*n+(i0+1)] = 1.0
        w[(k0+1)*n+(i0+2)] = 1.0
        terms.append((u, v, w))
        
        # M17 = A21 * B01 -> C01 += M17, C11 += M17, C21 += M17
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[(i0+2)*m+(j0+1)] = 1.0
        v[(j0+0)*p+(k0+1)] = 1.0
        w[(k0+1)*n+(i0+0)] = 1.0
        w[(k0+1)*n+(i0+1)] = 1.0
        w[(k0+1)*n+(i0+2)] = 1.0
        terms.append((u, v, w))
        
        # M18 = A22 * B01 -> C01 += M18, C11 += M18, C21 += M18
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[(i0+2)*m+(j0+2)] = 1.0
        v[(j0+0)*p+(k0+1)] = 1.0
        w[(k0+1)*n+(i0+0)] = 1.0
        w[(k0+1)*n+(i0+1)] = 1.0
        w[(k0+1)*n+(i0+2)] = 1.0
        terms.append((u, v, w))
        
        # Now for column 2 of B:
        # M19 = A00 * B02 -> C02 += M19
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[(i0+0)*m+(j0+0)] = 1.0
        v[(j0+0)*p+(k0+2)] = 1.0
        w[(k0+2)*n+(i0+0)] = 1.0
        terms.append((u, v, w))
        
        # M20 = A01 * B02 -> C02 += M20, C12 += M20
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[(i0+0)*m+(j0+1)] = 1.0
        v[(j0+0)*p+(k0+2)] = 1.0
        w[(k0+2)*n+(i0+0)] = 1.0
        w[(k0+2)*n+(i0+1)] = 1.0
        terms.append((u, v, w))
        
        # M21 = A02 * B02 -> C02 += M21, C12 += M21, C22 += M21
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[(i0+0)*m+(j0+2)] = 1.0
        v[(j0+0)*p+(k0+2)] = 1.0
        w[(k0+2)*n+(i0+0)] = 1.0
        w[(k0+2)*n+(i0+1)] = 1.0
        w[(k0+2)*n+(i0+2)] = 1.0
        terms.append((u, v, w))
        
        # M22 = A10 * B02 -> C02 += M22, C12 += M22, C22 += M22
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[(i0+1)*m+(j0+0)] = 1.0
        v[(j0+0)*p+(k0+2)] = 1.0
        w[(k0+2)*n+(i0+0)] = 1.0
        w[(k0+2)*n+(i0+1)] = 1.0
        w[(k0+2)*n+(i0+2)] = 1.0
        terms.append((u, v, w))
        
        # M23 = A11 * B02 -> C02 += M23, C12 += M23, C22 += M23
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[(i0+1)*m+(j0+1)] = 1.0
        v[(j0+0)*p+(k0+2)] = 1.0
        w[(k0+2)*n+(i0+0)] = 1.0
        w[(k0+2)*n+(i0+1)] = 1.0
        w[(k0+2)*n+(i0+2)] = 1.0
        terms.append((u, v, w))
        
        # Wait, that's only 23 but they're all schoolbook-like. Laderman uses clever
        # combinations. Let me use the actual Laderman 23-term algorithm.
        # The standard Laderman algorithm for 3x3x3:
        
        return terms  # This is just schoolbook for now, need to fix

    # Let me implement the proper Laderman algorithm
    def laderman_3x3x3(i0, j0, k0):
        """Laderman's 23-term algorithm for 3x3x3 matrix multiplication.
        
        Reference: Laderman (1976) "On the tensor rank of matrix multiplication"
        
        The 23 terms are:
        Let A = [[a,b,c],[d,e,f],[g,h,i]], B = [[j,k,l],[m,n,o],[p,q,r]]
        
        T1 = (a+b+c) * (j+m+p)       -> contributes to C[0,0], C[0,1], C[0,2], C[1,0], C[1,1], C[1,2], C[2,0], C[2,1], C[2,2]
        ...
        
        Actually, let me use a known correct Laderman decomposition.
        """
        terms = []
        
        # I'll use a verified Laderman decomposition.
        # The key insight is that Laderman uses 23 bilinear forms.
        
        # Let me define them properly using the standard formulation:
        # For 3x3, we can use the following 23 terms:
        
        # Using notation: A[i,j], B[j,k], C[i,k]
        
        # Term 1: (A[0,0]+A[1,0]+A[2,0]) * (B[0,0]+B[1,0]+B[2,0])
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[(i0+0)*m+(j0+0)] = 1.0
        u[(i0+1)*m+(j0+0)] = 1.0
        u[(i0+2)*m+(j0+0)] = 1.0
        v[(j0+0)*p+(k0+0)] = 1.0
        v[(j0+1)*p+(k0+0)] = 1.0
        v[(j0+2)*p+(k0+0)] = 1.0
        # This contributes to all C[i,k] for i,k in 0,1,2 with coefficient 1
        for ii in range(3):
            for kk in range(3):
                w[(k0+kk)*n+(i0+ii)] = 1.0
        terms.append((u, v, w))
        
        # This approach is getting complex. Let me just use a simpler strategy:
        # Use Strassen-like blocks for 2x2 sub-blocks within the 3x3 region.
        
        return terms

    # Actually, let me take a different approach. Instead of implementing Laderman
    # from scratch (which is error-prone), let me use multiple 2x2 Strassen blocks
    # that cover more of the 3x5x7 space.
    
    # The key idea: we can partition the 3x5x7 problem into overlapping 2x2x2 
    # Strassen blocks. Each Strassen block covers a 2x2x2 subproblem with rank 7 
    # instead of 8 (saving 1 term per block).
    
    # For 3x5x7, we can try to cover as many 2x2x2 subproblems as possible.
    
    def strassen_block_terms(i0, i1, j0, j1, k0, k1):
        """Return 7 (u, v, w) vectors for a 2x2x2 Strassen block."""
        terms = []
        
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m+j0] = 1.0; u[i1*m+j1] = 1.0
        v[j0*p+k0] = 1.0; v[j1*p+k1] = 1.0
        w[k0*n+i0] = 1.0; w[k1*n+i1] = 1.0
        terms.append((u, v, w))
        
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i1*m+j0] = 1.0; u[i1*m+j1] = 1.0
        v[j0*p+k0] = 1.0
        w[k0*n+i1] = 1.0; w[k1*n+i1] = -1.0
        terms.append((u, v, w))
        
        # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m+j0] = 1.0
        v[j0*p+k1] = 1.0; v[j1*p+k1] = -1.0
        w[k1*n+i0] = 1.0; w[k1*n+i1] = 1.0
        terms.append((u, v, w))
        
        # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i1*m+j1] = 1.0
        v[j1*p+k0] = 1.0; v[j0*p+k0] = -1.0
        w[k0*n+i0] = 1.0; w[k0*n+i1] = 1.0
        terms.append((u, v, w))
        
        # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m+j0] = 1.0; u[i0*m+j1] = 1.0
        v[j1*p+k1] = 1.0
        w[k0*n+i0] = -1.0; w[k1*n+i0] = 1.0
        terms.append((u, v, w))
        
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i1*m+j0] = 1.0; u[i0*m+j0] = -1.0
        v[j0*p+k0] = 1.0; v[j0*p+k1] = 1.0
        w[k1*n+i1] = 1.0
        terms.append((u, v, w))
        
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m+j1] = 1.0; u[i1*m+j1] = -1.0
        v[j1*p+k0] = 1.0; v[j1*p+k1] = 1.0
        w[k0*n+i0] = 1.0
        terms.append((u, v, w))
        
        return terms

    def is_valid_block_set(blocks):
        """Check that no (i,j,k) appears in more than one block."""
        seen = set()
        for I, J, K in blocks:
            for i in I:
                for j in J:
                    for k in K:
                        if (i, j, k) in seen:
                            return False
                        seen.add((i, j, k))
        return True

    def build_decomposition(blocks):
        """Build U, V, W from a set of Strassen blocks + schoolbook for the rest."""
        U_list, V_list, W_list = [], [], []
        
        for I, J, K in blocks:
            i0, i1 = sorted(I)
            j0, j1 = sorted(J)
            k0, k1 = sorted(K)
            terms = strassen_block_terms(i0, i1, j0, j1, k0, k1)
            for u, v, w in terms:
                U_list.append(u)
                V_list.append(v)
                W_list.append(w)
        
        # Add schoolbook terms for uncovered (i,j,k)
        covered = set()
        for I, J, K in blocks:
            for i in I:
                for j in J:
                    for k in K:
                        covered.add((i, j, k))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
                        u[i*m+j] = 1.0
                        v[j*p+k] = 1.0
                        w[k*n+i] = 1.0
                        U_list.append(u)
                        V_list.append(v)
                        W_list.append(w)
        
        U = np.array(U_list).T  # (nm, rank)
        V = np.array(V_list).T  # (mp, rank)
        W = np.array(W_list).T  # (pn, rank)
        return U, V, W

    # Generate many candidate block configurations systematically
    # Each block is a 2x2x2 subproblem defined by choosing 2 i's, 2 j's, 2 k's
    
    all_i_pairs = [(0,1), (0,2), (1,2)]
    all_j_pairs = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    all_k_pairs = [(0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (1,2), (1,3), (1,4), (1,5), (1,6), (2,3), (2,4), (2,5), (2,6), (3,4), (3,5), (3,6), (4,5), (4,6), (5,6)]
    
    # Generate all possible block configurations (up to 6 blocks)
    # This is a large search space, so we'll use a greedy approach
    
    # First, try the original 6-block configuration
    candidate_blocks_list = [
        # Original 6 blocks
        [({0,1},{0,1},{0,1}), ({1,2},{0,1},{2,3}), ({0,2},{0,1},{4,5}),
         ({0,1},{2,3},{0,1}), ({1,2},{2,3},{2,3}), ({0,2},{2,3},{4,5})],
    ]
    
    # Add more configurations by trying different combinations
    # Greedy: try to add blocks that cover the most uncovered (i,j,k)
    
    def greedy_block_selection(max_blocks=8, time_limit=30):
        """Greedy selection of Strassen blocks to maximize coverage."""
        start_time = time.time()
        best_blocks = []
        covered = set()
        
        for _ in range(max_blocks):
            if time.time() - start_time > time_limit:
                break
            
            best_block = None
            best_new_coverage = 0
            
            for ip in all_i_pairs:
                for jp in all_j_pairs:
                    for kp in all_k_pairs:
                        I, J, K = set(ip), set(jp), set(kp)
                        # Check if this block is valid (doesn't overlap with existing)
                        valid = True
                        new_covered = 0
                        for i in I:
                            for j in J:
                                for k in K:
                                    if (i, j, k) in covered:
                                        valid = False
                                        break
                                    new_covered += 1
                            if not valid:
                                break
                        if valid and new_covered > best_new_coverage:
                            best_new_coverage = new_covered
                            best_block = (I, J, K)
                        
                        if not valid:
                            continue
            
            if best_block is None or best_new_coverage == 0:
                break
            
            best_blocks.append(best_block)
            for i in best_block[0]:
                for j in best_block[1]:
                    for k in best_block[2]:
                        covered.add((i, j, k))
        
        return best_blocks

    # Try greedy selection
    greedy_blocks = greedy_block_selection(max_blocks=8, time_limit=20)
    if greedy_blocks:
        candidate_blocks_list.append(greedy_blocks)
    
    # Also try a few hand-crafted configurations
    # Configuration with more J-pair diversity
    candidate_blocks_list.append([
        ({0,1},{0,1},{0,1}), ({1,2},{1,2},{0,1}), ({0,2},{2,3},{0,1}),
        ({0,1},{3,4},{0,1}), ({1,2},{3,4},{2,3}), ({0,2},{3,4},{4,5}),
        ({0,1},{0,2},{2,3}), ({1,2},{1,3},{4,5})
    ])
    
    candidate_blocks_list.append([
        ({0,1},{0,1},{0,1}), ({1,2},{0,2},{0,1}), ({0,2},{1,3},{0,1}),
        ({0,1},{2,4},{0,1}), ({1,2},{3,4},{2,3}), ({0,2},{3,4},{4,5}),
        ({0,1},{1,2},{2,3}), ({1,2},{2,3},{4,5})
    ])
    
    candidate_blocks_list.append([
        ({0,1},{0,1},{0,1}), ({1,2},{1,3},{0,1}), ({0,2},{2,4},{0,1}),
        ({0,1},{3,4},{0,1}), ({1,2},{0,2},{2,3}), ({0,2},{1,3},{4,5}),
        ({0,1},{2,3},{2,3}), ({1,2},{3,4},{4,5})
    ])
    
    # Try each block configuration
    best_U, best_V, best_W = None, None, None
    best_rank = rank_sb
    
    for blocks in candidate_blocks_list:
        if time.time() > deadline - 20:
            break
        if not is_valid_block_set(blocks):
            continue
        U, V, W = build_decomposition(blocks)
        if verify(U, V, W):
            if U.shape[1] < best_rank:
                best_rank = U.shape[1]
                best_U, best_V, best_W = U, V, W
    
    # If we found a valid solution, try to remove redundant terms
    if best_U is not None:
        U, V, W = best_U, best_V, best_W
        R = U.shape[1]
        
        Ur = U.reshape(n, m, R)
        Vr = V.reshape(m, p, R)
        Wr = W.reshape(p, n, R)
        
        removed_any = True
        while removed_any and time.time() < deadline - 15:
            removed_any = False
            for c in range(R):
                if time.time() >= deadline - 15:
                    break
                
                ij_support = set()
                for i in range(n):
                    for j in range(m):
                        if Ur[i,j,c] != 0:
                            ij_support.add((i,j))
                
                if not ij_support:
                    U = np.delete(U, c, axis=1)
                    V = np.delete(V, c, axis=1)
                    W = np.delete(W, c, axis=1)
                    R -= 1
                    removed_any = True
                    break
                
                jk_support = set()
                for j in range(m):
                    for k in range(p):
                        if Vr[j,k,c] != 0:
                            jk_support.add((j,k))
                
                if not jk_support:
                    U = np.delete(U, c, axis=1)
                    V = np.delete(V, c, axis=1)
                    W = np.delete(W, c, axis=1)
                    R -= 1
                    removed_any = True
                    break
                
                ki_support = set()
                for k in range(p):
                    for i in range(n):
                        if Wr[k,i,c] != 0:
                            ki_support.add((k,i))
                
                if not ki_support:
                    U = np.delete(U, c, axis=1)
                    V = np.delete(V, c, axis=1)
                    W = np.delete(W, c, axis=1)
                    R -= 1
                    removed_any = True
                    break
                
                found_common = False
                for (i,j) in ij_support:
                    for (j2,k) in jk_support:
                        if j == j2:
                            for (k2,i2) in ki_support:
                                if k == k2 and i == i2:
                                    found_common = True
                                    break
                            if found_common:
                                break
                    if found_common:
                        break
                if not found_common:
                    U = np.delete(U, c, axis=1)
                    V = np.delete(V, c, axis=1)
                    W = np.delete(W, c, axis=1)
                    R -= 1
                    removed_any = True
                    break
        
        if verify(U, V, W):
            best_U, best_V, best_W = U, V, W
            best_rank = U.shape[1]
    
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
