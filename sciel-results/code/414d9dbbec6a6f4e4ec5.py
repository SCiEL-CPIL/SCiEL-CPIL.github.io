"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    nm, mp, pn = n * m, m * p, p * n  # 15, 35, 21
    
    def verify(U, V, W):
        """Verify that the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = 1.0
                    rhs = 0.0
                    for c in range(U.shape[1]):
                        rhs += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(lhs - rhs) > 1e-9:
                        return False
        return True
    
    def schoolbook():
        """Standard schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        for c in range(rank):
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W, rank
    
    # Build the tensor T[i,j,k] = 1 for all i,j,k
    # We want to find rank-1 decompositions: T = sum_c U_c ⊗ V_c ⊗ W_c
    # where U_c is a vector of length nm, V_c of length mp, W_c of length pn
    
    def try_strassen_block(U, V, W, I, J, K):
        """
        Try to replace schoolbook terms for (i,j,k) in I×J×K with a Strassen-like
        rank-7 decomposition. Returns (success, new_U, new_V, new_W, terms_to_remove)
        """
        i0, i1 = sorted(I)
        j0, j1 = sorted(J)
        k0, k1 = sorted(K)
        
        # The 8 schoolbook terms to be replaced:
        # (i0,j0,k0), (i0,j0,k1), (i0,j1,k0), (i0,j1,k1)
        # (i1,j0,k0), (i1,j0,k1), (i1,j1,k0), (i1,j1,k1)
        
        # Strassen 7 terms for 2x2x2:
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        # M2 = (A10+A11)B00       -> C10 += M2, C11 -= M2
        # M3 = A00(B01-B11)       -> C01 += M3, C11 += M3
        # M4 = A11(B10-B00)       -> C00 += M4, C10 += M4
        # M5 = (A00+A01)B11       -> C00 -= M5, C01 += M5
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        
        new_terms = []
        
        # M1: (A[i0,j0]+A[i1,j1])(B[j0,k0]+B[j1,k1])
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m+j0] = 1.0; u[i1*m+j1] = 1.0
        v[j0*p+k0] = 1.0; v[j1*p+k1] = 1.0
        w[k0*n+i0] = 1.0; w[k1*n+i1] = 1.0
        new_terms.append((u, v, w))
        
        # M2: (A[i1,j0]+A[i1,j1])B[j0,k0]
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i1*m+j0] = 1.0; u[i1*m+j1] = 1.0
        v[j0*p+k0] = 1.0
        w[k0*n+i1] = 1.0; w[k1*n+i1] = -1.0
        new_terms.append((u, v, w))
        
        # M3: A[i0,j0](B[j0,k1]-B[j1,k1])
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m+j0] = 1.0
        v[j0*p+k1] = 1.0; v[j1*p+k1] = -1.0
        w[k1*n+i0] = 1.0; w[k1*n+i1] = 1.0
        new_terms.append((u, v, w))
        
        # M4: A[i1,j1](B[j1,k0]-B[j0,k0])
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i1*m+j1] = 1.0
        v[j1*p+k0] = 1.0; v[j0*p+k0] = -1.0
        w[k0*n+i0] = 1.0; w[k0*n+i1] = 1.0
        new_terms.append((u, v, w))
        
        # M5: (A[i0,j0]+A[i0,j1])B[j1,k1]
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m+j0] = 1.0; u[i0*m+j1] = 1.0
        v[j1*p+k1] = 1.0
        w[k0*n+i0] = -1.0; w[k1*n+i0] = 1.0
        new_terms.append((u, v, w))
        
        # M6: (A[i1,j0]-A[i0,j0])(B[j0,k0]+B[j0,k1])
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i1*m+j0] = 1.0; u[i0*m+j0] = -1.0
        v[j0*p+k0] = 1.0; v[j0*p+k1] = 1.0
        w[k1*n+i1] = 1.0
        new_terms.append((u, v, w))
        
        # M7: (A[i0,j1]-A[i1,j1])(B[j1,k0]+B[j1,k1])
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m+j1] = 1.0; u[i1*m+j1] = -1.0
        v[j1*p+k0] = 1.0; v[j1*p+k1] = 1.0
        w[k0*n+i0] = 1.0
        new_terms.append((u, v, w))
        
        return new_terms
    
    def count_covered(blocks):
        """Count how many distinct (i,j,k) tuples are covered by the blocks."""
        covered = set()
        for I, J, K in blocks:
            for i in I:
                for j in J:
                    for k in K:
                        covered.add((i, j, k))
        return len(covered)
    
    # Strategy: Find the best set of disjoint 2x2x2 blocks to apply Strassen to.
    # Each block saves 1 rank (8 schoolbook -> 7 Strassen).
    # We want to maximize the number of blocks, subject to disjointness.
    
    # Generate all possible 2x2x2 blocks
    all_blocks = []
    for i0 in range(n):
        for i1 in range(i0+1, n):
            for j0 in range(m):
                for j1 in range(j0+1, m):
                    for k0 in range(p):
                        for k1 in range(k0+1, p):
                            all_blocks.append(({i0, i1}, {j0, j1}, {k0, k1}))
    
    # Greedy: pick blocks that don't overlap with already-picked ones
    # Try multiple orderings to find a good set
    import random
    random.seed(42)
    
    best_blocks = []
    best_count = 0
    
    # Try greedy with different random orderings
    for trial in range(200):
        if time.time() > deadline - 20:
            break
        shuffled = all_blocks[:]
        random.shuffle(shuffled)
        chosen = []
        used = set()
        for I, J, K in shuffled:
            # Check if this block overlaps with any chosen block
            overlap = False
            for i in I:
                for j in J:
                    for k in K:
                        if (i, j, k) in used:
                            overlap = True
                            break
                    if overlap:
                        break
                if overlap:
                    break
            if not overlap:
                chosen.append((I, J, K))
                for i in I:
                    for j in J:
                        for k in K:
                            used.add((i, j, k))
        if len(chosen) > best_count:
            best_count = len(chosen)
            best_blocks = chosen[:]
    
    # Also try a more systematic approach: sort by some heuristic
    # Blocks that use "middle" indices might be more flexible
    def block_heuristic(block):
        I, J, K = block
        # Prefer blocks that use indices closer to the middle
        center_i = (n-1)/2.0
        center_j = (m-1)/2.0
        center_k = (p-1)/2.0
        dist = 0
        for i in I:
            dist += abs(i - center_i)
        for j in J:
            dist += abs(j - center_j)
        for k in K:
            dist += abs(k - center_k)
        return dist
    
    sorted_blocks = sorted(all_blocks, key=block_heuristic)
    chosen = []
    used = set()
    for I, J, K in sorted_blocks:
        overlap = False
        for i in I:
            for j in J:
                for k in K:
                    if (i, j, k) in used:
                        overlap = True
                        break
                if overlap:
                    break
            if overlap:
                break
        if not overlap:
            chosen.append((I, J, K))
            for i in I:
                for j in J:
                    for k in K:
                        used.add((i, j, k))
    if len(chosen) > best_count:
        best_count = len(chosen)
        best_blocks = chosen[:]
    
    # Build the decomposition using the best blocks
    U_list = []
    V_list = []
    W_list = []
    
    covered = set()
    for I, J, K in best_blocks:
        terms = try_strassen_block(None, None, None, I, J, K)
        for u, v, w in terms:
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
        for i in I:
            for j in J:
                for k in K:
                    covered.add((i, j, k))
    
    # Add schoolbook terms for uncovered (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    u = np.zeros(nm)
                    v = np.zeros(mp)
                    w = np.zeros(pn)
                    u[i*m + j] = 1.0
                    v[j*p + k] = 1.0
                    w[k*n + i] = 1.0
                    U_list.append(u)
                    V_list.append(v)
                    W_list.append(w)
    
    rank = len(U_list)
    U_mat = np.array(U_list).T
    V_mat = np.array(V_list).T
    W_mat = np.array(W_list).T
    
    # Verify
    if verify(U_mat, V_mat, W_mat):
        # Try to reduce rank further by checking for linear dependencies
        # Use SVD to check if any rank-1 terms are redundant
        if time.time() < deadline - 30:
            # Check for linear dependencies in the combined tensor
            # Flatten each rank-1 term into a vector of length nm*mp*pn
            nmmpn = nm * mp * pn
            terms_flat = np.zeros((rank, nmmpn))
            for c in range(rank):
                u = U_mat[:, c]
                v = V_mat[:, c]
                w = W_mat[:, c]
                tensor = np.einsum('i,j,k->ijk', u, v, w)
                terms_flat[c] = tensor.flatten()
            
            # Try to remove terms by checking if removing a term still allows
            # the remaining terms to span the original tensor
            # This is expensive, so we do a limited version
            # Just check if any term is a linear combination of others
            # by looking at the row rank of terms_flat
            # But we need the sum to equal the target tensor, not just span
            
            # Alternative: try removing each term one at a time and check
            # if the remaining terms can still produce the correct result
            # This is O(rank * verify_cost) which is too expensive for large rank
            # So we skip this for now and just return what we have.
            
            pass
        
        return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
    else:
        # Fallback to schoolbook
        U_sb, V_sb, W_sb, _ = schoolbook()
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_mat.tolist()}

# EVOLVE-BLOCK-END
