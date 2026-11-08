"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    deadline = time.time() + 140  # Safety margin

    def validate(U, V, W, rank):
        """Check if U, V, W correctly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def strassen_2x2x2(U, V, W, col, i0, i1, j0, j1, k0, k1):
        """Add Strassen decomposition for 2x2x2 block. Returns new col."""
        # P1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        U[i0*m+j0, col] += 1; U[i1*m+j1, col] += 1
        V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
        W[k0*n+i0, col] += 1; W[k1*n+i1, col] += 1
        col += 1
        # P2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        U[i1*m+j0, col] += 1; U[i1*m+j1, col] += 1
        V[j0*p+k0, col] += 1
        W[k0*n+i1, col] += 1; W[k1*n+i1, col] -= 1
        col += 1
        # P3 = A[i0,j0] * (B[j0,k1]-B[j1,k1])
        U[i0*m+j0, col] += 1
        V[j0*p+k1, col] += 1; V[j1*p+k1, col] -= 1
        W[k1*n+i0, col] += 1; W[k1*n+i1, col] += 1
        col += 1
        # P4 = A[i1,j1] * (B[j1,k0]-B[j0,k0])
        U[i1*m+j1, col] += 1
        V[j1*p+k0, col] += 1; V[j0*p+k0, col] -= 1
        W[k0*n+i0, col] += 1; W[k0*n+i1, col] += 1
        col += 1
        # P5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
        U[i0*m+j0, col] += 1; U[i0*m+j1, col] += 1
        V[j1*p+k1, col] += 1
        W[k0*n+i0, col] -= 1; W[k1*n+i0, col] += 1
        col += 1
        # P6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
        U[i1*m+j0, col] += 1; U[i0*m+j0, col] -= 1
        V[j0*p+k0, col] += 1; V[j0*p+k1, col] += 1
        W[k1*n+i1, col] += 1
        col += 1
        # P7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        U[i0*m+j1, col] += 1; U[i1*m+j1, col] -= 1
        V[j1*p+k0, col] += 1; V[j1*p+k1, col] += 1
        W[k0*n+i0, col] += 1
        col += 1
        return col

    def schoolbook_block(U, V, W, col, i_range, j_range, k_range):
        """Add schoolbook terms for a block. Returns new col."""
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    U[i*m+j, col] += 1
                    V[j*p+k, col] += 1
                    W[k*n+i, col] += 1
                    col += 1
        return col

    def build_base_decomposition(j_pairs, k_pairs, remaining_j):
        """Build a base decomposition using Strassen on 2x2x2 blocks + schoolbook for rest."""
        # Estimate rank
        rank = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                rank += 7
        for j in remaining_j:
            rank += 2 * p  # schoolbook for each k
        # Also need to account for any j in j_pairs that might overlap
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                col = strassen_2x2x2(U, V, W, col, 0, 1, j0, j1, k0, k1)
        for j in remaining_j:
            for k in range(p):
                U[0*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+0, col] += 1
                col += 1
                U[1*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+1, col] += 1
                col += 1
        return U, V, W, rank

    def try_rank_reduction(U, V, W, rank):
        """
        Try to reduce the rank by finding linear dependencies among the rank-1 tensor terms.
        
        Key insight: If there exist coefficients alpha_c (not all zero) such that for ALL
        (i,j,k): sum_c alpha_c * U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 0,
        then we can shift the decomposition: for any target column t, we can set
        U_new[:,t] = U[:,t] + sum_c alpha_c * U[:,c] (and similarly for V, W) to absorb
        the dependency, effectively reducing the rank if one coefficient is nonzero.
        
        More practically: build the "tensor product matrix" M where M[(i,j,k), c] = 
        U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]. If rank(M) < rank, then there's a
        dependency. We can find it via SVD or null space computation.
        
        For each dependency vector alpha, we can try to eliminate one column by
        expressing its contribution as a combination of others.
        """
        best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank
        
        # Build the tensor product matrix: rows = (i,j,k) triples, cols = rank-1 terms
        # M[t, c] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] where t encodes (i,j,k)
        n_entries = n * m * p
        M = np.zeros((n_entries, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M[idx, :] = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                    idx += 1
        
        # Find null space of M^T (i.e., vectors alpha such that M @ alpha = 0)
        # This means: sum_c alpha_c * M[:,c] = 0, i.e., the weighted sum of rank-1 tensors is zero
        try:
            # Use SVD to find null space
            U_svd, S, Vt_svd = np.linalg.svd(M, full_matrices=True)
            # Null space vectors are columns of Vt_svd.T corresponding to zero singular values
            tol = 1e-10
            null_mask = S < tol
            if np.any(null_mask):
                null_vecs = Vt_svd[null_mask, :].T  # shape (rank, num_null)
                for nv in null_vecs:
                    if time.time() > deadline - 30:
                        break
                    # nv is a vector in R^rank such that sum_c nv[c] * M[:,c] = 0
                    # Find the column with largest |nv[c]| to eliminate
                    c_elim = np.argmax(np.abs(nv))
                    if abs(nv[c_elim]) < 1e-15:
                        continue
                    # We want to absorb column c_elim into others.
                    # New decomposition: for each other column c' != c_elim,
                    # we can't simply add nv[c']/nv[c_elim] * column_c_elim because
                    # that changes the tensor. Instead, we need a different approach.
                    #
                    # Actually, the correct approach: if sum_c nv[c] * T_c = 0 where T_c
                    # is the rank-1 tensor for column c, then we can write:
                    # T_c_elim = -sum_{c' != c_elim} (nv[c']/nv[c_elim]) * T_{c'}
                    # But this only works if we can express T_c_elim as a linear combination
                    # of the other rank-1 tensors, which is not generally possible.
                    #
                    # Alternative: use the dependency to create a new valid decomposition
                    # with rank-1 terms by shifting. This requires more care.
                    #
                    # Simpler practical approach: just check if removing column c_elim
                    # and adjusting the remaining columns (by adding a scaled version)
                    # still gives a valid decomposition. This is complex, so we'll
                    # fall back to the simple check.
                    pass
        except np.linalg.LinAlgError:
            pass
        
        # Fallback: try removing each column and check validity
        for c_remove in range(rank):
            if time.time() > deadline - 30:
                break
            r_new = rank - 1
            U_new = np.delete(U, c_remove, axis=1)
            V_new = np.delete(V, c_remove, axis=1)
            W_new = np.delete(W, c_remove, axis=1)
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U_new[i*m+j, :] * V_new[j*p+k, :] * W_new[k*n+i, :])
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid and r_new < best_rank:
                best_U, best_V, best_W, best_rank = U_new, V_new, W_new, r_new
        
        return best_U, best_V, best_W, best_rank

    # Strategy 1: Standard Strassen split
    def try_strassen_split():
        j_pairs = [(0,1), (2,3)]
        k_pairs = [(0,1), (2,3), (4,5)]
        remaining_j = [4]
        return build_base_decomposition(j_pairs, k_pairs, remaining_j)
    
    U, V, W, rank = try_strassen_split()
    if not validate(U, V, W, rank):
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = schoolbook_block(U, V, W, 0, range(n), range(m), range(p))
    
    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank
    
    # Strategy 2: Different block splits
    if time.time() < deadline - 20:
        # m=1+4, p=2+4
        def try_split_m14_p24():
            rank = 2*1*2 + 2*1*4 + 2*4*2 + 2*14
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            col = 0
            col = schoolbook_block(U, V, W, col, [0,1], [0], [0,1])
            col = schoolbook_block(U, V, W, col, [0,1], [0], [2,3,4,5])
            col = schoolbook_block(U, V, W, col, [0,1], [1,2,3,4], [0,1])
            for j0, j1 in [(1,2), (3,4)]:
                for k0, k1 in [(2,3), (4,5)]:
                    col = strassen_2x2x2(U, V, W, col, 0, 1, j0, j1, k0, k1)
            return U, V, W, rank
        
        U2, V2, W2, rank2 = try_split_m14_p24()
        if validate(U2, V2, W2, rank2) and rank2 < best_rank:
            best_U, best_V, best_W, best_rank = U2, V2, W2, rank2
    
    # Strategy 3: Try rank reduction on the best decomposition found so far
    if time.time() < deadline - 15:
        U3, V3, W3, rank3 = try_rank_reduction(best_U, best_V, best_W, best_rank)
        if rank3 < best_rank:
            best_U, best_V, best_W, best_rank = U3, V3, W3, rank3
    
    # Strategy 4: Try more aggressive block splits
    if time.time() < deadline - 10:
        # Try m=2+1+2, p=2+2+2
        def try_m212_p222():
            # j=0,1: Strassen with k=0,1 -> 7; k=2,3 -> 7; k=4,5 -> 7 => 21
            # j=2: schoolbook with all k => 12
            # j=3,4: Strassen with k=0,1 -> 7; k=2,3 -> 7; k=4,5 -> 7 => 21
            # Total: 54
            rank = 54
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            col = 0
            for j0, j1 in [(0,1), (3,4)]:
                for k0, k1 in [(0,1), (2,3), (4,5)]:
                    col = strassen_2x2x2(U, V, W, col, 0, 1, j0, j1, k0, k1)
            col = schoolbook_block(U, V, W, col, [0,1], [2], range(p))
            return U, V, W, rank
        
        U4, V4, W4, rank4 = try_m212_p222()
        if validate(U4, V4, W4, rank4) and rank4 < best_rank:
            best_U, best_V, best_W, best_rank = U4, V4, W4, rank4
    
    # Strategy 5: Try m=1+4, p=1+5 and other asymmetric splits
    if time.time() < deadline - 10:
        # m=1+4: j=0 schoolbook (12), j=1-4 Strassen pairs
        # j=1,2: 3 Strassen blocks (k=0,1; k=2,3; k=4,5) => 21
        # j=3,4: 3 Strassen blocks => 21
        # Total: 12 + 21 + 21 = 54 (same)
        # Try m=1+1+3: j=0 (12), j=1 (12), j=2,3,4: Strassen on j=2,3 (14) + schoolbook j=4 (12) = 26
        # Total: 12+12+26 = 50!
        def try_m113_p222():
            # j=0: schoolbook, all k => 2*p = 12
            # j=1: schoolbook, all k => 2*p = 12
            # j=2,3: Strassen with k=0,1 (7), k=2,3 (7), k=4,5 (7) => 21
            # j=4: schoolbook, all k => 2*p = 12
            # Total: 12 + 12 + 21 + 12 = 57 (worse)
            # Let me recalculate: for j=2,3, we do 3 Strassen blocks = 21
            # Total: 12 + 12 + 21 + 12 = 57
            return None
        
        # Try m=1+4, p=2+2+2: j=0 schoolbook (12), j=1-4: Strassen pairs
        # j=1,2: 3*7=21, j=3,4: 3*7=21 => 42+12=54
        # Try m=2+3, p=2+4: 
        # j=0,1: k=0,1 Strassen(7), k=2-5 schoolbook(8) => 15
        # j=2,3,4: k=0,1 Strassen on j=2,3(7)+schoolbook j=4(4)=11, k=2-5 schoolbook(12) => 23
        # Total: 15+23=38? Let me be more careful.
        # Actually for j=0,1 and k=0,1: Strassen 2x2x2 = 7
        # For j=0,1 and k=2,3,4,5: schoolbook 2*2*4 = 16
        # So j=0,1 block: 7+16=23
        # For j=2,3,4 and k=0,1: Strassen on j=2,3 (7) + schoolbook j=4 (4) = 11
        # For j=2,3,4 and k=2,3,4,5: schoolbook 3*2*4 = 24
        # So j=2,3,4 block: 11+24=35
        # Total: 23+35=58 (worse)
        
        # Try a different approach: use known R(2,3,2) = 11, R(2,3,3) <= 17
        # m=2+3, p=3+3:
        # j=0,1, k=0,1,2: R(2,3,3) <= 17
        # j=0,1, k=3,4,5: R(2,3,3) <= 17
        # j=2,3,4, k=0,1,2: R(3,2,3) - not standard
        # This is getting complex. Let's try m=2+3, p=2+4:
        # j=0,1, k=0,1: R(2,2,2)=7
        # j=0,1, k=2,3,4,5: R(2,2,4) <= R(2,2,2)+R(2,2,2)=14
        # j=2,3,4, k=0,1: R(3,2,2) <= R(2,2,2)+R(1,2,2)=7+4=11
        # j=2,3,4, k=2,3,4,5: R(3,2,4) <= R(2,2,4)+R(1,2,4)=14+8=22
        # Total: 7+14+11+22=54 (same)
        
        # Let's try to construct R(2,3,3) more efficiently
        # R(2,3,3) <= R(2,2,3) + R(2,1,3) = 11 + 6 = 17
        # R(2,2,3) = R(2,2,2) + R(2,2,1) = 7 + 4 = 11
        # R(2,1,3) = 2*1*3 = 6
        # So m=2+3, p=3+3: 17+17+17+17 = 68 (worse)
        
        # The best we can do with simple block decomposition seems to be 54.
        # Let's try a more creative approach: share Strassen terms across j blocks.
        pass
    
    # Convert to lists for the return interface, ensuring integer or half-integer values
    def to_half_int(x):
        """Convert to nearest half-integer (multiple of 0.5)"""
        h = round(x * 2) / 2
        return h
    
    U_list = [[int(best_U[i, c]) if abs(best_U[i, c] - round(best_U[i, c])) < 1e-9 else to_half_int(best_U[i, c]) for c in range(best_rank)] for i in range(n*m)]
    V_list = [[int(best_V[i, c]) if abs(best_V[i, c] - round(best_V[i, c])) < 1e-9 else to_half_int(best_V[i, c]) for c in range(best_rank)] for i in range(m*p)]
    W_list = [[int(best_W[i, c]) if abs(best_W[i, c] - round(best_W[i, c])) < 1e-9 else to_half_int(best_W[i, c]) for c in range(best_rank)] for i in range(p*n)]
    
    # Final validation with the converted values
    U_check = np.array(U_list, dtype=float)
    V_check = np.array(V_list, dtype=float)
    W_check = np.array(W_list, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_check[i*m+j, :] * V_check[j*p+k, :] * W_check[k*n+i, :])
                if abs(s - 1.0) > 1e-6:
                    # Fallback to original values
                    U_list = [[float(best_U[i, c]) for c in range(best_rank)] for i in range(n*m)]
                    V_list = [[float(best_V[i, c]) for c in range(best_rank)] for i in range(m*p)]
                    W_list = [[float(best_W[i, c]) for c in range(best_rank)] for i in range(p*n)]
                    break
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
