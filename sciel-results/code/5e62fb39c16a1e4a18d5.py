"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8

    def get_rank14():
        """
        Rank-14 decomposition for 2x2 * 2x4 using two Strassen 2x2 blocks.
        Returns (U, V, W) as numpy arrays of shape (4,14), (8,14), (8,14).
        """
        R = 14
        U = np.zeros((4, R), dtype=int)
        V = np.zeros((8, R), dtype=int)
        W = np.zeros((8, R), dtype=int)

        # Left block (cols 0,1 of B): Strassen on A * B_left where B_left is 2x2
        # B_left[0,0]=b00, B_left[0,1]=b01, B_left[1,0]=b10, B_left[1,1]=b11
        # V indices: j*p+k -> j*4+k
        #   b00 -> V[0], b01 -> V[1], b10 -> V[4], b11 -> V[5]
        # W indices: k*n+i -> k*2+i
        #   C[0,0] -> W[0], C[0,1] -> W[1], C[1,0] -> W[2], C[1,1] -> W[3]

        # L1: (a00+a11)(b00+b11)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[5,0] = 1
        W[0,0] = 1; W[3,0] = 1

        # L2: (a10+a11)b00
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[1,1] = 1; W[3,1] = -1

        # L3: a00(b01-b11)
        U[0,2] = 1
        V[1,2] = 1; V[5,2] = -1
        W[2,2] = 1; W[3,2] = 1

        # L4: a11(b10-b00)
        U[3,3] = 1
        V[4,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[1,3] = 1

        # L5: (a00+a01)b11
        U[0,4] = 1; U[1,4] = 1
        V[5,4] = 1
        W[0,4] = -1; W[2,4] = 1

        # L6: (a10-a00)(b00+b01)
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1

        # L7: (a01-a11)(b10+b11)
        U[1,6] = 1; U[3,6] = -1
        V[4,6] = 1; V[5,6] = 1
        W[0,6] = 1

        # Right block (cols 2,3 of B): Strassen on A * B_right where B_right is 2x2
        # B_right[0,0]=b02, B_right[0,1]=b03, B_right[1,0]=b12, B_right[1,1]=b13
        # V indices:
        #   b02 -> V[2], b03 -> V[3], b12 -> V[6], b13 -> V[7]
        # W indices:
        #   C[0,2] -> W[4], C[0,3] -> W[5], C[1,2] -> W[6], C[1,3] -> W[7]

        # R1: (a00+a11)(b02+b13)
        U[0,7] = 1; U[3,7] = 1
        V[2,7] = 1; V[7,7] = 1
        W[4,7] = 1; W[7,7] = 1

        # R2: (a10+a11)b02
        U[2,8] = 1; U[3,8] = 1
        V[2,8] = 1
        W[5,8] = 1; W[7,8] = -1

        # R3: a00(b03-b13)
        U[0,9] = 1
        V[3,9] = 1; V[7,9] = -1
        W[6,9] = 1; W[7,9] = 1

        # R4: a11(b12-b02)
        U[3,10] = 1
        V[6,10] = 1; V[2,10] = -1
        W[4,10] = 1; W[5,10] = 1

        # R5: (a00+a01)b13
        U[0,11] = 1; U[1,11] = 1
        V[7,11] = 1
        W[4,11] = -1; W[6,11] = 1

        # R6: (a10-a00)(b02+b03)
        U[2,12] = 1; U[0,12] = -1
        V[2,12] = 1; V[3,12] = 1
        W[7,12] = 1

        # R7: (a01-a11)(b12+b13)
        U[1,13] = 1; U[3,13] = -1
        V[6,13] = 1; V[7,13] = 1
        W[4,13] = 1

        return U, V, W

    def verify_np(U, V, W):
        """Fast numpy-based verification using einsum."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        R = Ua.shape[1]
        T = np.zeros((n, m, p))
        for r in range(R):
            u = Ua[:, r].reshape(n, m)
            v = Va[:, r].reshape(m, p)
            w = Wa[:, r].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True

    def verify_exact(U, V, W):
        """Exact verification using integer/rational arithmetic."""
        R = len(U[0])
        T = np.zeros((n, m, p), dtype=int)
        for r in range(R):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        if not np.all(T == 1):
            return False
        return True

    def naive_decomp():
        rank = n * m * p  # 16
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}

    def try_rank13():
        """
        Attempt to find a rank-13 decomposition.
        Strategy: For each pair of terms to remove from rank-14, check if the
        missing tensor is rank-1 (i.e., expressible as a single rank-1 term).
        """
        if time.time() > deadline - 30:
            return None
        
        U14, V14, W14 = get_rank14()
        T_matmul = np.ones((n, m, p), dtype=int)
        
        for remove_i, remove_j in itertools.combinations(range(14), 2):
            if time.time() > deadline - 15:
                return None
            
            # Compute T_missing = T_matmul - sum of remaining 12 terms
            T_missing = T_matmul.copy()
            for r in range(14):
                if r == remove_i or r == remove_j:
                    continue
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_missing[i, j, k] -= (
                                U14[i*m+j, r] * V14[j*p+k, r] * W14[k*n+i, r]
                            )
            
            # Check if T_missing is a rank-1 tensor: T[i,j,k] = u[i,j]*v[j,k]*w[k,i]
            # Necessary and sufficient condition: there exist u,v,w such that this holds.
            
            # Method: If T is nonzero, pick a reference (i0,j0,k0) with T[i0,j0,k0] != 0.
            # Set u[i0,j0] = 1. Then for all (i,j,k):
            #   v[j,k]*w[k,i] = T[i,j,k] / u[i,j]
            # This is underdetermined. Instead, use the following:
            # T[i,j,k] * T[i',j',k'] = T[i,j',k] * T[i',j,k'] * (w[k',i]*v[j',k'] / (w[k,i]*v[j,k]))
            # This doesn't simplify nicely.
            
            # Better: enumerate possible u vectors (small integers) and check.
            # u has 4 entries (i=0,1; j=0,1). Try u entries in {-2,-1,0,1,2}.
            # But u[i,j]=0 implies T[i,j,k]=0 for all k.
            
            # Determine which u[i,j] can be zero:
            u_support = set()
            for i in range(n):
                for j in range(m):
                    if any(T_missing[i, j, k] != 0 for k in range(p)):
                        u_support.add((i, j))
            
            if not u_support:
                # T_missing is zero! Rank 12 found!
                # Build rank-12 decomposition
                U12 = []
                V12 = []
                W12 = []
                for r in range(14):
                    if r != remove_i and r != remove_j:
                        U12.append(U14[:, r].tolist())
                        V12.append(V14[:, r].tolist())
                        W12.append(W14[:, r].tolist())
                # Transpose to (4,12), (8,12), (8,12)
                U12_mat = np.array(U12).T
                V12_mat = np.array(V12).T
                W12_mat = np.array(W12).T
                return (U12_mat.tolist(), V12_mat.tolist(), W12_mat.tolist())
            
            # Try small integer values for u
            u_values = list(u_support)
            for u_assign in itertools.product([-2,-1,1,2], repeat=len(u_values)):
                if time.time() > deadline - 10:
                    return None
                
                u_vec = np.zeros((n, m), dtype=int)
                for idx, (i, j) in enumerate(u_values):
                    u_vec[i, j] = u_assign[idx]
                
                # Now solve for v and w: T[i,j,k] = u[i,j]*v[j,k]*w[k,i]
                # For (i,j) with u[i,j] != 0: v[j,k]*w[k,i] = T[i,j,k]/u[i,j]
                
                v_vec = np.zeros((m, p), dtype=float)
                w_vec = np.zeros((p, n), dtype=float)
                consistent = True
                
                # Initialize v and w from one reference
                # Pick first (i,j) in u_support
                i0, j0 = u_values[0]
                for k in range(p):
                    val = T_missing[i0, j0, k] / u_vec[i0, j0]
                    # v[j0,k]*w[k,i0] = val
                    # Set w[k,i0] = 1, v[j0,k] = val
                    v_vec[j0, k] = val
                    w_vec[k, i0] = 1.0
                
                # Now for other (i,j):
                for (i, j) in u_values[1:]:
                    for k in range(p):
                        val = T_missing[i, j, k]
                        if val == 0:
                            continue
                        # v[j,k]*w[k,i] = val / u[i,j]
                        target = val / u_vec[i, j]
                        
                        # We know v[j,k] and w[k,i] partially
                        # If both are determined, check consistency
                        # If one is undetermined, we can set it
                        # This is complex; let's just check if the product matches
                        # using what we have. For now, skip full solve.
                        pass
                
                # Full verification would be complex. Skip for now.
            
            # Alternative simpler check: use the multiplicative consistency condition
            # For all i1,i2,j1,j2,k1,k2:
            # T[i1,j1,k1]*T[i2,j2,k2] = T[i1,j2,k1]*T[i2,j1,k2] * (w[k2,i1]*v[j2,k2])/(w[k1,i1]*v[j1,k1])
            # This still involves v and w.
            
            # Simplest check: the "Sylvester" type condition.
            # T is rank-1 iff for all i1,i2,j1,j2:
            #   T[i1,j1,k]*T[i2,j2,k] = T[i1,j2,k]*T[i2,j1,k] for all k
            # (i.e., the 2x2 minors of the "flattened" tensor are zero)
            
            valid = True
            for i1 in range(n):
                for i2 in range(n):
                    for j1 in range(m):
                        for j2 in range(m):
                            if i1 == i2 or j1 == j2:
                                continue
                            for k in range(p):
                                a = T_missing[i1, j1, k]
                                b = T_missing[i2, j2, k]
                                c = T_missing[i1, j2, k]
                                d = T_missing[i2, j1, k]
                                if a * b != c * d:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Additional check across k values:
            # T[i,j,k1]*T[i,j,k2] should be consistent with v[j,k]*w[k,i] structure
            # For fixed (i,j): T[i,j,k] = u[i,j]*v[j,k]*w[k,i]
            # So T[i,j,k1]*T[i,j,k2] = u[i,j]^2 * v[j,k1]*w[k1,i]*v[j,k2]*w[k2,i]
            # = u[i,j]^2 * (v[j,k1]*v[j,k2]) * (w[k1,i]*w[k2,i])
            # This should factor as A_{ij} * B_k1k2 where B doesn't depend on (i,j)
            # in a specific way. This is getting too complex.
            
            # Let's just try to construct v and w explicitly.
            # Normalize: for each j, set v[j,0] = 1 (if possible).
            # Then w[k,i] = T[i,j0,k] / (u[i,j0] * v[j0,k]) for some j0 with u[i,j0]!=0.
            
            # Find a j0 that works for all i
            v_vec = np.zeros((m, p), dtype=float)
            w_vec = np.zeros((p, n), dtype=float)
            found_solution = False
            
            for j0 in range(m):
                # Try setting v[j0, k] = 1 for all k
                v_vec[j0, :] = 1.0
                w_vec_ok = True
                for i in range(n):
                    if u_vec[i, j0] == 0:
                        continue
                    for k in range(p):
                        if T_missing[i, j0, k] != 0:
                            w_vec[k, i] = T_missing[i, j0, k] / (u_vec[i, j0] * 1.0)
                        else:
                            w_vec[k, i] = 0.0
                
                # Now verify for all (i,j,k)
                all_good = True
                for i in range(n):
                    for j in range(m):
                        if u_vec[i, j] == 0:
                            if any(T_missing[i, j, k] != 0 for k in range(p)):
                                all_good = False
                                break
                            continue
                        for k in range(p):
                            predicted = u_vec[i, j] * v_vec[j, k] * w_vec[k, i]
                            if abs(predicted - T_missing[i, j, k]) > 1e-9:
                                all_good = False
                                break
                        if not all_good:
                            break
                    if not all_good:
                        break
                
                if all_good:
                    found_solution = True
                    break
            
            if found_solution:
                # Build rank-13 decomposition
                U13 = []
                V13 = []
                W13 = []
                for r in range(14):
                    if r != remove_i and r != remove_j:
                        U13.append(U14[:, r].tolist())
                        V13.append(V14[:, r].tolist())
                        W13.append(W14[:, r].tolist())
                
                # Add the new rank-1 term
                U13.append([u_vec[i, j] for i in range(n) for j in range(m)])
                V13.append([v_vec[j, k] for j in range(m) for k in range(p)])
                W13.append([w_vec[k, i] for k in range(p) for i in range(n)])
                
                # Convert to (4,13), (8,13), (8,13)
                U13_mat = np.array(U13).T
                V13_mat = np.array(V13).T
                W13_mat = np.array(W13).T
                
                # Verify
                if verify_np(U13_mat.tolist(), V13_mat.tolist(), W13_mat.tolist()):
                    return (U13_mat.tolist(), V13_mat.tolist(), W13_mat.tolist())
        
        return None

    # Try rank 13 first (if we can find it)
    result13 = try_rank13()
    if result13 is not None:
        U13, V13, W13 = result13
        if verify_exact(U13, V13, W13):
            return {"U": U13, "V": V13, "W": W13}
    
    # Try the structured rank-14 decomposition
    U, V, W = get_rank14()
    
    # Verify before returning to ensure correctness
    if verify_np(U.tolist(), V.tolist(), W.tolist()):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to naive rank 16 if something went wrong (should not happen)
    return naive_decomp()
# EVOLVE-BLOCK-END
