import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 9]


def construct():
    """
    Find a low-rank exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Strategy: Start from the schoolbook (rank 144) and greedily merge rank-1 terms
    that are "compatible" — i.e., their supports overlap in a way that allows a
    combined rank-2 (or lower) decomposition of their sum. We use a structured
    ansatz: for each pair of schoolbook terms, check if their sum can be written
    as 2 or fewer rank-1 tensors. If so, replace the pair with the compressed form.
    
    Additionally, we try a direct algebraic construction using polynomial
    interpolation over the j-index to reduce rank.
    """
    n, m, p = 4, 4, 9
    deadline = time.time() + 110

    # Build schoolbook decomposition
    R_sb = n * m * p  # 144
    U = np.zeros((n * m, R_sb), dtype=float)
    V = np.zeros((m * p, R_sb), dtype=float)
    W = np.zeros((p * n, R_sb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1

    def verify(U, V, W):
        """Full verification: check all consistent and inconsistent indices."""
        # Consistent: T[a,b,c] = 1 where a=i*m+j, b=j*p+k, c=k*n+i
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a], V[b] * W[c])
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Inconsistent: T[a,b,c] = 0
        for a in range(n * m):
            i1, j1 = divmod(a, m)
            for b in range(m * p):
                j2, k2 = divmod(b, p)
                if j1 != j2:
                    # Check all c
                    for c in range(p * n):
                        val = np.dot(U[a], V[b] * W[c])
                        if abs(val) > 1e-9:
                            return False
                else:
                    # j1 == j2, check i and k consistency
                    for c in range(p * n):
                        k3, i3 = divmod(c, n)
                        if i1 != i3 or k2 != k3:
                            val = np.dot(U[a], V[b] * W[c])
                            if abs(val) > 1e-9:
                                return False
        return True

    # Quick check schoolbook
    if not verify(U, V, W):
        # Should never happen, but safety
        pass

    # Strategy 1: Polynomial interpolation in j to reduce rank
    # Key idea: For fixed (i, k), the tensor entry is 1 for all j in [0,4).
    # The "j-consistency" constraint means: term r contributes to U[i*m+j, r],
    # V[j'*p+k, r], W[k'*n+i', r] only when j=j', i=i', k=k'.
    #
    # Alternative view: Write the multiplication as:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # A rank-R decomposition means:
    # C[i,k] = sum_{r=1}^{R} (linear_form_r(A)) * (linear_form_r(B))
    # where linear_form_r(A) depends on A entries and linear_form_r(B) on B entries.
    #
    # In tensor notation with the trace convention:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{consistent}
    #
    # Let's try a different parameterization. Consider that for each r, the
    # non-zero pattern of U[:,r], V[:,r], W[:,r] defines a "support".
    # 
    # For the schoolbook, each term r corresponds to a unique (i,j,k) triple.
    # We can try to combine terms that share i or k.
    #
    # Construction attempt: Group by i.
    # For each i in [0,4), we have m*p = 36 terms (all j, k for that i).
    # Within a fixed i, the constraint is:
    # sum_{j,k} [term for (i,j,k)] must give C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # This is just the rank of the 4x9 matrix multiplication restricted to one row
    # of A and one "column-block" of B. The rank of multiplying a 1x4 row by a 4x9
    # matrix is at most 4 (since it's just a dot product for each output).
    # Actually, C[i,k] = sum_j A[i,j] B[j,k]. For fixed i, this is 4 separate
    # products A[i,j]*B[j,k] summed over j. The bilinear rank is m=4 (each j
    # gives one product). So for each i, we need rank at least 4, and we can do
    # exactly 4 by letting each term correspond to one j.
    #
    # So: R = n * m = 4 * 4 = 16? Let's check.
    # r indexes (i, j) with r = i*m + j
    # U[i'*m+j', r] = 1 if i'=i and j'=j, else 0  -> U is identity-like
    # V[j'*p+k', r] = 1 if j'=j (for any k'), else 0
    # W[k'*n+i', r] = 1 if i'=i (for any k'), else 0
    #
    # Check consistent: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # = sum_{(i',j')} [i'=i][j'=j] * [j'=j] * [i'=i]
    # = 1 * 1 * 1 = 1. Correct!
    #
    # Check inconsistent: say a=i1*m+j1, b=j2*p+k2, c=k3*n+i3 with j1!=j2.
    # sum_r U[a,r]*V[b,r]*W[c,r]
    # = sum_{(i',j')} [i1=i'][j1=j'] * [j2=j'] * [i3=i']
    # = [i1=i3] * [j1=j'] * [j2=j'] summed over j'
    # = [i1=i3] * sum_{j'} [j1=j'][j2=j']
    # = [i1=i3] * [j1==j2]
    # If j1!=j2, this is 0. Good.
    #
    # But what if j1==j2 but i1!=i3?
    # sum = [i1=i3] * [j1==j2] = 0 if i1!=i3. Good.
    #
    # What if i1==i3, j1==j2, but k2!=k3?
    # The sum doesn't depend on k2 or k3! It would be 1, but T should be 0.
    # PROBLEM: k is not checked.
    
    # So R=16 doesn't work because k consistency isn't enforced.
    
    # We need to also enforce k consistency. Let's try R = n*m*p but with a
    # different structure. Or use polynomial in k.
    
    # Actually, let me reconsider. The issue is that for fixed (i,j), the term
    # must "know" about k. In the schoolbook, each (i,j,k) is separate.
    
    # Let's try: R = m * p = 36, r = j*p + k
    # U[i*m+j', r] = 1 if j'=j (regardless of i)
    # V[j'*p+k', r] = 1 if j'=j and k'=k
    # W[k'*n+i', r] = 1 if k'=k (regardless of i')
    #
    # Consistent: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # = sum_{(j',k')} [j=j'] * [j=j',k=k'] * [k=k']
    # = 1. Correct!
    #
    # Inconsistent with j1!=j2:
    # sum_{(j',k')} [j1=j'] * [j2=j'] * [k2=k'] 
    # = sum_{j',k'} [j1=j'][j2=j'][k2=k']
    # = [j1==j2] * sum_{k'}[k2=k'] = [j1==j2]. If j1!=j2, 0. Good.
    #
    # Inconsistent with i1!=i3 (but j1==j2, k2==k3):
    # sum = [j1=j1] * [j2=j1,k2=k2] * [k2=k2] = 1. But T=0. PROBLEM.
    
    # Same issue: i not enforced.
    
    # The fundamental problem: we need to enforce all three: i consistency,
    # j consistency, k consistency. The schoolbook does this with one term per
    # (i,j,k). Can we do better?
    
    # Let's try R = n * p = 36, r = i*p + k
    # U[i'*m+j, r] = 1 if i'=i (regardless of j)
    # V[j'*p+k', r] = 1 if k'=k (regardless of j')
    # W[k'*n+i', r] = 1 if k'=k and i'=i
    #
    # Consistent: sum_{(i',k')} [i=i'] * [k=k'] * [k=k',i=i'] = 1. Good.
    # Inconsistent j1!=j2: sum doesn't depend on j. = [i1=i3][k2=k3]. If j1!=j2
    # but i1=i3, k2=k3, this gives 1. PROBLEM.
    
    # It seems hard to enforce all three with fewer than n*m*p terms using
    # simple product structures. Let me try a more sophisticated approach.
    
    # Use the fact that m=4 is small. For each (i,k) pair, we need:
    # sum_j A[i,j]*B[j,k] = C[i,k]
    # This is a dot product of length 4. The bilinear rank of a dot product of
    # length m is m (you need m multiplications). So for each (i,k), we need
    # m "multiplication slots". There are n*p = 36 such pairs, so naively
    # 36*4 = 144. But can we share multiplication slots?
    
    # A multiplication slot r corresponds to a product (linear in A)(linear in B).
    # For slot r to contribute to C[i,k], we need:
    # U[i*m+j, r] and V[j*p+k, r] to be nonzero for the j's involved.
    
    # Let's try a structured search: use small integers and try to find a
    # solution with R < 144 by solving a system of equations.
    
    # For a given R, we have R variables per row of U, V, W. The constraints are:
    # For each consistent (i,j,k): sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1
    # For each inconsistent (a,b,c): sum_r U[a,r] V[b,r] W[c,r] = 0
    
    # This is a system of trilinear equations in 3*R*n*m + R*m*p + R*p*n variables.
    # For R=144, that's 144*16 + 144*36 + 144*36 = 2304 + 5184 + 5184 = 12672 vars.
    # Too many for direct solve.
    
    # Alternative: fix U and V, solve for W linearly? No, it's trilinear.
    
    # Let me try a different strategy: exploit symmetry and use known results.
    # The rank of matrix multiplication (4,4,9) is at most the rank of (4,4,4) 
    # plus rank of (4,4,5) or similar decompositions.
    
    # Actually, let's just try to find any improvement by random search with
    # structured initializations, focusing on the j-consistency constraint.
    
    # New idea: For each j in [0,4), handle the j-block separately.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # If we could compute each A[i,j]*B[j,k] with shared structure...
    
    # Let's try: R = n * p + (m-1)*n*p = p*(n + (m-1)*n) = p*n*m = 144. Same.
    
    # I think the schoolbook might be close to optimal for this specific shape.
    # Let me verify by trying a few specific low-rank constructions more carefully.
    
    # Try R = 72: pair up k values. For k in {0,1}, {2,3}, {4,5}, {6,7}, and {8}
    # Use interpolation in k for pairs.
    
    # For a pair (k1, k2), can we compute A[i,j]*B[j,k1] + A[i,j]*B[j,k2]
    # in fewer than 2 terms? No, they're independent products.
    
    # I'll try one more thing: a randomized search with a smarter initialization.
    
    # Initialize U, V, W with the schoolbook pattern but perturb slightly,
    # then try to reduce rank by setting some columns to zero.
    
    # Actually, let's try the most promising structured approach:
    # Use the fact that 4x4 matrix multiplication has rank at most 49 (Strassen).
    # For 4x4 * 4x9, we can split B into three 4x3 or 4x4+4x4+4x1 blocks.
    
    # 4x4 * 4x4 has rank <= 49 (actually, the best known is 49 for 4x4, but
    # I don't have the explicit construction). 4x4 * 4x1 has rank 4.
    # 4x4 * 4x5: unknown, but at most 4*4*5=80, probably less.
    
    # Given I don't have explicit Strassen factors for 4x4, let me try to
    # construct them via a small search for the 4x4 case first.
    
    # For 4x4*4x4: n=4, m=4, p=4. U is (16,R), V is (16,R), W is (16,R).
    # Try to find R < 64.
    
    def find_4x4_decomp(max_R=50, time_budget=30):
        """Try to find a low-rank decomp for 4x4*4x4."""
        n4 = m4 = p4 = 4
        deadline4 = time.time() + time_budget
        
        # Schoolbook for 4x4
        R4 = 64
        U4 = np.zeros((16, R4))
        V4 = np.zeros((16, R4))
        W4 = np.zeros((16, R4))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U4[i*4+j, col] = 1
                    V4[j*4+k, col] = 1
                    W4[k*4+i, col] = 1
                    col += 1
        
        # Try random search with small integer entries
        np.random.seed(42)
        for R in range(48, 35, -2):
            if time.time() > deadline4:
                break
            for trial in range(50):
                if time.time() > deadline4:
                    break
                U_t = np.random.randint(-2, 3, (16, R))
                V_t = np.random.randint(-2, 3, (16, R))
                W_t = np.random.randint(-2, 3, (16, R))
                
                # Quick check: consistent indices
                ok = True
                for i in range(4):
                    for j in range(4):
                        for k in range(4):
                            val = np.dot(U_t[i*4+j], V_t[j*4+k] * W_t[k*4+i])
                            if abs(val - 1) > 1e-9:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                
                # Full check
                for a in range(16):
                    i1, j1 = divmod(a, 4)
                    for b in range(16):
                        j2, k2 = divmod(b, 4)
                        if j1 != j2:
                            for c in range(16):
                                if abs(np.dot(U_t[a], V_t[b] * W_t[c])) > 1e-9:
                                    ok = False
                                    break
                            if not ok:
                                break
                        else:
                            for c in range(16):
                                k3, i3 = divmod(c, 4)
                                if i1 != i3 or k2 != k3:
                                    if abs(np.dot(U_t[a], V_t[b] * W_t[c])) > 1e-9:
                                        ok = False
                                        break
                                if not ok:
                                    break
                        if not ok:
                            break
                    if not ok:
                        break
                if ok:
                    return U_t, V_t, W_t, R
        
        return None, None, None, R4

    # Don't spend too much time on this; just try briefly
    if time.time() < deadline - 60:
        U4, V4, W4, R4 = find_4x4_decomp(time_budget=25)
        if R4 < 64:
            print(f"Found 4x4 decomp with rank {R4}")
            # Use it to build 4x4*4x9: split into 4x4*4x4 + 4x4*4x4 + 4x4*4x1
            # Total rank: R4 + R4 + 4 = 2*R4 + 4
            R_total = 2 * R4 + 4
            U_tot = np.zeros((16, R_total))
            V_tot = np.zeros((36, R_total))
            W_tot = np.zeros((36, R_total))
            
            # Block 1: k in [0,4)
            U_tot[:, 0:R4] = U4
            V_tot[0:16, 0:R4] = V4  # j*p+k for k in [0,4), j in [0,4)
            W_tot[0:16, 0:R4] = W4   # k*n+i for k in [0,4)
            
            # Block 2: k in [4,8)
            U_tot[:, R4:2*R4] = U4
            for j in range(4):
                for k in range(4):
                    V_tot[j*9 + (k+4), R4:2*R4] = V4[j*4 + k, :]
            for k in range(4):
                for i in range(4):
                    W_tot[(k+4)*4 + i, R4:2*R4] = W4[k*4 + i, :]
            
            # Block 3: k=8
            for i in range(4):
                for j in range(4):
                    U_tot[i*4+j, 2*R4 + j] = 1
                V_tot[j*9 + 8, 2*R4 + j] = 1
            for i in range(4):
                W_tot[8*4 + i, 2*R4 + i] = 1
            
            if verify(U_tot, V_tot, W_tot):
                return {"U": U_tot.tolist(), "V": V_tot.tolist(), "W": W_tot.tolist()}
    
    # If we get here, return schoolbook
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
