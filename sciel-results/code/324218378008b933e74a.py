import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 3]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 by 2x3 matrix multiplication.
    Primary: known rank-11 decomposition (Strassen-like for 2x2 block + schoolbook for col 3).
    Secondary: systematic search for rank-10 using structured patterns.
    """
    n, m, p = 2, 2, 3
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s limit

    def verify(U, V, W):
        """Verify exact decomposition. U:(nm,rank), V:(mp,rank), W:(pn,rank)."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        target = 1 if j == jp else 0
                        val = 0
                        for r in range(rank):
                            val += U[i*m + j, r] * V[jp*p + k, r] * W[k*n + i, r]
                        if val != target:
                            return False
        return True

    def verify_list(U, V, W):
        """Verify from list-of-lists format."""
        rank = len(U[0])
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        return verify(U, V, W)

    # ============================================================
    # KNOWN RANK-11 DECOMPOSITION (verified, guaranteed valid)
    # ============================================================
    def rank11_solution():
        rank = 11
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)

        def set_u(i, j, r, val): U[i*m+j, r] = val
        def set_v(j, k, r, val): V[j*p+k, r] = val
        def set_w(k, i, r, val): W[k*n+i, r] = val

        # Term 1: (A00+A11)*(B00+B11)
        set_u(0,0,0,1); set_u(1,1,0,1)
        set_v(0,0,0,1); set_v(1,1,0,1)
        set_w(0,0,0,1); set_w(1,1,0,1)

        # Term 2: (A10+A11)*B00
        set_u(1,0,1,1); set_u(1,1,1,1)
        set_v(0,0,1,1)
        set_w(0,1,1,1); set_w(1,1,1,-1)

        # Term 3: A00*(B01-B11)
        set_u(0,0,2,1)
        set_v(0,1,2,1); set_v(1,1,2,-1)
        set_w(1,0,2,1); set_w(1,1,2,1)

        # Term 4: A11*(B10-B00)
        set_u(1,1,3,1)
        set_v(1,0,3,1); set_v(0,0,3,-1)
        set_w(0,0,3,1); set_w(0,1,3,1)

        # Term 5: (A00+A01)*B11
        set_u(0,0,4,1); set_u(0,1,4,1)
        set_v(1,1,4,1)
        set_w(0,0,4,-1); set_w(1,0,4,1)

        # Term 6: (A10-A00)*(B00+B01)
        set_u(1,0,5,1); set_u(0,0,5,-1)
        set_v(0,0,5,1); set_v(0,1,5,1)
        set_w(1,1,5,1)

        # Term 7: (A01-A11)*(B10+B11)
        set_u(0,1,6,1); set_u(1,1,6,-1)
        set_v(1,0,6,1); set_v(1,1,6,1)
        set_w(0,0,6,1)

        # Term 8: A00*B02
        set_u(0,0,7,1)
        set_v(0,2,7,1)
        set_w(2,0,7,1)

        # Term 9: A01*B12
        set_u(0,1,8,1)
        set_v(1,2,8,1)
        set_w(2,0,8,1)

        # Term 10: A10*B02
        set_u(1,0,9,1)
        set_v(0,2,9,1)
        set_w(2,1,9,1)

        # Term 11: A11*B12
        set_u(1,1,10,1)
        set_v(1,2,10,1)
        set_w(2,1,10,1)

        return U, V, W

    # ============================================================
    # SEARCH FOR RANK-10: Systematic structured search
    # ============================================================
    def search_rank10():
        """
        Search for rank-10 decomposition by:
        1. Using a rank-7 Strassen core for the 2x2*2x2 subproblem
        2. Adding 3 terms to handle the third column
        3. Trying structured patterns for the 3 extension terms
        """
        # Rank-7 Strassen decomposition for 2x2 * 2x2
        # U_core: (4,7), V_core: (4,7), W_core: (4,7)
        U_core = np.array([
            [1, 0, 1, 0, 1, 1, 0],  # A[0,0]
            [0, 1, 0, 1, 0, 1, 1],  # A[1,0]
            [0, 0, 1, 1, 1, 0, 1],  # A[0,1]
            [1, 1, 0, 0, 0, 0, 1]   # A[1,1]
        ], dtype=int)

        V_core = np.array([
            [1, 1, 1, 1, 0, 0, 1],  # B[0,0]
            [0, 0, 0, 0, 1, 1, 0],  # B[1,0]
            [0, 0, 0, 0, 0, 0, 1],  # B[0,1]
            [1, 1, 0, 0, 0, 0, 0]   # B[1,1]
        ], dtype=int)

        W_core = np.array([
            [1, 0, 0, 0, 0, 0, 0],  # C[0,0]
            [0, 1, 0, 0, 0, 0, 0],  # C[1,0]
            [0, 0, 1, 1, 1, 0, 0],  # C[0,1]
            [0, 0, 0, 0, 0, 0, 1]   # C[1,1]
        ], dtype=int)

        # Verify core
        def verify_core(U_c, V_c, W_c):
            rank = U_c.shape[1]
            for i in range(2):
                for j in range(2):
                    for jp in range(2):
                        for k in range(2):
                            target = 1 if j == jp else 0
                            val = 0
                            for r in range(rank):
                                val += U_c[i*2+j, r] * V_c[jp*2+k, r] * W_c[k*2+i, r]
                            if val != target:
                                return False
            return True

        if not verify_core(U_core, V_core, W_core):
            return None

        # Build full rank-10: 7 core + 3 extension
        U_full = np.zeros((4, 10), dtype=int)
        V_full = np.zeros((6, 10), dtype=int)
        W_full = np.zeros((6, 10), dtype=int)

        U_full[:, :7] = U_core
        V_full[:4, :7] = V_core
        W_full[:4, :7] = W_core
        # V_full[4:6, :7] = 0, W_full[4:6, :7] = 0 (already zero)

        # Now search for 3 extension terms (columns 7,8,9)
        # These must handle: C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
        #                  C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
        # And must NOT disturb the first two columns (C[*,0], C[*,1])
        # i.e., for k in {0,1}: sum_r U[i,j,r]*V[jp,k,r]*W[k,i,r] must be 0 for new terms
        
        # For the new terms (r=7,8,9), W[k,i,r] for k in {0,1} must be 0
        # to not disturb existing outputs. So W[0:4, 7:10] = 0.
        # Only W[4:6, 7:10] (C[0,2] and C[1,2]) can be non-zero.
        
        # Similarly, V[4:6, 7:10] (B[0,2] and B[1,2]) are the only B entries
        # that should be non-zero in new terms (to not disturb B[*,0], B[*,1]).
        
        # So the new terms have:
        # U: any 4x3 pattern
        # V: only rows 4,5 (B[0,2], B[1,2]) can be non-zero
        # W: only rows 4,5 (C[0,2], C[1,2]) can be non-zero
        
        # This is a 4x2x2 subproblem: A(2x2) * B(2x1) -> C(2x1)
        # The rank of this subproblem is at most 4 (schoolbook) but could be 3.
        # Actually for 2x2 * 2x1, the tensor is:
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # This is equivalent to matrix-vector multiplication 2x2 * 2x1
        # The rank of this tensor is 4 (it's the standard matrix multiplication for these dims)
        # Wait, actually it's a 2x2x1 tensor which has rank 4.
        # So we need at least 4 terms for the third column, making total rank 11.
        
        # BUT: we can share terms between the core and the extension if we allow
        # the new terms to have non-zero entries in V[0:4] and W[0:4] as long as
        # the total sum still works out.
        
        # Let's do a more general search: allow all entries in the 3 new terms.
        # Variables: U[4,3] + V[6,3] + W[6,3] = 12+18+18 = 48 variables
        # Constraints: for each (i,jp,k) tuple, the sum over all 10 terms must equal delta_{j,jp}
        # That's 2*2*3 = 12 constraints.
        
        # This is underdetermined, so we can try to find solutions.
        # Strategy: fix U and V patterns, solve for W.
        
        # Actually, let's try a different approach:
        # Use the fact that the 2x2x3 tensor can be seen as 2x2x2 (rank 7) + 2x2x1 (rank 4) = rank 11
        # But maybe there's overlap. Let's try to find if rank 10 is possible.
        
        # Try: allow the 3 new terms to have arbitrary values in all positions.
        # Use numpy to solve the linear system.
        
        # The constraint is: for each (i, jp, k):
        #   sum_{r=0}^{9} U[i,jp,r] * V[jp,k,r] * W[k,i,r] = delta_{jp, j} where j is determined by...
        # Wait, I need to be more careful.
        
        # The tensor T[i, j, k] = delta_{j, jp} where... no.
        # The matrix multiplication tensor: C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor is T[i,j,k] = delta_{j, j'} where j' is the index that matches.
        # Actually T[i,j,k] = 1 if j is the "middle" index that connects A[i,j] and B[j,k].
        # So T[i,j,k] = delta_{j, j} which is always 1? No.
        
        # The tensor for matrix multiplication: the (i,j,k)-th entry is 1 if and only if
        # the product A[i,j] * B[j,k] contributes to C[i,k].
        # So T[i,j,k] = 1 for all valid (i,j,k), and the decomposition is:
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        # Wait, that's not right either. Let me re-read the task.
        # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 2 matrix by a 2 by 3 matrix."
        # The output factor follows W[k*n+i,r].
        
        # So the tensor is: T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
        # T[i,j,k] = 1 if j is the "connecting" index, i.e., always 1 for valid indices?
        # No, the tensor for matrix multiplication is:
        # T[i,j,k] = delta_{j, j} which doesn't make sense.
        
        # Actually, the standard matrix multiplication tensor is:
        # T[i, j, k] = 1 for all (i,j,k) in the valid range
        # And the decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # This means sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all (i,j,k)
        
        # Wait, that can't be right because then the rank would be 1.
        
        # Let me re-read the verify function in the current program:
        # for i in range(n):
        #   for j in range(m):
        #     for jp in range(m):
        #       for k in range(p):
        #         target = 1 if j == jp else 0
        #         val = sum_r U[i*m+j, r] * V[jp*p+k, r] * W[k*n+i, r]
        #         if val != target: return False
        
        # So the tensor is: T[i, j, jp, k] = delta_{j, jp}
        # This is a 4-index tensor! The decomposition is:
        # T[i, j, jp, k] = sum_r U[i*m+j, r] * V[jp*p+k, r] * W[k*n+i, r]
        # = delta_{j, jp}
        
        # So for fixed (i, k), we have a matrix M[i,k] where M[i,k][j,jp] = delta_{j,jp} = I
        # And M[i,k] = sum_r U[i*m+j, r] * V[jp*p+k, r] * W[k*n+i, r]
        # This is: M[i,k] = (U_i * W_k) where U_i is a row vector indexed by j and W_k is a row vector indexed by jp
        # Wait, let me be more careful.
        
        # For fixed i, k:
        # sum_r U[i*m+j, r] * V[jp*p+k, r] * W[k*n+i, r] = delta_{j, jp}
        
        # Let's define for each r:
        # u_r[j] = U[i*m+j, r] for j in [0,m)
        # v_r[jp] = V[jp*p+k, r] for jp in [0,m)
        # w_r = W[k*n+i, r]
        
        # Then: sum_r w_r * u_r[j] * v_r[jp] = delta_{j, jp}
        # This means: sum_r w_r * outer(u_r, v_r) = I_m
        
        # So for each (i,k), we need: sum_r w_r * outer(u_r, v_r) = I_2
        
        # This is a decomposition of the identity matrix as a sum of rank-1 matrices!
        # The rank of I_2 is 2, so we need at least 2 terms for each (i,k) pair.
        # But the terms are shared across different (i,k) pairs.
        
        # There are n*p = 2*3 = 6 (i,k) pairs.
        # For each, we need sum_r w_r * outer(u_r, v_r) = I_2
        
        # This is a much clearer formulation!
        
        # The total rank is the number of terms r.
        # Lower bound: max over (i,k) of rank(I_2) = 2, but since terms are shared,
        # the actual lower bound is higher.
        
        # For the 2x2x2 case (p=2), there are 4 (i,k) pairs, and the known rank is 7.
        # For 2x2x3, there are 6 (i,k) pairs, and the known rank is 11.
        
        # Let's verify: can we do rank 10?
        # We need to find 10 vectors (u_r, v_r, w_r) such that for each of the 6 (i,k) pairs:
        # sum_r W[k*n+i, r] * outer(U[i*m+j, r] for j, V[jp*p+k, r] for jp) = I_2
        
        # This is a system of 6 * 4 = 24 equations (for the 4 entries of I_2 for each (i,k))
        # with 10 * (4 + 6 + 6) = 160 variables (but with structure).
        
        # Actually, the variables are:
        # U: 4 * 10 = 40
        # V: 6 * 10 = 60
        # W: 6 * 10 = 60
        # Total: 160 variables, 24 equations. Very underdetermined.
        
        # So rank 10 should be feasible! The question is whether there's an exact integer/half-integer solution.
        
        # Let's try to construct one.
        
        # Approach: Use the rank-7 Strassen core for the first 4 (i,k) pairs (k=0,1)
        # and add 3 more terms for the last 2 (i,k) pairs (k=2).
        
        # For the Strassen core, we have 7 terms that satisfy:
        # For k=0: sum_{r=0}^{6} W[0*n+i, r] * outer(U[i*m+j, r], V[jp*0+k, r]) = I_2
        # For k=1: sum_{r=0}^{6} W[1*n+i, r] * outer(U[i*m+j, r], V[jp*1+k, r]) = I_2
        
        # Wait, I'm getting confused with the indexing. Let me restart with the clearer formulation.
        
        # The constraint is: for each (i, k) in [0,n) x [0,p):
        #   sum_{r=0}^{R-1} W[k*n+i, r] * (U[i*m+0, r] * V[0*p+k, r], U[i*m+0, r] * V[1*p+k, r],
        #                                U[i*m+1, r] * V[0*p+k, r], U[i*m+1, r] * V[1*p+k, r]) = I_2
        
        # More precisely, for each (i,k,j,jp):
        #   sum_r U[i*m+j, r] * V[jp*p+k, r] * W[k*n+i, r] = delta_{j,jp}
        
        # Let me just do a numerical search for rank 10.
        
        # Use random initialization and gradient-free search (e.g., simulated annealing or random restarts)
        # to minimize the maximum absolute error.
        
        np.random.seed(42)
        best_error = float('inf')
        best_solution = None
        
        for trial in range(200):
            if time.time() > deadline:
                break
            
            # Random initialization with small integers
            U_rand = np.random.randint(-2, 3, size=(4, 10))
            V_rand = np.random.randint(-2, 3, size=(6, 10))
            W_rand = np.random.randint(-2, 3, size=(6, 10))
            
            # Compute error
            max_err = 0
            for i in range(2):
                for jp in range(2):
                    for k in range(3):
                        for j in range(2):
                            target = 1 if j == jp else 0
                            val = 0
                            for r in range(10):
                                val += U_rand[i*2+j, r] * V_rand[jp*3+k, r] * W_rand[k*2+i, r]
                            max_err = max(max_err, abs(val - target))
            
            if max_err < best_error:
                best_error = max_err
                best_solution = (U_rand.copy(), V_rand.copy(), W_rand.copy())
            
            if best_error == 0:
                break
        
        if best_error == 0:
            U_sol, V_sol, W_sol = best_solution
            if verify(U_sol, V_sol, W_sol):
                # Try to reduce rank
                for rank in range(1, 11):
                    if verify(U_sol[:, :rank], V_sol[:, :rank], W_sol[:, :rank]):
                        return U_sol[:, :rank], V_sol[:, :rank], W_sol[:, :rank]
                return U_sol, V_sol, W_sol
        
        return None

    # ============================================================
    # MAIN: Try rank 10, fall back to rank 11
    # ============================================================
    # First, try to find a rank-10 solution
    result_10 = search_rank10()
    if result_10 is not None:
        U, V, W = result_10
        if verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to known rank-11
    U, V, W = rank11_solution()
    if not verify(U, V, W):
        # Should not happen, but just in case
        raise ValueError("Rank-11 solution failed verification")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
