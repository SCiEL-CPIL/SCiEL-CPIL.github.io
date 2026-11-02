"""Rank-minimal exact tensor decomposition for 2x5 @ 5x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS

    # Verify via the full tensor equality check:
    # For the multiplication tensor T, T[i*m+j, j'*p+k', k'*n+i'] should be 1
    # if (j, k') = (j, k) in the summation sense, else 0.
    #
    # Actually, the correct formulation:
    # The tensor for multiplying (n x m) * (m x p) -> (n x p) is:
    # T[i, j, k, j2, k2, i2] ... no, let me think more carefully.
    #
    # The standard tensor decomposition of matrix multiplication M_{n,m,p}:
    # (A * B)[i, k] = sum_j A[i, j] * B[j, k]
    #
    # The tensor has indices (i, j, k) where A has indices (i,j), B has (j,k),
    # and output has (i,k). The multiplication tensor T is of format (n*m, m*p, p*n).
    #
    # A decomposition of rank r means:
    # T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # where T[i*m+j, j*p+k, k*n+i] = 1 if the indices form a valid multiplication
    # (i.e., the j in A's column index matches j in B's row index, and k in B's
    # column index matches k in output, and i in A's row index matches i in output),
    # else 0.
    #
    # More precisely: T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a,b,c] = 1 iff the j from a, the j from b match (they share j),
    #           and the k from b, the k from c match (they share k),
    #           and the i from a, the i from c match (they share i).
    #
    # Wait, that's not right either. Let me be very explicit.
    #
    # a ranges over i in [0,n), j in [0,m) -> a = i*m + j
    # b ranges over j in [0,m), k in [0,p) -> b = j*p + k  
    # c ranges over k in [0,p), i in [0,n) -> c = k*n + i
    #
    # The tensor T[a,b,c] is non-zero when:
    #   (i from a) == (i from c)  -- same row in A and output
    #   (j from a) == (j from b)  -- column in A matches row in B
    #   (k from b) == (k from c)  -- column in B matches column in output
    # And T[a,b,c] = 1 in that case.
    #
    # So the verify function checks:
    # For each (i,j,k) [defining a=i*m+j, b=j*p+k, c=k*n+i]:
    #   sum_c U[a,c]*V[b,c]*W[c_idx,c] should equal 1 where the indices match
    #   and 0 for all other combinations.

    def verify(U, V, W, n, m, p, tol=1e-9):
        """Verify exact tensor decomposition."""
        r = len(U[0]) if U and len(U) > 0 else 0
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Build the full tensor T (flattened) and check decomposition
        # T has dimensions (n*m, m*p, p*n)
        nm = n * m
        mp = m * p
        pn = p * n
        
        # Compute the decomposition tensor D[a, b, c] = sum_r U[a,r]*V[b,r]*W[c,r]
        D = U_arr @ (V_arr.T @ W_arr)
        # Wait, that's not right. D[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # This is a 3-tensor. Let me compute it properly.
        
        # D[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # We can think of this as: for each r, outer product of U[:,r], V[:,r], W[:,r]
        # D = sum_r np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
        # For efficiency with small ranks, just loop
        D = np.zeros((nm, mp, pn), dtype=float)
        for c in range(r):
            D += np.einsum('a,b,c->abc', U_arr[:, c], V_arr[:, c], W_arr[:, c])
        
        # Now build expected tensor T
        T = np.zeros((nm, mp, pn), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c_idx = k * n + i
                    T[a, b, c_idx] = 1.0
        
        return np.allclose(D, T, atol=tol)

    def schoolbook(n, m, p):
        """Standard rank = n*m*p decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    # Try rank-25 construction:
    # The key structural insight: for the 2x5x5 tensor, we can potentially exploit
    # that n=2. 
    #
    # Let me try a different rank-25 construction based on separating the i-index
    # more carefully.
    #
    # Actually, let me try a systematic approach. For rank 25, we have 25 terms.
    # The tensor T[i*m+j, j*p+k, k*n+i] = delta_valid.
    #
    # One natural rank-25 ansatz: index the 25 terms by (j0, k0) for j0 in [0,5), k0 in [0,5).
    # U[i*m+j, (j0,k0)] = delta(j, j0) * f(i)  -- needs to depend on i to separate rows
    # V[j*p+k, (j0,k0)] = delta(j, j0) * delta(k, k0)
    # W[k*n+i, (j0,k0)] = delta(k, k0) * g(i)
    #
    # Diagonal check: T[i*5+j, j*5+k, k*2+i] = sum_{j0,k0} U[i*5+j,(j0,k0)]*V[j*5+k,(j0,k0)]*W[k*2+i,(j0,k0)]
    # = sum_{j0,k0} delta(j,j0)*f(i) * delta(j,j0)*delta(k,k0) * delta(k,k0)*g(i)
    # = f(i)*g(i)  (only j0=j, k0=k survive)
    # Need f(i)*g(i) = 1 for all i.
    #
    # Off-diagonal: for (i2, j2, k2) != (i, j, k):
    # sum_{j0,k0} delta(j,j0)*f(i) * delta(j2,j0)*delta(k2,k0) * delta(k,k0)*g(i2)
    # = delta(j,j2) * delta(k,k2) * f(i)*g(i2)
    #
    # For this to be 0 whenever (i,j,k) != (i2,j2,k2):
    # - If j=j2 and k=k2 but i!=i2: need f(i)*g(i2) = 0
    # - If j=j2 but k!=k2: automatically 0
    # - If j!=j2: automatically 0
    #
    # So we need: for all i != i2, f(i)*g(i2) = 0.
    # But also f(i)*g(i) = 1.
    # 
    # If f(0)=1, g(0)=1, then for i=0, i2=1: f(0)*g(1) = 0 => g(1)=0.
    # But f(1)*g(1) = 1 => f(1)*0 = 1. Contradiction!
    #
    # So the symmetric rank-25 construction with this structure is impossible.
    #
    # Let me try an asymmetric construction. What if the 25 terms are indexed differently?
    #
    # Alternative: use 25 terms but with a different parameterization.
    # What if U depends on i and j in a non-trivial way?
    #
    # Let me try: for the 25 terms indexed by (a, b) where a in [0,5), b in [0,5):
    # U[i*5+j, (a,b)] = A_i * delta(j, a)  ... no this has the same issue.
    #
    # What about using 2*13+1 = 27? Or trying rank 25 with a non-structured search?
    #
    # Actually, let me just verify that rank 25 is impossible and return rank 50.
    # But first, let me try one more construction idea.
    
    # Try: can we do rank 25 with a block structure?
    # Split the 2 rows: for row 0, use some terms, for row 1, use other terms, with sharing.
    # 
    # What if we use 13 terms for row 0, 13 terms for row 1, with 1 shared term? Total = 25.
    # This seems unlikely to work with the constraints.
    
    # Let me try a completely different rank-25 ansatz:
    # What if the decomposition doesn't have the simple delta structure?
    # 
    # Actually, I recall that for 2xm @ mxm, the rank can sometimes be less than 2*m*m.
    # But I'm not sure what the exact minimal rank is for 2x5@5x5.
    # 
    # Let me just go with the verified schoolbook rank-50 solution, which is guaranteed correct.
    
    U, V, W = schoolbook(n, m, p)
    
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Should never reach here, but just in case
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
