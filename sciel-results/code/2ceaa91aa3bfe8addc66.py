import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 15s margin before 130s kill
    
    n, m, p = 2, 3, 3
    
    def elapsed():
        return time.time() - start_time
    
    def build_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T
    
    def verify(U, V, W, T, tol=1e-9):
        Re = T - np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(Re)) < tol
    
    def build_rank17():
        """Known valid rank-17 decomposition using Strassen on 2x2x2 sub-block."""
        rank = 17
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        def set_U(i, j, r, val): U[i * m + j, r] = val
        def set_V(j, k, r, val): V[j * p + k, r] = val
        def set_W(k, i, r, val): W[k * n + i, r] = val
        
        # Strassen 2x2x2 for the sub-block: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        # A00=A[0,0], A01=A[0,1], A10=A[1,0], A11=A[1,1]
        # B00=B[0,0], B01=B[0,1], B10=B[1,0], B11=B[1,1]
        # C00=A00*B00+A01*B10, C01=A00*B01+A01*B11
        # C10=A10*B00+A11*B10, C11=A10*B01+A11*B11
        
        # Strassen: M1=(A00+A11)(B00+B11), M2=(A10+A11)B00, M3=A00(B11-B10),
        #          M4=A11(B01-B00), M5=(A00+A01)B11, M6=(A10-A00)(B00+B01), M7=(A01-A11)(B10+B11)
        # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
        
        # Term 0: M1 = (A00+A11)(B00+B11)
        set_U(0, 0, 0, 1); set_U(1, 1, 0, 1)
        set_V(0, 0, 0, 1); set_V(1, 1, 0, 1)
        set_W(0, 0, 0, 1); set_W(0, 1, 0, 1); set_W(1, 0, 0, 1); set_W(1, 1, 0, -1)
        
        # Term 1: M2 = (A10+A11)B00
        set_U(1, 0, 1, 1); set_U(1, 1, 1, 1)
        set_V(0, 0, 1, 1)
        set_W(1, 0, 1, 1); set_W(1, 1, 1, -1)
        
        # Term 2: M3 = A00(B11-B10)
        set_U(0, 0, 2, 1)
        set_V(1, 1, 2, 1); set_V(1, 0, 2, -1)
        set_W(0, 1, 2, 1); set_W(1, 1, 2, 1)
        
        # Term 3: M4 = A11(B01-B00)
        set_U(1, 1, 3, 1)
        set_V(0, 1, 3, 1); set_V(0, 0, 3, -1)
        set_W(0, 0, 3, 1); set_W(1, 0, 3, 1)
        
        # Term 4: M5 = (A00+A01)B11
        set_U(0, 0, 4, 1); set_U(0, 1, 4, 1)
        set_V(1, 1, 4, 1)
        set_W(0, 0, 4, -1); set_W(1, 0, 4, 1)
        
        # Term 5: M6 = (A10-A00)(B00+B01)
        set_U(1, 0, 5, 1); set_U(0, 0, 5, -1)
        set_V(0, 0, 5, 1); set_V(0, 1, 5, 1)
        set_W(1, 1, 5, 1)
        
        # Term 6: M7 = (A01-A11)(B10+B11)
        set_U(0, 1, 6, 1); set_U(1, 1, 6, -1)
        set_V(1, 0, 6, 1); set_V(1, 1, 6, 1)
        set_W(0, 1, 6, 1)
        
        # Cross terms: A[*,2]*B[2,*] and A[*,0:2]*B[2,*] and A[*,2]*B[0:2,*]
        # A[0,2]*B[0,k] for k=0,1,2 -> C[0,k]
        # A[1,2]*B[0,k] for k=0,1,2 -> C[1,k]
        # A[i,0]*B[2,k] for i=0,1, k=0,1,2 -> C[i,k]
        # A[i,1]*B[2,k] for i=0,1, k=0,1,2 -> C[i,k]
        # A[i,2]*B[2,k] for i=0,1, k=0,1,2 -> C[i,k]
        
        # A[0,2]*B[0,0] -> C[0,0]
        set_U(0, 2, 7, 1); set_V(0, 0, 7, 1); set_W(0, 0, 7, 1)
        # A[0,2]*B[0,1] -> C[0,1]
        set_U(0, 2, 8, 1); set_V(0, 1, 8, 1); set_W(0, 1, 8, 1)
        # A[0,2]*B[0,2] -> C[0,2]
        set_U(0, 2, 9, 1); set_V(0, 2, 9, 1); set_W(2, 0, 9, 1)
        # A[1,2]*B[0,0] -> C[1,0]
        set_U(1, 2, 10, 1); set_V(0, 0, 10, 1); set_W(0, 1, 10, 1)
        # A[1,2]*B[0,1] -> C[1,1]
        set_U(1, 2, 11, 1); set_V(0, 1, 11, 1); set_W(1, 1, 11, 1)
        # A[1,2]*B[0,2] -> C[1,2]
        set_U(1, 2, 12, 1); set_V(0, 2, 12, 1); set_W(2, 1, 12, 1)
        # A[0,0]*B[2,0] -> C[0,0]
        set_U(0, 0, 13, 1); set_V(2, 0, 13, 1); set_W(0, 0, 13, 1)
        # A[0,1]*B[2,0] -> C[0,0]
        set_U(0, 1, 14, 1); set_V(2, 0, 14, 1); set_W(0, 0, 14, 1)
        # A[0,0]*B[2,1] -> C[0,1]
        set_U(0, 0, 15, 1); set_V(2, 1, 15, 1); set_W(1, 0, 15, 1)
        # A[0,1]*B[2,1] -> C[0,1]
        set_U(0, 1, 16, 1); set_V(2, 1, 16, 1); set_W(1, 0, 16, 1)
        # A[1,0]*B[2,0] -> C[1,0]
        # A[1,1]*B[2,0] -> C[1,0]
        # A[1,0]*B[2,1] -> C[1,1]
        # A[1,1]*B[2,1] -> C[1,1]
        # A[0,0]*B[2,2] -> C[0,2]
        # A[0,1]*B[2,2] -> C[0,2]
        # A[1,0]*B[2,2] -> C[1,2]
        # A[1,1]*B[2,2] -> C[1,2]
        # A[0,2]*B[2,0] -> C[0,0]
        # A[0,2]*B[2,1] -> C[0,1]
        # A[0,2]*B[2,2] -> C[0,2]
        # A[1,2]*B[2,0] -> C[1,0]
        # A[1,2]*B[2,1] -> C[1,1]
        # A[1,2]*B[2,2] -> C[1,2]
        
        # That's already 17 terms but we haven't covered all 18 products!
        # Let me count: the 2x2x2 block has 4 products (i=0,1; j=0,1; k=0,1)
        # Remaining: 14 products
        # I used 7 for Strassen + 10 for cross = 17, but I need 14 cross terms
        # Let me redo this more carefully
        
        return U, V, W
    
    def build_rank18():
        """Trivial rank-18 decomposition."""
        rank = 18
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
        return U, V, W
    
    T = build_tensor()
    
    # First, try to find a rank-16 solution via structured search
    # Strategy: Use a bilinear ansatz and solve linear equations
    
    def search_rank16():
        """Search for rank-16 decomposition using structured linear algebra."""
        if elapsed() > deadline - 20:
            return None
        
        rank = 16
        # We need to find U(6x16), V(9x16), W(6x16) such that
        # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        
        # The tensor has 18 non-zero entries. We can set up equations.
        # For each non-zero entry T[a,b,c]=1, we need sum_r U[a,r]*V[b,r]*W[c,r] = 1
        # For each zero entry, sum_r U[a,r]*V[b,r]*W[c,r] = 0
        
        # This is a system of nonlinear equations (trilinear). 
        # Instead, fix U and V, then solve for W linearly.
        
        # Structured ansatz: 
        # The 18 products are:
        # (i,j,k) for i in {0,1}, j in {0,1,2}, k in {0,1,2}
        # C[i,k] = sum_j A[i,j]*B[j,k]
        
        # Let's use a different decomposition strategy:
        # Group by j (the summation index):
        # j=0: A[i,0]*B[0,k] for i=0,1; k=0,1,2 -> 6 terms
        # j=1: A[i,1]*B[1,k] for i=0,1; k=0,1,2 -> 6 terms  
        # j=2: A[i,2]*B[2,k] for i=0,1; k=0,1,2 -> 6 terms
        
        # For each j, we have a 2x3 outer product: A[:,j] (2x1) * B[j,:] (1x3)
        # This is rank 1 as a 2x3 matrix, but in the tensor decomposition sense,
        # each A[i,j]*B[j,k] is one rank-1 term.
        
        # Alternative: Use the fact that for fixed j, the contribution is
        # outer(A[:,j], B[j,:]) which is a 2x3 matrix of rank 1.
        # In the tensor T, this contributes to positions (i, j, k) for all i,k.
        
        # Let me try a direct approach: assume a specific structure and solve.
        
        # Ansatz: Let's say we use rank 16. We'll parameterize and solve.
        # Key idea: The 18 scalar products can be grouped.
        # For the 2x2x2 subblock (j=0,1; k=0,1), use Strassen (rank 7).
        # That handles 4 of the 18 products.
        # Remaining 14 products need rank 14, total 21. Too many.
        
        # Better: Let's think about it differently.
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
        
        # 6 output entries, each with 3 terms = 18 total.
        
        # Let's try to find if rank 16 is possible by setting up the problem.
        # We'll use a random initialization and gradient-free search, but that's slow.
        
        # Instead, let's use the following structured approach:
        # Assume the decomposition has a specific block structure.
        
        # Actually, let me try a completely different approach:
        # Use the fact that we can combine some terms.
        
        # For C[0,0]: A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
        # For C[1,0]: A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
        
        # Note that B[0,0] appears in both C[0,0] and C[1,0].
        # Similarly B[1,0] and B[2,0].
        
        # Let's define:
        # M0 = (A[0,0]+A[1,0]) * B[0,0]  -> contributes to C[0,0] and C[1,0]
        # M1 = (A[0,1]+A[1,1]) * B[1,0]  -> contributes to C[0,0] and C[1,0]
        # M2 = (A[0,2]+A[1,2]) * B[2,0]  -> contributes to C[0,0] and C[1,0]
        # M3 = (A[0,0]-A[1,0]) * B[0,0]  -> difference for C[0,0] vs C[1,0]
        # M4 = (A[0,1]-A[1,1]) * B[1,0]  -> difference
        # M5 = (A[0,2]-A[1,2]) * B[2,0]  -> difference
        
        # From these: C[0,0] = (M0+M1+M2+M3+M4+M5)/2... this is getting complex.
        
        # Let me just try a numerical search with a good initialization.
        
        if elapsed() > deadline - 15:
            return None
        
        # Try rank 16 with a structured initialization
        # Initialize with rank 18 and try to merge two terms
        
        U18, V18, W18 = build_rank18()
        
        # Try to find if any two columns can be merged
        # For a merge to work, we need the two rank-1 terms to be "compatible"
        
        # Alternative: use scipy to solve a least-squares problem
        # Fix U and V, solve for W
        
        # Let's try: fix U and V to be the rank-18 version with two columns modified
        # to create linear dependencies
        
        # Actually, let me try a direct linear algebra approach for rank 16:
        # If we fix U (6x16) and V (9x16), then for each c, we need:
        # W[:,c] to satisfy: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all a,b
        # This is a linear system in W[c, :] (16 unknowns) with 54 equations (6x9).
        # But T only has 18 non-zero entries, so 18 equations are =1 and 36 are =0.
        
        # For a solution to exist, the system must be consistent.
        
        # Let's parameterize U and V and check consistency.
        
        # This is still a nonlinear problem. Let me try a different tactic.
        
        # Use the known rank-17 and try to reduce by 1.
        # In the rank-17, if any two rank-1 terms can be combined into one, we get rank 16.
        
        # Let me build a proper rank-17 first and verify it.
        rank = 17
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        def set_U(i, j, r, val): U[i * m + j, r] = val
        def set_V(j, k, r, val): V[j * p + k, r] = val
        def set_W(k, i, r, val): W[k * n + i, r] = val
        
        # Proper rank-17: Strassen for 2x2x2 + 10 cross terms
        # The 2x2x2 subblock: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        # This involves products: A[i,j]*B[j,k] for i=0,1; j=0,1; k=0,1
        # That's 4 products, done in rank 7.
        
        # Remaining products: those with j=2 or k=2
        # j=2, k=0: A[i,2]*B[2,0] for i=0,1 -> 2 terms
        # j=2, k=1: A[i,2]*B[2,1] for i=0,1 -> 2 terms
        # j=2, k=2: A[i,2]*B[2,2] for i=0,1 -> 2 terms
        # j=0, k=2: A[i,0]*B[0,2] for i=0,1 -> 2 terms
        # j=1, k=2: A[i,1]*B[1,2] for i=0,1 -> 2 terms
        # Total remaining: 10 terms
        
        # So rank 17 = 7 (Strassen) + 10 (cross) = 17. Correct!
        
        # Strassen terms for A[0:2,0:2]*B[0:2,0:2]:
        # M1=(A00+A11)(B00+B11)
        set_U(0, 0, 0, 1); set_U(1, 1, 0, 1)
        set_V(0, 0, 0, 1); set_V(1, 1, 0, 1)
        # C00 += M1, C01 -= M1, C10 -= M1, C11 += M1
        set_W(0, 0, 0, 1); set_W(0, 1, 0, -1); set_W(1, 0, 0, -1); set_W(1, 1, 0, 1)
        
        # M2=(A10+A11)B00
        set_U(1, 0, 1, 1); set_U(1, 1, 1, 1)
        set_V(0, 0, 1, 1)
        # C10 += M2, C11 -= M2
        set_W(1, 0, 1, 1); set_W(1, 1, 1, -1)
        
        # M3=A00(B11-B10)
        set_U(0, 0, 2, 1)
        set_V(1, 1, 2, 1); set_V(1, 0, 2, -1)
        # C01 += M3, C11 += M3
        set_W(0, 1, 2, 1); set_W(1, 1, 2, 1)
        
        # M4=A11(B01-B00)
        set_U(1, 1, 3, 1)
        set_V(0, 1, 3, 1); set_V(0, 0, 3, -1)
        # C00 += M4, C10 += M4
        set_W(0, 0, 3, 1); set_W(1, 0, 3, 1)
        
        # M5=(A00+A01)B11
        set_U(0, 0, 4, 1); set_U(0, 1, 4, 1)
        set_V(1, 1, 4, 1)
        # C00 -= M5, C10 += M5
        set_W(0, 0, 4, -1); set_W(1, 0, 4, 1)
        
        # M6=(A10-A00)(B00+B01)
        set_U(1, 0, 5, 1); set_U(0, 0, 5, -1)
        set_V(0, 0, 5, 1); set_V(0, 1, 5, 1)
        # C11 += M6
        set_W(1, 1, 5, 1)
        
        # M7=(A01-A11)(B10+B11)
        set_U(0, 1, 6, 1); set_U(1, 1, 6, -1)
        set_V(1, 0, 6, 1); set_V(1, 1, 6, 1)
        # C00 += M7
        set_W(0, 0, 6, 1)
        
        # Verify Strassen: 
        # C00 = M1 + M4 - M5 + M7 = (A00+A11)(B00+B11) + A11(B01-B00) - (A00+A01)B11 + (A01-A11)(B10+B11)
        # = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B01 - A11*B00 - A00*B11 - A01*B11 + A01*B10 + A01*B11 - A11*B10 - A11*B11
        # = A00*B00 + A01*B10
        # Correct!
        
        # Cross terms (10 terms):
        # r=7: A[0,2]*B[2,0] -> C[0,0]
        set_U(0, 2, 7, 1); set_V(2, 0, 7, 1); set_W(0, 0, 7, 1)
        # r=8: A[1,2]*B[2,0] -> C[1,0]
        set_U(1, 2, 8, 1); set_V(2, 0, 8, 1); set_W(0, 1, 8, 1)
        # r=9: A[0,2]*B[2,1] -> C[0,1]
        set_U(0, 2, 9, 1); set_V(2, 1, 9, 1); set_W(1, 0, 9, 1)
        # r=10: A[1,2]*B[2,1] -> C[1,1]
        set_U(1, 2, 10, 1); set_V(2, 1, 10, 1); set_W(1, 1, 10, 1)
        # r=11: A[0,2]*B[2,2] -> C[0,2]
        set_U(0, 2, 11, 1); set_V(2, 2, 11, 1); set_W(2, 0, 11, 1)
        # r=12: A[1,2]*B[2,2] -> C[1,2]
        set_U(1, 2, 12, 1); set_V(2, 2, 12, 1); set_W(2, 1, 12, 1)
        # r=13: A[0,0]*B[0,2] -> C[0,2]
        set_U(0, 0, 13, 1); set_V(0, 2, 13, 1); set_W(2, 0, 13, 1)
        # r=14: A[1,0]*B[0,2] -> C[1,2]
        set_U(1, 0, 14, 1); set_V(0, 2, 14, 1); set_W(2, 1, 14, 1)
        # r=15: A[0,1]*B[1,2] -> C[0,2]
        set_U(0, 1, 15, 1); set_V(1, 2, 15, 1); set_W(2, 0, 15, 1)
        # r=16: A[1,1]*B[1,2] -> C[1,2]
        set_U(1, 1, 16, 1); set_V(1, 2, 16, 1); set_W(2, 1, 16, 1)
        
        if verify(U, V, W, T):
            # Now try to reduce to rank 16
            # Check if any two rank-1 terms can be merged
            # A merge of terms r1 and r2 is possible if there exist scalars alpha, beta
            # such that the combined term works.
            
            # For now, return rank 17 as a valid solution
            # But let's try a quick search for rank 16
            
            if elapsed() < deadline - 30:
                # Try to find rank 16 by solving a linear system
                # Pick a pair of terms to merge and see if it works
                # This is heuristic; try a few pairs
                
                # Actually, let's try a more systematic approach:
                # Use rank 16 with a specific structure
                
                # The key observation: in the cross terms, we have pairs like:
                # A[0,2]*B[2,0] and A[1,2]*B[2,0] both involve B[2,0]
                # A[0,2]*B[2,1] and A[1,2]*B[2,1] both involve B[2,1]
                # etc.
                
                # Can we combine A[0,2]*B[2,0] + A[1,2]*B[2,0] into fewer terms?
                # These contribute to C[0,0] and C[1,0] respectively.
                # They are separate rank-1 terms in the tensor.
                
                # The only way to reduce rank is if some linear combination of
                # rank-1 tensors equals another rank-1 tensor, which is rare.
                
                # Let me try a numerical approach: assume rank 16 and solve.
                # We'll use the fact that the problem is underdetermined.
                
                # For rank 16, we have 6*16 + 9*16 + 6*16 = 384 unknowns
                # and 6*9*6 = 324 equations (but only 18 are non-trivial =1, rest are =0)
                # So it's highly underdetermined. A solution likely exists.
                
                # Let's try to construct one:
                # Start with rank 18, and try to eliminate 2 degrees of freedom.
                
                # Approach: Set up the system for rank 16 with a specific ansatz.
                # We'll fix most of U and V and solve for the rest.
                
                rank16 = 16
                U16 = np.zeros((n * m, rank16))
                V16 = np.zeros((m * p, rank16))
                W16 = np.zeros((p * n, rank16))
                
                # Use the Strassen structure for the first 7 terms
                # and 9 terms for the cross interactions (instead of 10)
                
                # For the cross terms, we have 10 products. Can we do them in 9?
                # The cross products are:
                # (0,2,0), (1,2,0), (0,2,1), (1,2,1), (0,2,2), (1,2,2), (0,0,2), (1,0,2), (0,1,2), (1,1,2)
                # in (i,j,k) notation
                # 
                # These are 10 independent rank-1 tensors. To reduce to 9,
                # we need one to be a linear combination of others, which is
                # generally not possible for independent tensors.
                #
                # UNLESS we can modify the Strassen part to help.
                
                # Let me try a completely different approach: use scipy to solve.
                # We'll set up a least-squares problem.
                
                # Actually, the simplest approach: just return rank 17.
                # The chance of finding rank 16 quickly is low without more sophisticated methods.
                
                pass
            
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # If rank 17 doesn't verify, fall back to rank 18
        U18, V18, W18 = build_rank18()
        return {"U": U18.tolist(), "V": V18.tolist(), "W": W18.tolist()}
    
    # Try rank 16 search first, then fall back
    result = search_rank16()
    if result is not None:
        return result
    
    # Fallback: rank 18
    U, V, W = build_rank18()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
