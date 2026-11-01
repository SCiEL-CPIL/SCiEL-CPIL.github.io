import time
import numpy as np

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    start_time = time.time()
    
    # Try rank-8 decomposition for 2x2x3, giving 3*8 = 24 total
    # If that fails, fall back to rank-32 (4x7 + 1x4)
    
    # First, let's try to construct a valid rank-8 decomposition for 2x2x3
    # and verify it. If valid, use 3 blocks of rank 8.
    
    def build_rank8_223():
        """Build rank-8 decomposition for 2x2x3 matrix multiplication.
        A is 2x2, B is 2x3, C is 2x3.
        A indices: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
        B indices: 0=B[0,0], 1=B[0,1], 2=B[0,2], 3=B[1,0], 4=B[1,1], 5=B[1,2]
        C indices: 0=C[0,0], 1=C[0,1], 2=C[0,2], 3=C[1,0], 4=C[1,1], 5=C[1,2]
        """
        terms = []
        
        # Use a known rank-8 structure based on grouping:
        # Split B columns into (col0, col1) and (col2), use Strassen-like for first two,
        # and handle col2 cleverly, or use a full rank-8.
        
        # Actually, let me use the approach: treat 2x2x3 as combining a 2x2x2 
        # with additional structure. A valid rank-8 decomposition:
        
        # Term 1: (A00+A11)(B00+B10) -> C[0,0]+C[1,1]... no, need to be careful.
        # C[i,j] = sum_k A[i,k]*B[k,j]
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
        
        # Let me try a different approach: use the known result that 
        # rank(2,2,3) <= 8 and construct it explicitly.
        
        # One construction: use 4 terms for column 0 (schoolbook for 2x2x1),
        # and 4 terms that share with column 1 and 2.
        # Actually, let me just use a verified construction.
        
        # Standard rank-8 for [2,2,3]:
        # Based on the idea of using 7 Strassen-like terms for a 2x2x2 sub-problem
        # plus 1 additional, but that gives 8 for 2x2x2 not 2x2x3.
        
        # Let me try: 4 schoolbook terms for col 2, and 4 shared terms for cols 0,1.
        # For cols 0,1: C[:,0] and C[:,1] depend on B[:,0] and B[:,1]
        # This is essentially a 2x2x2 problem (A @ [B0, B1]) with output 2x2
        # Rank 7 for that part + 4 for col 2 = 11. Not good.
        
        # Better: interleave. Let me use the explicit rank-8 from literature.
        # The rank-8 decomposition of [2,2,3]:
        
        # I'll define terms as (U_list, V_list, W_list) where each is a sparse dict.
        
        # After careful derivation, here's a valid rank-8:
        # Let me denote A elements: a=A00, b=A01, c=A10, d=A11
        # B elements: e=B00, f=B01, g=B02, h=B10, i=B11, j=B12
        
        # T1 = (a+d)(e+h) -> contributes to C00, C11 (from A@B[:,0] part)
        #    W: C00=1, C11=1
        terms.append(({0:1, 3:1}, {0:1, 3:1}, {0:1, 4:1}))
        
        # T2 = (c+d)e -> C10, and we need to handle it
        #    W: C10=1
        terms.append(({2:1, 3:1}, {0:1}, {3:1}))
        
        # T3 = a(f-i) -> C01, C11
        #    W: C01=1, C11=1  
        terms.append(({0:1}, {1:1, 4:-1}, {1:1, 4:1}))
        
        # T4 = d(h-e) -> C00, C10
        #    W: C00=1, C10=1
        terms.append(({3:1}, {3:1, 0:-1}, {0:1, 3:1}))
        
        # T5 = (a+b)h -> C00, C01... let me check
        #    (a+b)h = ah + bh -> contributes to C00 (ah) and C10 (bh)? No.
        #    C00 = ae + bh, C10 = ce + dh
        #    Let me restart with a cleaner approach.
        
        # Let me just verify computationally. I'll generate candidate decompositions
        # and check them.
        terms = []
        
        # Use the following rank-8 construction:
        # Combine Strassen for [A | B[:,0]] (2x2 times 2x2) giving 7 terms for C[:,0]
        # and C[:,1], but that's only for 2 output columns.
        
        # Actually, the cleanest approach: I know rank(2,2,3) = 8.
        # Let me use the explicit coefficients from a reference construction.
        
        # Here's a verified rank-8 for tensor [2,2,3]:
        # (Using the convention: C[i,j] = sum_k A[i,k] B[k,j])
        
        # Term structure (U: 4-vector, V: 6-vector, W: 6-vector):
        # t1: U=[1,0,0,1], V=[1,0,0,1,0,0], W=[1,0,0,0,1,0]  -> (a+d)(e+h)
        #     Check: a*e+a*h+d*e+d*h -> C00 gets a*e (good, C00=ae+bh), C11 gets d*h (C11=ci+dj... no C11=ci+dj? No C11 = c*f+d*i)
        #     Hmm, W=[1,0,0,0,1,0] means C00+=1, C11+=1
        #     a*e contributes to C00 ✓, a*h contributes to C10 ✗ (should be C00 or C10)
        #     Wait: a*h: a=A[0,0], h=B[1,0]. C[0,0] = A[0,0]*B[0,0]+A[0,1]*B[1,0] = ae+bh
        #     So a*h is NOT in C[0,0]. It's... a=A[0,0], h=B[1,0]. 
        #     C[i,j] = sum_k A[i,k]*B[k,j]. C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = ae + bh.
        #     a*h = A[0,0]*B[1,0] - this is NOT a standard product term. 
        #     B[1,0] is the (1,0) entry of B, which multiplies A[i,1] for C[i,0].
        #     So A[0,0]*B[1,0] doesn't appear directly. It's not in the tensor.
        
        # I'm making errors. Let me be very careful.
        # C[i,j] = sum_k A[i,k] * B[k,j]
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]  
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
        
        # So the tensor has 24 nonzero entries.
        
        # For Strassen on 2x2x2 (columns 0,1 of B, giving C columns 0,1):
        # M1 = (a+d)(e+h) where e=B[0,0], h=B[1,0]
        #   a*e = A[0,0]*B[0,0] ✓ (in C[0,0])
        #   a*h = A[0,0]*B[1,0] ✗ (not in any C[i,j] directly; A[0,0] multiplies B[0,j])
        #   d*e = A[1,1]*B[0,0] ✗ (A[1,1] multiplies B[1,j])
        #   d*h = A[1,1]*B[1,0] ✓ (in C[1,0])
        #   So (a+d)(e+h) = ae + ah + de + dh. Only ae and dh are "valid" tensor entries.
        #   ae is in C[0,0], dh is in C[1,0].
        #   ah = A[0,0]*B[1,0]: this is NOT a tensor entry.
        #   de = A[1,1]*B[0,0]: this is NOT a tensor entry.
        
        # Wait, I think I've been confusing myself. In Strassen for matrix multiplication C=A*B:
        # C[i,j] = sum_k A[i,k]*B[k,j]
        # The "sub-matrices" in Strassen partition A, B, C into blocks.
        # For 2x2*2x2: A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[p,q],[r,s]]
        # p = ae+bg, q = af+bh, r = ce+dg, s = cf+dh
        # 
        # In our case for 2x2 * 2x3:
        # A = [[a,b],[c,d]] (2x2)
        # B = [[e,f,g],[h,i,j]] (2x3)
        # C = [[p,q,r],[s,t,u]] (2x3) where
        # p = ae+bh, q = af+bi, r = ag+bj
        # s = ce+dh, t = cf+di, u = cg+dj
        #
        # So if I apply Strassen to the first two columns:
        # Treat B' = [[e,f],[h,i]] (2x2), C' = [[p,q],[s,t]] (2x2)
        # Strassen gives rank 7 for computing C' from A and B'.
        # Then I need 4 more terms for r and u (column 2 of C), using g and j.
        # Total: 7 + 4 = 11. That's rank 11 for 2x2x3, not 8.
        #
        # To get rank 8, I need to share some of the column-2 computation with the first two.
        
        # Let me try a different grouping. What if I don't use Strassen but find
        # a clever 8-term decomposition directly?
        
        # Actually, let me reconsider. The minimum rank for [2,2,3] might actually be 8,
        # but the construction is non-trivial. Let me try to find it by optimization.
        
        # For now, let me just go with the known-good rank-32 solution but see if I can
        # do better with a different block decomposition.
        
        # Alternative: use rank 8 for 2x2x2 is not possible (min is 7).
        # What about 2x2x4? Min rank is 14 (two Strassen blocks).
        # 2x2x9: could be 2*(rank 2x2x4) + rank(2x2x1) = 2*14+4 = 32. Same as current.
        # Or rank(2x2x3) + rank(2x2x3) + rank(2x2x3) = 3*8 = 24 if rank(2,2,3)=8.
        
        # Let me just try to verify if a rank-8 exists by solving the system.
        # But that's expensive. Instead, let me use the fallback rank-32.
        
        return None  # Indicate we can't easily construct rank-8
    
    # Since constructing a verified rank-8 for 2x2x3 is complex and error-prone,
    # let me try a different optimization of the rank-32 approach.
    
    # Current: 4 blocks of 2x2x2 (Strassen, rank 7) + 1 block of 2x2x1 (schoolbook, rank 4) = 32
    
    # Alternative: 2 blocks of 2x2x4 (rank 14 each) + 1 block of 2x2x1 (rank 4) = 32. Same.
    
    # What if I use a non-uniform partition? E.g., 3 blocks:
    # Block 1: columns 0,1 (2x2x2, rank 7)
    # Block 2: columns 2,3,4 (2x2x3, rank ?)
    # Block 3: columns 5,6,7,8 (2x2x4, rank 14)
    # Total: 7 + rank(2,2,3) + 14
    
    # If rank(2,2,3) = 8, total = 29. That would be an improvement!
    
    # Let me try to construct a rank-8 for 2x2x3 more carefully.
    
    def try_rank8_223():
        """Attempt to construct and verify a rank-8 decomposition for 2x2x3."""
        # C[i,j] = sum_k A[i,k]*B[k,j]
        # A: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
        # B: e=B[0,0], f=B[0,1], g=B[0,2], h=B[1,0], i=B[1,1], j=B[1,2]
        # C: p=C[0,0]=ae+bh, q=C[0,1]=af+bi, r=C[0,2]=ag+bj
        #     s=C[1,0]=ce+dh, t=C[1,1]=cf+di, u=C[1,2]=cg+dj
        
        # I'll use a parametric approach and solve for the 8 terms.
        # Each term r has U_r (4), V_r (6), W_r (6).
        # The constraint is: sum_r U_r[i]*V_r[k]*W_r[j] = T[i,k,j]
        # where T is the multiplication tensor.
        
        # This is a nonlinear system. Let me try a specific ansatz.
        
        # Ansatz: Use Strassen-like structure for columns 0,1 and extend to column 2.
        
        # Strassen terms for (A, B[:,0:2]) -> C[:,0:2]:
        # M1 = (a+d)(e+h)  [U=[1,0,0,1], V=[1,0,0,1,0,0], W for p and t]
        #   Expands: ae+ah+de+dh
        #   ae -> p, dh -> s, ah and de are "errors" that must cancel
        # M2 = (c+d)e     [U=[0,0,1,1], V=[1,0,0,0,0,0], W for s]
        #   Expands: ce+de
        #   ce -> s, de cancels with de from M1
        # M3 = a(f-i)     [U=[1,0,0,0], V=[0,1,0,0,-1,0], W for q,t]
        #   Expands: af-ai
        #   af -> q, ai... A[0,0]*B[1,1]? Not a tensor entry. Error.
        
        # I see the issue - Strassen works for square matrices because the 
        # "cross terms" correspond to other entries of the output. For 
        # 2x2*2x3, the cross terms don't all map to valid output entries.
        
        # Let me try yet another approach: direct optimization with a specific structure.
        
        # Actually, I recall that rank([2,2,3]) might be 8, achieved by:
        # Using the fact that you can "fold" the third column.
        
        # Let me just go with rank 32 but try to see if there's a simpler improvement.
        
        # What about using a 2x2x5 block? rank(2,2,5) <= 2*rank(2,2,2)+rank(2,2,1) = 18
        # And 2x2x4 block: rank <= 14
        # 2x2x5 + 2x2x4 = 18+14 = 32. Still the same.
        
        # I think 32 might be optimal or near-optimal for this problem with 
        # the simple block decomposition approach. Let me just return the rank-32 solution.
        
        return None
    
    # Fall back to the known rank-32 solution
    terms = []
    
    # Strassen blocks for k-pairs: (0,1), (2,3), (4,5), (6,7)
    for block_idx in range(4):
        k0 = 2 * block_idx
        k1 = k0 + 1
        
        # M1 = (a+d)(e+h) -> C[0,k0] +1, C[1,k1] +1
        U1 = {0: 1, 3: 1}
        V1 = {k0: 1, 9+k1: 1}
        W1 = {2*k0: 1, 2*k1+1: 1}
        terms.append((U1, V1, W1))
        
        # M2 = (c+d)e -> C[1,k0] +1, C[1,k1] -1
        U2 = {2: 1, 3: 1}
        V2 = {k0: 1}
        W2 = {2*k0+1: 1, 2*k1+1: -1}
        terms.append((U2, V2, W2))
        
        # M3 = a(f-h) -> C[0,k1] +1, C[1,k1] +1
        U3 = {0: 1}
        V3 = {k1: 1, 9+k1: -1}
        W3 = {2*k1: 1, 2*k1+1: 1}
        terms.append((U3, V3, W3))
        
        # M4 = d(g-e) -> C[0,k0] +1, C[1,k0] +1
        U4 = {3: 1}
        V4 = {9+k0: 1, k0: -1}
        W4 = {2*k0: 1, 2*k0+1: 1}
        terms.append((U4, V4, W4))
        
        # M5 = (a+b)h -> C[0,k0] -1, C[0,k1] +1
        U5 = {0: 1, 1: 1}
        V5 = {9+k1: 1}
        W5 = {2*k0: -1, 2*k1: 1}
        terms.append((U5, V5, W5))
        
        # M6 = (c-a)(e+f) -> C[1,k1] +1
        U6 = {2: 1, 0: -1}
        V6 = {k0: 1, k1: 1}
        W6 = {2*k1+1: 1}
        terms.append((U6, V6, W6))
        
        # M7 = (b-d)(g+h) -> C[0,k0] +1
        U7 = {1: 1, 3: -1}
        V7 = {9+k0: 1, 9+k1: 1}
        W7 = {2*k0: 1}
        terms.append((U7, V7, W7))
    
    # Schoolbook for k=8
    k = 8
    terms.append(({0: 1}, {k: 1}, {2*k: 1}))
    terms.append(({1: 1}, {9+k: 1}, {2*k: 1}))
    terms.append(({2: 1}, {k: 1}, {2*k+1: 1}))
    terms.append(({3: 1}, {9+k: 1}, {2*k+1: 1}))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(18)]
    W = [[0] * rank for _ in range(18)]
    
    for t, (Ud, Vd, Wd) in enumerate(terms):
        for idx, val in Ud.items():
            U[idx][t] = val
        for idx, val in Vd.items():
            V[idx][t] = val
        for idx, val in Wd.items():
            W[idx][t] = val
    
    # Verify the decomposition
    def verify(U, V, W, n=2, m=2, p=9):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta terms
        # Actually, the convention is:
        # U has shape (n*m, rank), indexed by (i,j) -> i*m+j
        # V has shape (m*p, rank), indexed by (j,k) -> j*p+k  
        # W has shape (p*n, rank), indexed by (k,i) -> k*n+i
        # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # should equal 1 if (i,j,k) is a valid multiplication term, 0 otherwise.
        # For matrix mult: T[i,j,k] = 1 if i,j,k form a valid term, i.e., always 1 for the "multiplication" pattern
        # Actually T[i,j,k] = delta_{i,j,k} where the tensor is the multiplication tensor.
        # The multiplication tensor for C=A*B has T[i,j,k] = 1 for all valid (i,j,k) in the sum.
        # Specifically, T[i,j,k] = 1 if j is the contraction index, i.e., for each (i,j,k), 
        # T[i,j,k] = 1 means A[i,j]*B[j,k] contributes to C[i,k].
        # So T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in the right shape).
        
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # Reshape: U -> (n, m, rank), V -> (m, p, rank), W -> (p, n, rank)
        U_r = U_arr.reshape(n, m, -1)
        V_r = V_arr.reshape(m, p, -1)
        W_r = W_arr.reshape(p, n, -1)
        
        # T[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        # Use einsum
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        
        # Expected: T[i,j,k] = 1 for all i,j,k
        expected = np.ones((n, m, p))
        
        if not np.allclose(T, expected, atol=1e-9):
            diff = np.abs(T - expected)
            max_diff = np.max(diff)
            print(f"Verification failed! Max diff: {max_diff}")
            return False
        return True
    
    if not verify(U, V, W):
        # If verification fails, return schoolbook rank 36 as ultimate fallback
        rank36 = 36
        U36 = [[0]*rank36 for _ in range(4)]
        V36 = [[0]*rank36 for _ in range(18)]
        W36 = [[0]*rank36 for _ in range(18)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U36[i*m+j][r] = 1
                    V36[j*p+k][r] = 1
                    W36[k*n+i][r] = 1
        return {"U": U36, "V": V36, "W": W36}
    
    return {"U": U, "V": V, "W": W}
