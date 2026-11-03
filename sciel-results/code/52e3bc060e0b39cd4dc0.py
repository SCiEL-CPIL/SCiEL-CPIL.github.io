"""Matrix multiplication 3x5x7 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n=3, m=5, p=7, tol=1e-9):
    """Verify that U, V, W form a valid decomposition of the matmul tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check all tensor entries: sum_c U[i*m+j,c]*V[j'*p+k,c]*W[k'*n+i',c] = delta_{j,j'}delta_{k,k'}delta_{i,i'}
    # We check all (i,j,k,i2,j2,k2) combinations that should be 1, and a sample of those that should be 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = np.dot(U_arr[i*m+j, :], np.multiply(V_arr[j*p+k, :], W_arr[k*n+i, :]))
                if abs(lhs - 1.0) > tol:
                    return False
    # Check a sample of zero entries
    rng = np.random.RandomState(12345)
    for _ in range(2000):
        i = rng.randint(n)
        j = rng.randint(m)
        k = rng.randint(p)
        i2 = rng.randint(n)
        j2 = rng.randint(m)
        k2 = rng.randint(p)
        if i == i2 and j == j2 and k == k2:
            continue
        lhs = np.dot(U_arr[i*m+j, :], np.multiply(V_arr[j2*p+k2, :], W_arr[k2*n+i2, :]))
        if abs(lhs) > tol:
            return False
    return True

def _schoolbook(n=3, m=5, p=7):
    """Standard schoolbook decomposition, rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    i_idx, j_idx, k_idx = np.indices((n, m, p)).reshape(3, -1)
    i_flat = i_idx.ravel()
    j_flat = j_idx.ravel()
    k_flat = k_idx.ravel()
    
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    cols = np.arange(rank)
    
    U[u_rows, cols] = 1
    V[v_rows, cols] = 1
    W[w_rows, cols] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _strassen_block(U, V, W, i0, j0, k0, col_offset, n=3, m=5, p=7):
    """
    Apply Strassen decomposition to a 2x2x2 sub-block starting at (i0, j0, k0).
    Uses 7 multiplications instead of 8, reducing rank by 1 per block.
    Strassen formulas for C = A*B where A is 2x2, B is 2x2:
      M1 = (A00+A11)(B00+B11)
      M2 = (A10+A11)B00
      M3 = A00(B10-B11)
      M4 = A11(B01-B00)
      M5 = (A00+A01)B11
      M6 = (A10-A00)(B00+B10)
      M7 = (A01-A11)(B01+B11)
      C00 = M1 + M4 - M5 + M7
      C01 = M3 + M5
      C10 = M2 + M4
      C11 = M1 - M2 + M3 + M6
    """
    # The 8 schoolbook terms for positions (i0+di, j0+dj, k0+dk) where di,dj,dk in {0,1}
    # Each term c has U[i*m+j, c]=1, V[j'*p+k, c]=1, W[k'*n+i', c]=1
    # For Strassen, we replace 8 rank-1 terms with 7.
    
    # U indices: i*m + j for i in {i0, i0+1}, j in {j0, j0+1}
    # V indices: j'*p + k for j' in {j0, j0+1}, k in {k0, k0+1}
    # W indices: k'*n + i' for k' in {k0, k0+1}, i' in {i0, i0+1}
    
    # The 7 Strassen "multiplications" correspond to 7 new rank-1 terms
    # We need to express each C element as a linear combination of these 7 terms.
    
    # For the matmul tensor, the (i,j,k) entry of the tensor is delta_{j,j'}delta_{k,k'}delta_{i,i'}
    # In a 2x2x2 block, the 8 entries are all 1s on the diagonal (i,j,k) = (i0+di, j0+dj, k0+dk)
    # with i'=i, j'=j, k'=k.
    
    # Actually, let me think about this more carefully.
    # The tensor T has T[i,j,k; i',j',k'] = delta_{j,j'} delta_{k,k'} delta_{i,i'}
    # For a 2x2x2 block at (i0,j0,k0), we look at i in {i0,i0+1}, j in {j0,j0+1}, k in {k0,k0+1}
    # and i' in {i0,i0+1}, j' in {j0,j0+1}, k' in {k0,k0+1}
    # The nonzero entries are when i=i', j=j', k=k'
    # So we have 8 nonzero entries: (i0,j0,k0), (i0,j0+1,k0), (i0,j0,k0+1), (i0,j0+1,k0+1),
    #   (i0+1,j0,k0), (i0+1,j0+1,k0), (i0+1,j0,k0+1), (i0+1,j0+1,k0+1)
    
    # In the tensor decomposition, each rank-1 term is U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c]
    # For a schoolbook term at (i,j,k), we have U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1
    # This contributes 1 to T[i,j,k; i,j,k] and 0 elsewhere.
    
    # For Strassen, we want to express the 8 diagonal entries as 7 rank-1 terms.
    # The key insight: in the 2x2x2 sub-tensor, the 8 entries can be written as 7 products.
    
    # Let's define the 7 Strassen terms for this block:
    # Term 1: U has +1 at (i0,j0) and +1 at (i0+1,j0+1); V has +1 at (j0,k0) and +1 at (j0+1,k0+1); W has +1 at (k0,i0) and +1 at (k0+1,i0+1)
    # Term 2: U has +1 at (i0+1,j0) and +1 at (i0+1,j0+1); V has +1 at (j0,k0); W has +1 at (k0,i0+1)
    # Term 3: U has +1 at (i0,j0); V has +1 at (j0+1,k0) and -1 at (j0+1,k0+1); W has +1 at (k0+1,i0)
    # Term 4: U has +1 at (i0+1,j0+1); V has +1 at (j0,k0+1) and -1 at (j0,k0); W has +1 at (k0+1,i0+1)
    # Term 5: U has +1 at (i0,j0) and +1 at (i0,j0+1); V has +1 at (j0+1,k0+1); W has +1 at (k0+1,i0)
    # Term 6: U has +1 at (i0+1,j0) and -1 at (i0,j0); V has +1 at (j0,k0) and +1 at (j0,k0+1); W has +1 at (k0,i0+1)
    # Term 7: U has +1 at (i0,j0+1) and -1 at (i0+1,j0+1); V has +1 at (j0+1,k0+1) and +1 at (j0+1,k0); W has +1 at (k0+1,i0)
    
    # Let me verify this works. The contribution to T[i,j,k; i',j',k'] from term t is
    # U[i*m+j,t] * V[j'*p+k,t] * W[k'*n+i',t]
    
    # For entry (i0,j0,k0; i0,j0,k0):
    # Term 1: U[i0*m+j0,1]=1, V[j0*p+k0,1]=1, W[k0*n+i0,1]=1 -> 1
    # Term 2: U[i0*m+j0,2]=0 -> 0
    # Term 3: U[i0*m+j0,3]=1, V[j0*p+k0,3]=0 -> 0
    # Term 4: U[i0*m+j0,4]=0 -> 0
    # Term 5: U[i0*m+j0,5]=1, V[j0*p+k0,5]=0 -> 0
    # Term 6: U[i0*m+j0,6]=-1, V[j0*p+k0,6]=1, W[k0*n+i0,6]=0 -> 0
    # Term 7: U[i0*m+j0,7]=0 -> 0
    # Total: 1 ✓
    
    # For entry (i0,j0+1,k0; i0,j0+1,k0):
    # Term 1: U[i0*m+j0+1,1]=0 -> 0
    # Term 2: U[i0*m+j0+1,2]=0 -> 0
    # Term 3: U[i0*m+j0+1,3]=0 -> 0
    # Term 4: U[i0*m+j0+1,4]=0 -> 0
    # Term 5: U[i0*m+j0+1,5]=1, V[j0+1*p+k0,5]=0 -> 0
    # Term 6: U[i0*m+j0+1,6]=0 -> 0
    # Term 7: U[i0*m+j0+1,7]=1, V[j0+1*p+k0,7]=1, W[k0*n+i0,7]=1 -> 1
    # Total: 1 ✓
    
    # For entry (i0,j0,k0+1; i0,j0,k0+1):
    # Term 1: U[i0*m+j0,1]=1, V[j0*p+k0+1,1]=0 -> 0
    # Term 2: U[i0*m+j0,2]=0 -> 0
    # Term 3: U[i0*m+j0,3]=1, V[j0*p+k0+1,3]=0 -> 0
    # Term 4: U[i0*m+j0,4]=0 -> 0
    # Term 5: U[i0*m+j0,5]=1, V[j0*p+k0+1,5]=0 -> 0
    # Term 6: U[i0*m+j0,6]=-1, V[j0*p+k0+1,6]=1, W[k0+1*n+i0,6]=0 -> 0
    # Term 7: U[i0*m+j0,7]=0 -> 0
    # Total: 0 ✗ (should be 1)
    
    # Hmm, this doesn't work. Let me reconsider.
    
    # Actually, I think the issue is that Strassen's algorithm works for matrix multiplication C=A*B,
    # where the tensor is T[A,B,C] with A being 2x2, B being 2x2, C being 2x2.
    # The matmul tensor for 2x2x2 has entries T[a1,a2,b1,b2,c1,c2] = delta_{a2,b1} * delta_{b2,c1} * delta_{a1,c2}
    # Wait no. Let me be more careful.
    
    # For multiplying A (2x2) by B (2x2) to get C (2x2):
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # The tensor is T[i,k,j; i',k',j'] where the "input" indices are (i,k) for A, (k',j') for B,
    # and the "output" index is (i',j').
    # T[i,k,j; i',k',j'] = delta_{k,k'} * delta_{i,i'} * delta_{j,j'}
    
    # So in our notation with n=m=p=2:
    # U index: i*m + k = 2i + k (for A)
    # V index: k'*p + j' = 2k' + j' (for B)
    # W index: j'*n + i' = 2j' + i' (for C)
    # T[i,k,j; i',k',j'] = delta_{k,k'} * delta_{i,i'} * delta_{j,j'}
    
    # The 8 nonzero entries are (i,k,j) = (i',k',j') for all i,k,j in {0,1}
    # i.e., (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)
    
    # Strassen replaces these 8 rank-1 terms with 7.
    # The 7 Strassen terms for 2x2x2:
    # Let me use the standard Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M3+M6
    
    # In tensor form, each M_t is a product of a linear combination of A entries and a linear combination of B entries.
    # M1 = (A00+A11)(B00+B11): U has +1 at (0,0) and +1 at (1,1); V has +1 at (0,0) and +1 at (1,1); W has +1 at (0,0)
    # Wait, that's not right either. Let me think again.
    
    # Actually, in the tensor decomposition, a rank-1 term is:
    # U[i*m+k, t] * V[k'*p+j', t] * W[j'*n+i', t]
    # This contributes U[i*m+k,t] * V[k'*p+j',t] * W[j'*n+i',t] to T[i,k,j; i',k',j']
    
    # For M1 = (A00+A11)(B00+B11), this means:
    # U[0*m+0, t1] = 1, U[1*m+1, t1] = 1 (A00 and A11)
    # V[0*p+0, t1] = 1, V[1*p+1, t1] = 1 (B00 and B11)
    # W[0*n+0, t1] = 1 (C00)
    # But wait, M1 contributes to C00, so W should only have +1 at (0,0) for C00.
    # But M1 also contributes to C11 with coefficient -1, so W[1*n+1, t1] = -1.
    
    # Let me redo this properly. Each Strassen term t contributes to multiple C entries.
    # M1 contributes to C00 with +1 and C11 with -1.
    # So: U[0, t1]=1, U[3, t1]=1; V[0, t1]=1, V[3, t1]=1; W[0, t1]=1, W[3, t1]=-1
    # (using 0-indexed: A00 is at U[0], A11 at U[3], B00 at V[0], B11 at V[3], C00 at W[0], C11 at W[3])
    
    # Let me verify: T[0,0,0; 0,0,0] from t1: U[0,1]*V[0,1]*W[0,1] = 1*1*1 = 1
    # T[0,0,1; 0,0,1] from t1: U[0,1]*V[0,1]*W[1,1] = 1*1*0 = 0
    # T[1,1,1; 1,1,1] from t1: U[3,1]*V[3,1]*W[3,1] = 1*1*(-1) = -1
    # So M1 contributes +1 to (0,0,0) and -1 to (1,1,1).
    
    # Now let's list all 7 terms:
    # t1 (M1): U[0]=1,U[3]=1; V[0]=1,V[3]=1; W[0]=1,W[3]=-1
    # t2 (M2): U[2]=1,U[3]=1; V[0]=1; W[2]=1
    # t3 (M3): U[0]=1; V[2]=1,V[3]=-1; W[1]=1
    # t4 (M4): U[3]=1; V[1]=1,V[0]=-1; W[2]=1
    # t5 (M5): U[0]=1,U[1]=1; V[3]=1; W[1]=1
    # t6 (M6): U[2]=1,U[0]=-1; V[0]=1,V[2]=1; W[3]=1
    # t7 (M7): U[1]=1,U[3]=-1; V[3]=1,V[1]=1; W[0]=1
    
    # Let me verify all 8 entries:
    # (0,0,0): t1:1*1*1=1, t2:0, t3:0, t4:0, t5:0, t6:0, t7:0 -> 1 ✓
    # (0,0,1): t1:1*1*0=0, t2:0, t3:1*0*0=0, t4:0, t5:1*0*1=0, t6:0, t7:0*1*1=0 -> 0 ✗ (should be 1)
    
    # Hmm, still not working. Let me look up the correct Strassen tensor decomposition.
    
    # Actually, I think the issue is that I'm confusing the tensor structure. Let me restart.
    
    # The matmul tensor for 2x2x2: T is a 3-tensor of shape (4,4,4) where
    # T[a,b,c] = 1 if (a,b,c) corresponds to a valid (i,k,j; i',k',j') with i=i', k=k', j=j'
    # Using a = 2i+k, b = 2k'+j', c = 2j'+i'
    # T[2i+k, 2k'+j', 2j'+i'] = 1 if i=i', k=k', j=j'
    
    # So T[a,b,c] = 1 iff there exist i,k,j,i',k',j' such that a=2i+k, b=2k'+j', c=2j'+i', and i=i', k=k', j=j'
    # This means a=2i+k, b=2k+j', c=2j'+i with i,k,j' free.
    # So T[2i+k, 2k+j', 2j'+i] = 1 for all i,k,j' in {0,1}
    
    # The 8 nonzero entries:
    # i=0,k=0,j'=0: T[0,0,0]
    # i=0,k=0,j'=1: T[0,1,2]
    # i=0,k=1,j'=0: T[1,0,0]
    # i=0,k=1,j'=1: T[1,1,2]
    # i=1,k=0,j'=0: T[2,0,1]
    # i=1,k=0,j'=1: T[2,1,3]
    # i=1,k=1,j'=0: T[3,0,1]
    # i=1,k=1,j'=1: T[3,1,3]
    
    # Now Strassen: we want to write these 8 entries as 7 rank-1 terms.
    # A rank-1 term is u * v * w where u is a 4-vector, v is a 4-vector, w is a 4-vector.
    # The 7 Strassen terms:
    
    # Let me use the known Strassen decomposition:
    # t1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,-1]
    # t2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]
    # t3: u=[1,0,0,0], v=[0,0,1,-1], w=[0,1,0,0]
    # t4: u=[0,0,0,1], v=[0,1,-1,0], w=[0,0,1,0]
    # t5: u=[1,1,0,0], v=[0,0,0,1], w=[0,1,0,0]
    # t6: u=[-1,0,1,0], v=[1,0,1,0], w=[0,0,0,1]
    # t7: u=[0,1,0,-1], v=[0,1,0,1], w=[1,0,0,0]
    
    # Verify T[0,0,0]: t1:1*1*1=1, others:0 -> 1 ✓
    # Verify T[0,1,2]: t1:1*0*0=0, t3:1*0*0=0, t5:1*0*0=0, t7:0*1*0=0 -> 0 ✗ (should be 1)
    
    # Still not working. I think I need to be more careful with the index mapping.
    
    # Let me just hardcode the known Strassen decomposition for 2x2x2 and verify it numerically.
    
    # Actually, let me take a different approach. I'll use the fact that Strassen's algorithm
    # gives a rank-7 decomposition of the 2x2x2 matmul tensor. I'll construct it by
    # solving the linear system.
    
    pass

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition of the 2x2x2 matmul tensor.
    U, V, W are 4x7 matrices.
    """
    # Strassen's 7 multiplications for 2x2 matrix multiplication
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-g)
    # M4 = d(h-f)
    # M5 = (a+b)h
    # M6 = (c-a)(e+g)
    # M7 = (b-d)(f+h)
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # In tensor form, the matmul tensor T has T[i,k,j; i',k',j'] = delta_{i,i'}delta_{k,k'}delta_{j,j'}
    # With a=2i+k, b=2k'+j', c=2j'+i'
    
    # Let me construct U, V, W directly from the Strassen formulas.
    # A entries: a=A[0,0] at index 0, b=A[0,1] at index 1, c=A[1,0] at index 2, d=A[1,1] at index 3
    # B entries: e=B[0,0] at index 0, f=B[0,1] at index 1, g=B[1,0] at index 2, h=B[1,1] at index 3
    # C entries: C[0,0] at index 0, C[0,1] at index 1, C[1,0] at index 2, C[1,1] at index 3
    
    # But wait, in our tensor convention:
    # U index: i*m + k where m=2, so U[2i+k]
    # V index: k'*p + j' where p=2, so V[2k'+j']
    # W index: j'*n + i' where n=2, so W[2j'+i']
    
    # A[i,k] = U[2i+k]: a=U[0], b=U[1], c=U[2], d=U[3]
    # B[k',j'] = V[2k'+j']: e=V[0], f=V[1], g=V[2], h=V[3]
    # C[j',i'] = W[2j'+i']: C00=W[0], C01=W[1], C10=W[2], C11=W[3]
    
    # Hmm wait, that's not right. Let me re-examine.
    # W index: j'*n + i' = 2j' + i'
    # So W[0] = C[0,0], W[1] = C[0,1], W[2] = C[1,0], W[3] = C[1,1]
    # Yes that's correct.
    
    # Now, M1 = (a+d)(e+h): U[0]=1, U[3]=1; V[0]=1, V[3]=1; contributes to C00 (+1) and C11 (-1)
    # So W[0]=1, W[3]=-1
    
    # M2 = (c+d)e: U[2]=1, U[3]=1; V[0]=1; contributes to C10 (+1)
    # W[2]=1
    
    # M3 = a(f-g): U[0]=1; V[1]=1, V[2]=-1; contributes to C01 (+1)
    # W[1]=1
    
    # M4 = d(h-f): U[3]=1; V[3]=1, V[1]=-1; contributes to C00 (+1) and C10 (+1)
    # W[0]=1, W[2]=1
    
    # M5 = (a+b)h: U[0]=1, U[1]=1; V[3]=1; contributes to C01 (+1) and C00 (-1)
    # W[1]=1, W[0]=-1
    
    # M6 = (c-a)(e+g): U[2]=1, U[0]=-1; V[0]=1, V[2]=1; contributes to C11 (+1)
    # W[3]=1
    
    # M7 = (b-d)(f+h): U[1]=1, U[3]=-1; V[1]=1, V[3]=1; contributes to C00 (+1)
    # W[0]=1
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # t1 (M1)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = -1
    
    # t2 (M2)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # t3 (M3)
    U[0, 2] = 1
    V[1, 2] = 1; V[2, 2] = -1
    W[1, 2] = 1
    
    # t4 (M4)
    U[3, 3] = 1
    V[3, 3] = 1; V[1, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # t5 (M5)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[0, 4] = -1
    
    # t6 (M6)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # t7 (M7)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _verify_strassen_2x2x2():
    """Verify the Strassen 2x2x2 decomposition."""
    U, V, W = _strassen_2x2x2()
    # Check all 8 nonzero entries
    # T[2i+k, 2k'+j', 2j'+i'] should be 1 when i=i', k=k', j=j'
    # i.e., T[2i+k, 2k+j', 2j'+i] = 1
    for i in range(2):
        for k in range(2):
            for j in range(2):
                a = 2*i + k
                b = 2*k + j
                c = 2*j + i
                val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                if abs(val - 1.0) > 1e-9:
                    print(f"FAIL: T[{a},{b},{c}] = {val}")
                    return False
    # Check some zero entries
    for a in range(4):
        for b in range(4):
            for c in range(4):
                # Check if this should be zero
                # T[a,b,c] = 1 iff there exist i,k,j' with a=2i+k, b=2k+j', c=2j'+i
                found = False
                for i in range(2):
                    for k in range(2):
                        for jp in range(2):
                            if 2*i+k == a and 2*k+jp == b and 2*jp+i == c:
                                found = True
                if not found:
                    val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                    if abs(val) > 1e-9:
                        print(f"FAIL: T[{a},{b},{c}] should be 0 but is {val}")
                        return False
    return True

def _strassen_decomposition(n=3, m=5, p=7):
    """
    Build a decomposition using Strassen on 2x2x2 sub-blocks where possible,
    and schoolbook for the rest.
    """
    # We can fit 2x2x2 blocks at positions:
    # i0 in {0,1} (since n=3, i0 can be 0 or 1, giving i in {i0, i0+1} subset of {0,1,2})
    # j0 in {0,1,2} (since m=5, j0 can be 0,1,2,3 but j0+1<=4, so j0 in {0,1,2,3})
    # k0 in {0,1,2,3,4,5} (since p=7, k0+1<=6, so k0 in {0,1,2,3,4,5})
    
    # We want to maximize the number of disjoint 2x2x2 blocks.
    # Each block covers 2 i-values, 2 j-values, 2 k-values.
    # n=3: can fit 1 block covering i={0,1} or i={1,2}
    # m=5: can fit 2 blocks covering j={0,1},{2,3} or j={1,2},{3,4} etc.
    # p=7: can fit 3 blocks covering k={0,1},{2,3},{4,5}
    
    # Strategy: use i0=0 (covers i=0,1), leaving i=2 for schoolbook.
    # Use j0=0,2 (covers j=0,1 and j=2,3), leaving j=4 for schoolbook.
    # Use k0=0,2,4 (covers k=0,1 and k=2,3 and k=4,5), leaving k=6 for schoolbook.
    
    # Number of Strassen blocks: 1 * 2 * 3 = 6 blocks, each saving 1 rank.
    # Total rank = 105 - 6 = 99.
    
    # Let's build this.
    # First, create the full schoolbook decomposition.
    rank_sb = n * m * p
    U = np.zeros((n * m, rank_sb), dtype=float)
    V = np.zeros((m * p, rank_sb), dtype=float)
    W = np.zeros((p * n, rank_sb), dtype=float)
    
    # Assign schoolbook terms to all positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
    
    # Now, for each 2x2x2 block, replace the 8 schoolbook terms with 7 Strassen terms.
    # We'll do this by modifying the U, V, W matrices.
    
    # Strassen 2x2x2 base decomposition
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Define which blocks to use
    i_blocks = [0]  # covers i=0,1
    j_blocks = [0, 2]  # covers j=0,1 and j=2,3
    k_blocks = [0, 2, 4]  # covers k=0,1 and k=2,3 and k=4,5
    
    # For each block, we need to:
    # 1. Zero out the 8 schoolbook terms for this block
    # 2. Add the 7 Strassen terms
    
    # The 8 schoolbook terms for block (i0, j0, k0) are at columns:
    # col(i0+dI, j0+dJ, k0+dK) for dI,dJ,dK in {0,1}
    
    # We'll use a new set of columns for the Strassen terms.
    # Actually, we can reuse the existing columns by modifying them.
    # But it's cleaner to allocate new columns and remove the old ones.
    
    # Let's build the final decomposition from scratch.
    U_final = []
    V_final = []
    W_final = []
    
    # Track which (i,j,k) positions are covered by Strassen blocks
    covered = set()
    for i0 in i_blocks:
        for j0 in j_blocks:
            for k0 in k_blocks:
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            covered.add((i0+di, j0+dj, k0+dk))
    
    # Add Strassen blocks
    for i0 in i_blocks:
        for j0 in j_blocks:
            for k0 in k_blocks:
                # Add 7 Strassen terms
                for t in range(7):
                    u_vec = np.zeros(n * m)
                    v_vec = np.zeros(m * p)
                    w_vec = np.zeros(p * n)
                    
                    for a in range(4):
                        if U_s[a, t] != 0:
                            # a = 2*di + dk where di is the i-offset, dk is the k-offset
                            di = a // 2
                            dk = a % 2
                            u_vec[(i0 + di) * m + (j0 + dk)] = U_s[a, t]
                        if V_s[a, t] != 0:
                            # a = 2*dk + dj where dk is the k-offset, dj is the j-offset
                            dk = a // 2
                            dj = a % 2
                            v_vec[(j0 + dk) * p + (k0 + dj)] = V_s[a, t]
                        if W_s[a, t] != 0:
                            # a = 2*dj + di where dj is the j-offset, di is the i-offset
                            dj = a // 2
                            di = a % 2
                            w_vec[(k0 + dj) * n + (i0 + di)] = W_s[a, t]
                    
                    U_final.append(u_vec)
                    V_final.append(v_vec)
                    W_final.append(w_vec)
    
    # Add schoolbook terms for uncovered positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    u_vec = np.zeros(n * m)
                    v_vec = np.zeros(m * p)
                    w_vec = np.zeros(p * n)
                    u_vec[i * m + j] = 1
                    v_vec[j * p + k] = 1
                    w_vec[k * n + i] = 1
                    U_final.append(u_vec)
                    V_final.append(v_vec)
                    W_final.append(w_vec)
    
    U_final = np.array(U_final).T  # (n*m, rank)
    V_final = np.array(V_final).T  # (m*p, rank)
    W_final = np.array(W_final).T  # (p*n, rank)
    
    rank = U_final.shape[1]
    print(f"Strassen decomposition rank: {rank}")
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication.
    Returns dict with 'U', 'V', 'W' factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Try Strassen-based decomposition
    try:
        sol = _strassen_decomposition(n, m, p)
        if _verify(sol["U"], sol["V"], sol["W"], n, m, p, tol=1e-8):
            return sol
    except Exception as e:
        print(f"Strassen failed: {e}")
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
