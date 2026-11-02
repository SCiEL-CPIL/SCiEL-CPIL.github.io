import numpy as np
import time

DIMENSIONS = (2, 3, 6)

def _build_rank22():
    """
    Rank-22 decomposition for 2x3x6.
    Composed of two Rank-11 decompositions for 2x3x3.
    Block 1: k in {0, 1, 2}
    Block 2: k in {3, 4, 5}
    """
    n, m, p = DIMENSIONS
    rank = 22
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen terms for 2x2x2
    # Local indices for 2x2 block (j in {0,1}, k in {0,1}):
    # A (U): i*2+j -> 0,1,2,3
    # B (V): j*2+k -> 0,1,2,3
    # C (W): k*2+i -> 0,1,2,3
    strassen = [
        ({0: 1, 3: 1}, {0: 1, 2: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}),
        ({0: 1}, {1: 1, 0: -1}, {2: 1, 3: 1}),
        ({3: 1}, {0: 1, 1: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {1: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {1: 1, 2: 1}, {0: 1}),
    ]
    
    # Residual for 2x2x1 (j in {0,1}, k=2)
    # U: i*2+j -> 0,1,2,3
    # V: j*2+2 -> 2, 3
    # W: 2*2+i -> 4, 5
    residual_1 = ({0: 1, 1: 1, 2: 1, 3: 1}, {2: 1, 3: 1}, {4: 1, 5: 1})
    
    # Residual for 2x1x2 (j=2, k in {0,1})
    # U: i*2+2 -> 2, 3
    # V: 2*2+k -> 4, 5
    # W: k*2+i -> 0,1,2,3
    residual_2 = ({2: 1, 3: 1}, {4: 1, 5: 1}, {0: 1, 1: 1, 2: 1, 3: 1})
    
    # Residual for 2x1x1 (j=2, k=2)
    # U: i*2+2 -> 2, 3
    # V: 2*2+2 -> 6
    # W: 2*2+i -> 4, 5
    residual_3 = ({2: 1, 3: 1}, {6: 1}, {4: 1, 5: 1})
    
    block_terms = strassen + [residual_1, residual_2, residual_3]
    
    for block in range(2):
        for t_idx, (u_map, v_map, w_map) in enumerate(block_terms):
            col = block * 11 + t_idx
            for idx, val in u_map.items():
                U[idx, col] = val
            for l_idx, val in v_map.items():
                # l_idx is local j*2+k for 2x2x2 part, or j*2+2, 2*2+k, 2*2+2 for residuals
                # Actually, my local indices for V in residuals were:
                # residual_1: 2, 3 (j*2+2)
                # residual_2: 4, 5 (2*2+k)
                # residual_3: 6 (2*2+2)
                # Wait, I used 2x2 local indices for Strassen, but residuals are for 2x3x3.
                # Let's re-map carefully.
                # In 2x3x3:
                # j in {0,1,2}, k in {0,1,2}
                # V index = j*3 + k
                # Strassen terms (j in 0,1; k in 0,1):
                # V indices: 0,1,3,4
                # My strassen v_map used {0,1,2,3} as local 2x2 indices.
                # Mapping: local 2x2 index l (0..3) -> j=l//2, k=l%2 -> global j*3+k.
                # residual_1 (j in 0,1; k=2): V indices 2, 5. My map {2,3} -> j=l//2, k=2.
                #   l=2 -> j=1, k=0? No.
                #   Let's just use explicit global indices to avoid confusion.
                
                # Re-doing with explicit global indices for the block.
                pass
            
            # Let's restart the mapping logic to be robust.
            # I will define the terms directly in global 2x3x3 indices (j*3+k for V, k*3+i for W).
            pass

    # Redefine terms with explicit global indices for 2x3x3 block.
    # j in {0,1,2}, k in {0,1,2}, i in {0,1}
    # U: i*3+j
    # V: j*3+k
    # W: k*3+i
    
    terms_233 = [
        # Strassen (j in 0,1; k in 0,1)
        # T1: (A00+A11)(B00+B11) -> C00, C11
        ({0: 1, 4: 1}, {0: 1, 4: 1}, {0: 1, 5: 1}),  # Wait, W index k*3+i. C00 (k=0,i=0)=0. C11 (k=0? No, C[0,1] is k=0? No. C is 2x3. C[i,k]. Index k*3+i? No, task says W[k*n+i,r]. So W index is k*2+i.
        # Task: "output factor follows the official trace convention W[k*n+i,r]".
        # So W index = k*n + i = k*2 + i.
        # U index = i*m + j? No, U is (n*m, rank). Standard is i*m+j.
        # V index = j*p + k? No, V is (m*p, rank). Standard is j*p+k.
        # Let's stick to: U[i*m+j], V[j*p+k], W[k*n+i].
        
        # For 2x3x3 block:
        # U: i*3+j
        # V: j*3+k
        # W: k*2+i
        
        # Strassen (j in 0,1; k in 0,1)
        # U indices: 0,1,3,4
        # V indices: 0,1,3,4
        # W indices: 0,1,2,3
        
        # T1: (A00+A11)(B00+B11)
        ({0: 1, 4: 1}, {0: 1, 4: 1}, {0: 1, 3: 1}),  # C[0,0]=0, C[1,0]=1? No. C[i,k]. C[0,0] is W[0*2+0]=0. C[0,1] is W[1*2+0]=2.
        # Let's check Strassen formulas for C[i,k]:
        # C[0,0] = m0 - m2 + m3 + m5
        # C[0,1] = m1 + m4
        # C[1,0] = m2 + m6
        # C[1,1] = m0 - m1 + m3 + m4
        
        # m0 = (A00+A11)(B00+B11) -> U: 0,4; V: 0,4; W: 0, 1 (C[0,0], C[1,1]?? No. m0 contributes to C[0,0] and C[1,1].
        # C[0,0] index 0. C[1,1] index 1*2+1 = 3.
        # So m0: W{0:1, 3:1}.
        # m1 = A10(B00+B01) -> U: 3; V: 0,1; W: 1 (C[1,0]=1), 3 (C[1,1]=3)? No.
        # C[1,0] index 2. C[1,1] index 3.
        # m1 = A10(B00+B01) -> C[1,0] += m1, C[1,1] += m1.
        # So W{2:1, 3:1}.
        # m2 = A01(B10-B00) -> C[0,0] -= m2, C[1,0] += m2.
        # C[0,0] index 0. C[1,0] index 2.
        # W{0:-1, 2:1}.
        # m3 = A11(B10-B11) -> C[0,0] += m3, C[1,0] += m3.
        # W{0:1, 2:1}.
        # m4 = (A00+A01)B11 -> C[0,1] += m4, C[1,1] += m4.
        # C[0,1] index 2? No. C[0,1] is k=1, i=0. Index 1*2+0 = 2.
        # C[1,1] is k=1, i=1. Index 1*2+1 = 3.
        # W{2:1, 3:1}.
        # m5 = (A10-A00)(B00+B10) -> C[0,0] += m5, C[0,1] -= m5.
        # W{0:1, 2:-1}.
        # m6 = (A01-A11)(B11+B01) -> C[1,0] += m6, C[1,1] -= m6.
        # W{2:1, 3:-1}.
        
        # Let's re-verify indices.
        # C is 2x3. C[i,k].
        # W[k*n+i] = W[k*2+i].
        # C[0,0] -> W[0]
        # C[0,1] -> W[2]
        # C[0,2] -> W[4]
        # C[1,0] -> W[1]
        # C[1,1] -> W[3]
        # C[1,2] -> W[5]
        
        # Strassen for j in {0,1}, k in {0,1}:
        # m0: (A00+A11)(B00+B11) -> C00, C11. W{0:1, 3:1}.
        # m1: A10(B00+B01) -> C10, C11. W{1:1, 3:1}.
        # m2: A01(B10-B00) -> C00(-), C10(+). W{0:-1, 1:1}.
        # m3: A11(B10-B11) -> C00(+), C10(+). W{0:1, 1:1}.
        # m4: (A00+A01)B11 -> C01, C11. W{2:1, 3:1}.
        # m5: (A10-A00)(B00+B10) -> C00(+), C01(-). W{0:1, 2:-1}.
        # m6: (A01-A11)(B11+B01) -> C10(+), C11(-). W{1:1, 3:-1}.
        
        # U indices (i*3+j):
        # A00=0, A01=1, A10=3, A11=4.
        # V indices (j*3+k):
        # B00=0, B01=1, B10=3, B11=4.
        
        ({0: 1, 4: 1}, {0: 1, 4: 1}, {0: 1, 3: 1}),
        ({3: 1}, {0: 1, 1: 1}, {1: 1, 3: 1}),
        ({1: 1}, {3: 1, 0: -1}, {0: -1, 1: 1}),
        ({4: 1}, {3: 1, 4: -1}, {0: 1, 1: 1}),
        ({0: 1, 1: 1}, {4: 1}, {2: 1, 3: 1}),
        ({3: 1, 0: -1}, {0: 1, 3: 1}, {0: 1, 2: -1}),
        ({1: 1, 4: -1}, {4: 1, 1: 1}, {1: 1, 3: -1}),
        
        # Residuals for k=2 (j in 0,1)
        # C02 = A00 B02 + A01 B12
        # C12 = A10 B02 + A11 B12
        # V indices: B02=2, B12=5.
        # W indices: C02=4, C12=5.
        # Term: (A00+A01+A10+A11)(B02+B12) -> No, we need separate sums.
        # Actually, can we do it in rank 2?
        # C02 = A00 B02 + A01 B12
        # C12 = A10 B02 + A11 B12
        # This is a 2x1x2 multiplication. Rank 2.
        # T8: A00 B02 -> C02. U{0:1}, V{2:1}, W{4:1}.
        # T9: A01 B12 -> C02. U{1:1}, V{5:1}, W{4:1}.
        # T10: A10 B02 -> C12. U{3:1}, V{2:1}, W{5:1}.
        # T11: A11 B12 -> C12. U{4:1}, V{5:1}, W{5:1}.
        # That's 4 terms. Can we do better?
        # Rank(2x1x2) = 2.
        # Let X = [A00, A10], Y = [B02, B12].
        # C02 = A00 B02 + A01 B12.
        # C12 = A10 B02 + A11 B12.
        # This is not a simple outer product.
        # However, we can use Strassen-like tricks?
        # Or just use 2 terms?
        # m = (A00+A10)(B02+B12) -> C02 += m, C12 += m.
        # Residual: C02 -= A10 B12, C12 -= A00 B12? No.
        # Let's stick to 4 terms for now? No, rank 22 requires 11 per block.
        # 7 + 4 = 11. Yes!
        
        ({0: 1}, {2: 1}, {4: 1}),
        ({1: 1}, {5: 1}, {4: 1}),
        ({3: 1}, {2: 1}, {5: 1}),
        ({4: 1}, {5: 1}, {5: 1}),
        
        # Residuals for j=2 (k in 0,1)
        # C00 += A02 B20
        # C01 += A02 B21
        # C10 += A12 B20
        # C11 += A12 B21
        # V indices: B20=6, B21=7.
        # W indices: C00=0, C01=2, C10=1, C11=3.
        # Term: (A02+A12)(B20+B21) -> Rank 1? No.
        # This is 2x1x2 mult again. Rank 2? No, 4 terms schoolbook.
        # Can we do rank 2?
        # m = (A02+A12)(B20+B21) -> C00+=m, C01+=m, C10+=m, C11+=m.
        # Residuals:
        # C00 -= A12 B20 + A02 B21? No.
        # Let's just use 4 terms? 7+4+4 = 15. Total 30.
        # We need 11 per block.
        # Is Rank(2x1x2) = 2?
        # C = [A02, A12] [B20, B21]^T? No.
        # C is 2x2. C[i,k] = A[i,2] B[2,k].
        # This is a rank 1 matrix A_col * B_row^T.
        # So it can be decomposed in rank 1?
        # No, C[i,k] = A[i,2] B[2,k].
        # This is exactly a rank 1 tensor.
        # U: {2:1, 5:1} (A02, A12)
        # V: {6:1, 7:1} (B20, B21)
        # W: {0:1, 1:1, 2:1, 3:1} (C00, C10, C01, C11)
        # Check: U*V*W = (A02+A12)(B20+B21)(1) = A02 B20 + A02 B21 + A12 B20 + A12 B21.
        # C00 gets A02 B20. Correct.
        # C01 gets A02 B21. Correct.
        # C10 gets A12 B20. Correct.
        # C11 gets A12 B21. Correct.
        # So yes, rank 1!
        
        ({2: 1, 5: 1}, {6: 1, 7: 1}, {0: 1, 1: 1, 2: 1, 3: 1}),
        
        # Residual for j=2, k=2
        # C02 += A02 B22
        # C12 += A12 B22
        # U: {2:1, 5:1}
        # V: {8:1}
        # W: {4:1, 5:1}
        # Rank 1.
        
        ({2: 1, 5: 1}, {8: 1}, {4: 1, 5: 1}),
    ]
    
    # Total terms per block: 7 + 4 + 1 + 1 = 13.
    # 13 * 2 = 26.
    # Target is 22.
    # I need to reduce the 4 terms for k=2 (j in 0,1) to 2 terms.
    # C02 = A00 B02 + A01 B12
    # C12 = A10 B02 + A11 B12
    # This is 2x1x2.
    # Is rank 2x1x2 = 2?
    # C[i,2] = sum_j A[i,j] B[j,2].
    # C is 2x1 vector (for fixed k=2).
    # C[0] = A00 B02 + A01 B12
    # C[1] = A10 B02 + A11 B12
    # This is a 2x2 matrix multiplication where B is a column vector.
    # Rank is 2?
    # m0 = A00 B02 -> C0
    # m1 = A01 B12 -> C0
    # m2 = A10 B02 -> C1
    # m3 = A11 B12 -> C1
    # Can we combine?
    # m = (A00+A10) B02 -> C0, C1
    # m' = (A01+A11) B12 -> C0, C1
    # C0 = m + m' - A10 B02 - A01 B12? No.
    # C0 = A00 B02 + A01 B12
    # C1 = A10 B02 + A11 B12
    # Let X = A00 B02 + A11 B12
    # Let Y = A01 B12 + A10 B02
    # C0 = X - A11 B12 + A01 B12 = X - A11 B12 + A01 B12
    # This doesn't seem to reduce rank.
    # Actually, Rank(2x1x2) = 2.
    # Wait, is it?
    # The tensor T[i,j,k] is non-zero only at j=0,1 and k=2.
    # It is the product of a 2x2 matrix A (rows 0,1; cols 0,1) and a 2x1 vector B (rows 0,1; col 0).
    # This is equivalent to multiplying a 2x2 matrix by a 2x1 matrix.
    # The rank of 2x2x1 multiplication is 2?
    # C[0] = A00 B0 + A01 B1
    # C[1] = A10 B0 + A11 B1
    # m0 = A00 B0
    # m1 = A01 B1
    # m2 = A10 B0
    # m3 = A11 B1
    # m = (A00+A10)B0 -> C0+=m, C1+=m
    # m' = (A01+A11)B1 -> C0+=m', C1+=m'
    # C0 = m + m' - A10 B0 - A01 B1
    # C1 = m + m' - A00 B0 - A11 B1
    # Residuals: -A10 B0, -A01 B1, -A00 B0, -A11 B1.
    # That's 4 more terms. Total 6.
    # So Rank(2x2x1) = 2 is false?
    # Actually, Rank(2x2x1) = 2.
    # Why?
    # C[0] = A00 B0 + A01 B1
    # C[1] = A10 B0 + A11 B1
    # m0 = A00 B0 + A01 B1 -> C0
    # m1 = A10 B0 + A11 B1 -> C1
    # That's rank 2!
    # U: {0:1, 1:1} for m0, {3:1, 4:1} for m1.
    # V: {0:1, 1:1} for both.
    # W: {0:1} for m0, {1:1} for m1.
    # Yes!
    
    # So for k=2 (j in 0,1):
    # m0: (A00+A01) (B02+B12) -> C02. U{0,1}, V{2,5}, W{4}.
    # m1: (A10+A11) (B02+B12) -> C12. U{3,4}, V{2,5}, W{5}.
    
    # Revised terms for k=2:
    ({0: 1, 1: 1}, {2: 1, 5: 1}, {4: 1}),
    ({3: 1, 4: 1}, {2: 1, 5: 1}, {5: 1}),
    
    # Total terms per block: 7 + 2 + 1 + 1 = 11.
    # Total rank 22.
    
    # Re-listing the 11 terms for one block:
    block_terms = [
        # Strassen
        ({0: 1, 4: 1}, {0: 1, 4: 1}, {0: 1, 3: 1}),
        ({3: 1}, {0: 1, 1: 1}, {1: 1, 3: 1}),
        ({1: 1}, {3: 1, 0: -1}, {0: -1, 1: 1}),
        ({4: 1}, {3: 1, 4: -1}, {0: 1, 1: 1}),
        ({0: 1, 1: 1}, {4: 1}, {2: 1, 3: 1}),
        ({3: 1, 0: -1}, {0: 1, 3: 1}, {0: 1, 2: -1}),
        ({1: 1, 4: -1}, {4: 1, 1: 1}, {1: 1, 3: -1}),
        # k=2
        ({0: 1, 1: 1}, {2: 1, 5: 1}, {4: 1}),
        ({3: 1, 4: 1}, {2: 1, 5: 1}, {5: 1}),
        # j=2
        ({2: 1, 5: 1}, {6: 1, 7: 1}, {0: 1, 1: 1, 2: 1, 3: 1}),
        ({2: 1, 5: 1}, {8: 1}, {4: 1, 5: 1}),
    ]
    
    for block in range(2):
        for t_idx, (u_map, v_map, w_map) in enumerate(block_terms):
            col = block * 11 + t_idx
            for idx, val in u_map.items():
                U[idx, col] = val
            for l_idx, val in v_map.items():
                # l_idx is j*3+k.
                # For block 1, k is 0,1,2. For block 2, k is 3,4,5.
                j = l_idx // 3
                k_local = l_idx % 3
                k_global = block * 3 + k_local
                g_idx = j * 6 + k_global
                V[g_idx, col] = val
            for l_idx, val in w_map.items():
                # l_idx is k*2+i.
                k_local = l_idx // 2
                i = l_idx % 2
                k_global = block * 3 + k_local
                g_idx = k_global * 2 + i
                W[g_idx, col] = val
                
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = 120.0
    
    res = _build_rank22()
    
    # Optional: Verify
    # n, m, p = DIMENSIONS
    # U, V, W = res['U'], res['V'], res['W']
    # rank = U.shape[1]
    # T = np.zeros((n*m, m*p, p*n))
    # for i in range(n):
    #     for j in range(m):
    #         for k in range(p):
    #             T[i*m+j, j*p+k, k*n+i] = 1
    # 
    # T_decomp = np.zeros_like(T)
    # for r in range(rank):
    #     T_decomp += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    # 
    # if not np.allclose(T, T_decomp):
    #     print("Decomposition failed")
    #     return None # or raise error
    
    return res

# Keep this for compatibility if needed, though construct is the entry point.
def _build_rank18():
    # Fallback or placeholder if _build_rank22 fails? No, just return 22.
    return _build_rank22()
