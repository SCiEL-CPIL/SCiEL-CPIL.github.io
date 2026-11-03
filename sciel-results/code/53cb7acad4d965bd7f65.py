import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = 110.0  # 10s margin before 120s limit

    n, m, p = DIMENSIONS

    # Schoolbook fallback (rank 54)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U_school[i * m + k][col] = 1
                V_school[k * p + j][col] = 1
                W_school[j * n + i][col] = 1
                col += 1

    # Attempt rank-45 decomposition
    # Split B (3x6) into three 3x2 blocks: B[:,0:2], B[:,2:4], B[:,4:6]
    # For each 3x3x2 block, use rank-15 decomposition
    # Total rank: 3 * 15 = 45

    def build_3x3x2_rank15(offset_j):
        """
        Build rank-15 decomposition for 3x3x2 matrix multiplication.
        A is 3x3, B is 3x2 (columns offset_j, offset_j+1), C is 3x2.
        Returns list of (u_idx, v_idx, w_idx, u_val, v_val, w_val)
        where u_idx is in [0,9), v_idx in [0,6) relative to this block, w_idx in [0,6) relative.
        
        Tensor: C[i, j] = sum_k A[i,k] * B[k, j]
        i in [0,3), j in [0,2), k in [0,3)
        
        U index: i*m + k = i*3 + k (9 entries)
        V index: k*p + j = k*2 + j (6 entries for 3x2)
        W index: j*n + i = j*3 + i (6 entries for 2x3)
        """
        terms = []
        
        # Strassen decomposition for the 2x2x2 sub-block (i,k in {0,1}, j in {0,1})
        # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
        # M2 = (A[0,0]+A[0,1])B[1,1]
        # M3 = A[0,0](B[0,0]-B[1,0])
        # M4 = A[1,1](B[1,0]-B[1,1])  -- wait, let me use standard Strassen
        # Standard Strassen for C = A*B where A,B are 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B01
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M2+M3-M4+M6
        # C10 = M2+M3+M5-M7
        # C11 = M1-M2+M4+M6
        
        # For our tensor T[i,j,k] = A[i,k]*B[k,j]:
        # A is 2x2 with A[i,k], B is 2x2 with B[k,j]
        # C[i,j] = sum_k A[i,k]*B[k,j]
        
        # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,0])  -- wait, B[k,j] so B[0,0] and B[1,0]
        # Actually B is indexed B[k,j], so B[0,0]=B00, B[0,1]=B01, B[1,0]=B10, B[1,1]=B11
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,0])
        # This contributes to: A part: (0,0) and (1,1), B part: (0,0) and (1,0)
        # As rank-1 tensors: u has entries at (0,0) and (1,1) in A-space, v has entries at (0,0) and (1,0) in B-space
        # u_idx for A[i,k]: (0,0)->0, (1,1)->4
        # v_idx for B[k,j]: (0,0)->0, (1,0)->2
        # w_idx for C[i,j]: all 4 entries (0,0),(0,1),(1,0),(1,1) -> 0,1,3,4
        
        # Let me define the 7 Strassen terms:
        # Each term is (u_dict, v_dict, w_dict) where dicts map index->value
        
        strassen_terms = []
        
        # M1 = (A00+A11)(B00+B10)
        # u: {0:1, 4:1}, v: {0:1, 2:1}, w: {0:1, 1:1, 3:1, 4:1}
        strassen_terms.append(({0:1, 4:1}, {0:1, 2:1}, {0:1, 1:1, 3:1, 4:1}))
        
        # M2 = (A10+A11)(B01)
        # u: {3:1, 4:1}, v: {1:1}, w: {1:1, 4:1}  -- C01 and C11 get M2
        # C[i,j]: M2 contributes to C[0,1] and C[1,1]
        # w: {1:1, 4:1}
        strassen_terms.append(({3:1, 4:1}, {1:1}, {1:1, 4:1}))
        
        # M3 = A00(B01-B11)
        # u: {0:1}, v: {1:1, 3:-1}, w: {1:1, 4:-1}
        strassen_terms.append(({0:1}, {1:1, 3:-1}, {1:1, 4:-1}))
        
        # M4 = A11(B10-B00)
        # u: {4:1}, v: {2:1, 0:-1}, w: {0:1, 3:-1}
        strassen_terms.append(({4:1}, {2:1, 0:-1}, {0:1, 3:-1}))
        
        # M5 = (A00+A01)(B11)
        # u: {0:1, 1:1}, v: {3:1}, w: {0:-1, 3:1}  -- C00 -= M5, C10 += M5
        strassen_terms.append(({0:1, 1:1}, {3:1}, {0:-1, 3:1}))
        
        # M6 = (A10-A00)(B00+B10)
        # u: {3:1, 0:-1}, v: {0:1, 2:1}, w: {1:1, 3:1}  -- C01 += M6, C10 += M6
        strassen_terms.append(({3:1, 0:-1}, {0:1, 2:1}, {1:1, 3:1}))
        
        # M7 = (A01-A11)(B01+B11)
        # u: {1:1, 4:-1}, v: {1:1, 3:1}, w: {0:1, 4:-1}  -- C00 += M7, C11 -= M7
        strassen_terms.append(({1:1, 4:-1}, {1:1, 3:1}, {0:1, 4:-1}))
        
        # Now handle the "border" terms where i=2 or k=2
        # Terms: C[i,j] += A[i,k]*B[k,j] where (i,k) not both in {0,1}
        
        # i=2, k=0, j=0: A[2,0]*B[0,0] -> u:{6:1}, v:{0:1}, w:{2:1}
        terms.append(({6:1}, {0:1}, {2:1}))
        # i=2, k=0, j=1: A[2,0]*B[0,1] -> u:{6:1}, v:{1:1}, w:{5:1}
        terms.append(({6:1}, {1:1}, {5:1}))
        # i=2, k=1, j=0: A[2,1]*B[1,0] -> u:{7:1}, v:{2:1}, w:{2:1}
        terms.append(({7:1}, {2:1}, {2:1}))
        # i=2, k=1, j=1: A[2,1]*B[1,1] -> u:{7:1}, v:{3:1}, w:{5:1}
        terms.append(({7:1}, {3:1}, {5:1}))
        # i=2, k=2, j=0: A[2,2]*B[2,0] -> u:{8:1}, v:{4:1}, w:{2:1}
        terms.append(({8:1}, {4:1}, {2:1}))
        # i=2, k=2, j=1: A[2,2]*B[2,1] -> u:{8:1}, v:{5:1}, w:{5:1}
        terms.append(({8:1}, {5:1}, {5:1}))
        # i=0, k=2, j=0: A[0,2]*B[2,0] -> u:{2:1}, v:{4:1}, w:{0:1}
        terms.append(({2:1}, {4:1}, {0:1}))
        # i=0, k=2, j=1: A[0,2]*B[2,1] -> u:{2:1}, v:{5:1}, w:{1:1}
        terms.append(({2:1}, {5:1}, {1:1}))
        # i=1, k=2, j=0: A[1,2]*B[2,0] -> u:{5:1}, v:{4:1}, w:{3:1}
        terms.append(({5:1}, {4:1}, {3:1}))
        # i=1, k=2, j=1: A[1,2]*B[2,1] -> u:{5:1}, v:{5:1}, w:{4:1}
        terms.append(({5:1}, {5:1}, {4:1}))
        
        # Total: 7 (Strassen) + 10 (border) = 17. But we need 15!
        # We need to combine some terms. Let me reconsider.
        
        # Actually, let me reconsider the approach. The rank of 3x3x2 is 15, not 17.
        # I need a smarter construction. Let me use a different decomposition.
        
        # Alternative: Use the fact that 3x3x2 can be decomposed more efficiently.
        # Let me try a different approach: express all 18 terms as linear combinations of 15 rank-1 tensors.
        
        # Actually, let me just verify if 17 is acceptable and use rank 51 (3*17).
        # But the goal is rank 45. Let me think harder.
        
        # For 3x3x2, one known rank-15 construction:
        # Use 7 Strassen terms for the 2x2x2 core, and 8 terms for the border.
        # The border has 10 terms, but we can combine 2 pairs.
        
        # Pair 1: A[2,0]*B[0,0] and A[2,1]*B[1,0] can be combined?
        # No, they have different A indices.
        
        # Actually, I think the issue is that my Strassen construction might not be optimal
        # for this embedding. Let me just use rank 17 per block = 51 total, or find the true rank-15.
        
        # For now, let me return the 17-term version and use rank 51.
        # Actually, let me try to get 15 by being more clever.
        
        # Hmm, let me just go with the schoolbook for safety and return rank 54.
        # The risk of an incorrect decomposition is worse than a slightly higher rank.
        
        return None  # Signal that this construction is not ready
    
    # Since the rank-15 construction is complex and error-prone, let me try a different approach:
    # Use Strassen-based decomposition for the full 3x3x6 directly.
    
    # Actually, let me try to construct rank-45 more carefully.
    # The key insight: for 3x3x2, the rank is 15. Let me construct it explicitly.
    
    # For 3x3x2, I'll use the following rank-15 construction:
    # 7 terms from Strassen on the 2x2x2 sub-block (i,k in {0,1}, j in {0,1})
    # 8 terms for the remaining border, with 2 terms combined
    
    # Let me try: instead of separate terms for i=2,k=0 and i=2,k=1 with same j,
    # I can combine them if they share the same B factor pattern.
    
    # Actually, the cleanest way: let me just verify the 17-term construction works
    # and use rank 51. That's still better than 54.
    
    # Let me build rank-51 (3 blocks of 17):
    U_51 = [[0] * 51 for _ in range(9)]
    V_51 = [[0] * 51 for _ in range(18)]
    W_51 = [[0] * 51 for _ in range(18)]
    
    col = 0
    for block in range(3):
        j_offset = block * 2
        # Strassen 7 terms
        strassen = [
            ({0:1, 4:1}, {0:1, 2:1}, {0:1, 1:1, 3:1, 4:1}),
            ({3:1, 4:1}, {1:1}, {1:1, 4:1}),
            ({0:1}, {1:1, 3:-1}, {1:1, 4:-1}),
            ({4:1}, {2:1, 0:-1}, {0:1, 3:-1}),
            ({0:1, 1:1}, {3:1}, {0:-1, 3:1}),
            ({3:1, 0:-1}, {0:1, 2:1}, {1:1, 3:1}),
            ({1:1, 4:-1}, {1:1, 3:1}, {0:1, 4:-1}),
        ]
        for u_dict, v_dict, w_dict in strassen:
            for u_idx, u_val in u_dict.items():
                U_51[u_idx][col] += u_val
            for v_idx, v_val in v_dict.items():
                V_51[v_idx + j_offset][col] += v_val
            for w_idx, w_val in w_dict.items():
                # w_idx is j*3+i where j in {0,1} relative to block
                # global j = j_offset + j_local
                j_local, i_local = divmod(w_idx, 3)
                global_j = j_offset + j_local
                W_51[global_j * 3 + i_local][col] += w_val
            col += 1
        
        # Border 10 terms
        border = [
            ({6:1}, {0:1}, {2:1}),
            ({6:1}, {1:1}, {5:1}),
            ({7:1}, {2:1}, {2:1}),
            ({7:1}, {3:1}, {5:1}),
            ({8:1}, {4:1}, {2:1}),
            ({8:1}, {5:1}, {5:1}),
            ({2:1}, {4:1}, {0:1}),
            ({2:1}, {5:1}, {1:1}),
            ({5:1}, {4:1}, {3:1}),
            ({5:1}, {5:1}, {4:1}),
        ]
        for u_dict, v_dict, w_dict in border:
            for u_idx, u_val in u_dict.items():
                U_51[u_idx][col] += u_val
            for v_idx, v_val in v_dict.items():
                V_51[v_idx + j_offset][col] += v_val
            for w_idx, w_val in w_dict.items():
                j_local, i_local = divmod(w_idx, 3)
                global_j = j_offset + j_local
                W_51[global_j * 3 + i_local][col] += w_val
            col += 1
    
    # Verify rank-51 decomposition
    def verify(U, V, W, n, m, p):
        """Verify that the decomposition correctly computes C = A*B"""
        # Build the tensor from U, V, W
        # T[i,j,k] should equal sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
        # Wait, the decomposition is: C[i,j] = sum_k A[i,k] B[k,j]
        # = sum_r (sum_k U[i*m+k,r] A[i,k]) (sum_k V[k*p+j,r] B[k,j]) ... no
        # Actually: T[i,j,k] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
        # And the product C[i,j] = sum_k T[i,j,k] A[i,k] B[k,j]
        # So we need: for all i,j: C[i,j] = sum_k A[i,k] B[k,j] = sum_r W[j*n+i,r] * (sum_k U[i*m+k,r]*A[i,k]) * (sum_k V[k*p+j,r]*B[k,j])
        # Hmm, that's not quite right either.
        
        # The tensor decomposition means:
        # T[i,j,k] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
        # And the matrix product is: C[i,j] = sum_k A[i,k] * B[k,j]
        # The bilinear form: C[i,j] = sum_{i',j',k'} T[i',j',k'] * A[i',k'] * B[k',j']
        # So we need T[i,j,k] = delta[i,j,k] (the multiplication tensor)
        
        # Let me verify by checking that sum_r U[i*m+k,r]*V[k*p+j,r]*W[j*n+i,r] = 1 if this is a valid multiplication term, else 0
        # Actually for matrix multiplication, T[i,j,k] = 1 for all i,j,k (each term A[i,k]*B[k,j] appears exactly once)
        # Wait no. The tensor T[i,j,k] represents the coefficient of A[i,k]*B[k,j] in C[i,j].
        # For standard matrix multiplication, each product A[i,k]*B[k,j] contributes to C[i,j] with coefficient 1.
        # So T[i,j,k] = 1 for all i,j,k.
        
        # The rank-r decomposition: T[i,j,k] = sum_{r=0}^{r-1} U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
        # We need this to equal 1 for all i,j,k.
        
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    total = 0
                    for r in range(len(U[0])):
                        total += U[i*m+k][r] * V[k*p+j][r] * W[j*n+i][r]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    if verify(U_51, V_51, W_51, n, m, p):
        return {"U": U_51, "V": V_51, "W": W_51}
    
    # If verification fails, return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
