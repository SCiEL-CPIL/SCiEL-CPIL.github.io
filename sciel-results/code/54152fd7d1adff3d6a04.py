import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = 145

    n, m, p = DIMENSIONS

    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Field-theoretic decomposition using F_13 structure
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Map column k to field element alpha_k in F_13
    # B[j,k] = B_j(alpha_k) where B_j is polynomial of degree < 13
    # C[i,k] = sum_j A[i,j] * B_j(alpha_k) = P_i(alpha_k) where P_i is polynomial of degree < 13
    # 
    # Rank-1 terms: for each degree d (0..12), we need terms that capture
    # the coefficient of x^d in P_i, which is sum_j A[i,j] * B_j[d]
    # 
    # This gives n * (p) = 2*13 = 26 terms if we can separate A[i,j]*B_j[d]
    # But we need rank-1: U[i,j,t] * V[j,k,t] * W[k,i,t]
    # 
    # Let's use: for each d in 0..12, for each i in 0..1:
    #   U[i*m+j, t] = A[i,j] (but A is input, not fixed!)
    # 
    # This doesn't work directly because A is variable. Let me reconsider.
    # 
    # The tensor is T[i,j,k] = delta_{i,k} (no, that's wrong)
    # Actually T[i,j,k] contributes to C[i,k] via A[i,j]*B[j,k]
    # The multiplication tensor has entries M[i,j,k; i',j',k'] ... 
    # 
    # Let me use the standard: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor: T[i,j,k] = 1 (the "structure" tensor)
    # Rank-1 decomposition: T[i,j,k] = sum_t U[i,j,t] V[j,k,t] W[k,i,t]... 
    # Wait, the convention is W[k*n+i, r] so W index is (k,i)
    # 
    # We need: for all i,j,k: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = delta_{i,k}? No.
    # 
    # The matrix multiplication tensor: C[i,k] = sum_j A[i,j] B[j,k]
    # This is bilinear. The tensor T has T[i,j,k] = 1 for the "identity" in the sense
    # that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k] where T[i,j,k]=1 always.
    # 
    # Wait no. C[i,k] = sum_j A[i,j] B[j,k]. So the "tensor" relating A, B to C is:
    # C[i,k] = sum_{j,t} (A[i,j]) (B[j,k]) * [coefficient]
    # The coefficient is 1 for each (i,j,k) triple. So T[i,j,k] = 1 for all i,j,k.
    # 
    # We need: sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t] = 1 for all i, j, k
    # 
    # This is a rank-1 decomposition of the all-ones tensor of size n*m x m*p x p*n
    # with the constraint that the third index is (k,i) not (i,k).
    # 
    # For the all-ones tensor, rank = max(n*m, m*p, p*n) in general? No, that's for 
    # matrix rank. For tensor rank of all-ones tensor of size a x b x c, it's 
    # related to the border rank.
    # 
    # Actually, let me just try: can we decompose T[i,j,k]=1 with fewer terms?
    # 
    # Using the field approach: pick distinct alpha_0, ..., alpha_{12} in F_13.
    # For each k, define w_k = alpha_k.
    # We want sum_t U[i,j,t] V[j,k,t] W[k,i,t] = 1.
    # 
    # Try: let there be p terms (t=0..12). For term t, let:
    #   V[j,k,t] = 1 if k==t else 0 (delta)
    #   W[k,i,t] = 1 (constant)  
    #   U[i,j,t] = ?
    # Then sum_t U[i,j,t] * delta(k,t) * 1 = U[i,j,k] = 1 for all i,j,k.
    # That's just p = 13 terms! But wait, that gives rank 13.
    # 
    # Let me verify: U[i*m+j, t] = 1 for all i,j,t. V[j*p+k, t] = 1 if k==t else 0.
    # W[k*n+i, t] = 1 for all k,i,t.
    # Then sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = sum_t 1 * delta(k,t) * 1 = 1. YES!
    # 
    # So rank 13 works! Let me verify this is valid.
    
    # Rank 13 decomposition
    total_rank = p  # 13
    U = [[1] * total_rank for _ in range(n * m)]   # 4 rows, all ones
    V = [[0] * total_rank for _ in range(m * p)]   # 26 rows
    W = [[1] * total_rank for _ in range(p * n)]   # 26 rows, all ones
    
    for j in range(m):
        for k in range(p):
            V[j * p + k][k] = 1
    
    # Verify numerically
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test

    # C[i,k] = sum_j A[i,j] B[j,k]
    # Our decomposition: C[i,k] = sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t] summed over j
    # = sum_j sum_t A[i,j] * U[i*m+j, t] * B[j,k] * V[j*p+k, t] * W[k*n+i, t]
    # Wait, I need to be more careful. The decomposition is:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_t (sum_j A[i,j] U[i*m+j,t]) (sum_k' B[j,k'] V[j*p+k',t]) ... no
    # 
    # The tensor decomposition means: the bilinear form C[i,k] = sum_j A[i,j] B[j,k]
    # can be written as C[i,k] = sum_t (sum_j A[i,j] U[i*m+j,t]) (sum_k B[j,k] V[j*p+k,t]) W[k*n+i,t]
    # No, that's not right either.
    # 
    # The standard tensor decomposition for matrix multiplication:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_t (sum_j A[i,j] U[i*m+j, t]) * (sum_k B[j,k] V[j*p+k, t]) ... no
    # 
    # Actually: C[i,k] = sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t] is NOT the right formula.
    # 
    # The correct formula is: C[i,k] = sum_j A[i,j] B[j,k] = sum_t (sum_j A[i,j] U[i*m+j,t]) (sum_k B[j,k] V[j*p+k,t]) ... 
    # 
    # No. The rank-r decomposition of the multiplication tensor means:
    # C[i,k] = sum_t (sum_j A[i,j] U[i*m+j, t]) * (sum_k B[j,k] V[j*p+k, t]) * W[k*n+i, t]
    # That doesn't make sense dimensionally.
    # 
    # Let me re-read the task: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "that exactly decompose the matrix multiplication tensor"
    # 
    # The matrix multiplication tensor T of size (n*m) x (m*p) x (p*n) has:
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if j==j' and i==i' and k==k', else 0? 
    # 
    # No. The tensor for matrix multiplication A(n,m) * B(m,p) = C(n,p):
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has entries T[a, b, c] where a indexes (i,j), b indexes (j',k), c indexes (k',i')
    # T[i*m+j, j'*p+k, k'*n+i'] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    # 
    # So T[a,b,c] = 1 iff the j-index of a equals the j'-index of b, 
    #             and the i-index of c equals the i-index of a,
    #             and the k-index of c equals the k-index of b.
    # 
    # A rank-1 term is U[a,t] * V[b,t] * W[c,t].
    # We need: sum_t U[i*m+j, t] V[j'*p+k, t] W[k'*n+i', t] = delta_{j,j'} delta_{i,i'} delta_{k,k'}
    # 
    # So for the all-ones approach: U[i*m+j, t] = 1, V[j*p+k, t] = delta_{k,t}, W[k*n+i, t] = 1
    # LHS = sum_t 1 * delta_{k,t} * 1 = delta_{k,t} summed over t... = 1 if there exists t=k, which is always 1.
    # But we need delta_{j,j'} delta_{i,i'} delta_{k,k'}, which is 0 unless j=j', i=i', k=k'.
    # 
    # So my rank-13 attempt is WRONG. It gives 1 for all (i,j,k',i') not just the diagonal.
    # 
    # Let me fix: I need the decomposition to satisfy:
    # sum_t U[i*m+j, t] V[j'*p+k, t] W[k'*n+i', t] = delta_{j,j'} delta_{i,i'} delta_{k,k'}
    # 
    # This is the identity tensor. The rank of this tensor is what we're looking for.
    # 
    # For delta_{j,j'}: we can use V[j*p+k, t] with t encoding j. 
    # For delta_{i,i'}: W[k*n+i, t] with t encoding i.
    # For delta_{k,k'}: V[j*p+k, t] with t encoding k.
    # 
    # So t needs to encode (j, i, k) which is n*m*p = 52 values. That's schoolbook.
    # 
    # But we can do better. Let's think about it as:
    # T[a, b, c] = delta_{j(a), j(b)} * delta_{i(c), i(a)} * delta_{k(c), k(b)}
    # 
    # where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # 
    # This tensor factors as a product of three Kronecker deltas. The tensor rank of a product 
    # of Kronecker deltas... 
    # 
    # Let me try a different approach. Use the fact that:
    # delta_{j,j'} can be written using a basis: sum_t phi_j(t) phi_{j'}(t) for orthonormal phi
    # Similarly for the others.
    # 
    # If we use t to range over m values for j, n values for i, and p values for k, 
    # we get m*n*p = 52 terms (schoolbook).
    # 
    # But we can combine: let t range over some set, and use:
    # U[i*m+j, t] = f_j(t)
    # V[j'*p+k, t] = g_{j',k}(t)  
    # W[k'*n+i', t] = h_{i',k'}(t)
    # 
    # Need: sum_t f_j(t) g_{j',k}(t) h_{i',k'}(t) = delta_{j,j'} delta_{i,i'} delta_{k,k'}
    # 
    # This is hard to satisfy with fewer than 52 terms in general.
    # 
    # Let me go back to the Strassen approach which gives rank 46, and verify it's correct.
    
    # Strassen for 2x2 blocks + Schoolbook for remainder
    # p=13 -> 6 blocks of 2 cols (12 cols) + 1 col of 1
    # Rank: 6 * 7 + 4 = 46
    
    total_rank = 46
    U = [[0] * total_rank for _ in range(n * m)]   # 4 rows
    V = [[0] * total_rank for _ in range(m * p)]   # 26 rows
    W = [[0] * total_rank for _ in range(p * n)]   # 26 rows

    col_idx = 0
    
    # Helper to add Strassen terms for a 2x2 block at columns c0, c1
    def add_strassen_block(c0, c1, start_col):
        # For multiplying A (2x2) by B_block (2x2), C_block (2x2) = A * B_block
        # A = [[a, b], [c, d]] = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
        # B_block = [[e, f], [g, h]] = [[B[0,c0], B[0,c1]], [B[1,c0], B[1,c1]]]
        # C_block = [[i, j], [k, l]] = [[C[0,c0], C[0,c1]], [C[1,c0], C[1,c1]]]
        # 
        # Strassen:
        # M1 = (a+b)(e+f) = i + l  -> C[0,c0] += M1, C[1,c1] += M1
        # M2 = (c+d)e = k  -> C[1,c0] += M2
        # M3 = a(h-f) = j - i  -> C[0,c1] += M3, C[0,c0] -= M3 (so net C[0,c0] = M1 - M3 = i)
        # M4 = d(g-e) = l - k  -> C[1,c1] += M4, C[1,c0] -= M4 (so net C[1,c0] = M2 - M4 = k)
        # M5 = (a+d)(e+h) = j + k  -> C[0,c1] += M5, C[1,c0] += M5 (wait, let me recheck)
        # 
        # Standard Strassen:
        # M1 = (a+b)(e+f)
        # M2 = (c+d)(e)
        # M3 = (a)(h-f)
        # M4 = (d)(g-e)
        # M5 = (a+d)(e+h)
        # M6 = (b+c)(g-e)
        # M7 = (b+d)(f-h)
        # 
        # C[0,0] = M1 - M3 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M4 - M5 + M6
        # 
        # Let me verify:
        # C[0,0] = ae + bg
        # M1 - M3 + M7 = (a+b)(e+f) - a(h-f) + (b+d)(f-h)
        # = ae + af + be + bf - ah + af + bf - bh + df - dh
        # = ae + 2af + be + 2bf - ah - bh + df - dh
        # That doesn't look right. Let me use the standard formula.
        # 
        # Standard Strassen for 2x2:
        # p1 = (a + b) * (e + f)
        # p2 = (c + d) * e
        # p3 = a * (h - f)
        # p4 = d * (g - e)
        # p5 = (a + d) * (e + h)
        # p6 = (b + c) * (g - e)
        # p7 = (b + d) * (f - h)
        # 
        # C[0,0] = p1 + p4 - p5 + p7
        # C[0,1] = p3 + p5
        # C[1,0] = p2 + p4
        # C[1,1] = p1 - p3 + p6 - p7
        # 
        # Verify C[0,0] = ae + bg:
        # p1 + p4 - p5 + p7 = (a+b)(e+f) + d(g-e) - (a+d)(e+h) + (b+d)(f-h)
        # = ae + af + be + bf + dg - de - ae - ah - de - dh + bf - bh + df - dh
        # = (ae - ae) + af + be + 2bf + dg - 2de - ah - 2dh + df - bh
        # Hmm, this is getting messy. Let me just use the known-correct Strassen formula.
        # 
        # Actually, let me just verify numerically with the current code and fix if needed.
        
        # Using the formula from the original code, let me verify it's correct:
        # Term 1: (a+b)(e+f) -> contributes to C[0,c0] and C[1,c1]
        #   U[0] += 1 (a), U[1] += 1 (b)
        #   V[c0] += 1 (e), V[c1] += 1 (f)  
        #   W[2*c0] += 1 (C[0,c0]), W[2*c1+1] += 1 (C[1,c1])
        # This gives: (a+b)(e+f) * C[0,c0] and (a+b)(e+f) * C[1,c1]
        # But we need it to contribute to both C[0,c0] AND C[1,c1] with the SAME coefficient.
        # In the tensor decomposition, each term t contributes U[i*m+j,t]*V[j'*p+k,t]*W[k'*n+i',t]
        # to T[i*m+j, j'*p+k, k'*n+i'].
        # 
        # For this term: U[0,t]=1, U[1,t]=1, V[c0,t]=1, V[c1,t]=1, W[2*c0,t]=1, W[2*c1+1,t]=1
        # This means: T[0, c0, 2*c0] += 1, T[0, c1, 2*c0] += 1, T[1, c0, 2*c0] += 1, T[1, c1, 2*c0] += 1
        #             T[0, c0, 2*c1+1] += 1, T[0, c1, 2*c1+1] += 1, T[1, c0, 2*c1+1] += 1, T[1, c1, 2*c1+1] += 1
        # 
        # But we need T[i*m+j, j'*p+k, k'*n+i'] = delta_{j,j'} delta_{i,i'} delta_{k,k'}
        # So T[0, c0, 2*c0] should be 1 (i=0,j=0, j'=0,k=c0, k'=c0,i'=0: all match) ✓
        # T[0, c1, 2*c0] should be 0 (i=0,j=0, j'=0,k=c1, k'=c0,i'=0: k≠k') ✗
        # 
        # So the current decomposition is WRONG! It's adding terms where k≠k'.
        # 
        # I think the issue is that the W matrix should have W[k*n+i, t] where the term contributes
        # to C[i,k] = sum_j A[i,j] B[j,k]. The tensor entry is T[i*m+j, j'*p+k, k'*n+i'] 
        # and it should be 1 only when j=j', i=i', k=k'.
        # 
        # For Strassen, the trick is that we're computing C = A*B using 7 multiplications,
        # but the decomposition of the TENSOR is different.
        # 
        # Let me reconsider. The bilinear map is: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k].
        # A rank-1 term in the tensor decomposition corresponds to:
        # C[i,k] += (sum_j A[i,j] U[i*m+j,t]) * (sum_k B[j,k] V[j*p+k,t]) * W[k*n+i,t]
        # No, that's not right either.
        # 
        # I think the correct interpretation is:
        # C[i,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] * A[i,j] * B[j,k]
        # No, A and B are the inputs.
        # 
        # Let me look at this differently. The tensor T has:
        # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i (i.e., the indices are consistent)
        # 
        # A rank-r decomposition: T[a,b,c] = sum_{t=1}^r U[a,t] V[b,t] W[c,t]
        # 
        # So we need: sum_t U[i*m+j, t] V[j'*p+k, t] W[k'*n+i', t] = 1 if j=j', i=i', k=k' else 0
        # 
        # For the Strassen approach to work, we need to be decomposing this specific tensor.
        # The current code seems to be doing something different (computing C = A*B directly
        # rather than decomposing the tensor).
        # 
        # Let me just do a clean schoolbook and verify, then try to improve.
        
        pass  # Disable the broken Strassen for now

    # Clean schoolbook decomposition
    # T[i*m+j, j'*p+k, k'*n+i'] = delta_{j,j'} delta_{i,i'} delta_{k,k'}
    # Schoolbook: one term for each (i,j,k) triple
    total_rank = n * m * p  # 52
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Verify
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check: sum_t U[i*m+j, t] V[j'*p+k, t] W[k'*n+i', t] should equal
    # delta_{j,j'} delta_{i,i'} delta_{k,k'}
    error = 0.0
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            expected = 1.0 if (j == jp and i == ip and k == kp) else 0.0
                            actual = np.sum(U_np[i*m+j, :] * V_np[jp*p+k, :] * W_np[kp*n+ip, :])
                            error += abs(actual - expected)
    
    if error < 1e-10:
        best_result = {"U": U, "V": V, "W": W}
        best_rank = total_rank

    if time.time() - start_time > deadline - 5:
        return best_result

    return best_result
