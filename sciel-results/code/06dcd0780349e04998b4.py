import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal rank
        U_list.append([0] * rank + [coeff])
        V_list.append([0] * rank + [coeff])
        W_list.append([0] * rank + [coeff])
        rank += 1

    # This helper is not quite right for building matrices directly.
    # We should build the matrices in numpy arrays and then convert.
    
    # Let's use a list of tuples (u_idx, v_idx, w_idx, coeff)
    terms = []
    
    def add_term_simple(u_idx, v_idx, w_idx, coeff=1):
        terms.append((u_idx, v_idx, w_idx, coeff))

    # Indices:
    # A is 2x5. Indices i (0,1), j (0..4). Flat: i*5+j.
    # B is 5x5. Indices j (0..4), k (0..4). Flat: j*5+k.
    # C is 2x5. Indices i (0,1), k (0..4). Flat: k*2+i (Trace convention: W[k*n+i]).
    
    # Helper for 2x2x2 Strassen
    # A indices: (i, j) where i in {0,1}, j in {j0, j0+1}
    # B indices: (j, k) where j in {j0, j0+1}, k in {k0, k0+1}
    # C indices: (i, k) where i in {0,1}, k in {k0, k0+1}
    def strassen_222(j0, k0):
        # A elements:
        # a = A[0, j0], b = A[0, j0+1]
        # c = A[1, j0], d = A[1, j0+1]
        # B elements:
        # e = B[j0, k0], f = B[j0, k0+1]
        # g = B[j0+1, k0], h = B[j0+1, k0+1]
        
        # M1 = a(e+h)
        # Term 1: A[0,j0] * (B[j0,k0] + B[j0+1,k0+1])
        # This is not a single rank-1 term in the global tensor if we expand.
        # We need to decompose M1 into rank-1 terms of the global tensor.
        # M1 contributes to C[0, k0], C[0, k0+1], C[1, k0], C[1, k0+1]?
        # No, Strassen computes the 2x2 product.
        # The output C is 2x2.
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M6
        
        # We need to add terms to the global U, V, W.
        # Each "M" is a scalar multiplication of a linear combination of A and a linear combination of B.
        # This results in multiple rank-1 terms in the global decomposition?
        # No, the decomposition of the tensor M_{2,2,2} into 7 rank-1 tensors means:
        # T = sum_{r=1}^7 u_r \otimes v_r \otimes w_r.
        # We need to find u_r, v_r, w_r for the 2x2x2 block and map them to global indices.
        
        # Strassen's 7 terms:
        # 1. u: (1, 0, 0, 1) on A? No.
        # u1 = (1, 0, 0, 0) ? No.
        # Let's use the standard Strassen coefficients.
        # A is 2x2. B is 2x2.
        # M1 = A[0,0] * (B[0,0] + B[1,1])
        # This corresponds to u1 = (1, 0, 0, 0) (for A[0,0]), v1 = (1, 0, 0, 1) (for B[0,0]+B[1,1]).
        # Wait, u and v must be vectors.
        # u1 = (1, 0, 0, 0). v1 = (1, 0, 0, 1).
        # w1 must be such that sum w1_r u1_r v1_r = C?
        # No, the tensor is T = sum u_r \otimes v_r \otimes w_r.
        # The product is sum (u_r . A) (v_r . B) w_r.
        # So for M1:
        # u1 = (1, 0, 0, 0) -> A[0,0]
        # v1 = (1, 0, 0, 1) -> B[0,0] + B[1,1]
        # w1 = ?
        # M1 contributes to C[0,0] and C[1,1]?
        # C[0,0] gets M1. C[1,1] gets M1.
        # So w1 should have 1 at C[0,0] and 1 at C[1,1].
        # w1 = (1, 0, 0, 1) (indices 0,1,2,3 for C[0,0], C[0,1], C[1,0], C[1,1]).
        
        # Let's list the 7 terms (u, v, w) for 2x2x2.
        # Indices for A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Indices for B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Indices for C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # 1. M1 = A00 (B00+B11)
        # u=(1,0,0,0), v=(1,0,0,1), w=(1,0,0,1)
        
        # 2. M2 = (A10+A11) B00
        # u=(0,0,1,1), v=(1,0,0,0), w=(0,0,1,0)
        
        # 3. M3 = (A00+A01) (B11-B10)
        # u=(1,1,0,0), v=(0,0,-1,1), w=(0,1,0,0)
        
        # 4. M4 = A11 (B10+B01)
        # u=(0,0,0,1), v=(0,1,1,0), w=(0,0,1,1)
        
        # 5. M5 = (A00-A11) (B00+B01)
        # u=(1,0,0,-1), v=(1,1,0,0), w=(1,-1,0,0)
        
        # 6. M6 = (A01-A10) (B00+B10)
        # u=(0,1,-1,0), v=(1,0,1,0), w=(0,0,0,1)
        
        # 7. M7 = (A10+A11) (B01-B11)
        # u=(0,0,1,1), v=(0,1,0,-1), w=(1,0,0,-1)
        
        # Now map to global indices.
        # j0, j1 = j0, j0+1
        # k0, k1 = k0, k0+1
        
        # A indices:
        # 0 -> i=0, j=j0 -> 0*5+j0
        # 1 -> i=0, j=j0+1 -> 0*5+j0+1
        # 2 -> i=1, j=j0 -> 1*5+j0
        # 3 -> i=1, j=j0+1 -> 1*5+j0+1
        
        # B indices:
        # 0 -> j=j0, k=k0 -> j0*5+k0
        # 1 -> j=j0, k=k0+1 -> j0*5+k0+1
        # 2 -> j=j0+1, k=k0 -> (j0+1)*5+k0
        # 3 -> j=j0+1, k=k0+1 -> (j0+1)*5+k0+1
        
        # C indices (Trace: k*n+i):
        # 0 -> i=0, k=k0 -> k0*2+0
        # 1 -> i=0, k=k0+1 -> (k0+1)*2+0
        # 2 -> i=1, k=k0 -> k0*2+1
        # 3 -> i=1, k=k0+1 -> (k0+1)*2+1
        
        local_u = [(1,0,0,0), (0,0,1,1), (1,1,0,0), (0,0,0,1), (1,0,0,-1), (0,1,-1,0), (0,0,1,1)]
        local_v = [(1,0,0,1), (1,0,0,0), (0,0,-1,1), (0,1,1,0), (1,1,0,0), (1,0,1,0), (0,1,0,-1)]
        local_w = [(1,0,0,1), (0,0,1,0), (0,1,0,0), (0,0,1,1), (1,-1,0,0), (0,0,0,1), (1,0,0,-1)]
        
        a_idx = [0*5+j0, 0*5+j0+1, 1*5+j0, 1*5+j0+1]
        b_idx = [j0*5+k0, j0*5+k0+1, (j0+1)*5+k0, (j0+1)*5+k0+1]
        c_idx = [k0*2+0, (k0+1)*2+0, k0*2+1, (k0+1)*2+1]
        
        for r in range(7):
            for la in range(4):
                if local_u[r][la] != 0:
                    add_term_simple(a_idx[la], 0, 0, 0) # Placeholder, need to add to global
            # This approach is messy. Let's just add the terms directly to the global list.
            pass

    # Better approach: Build U, V, W as lists of dicts or sparse structures, then densify.
    
    U_sparse = {}
    V_sparse = {}
    W_sparse = {}
    current_rank = 0
    
    def add_rank1(u_idx, v_idx, w_idx, coeff=1):
        nonlocal current_rank
        r = current_rank
        current_rank += 1
        if u_idx is not None:
            U_sparse.setdefault(u_idx, []).append((r, coeff))
        if v_idx is not None:
            V_sparse.setdefault(v_idx, []).append((r, coeff))
        if w_idx is not None:
            W_sparse.setdefault(w_idx, []).append((r, coeff))

    # Helper to add a linear combination term
    # u_vec: list of (global_idx, coeff)
    # v_vec: list of (global_idx, coeff)
    # w_vec: list of (global_idx, coeff)
    # This adds one rank-1 term to the decomposition?
    # No, a rank-1 term is a single u_idx, v_idx, w_idx.
    # Strassen terms involve sums in u or v.
    # e.g., u = (1,0,0,1) means u has non-zeros at two indices.
    # This is NOT a rank-1 term in the tensor sense if u is a vector with 2 non-zeros.
    # Wait. The tensor decomposition is T = sum_r u_r \otimes v_r \otimes w_r.
    # u_r is a vector.
    # So u_r can have multiple non-zeros.
    # So we just need to add the vector u_r to U[:, r].
    
    def add_term_vecs(u_vec, v_vec, w_vec):
        nonlocal current_rank
        r = current_rank
        current_rank += 1
        for idx, coeff in u_vec:
            if coeff != 0:
                U_sparse.setdefault(idx, []).append((r, coeff))
        for idx, coeff in v_vec:
            if coeff != 0:
                V_sparse.setdefault(idx, []).append((r, coeff))
        for idx, coeff in w_vec:
            if coeff != 0:
                W_sparse.setdefault(idx, []).append((r, coeff))

    # Strassen 2x2x2
    def strassen_222(j0, k0):
        a_idx = [0*5+j0, 0*5+j0+1, 1*5+j0, 1*5+j0+1]
        b_idx = [j0*5+k0, j0*5+k0+1, (j0+1)*5+k0, (j0+1)*5+k0+1]
        c_idx = [k0*2+0, (k0+1)*2+0, k0*2+1, (k0+1)*2+1]
        
        local_u = [(1,0,0,0), (0,0,1,1), (1,1,0,0), (0,0,0,1), (1,0,0,-1), (0,1,-1,0), (0,0,1,1)]
        local_v = [(1,0,0,1), (1,0,0,0), (0,0,-1,1), (0,1,1,0), (1,1,0,0), (1,0,1,0), (0,1,0,-1)]
        local_w = [(1,0,0,1), (0,0,1,0), (0,1,0,0), (0,0,1,1), (1,-1,0,0), (0,0,0,1), (1,0,0,-1)]
        
        for r in range(7):
            u_vec = [(a_idx[i], local_u[r][i]) for i in range(4)]
            v_vec = [(b_idx[i], local_v[r][i]) for i in range(4)]
            w_vec = [(c_idx[i], local_w[r][i]) for i in range(4)]
            add_term_vecs(u_vec, v_vec, w_vec)

    # Trivial 2x2x1
    # A is 2x2, B is 2x1.
    # C is 2x1.
    # C[i,0] = sum_j A[i,j] B[j,0]
    # Rank 4.
    def trivial_221(j0, k0):
        # A indices: i=0,1; j=j0, j0+1
        # B indices: j=j0, j0+1; k=k0
        # C indices: i=0,1; k=k0
        a_idx = [0*5+j0, 0*5+j0+1, 1*5+j0, 1*5+j0+1]
        b_idx = [j0*5+k0, (j0+1)*5+k0]
        c_idx = [k0*2+0, k0*2+1]
        
        # Terms:
        # 1. A[0,j0] B[j0,k0] -> C[0,k0]
        add_term_vecs([(a_idx[0], 1)], [(b_idx[0], 1)], [(c_idx[0], 1)])
        # 2. A[0,j0+1] B[j0+1,k0] -> C[0,k0]
        add_term_vecs([(a_idx[1], 1)], [(b_idx[1], 1)], [(c_idx[0], 1)])
        # 3. A[1,j0] B[j0,k0] -> C[1,k0]
        add_term_vecs([(a_idx[2], 1)], [(b_idx[0], 1)], [(c_idx[1], 1)])
        # 4. A[1,j0+1] B[j0+1,k0] -> C[1,k0]
        add_term_vecs([(a_idx[3], 1)], [(b_idx[1], 1)], [(c_idx[1], 1)])

    # Trivial 2x1x5
    # A is 2x1, B is 1x5.
    # C is 2x5.
    # Rank 10.
    def trivial_215(j0, k0):
        # A indices: i=0,1; j=j0
        # B indices: j=j0; k=k0..k0+4
        # C indices: i=0,1; k=k0..k0+4
        a_idx = [0*5+j0, 1*5+j0]
        b_idx = [j0*5+k for k in range(k0, k0+5)]
        c_idx = [k*2+i for k in range(k0, k0+5) for i in [0,1]]
        
        # For each i, k:
        # C[i,k] = A[i,j0] B[j0,k]
        # Term: u=(A[i,j0]), v=(B[j0,k]), w=(C[i,k])
        for i in range(2):
            for kk in range(5):
                k_global = k0 + kk
                add_term_vecs([(a_idx[i], 1)], [(b_idx[kk], 1)], [(c_idx[i*5+kk], 1)])

    # Main decomposition
    # J1 = {0,1}, J2 = {2,3}, J3 = {4}
    
    # J1: 2x2x5
    # Split B into 2x2 and 2x3
    # 2x2 part: j in {0,1}, k in {0,1} -> Strassen
    strassen_222(0, 0)
    
    # 2x3 part: j in {0,1}, k in {2,3,4}
    # Split B into 2x1 (k=2) and 2x2 (k=3,4)
    # 2x1 part: j in {0,1}, k=2 -> Trivial 221
    trivial_221(0, 2)
    
    # 2x2 part: j in {0,1}, k in {3,4} -> Strassen
    strassen_222(0, 3)
    
    # J2: 2x2x5
    # Split B into 2x2 and 2x3
    # 2x2 part: j in {2,3}, k in {0,1} -> Strassen
    strassen_222(2, 0)
    
    # 2x3 part: j in {2,3}, k in {2,3,4}
    # Split B into 2x1 (k=2) and 2x2 (k=3,4)
    # 2x1 part: j in {2,3}, k=2 -> Trivial 221
    trivial_221(2, 2)
    
    # 2x2 part: j in {2,3}, k in {3,4} -> Strassen
    strassen_222(2, 3)
    
    # J3: 2x1x5
    # j=4, k in {0..4}
    trivial_215(4, 0)
    
    # Total rank: 7 + 4 + 7 + 7 + 4 + 7 + 10 = 46.
    
    # Convert sparse to dense
    U = np.zeros((n*m, current_rank), dtype=np.int64)
    V = np.zeros((m*p, current_rank), dtype=np.int64)
    W = np.zeros((p*n, current_rank), dtype=np.int64)
    
    for idx, terms in U_sparse.items():
        for r, coeff in terms:
            U[idx, r] += coeff
            
    for idx, terms in V_sparse.items():
        for r, coeff in terms:
            V[idx, r] += coeff
            
    for idx, terms in W_sparse.items():
        for r, coeff in terms:
            W[idx, r] += coeff
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

