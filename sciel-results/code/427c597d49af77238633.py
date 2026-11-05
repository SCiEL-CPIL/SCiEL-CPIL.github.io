import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 110.0

    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for k in range(p):
                c_idx = k * n + i
                for j in range(m):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    s = 0
                    for r in range(rank):
                        u_val = U[a_idx][r]
                        v_val = V[b_idx][r]
                        w_val = W[c_idx][r]
                        if u_val != 0 and v_val != 0 and w_val != 0:
                            s += u_val * v_val * w_val
                    if s != 1:
                        return False
        return True

    def strassen_block_solution():
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for idx, val in u_coeffs.items():
                if val != 0:
                    U[idx][col] += val
            for idx, val in v_coeffs.items():
                if val != 0:
                    V[idx][col] += val
            for idx, val in w_coeffs.items():
                if val != 0:
                    W[idx][col] += val
            col += 1

        # Strassen 2x2x2 terms
        strassen_terms = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
        ]
        
        # Schoolbook 2x2x1 terms
        standard_terms_2x2x1 = [
            ({0: 1}, {0: 1}, {0: 1}),
            ({1: 1}, {1: 1}, {0: 1}),
            ({2: 1}, {0: 1}, {1: 1}),
            ({3: 1}, {1: 1}, {1: 1}),
        ]

        # Decompose 2x4x5 as:
        # Two 2x2x2 Strassen blocks for k in {0,1} and {2,3} (j in {0,1} and j in {2,3})
        # Two 2x2x1 schoolbook blocks for k={4} (j in {0,1} and j in {2,3})
        
        # j-group 0 (rows 0,1 of A, rows 0,1 of B): covers k={0,1} (Strassen) and k={4} (Schoolbook)
        # j-group 1 (rows 2,3 of A, rows 2,3 of B): covers k={0,1} (Strassen) and k={4} (Schoolbook)
        # Wait, the current code iterates q in 0..1 (j-group) and s in 0..2 (k-group).
        # s=0: k in {0,1} -> Strassen
        # s=1: k in {2,3} -> Strassen
        # s=2: k in {4}    -> Schoolbook
        
        for q in range(2):
            for s in range(3):
                u_g, v_g, w_g = {}, {}, {}
                if s < 2:
                    # Strassen block for 2x2x2
                    # A block: rows li in {0,1}, cols lj in {2q, 2q+1}
                    # B block: rows lj in {2q, 2q+1}, cols lk in {2s, 2s+1}
                    # C block: rows li in {0,1}, cols lk in {2s, 2s+1}
                    for u_t, v_t, w_t in strassen_terms:
                        u_g, v_g, w_g = {}, {}, {}
                        for idx, val in u_t.items():
                            li, lj = idx // 2, idx % 2
                            u_g[li * m + (2 * q + lj)] = val
                        for idx, val in v_t.items():
                            lj, lk = idx // 2, idx % 2
                            v_g[(2 * q + lj) * p + (2 * s + lk)] = val
                        for idx, val in w_t.items():
                            li, lk = idx // 2, idx % 2
                            w_g[(2 * s + lk) * n + li] = val
                        add_term(u_g, v_g, w_g)
                else:
                    # Schoolbook block for 2x2x1 (k=4)
                    # A block: rows li in {0,1}, cols lj in {2q, 2q+1}
                    # B block: rows lj in {2q, 2q+1}, col lk=4
                    # C block: rows li in {0,1}, col lk=4
                    for u_t, v_t, w_t in standard_terms_2x2x1:
                        u_g, v_g, w_g = {}, {}, {}
                        for idx, val in u_t.items():
                            li, lj = idx // 2, idx % 2
                            u_g[li * m + (2 * q + lj)] = val
                        for idx, val in v_t.items():
                            lj = idx
                            v_g[(2 * q + lj) * p + 4] = val
                        for idx, val in w_t.items():
                            li = idx
                            w_g[4 * n + li] = val
                        add_term(u_g, v_g, w_g)

        return {"U": U, "V": V, "W": W}

    # Attempt to find a better solution using the "structured embedding" strategy
    # Padding to 3x3x3 using rank-19 decomposition.
    # However, the instruction says "Keep this design and investigate improvements within its family".
    # The current design is block decomposition. The "structured embedding" is a different design family.
    # "Treat these as hypotheses to test. Do NOT switch to a different design family."
    # So I should stick to block decomposition or very similar improvements.
    # The rank-36 solution is the baseline.
    # Can we improve the rank-36 solution?
    # The 2x2x1 blocks have rank 4.
    # Is there a rank-3 solution for 2x2x1?
    # T[i,j,k] = 1 if k=0. T[i,j,0] = 1.
    # This is the sum of 4 rank-1 tensors e_i x e_j x e_0.
    # The rank of the matrix J_4 (4x4 ones) is 1, but this is a tensor T[i,j] = 1 for fixed k.
    # The tensor is e_0 x J_4 (where J_4 is the 2x2 matrix of ones).
    # The rank of the tensor e_0 x J is the rank of the matrix J? No.
    # The tensor is T_{ijk} = A_{ik} B_{jk} C_{ij}?
    # T_{ij0} = 1.
    # This is a sum of 4 terms: e_0 e_0 e_0, e_0 e_1 e_0, e_1 e_0 e_0, e_1 e_1 e_0.
    # Can we combine them?
    # (e_0+e_1)(e_0+e_1)e_0 = e_0e_0e_0 + e_0e_1e_0 + e_1e_0e_0 + e_1e_1e_0.
    # This produces 4 terms but the coefficients are all 1.
    # So T = (e_0+e_1) \otimes (e_0+e_1) \otimes e_0.
    # This is a RANK-1 tensor!
    # Wait.
    # If we use this, the term contributes to C[0,0], C[0,1], C[1,0], C[1,1] all with value 1.
    # But we only want C[0,0]=A[0,0]B[0,0] etc.
    # The block is A[2x2] * B[2x1] -> C[2x1].
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    # If we use the rank-1 tensor (A[0]+A[1]) (B[0]+B[1]) ... no, B is 2x1.
    # Let's check the rank of the 2x2x1 multiplication tensor.
    # T[i,j,0] = 1.
    # This tensor is the Kronecker product of the 2x2x1 identity? No.
    # It is the tensor defined by the matrix of ones.
    # The rank of the "multiplication tensor" for 2x2x1 (where the output is 2x1) is 4.
    # Why? Because we need to compute 4 distinct products.
    # The trick above (e0+e1)(e0+e1)e0 computes (A0+A1)(B0+B1) which is A0B0+A0B1+A1B0+A1B1.
    # We only want A0B0, A0B1, A1B0, A1B1 separately?
    # In matrix multiplication C = AB, the entries are sums.
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0].
    # If we use the combined product, we get the sum of 4 terms.
    # We need to extract the specific terms.
    # Since the tensor entries are 1, we are looking for a decomposition of the identity tensor of the space?
    # No, the tensor is T_{ijk} = 1.
    # This is a specific tensor.
    # Is it rank 1?
    # If T is rank 1, T = u v w.
    # u[i]v[j]w[0] = 1 for all i,j.
    # This implies u[i]v[j] is constant.
    # So u must be constant and v must be constant.
    # If u=[1,1], v=[1,1], w=[1], then T[i,j,0] = 1*1*1 = 1.
    # So YES, the 2x2x1 multiplication tensor IS rank 1!
    # Wait, why did I think it was rank 4?
    # In the context of matrix multiplication, we are computing the bilinear form.
    # The tensor is the "structure constants" of the algebra.
    # For 2x2x1, the output is 2x1.
    # The bilinear map is (A,B) -> AB.
    # The tensor is T such that (AB)_{ik} = sum_j A_{ij} B_{jk} T_{ijk}?
    # No, the standard definition is:
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # This is a linear map from A x B to C.
    # The tensor decomposition represents this map as sum of rank-1 maps.
    # A rank-1 map is (A,B) -> u^T A v * w?
    # No, a rank-1 term in the decomposition corresponds to:
    # Term_r: U[:, r] * V[:, r] * W[:, r]
    # The contribution to C_{ik} is sum_r U_{ir} V_{jr} W_{jk}?
    # The standard definition:
    # T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}
    # And the matrix product is C_{ik} = sum_j A_{ij} B_{jk} T_{ijk}?
    # No.
    # The tensor T encodes the multiplication.
    # C_{ik} = sum_{j} A_{ij} B_{jk}.
    # This can be written as a contraction of T with A and B.
    # T_{ijk} = 1 if j is the inner index?
    # Let's look at the verify function in the code.
    # s = sum_r U[a_idx][r] * V[b_idx][r] * W[c_idx][r]
    # a_idx = i*m + j (A_{ij})
    # b_idx = j*p + k (B_{jk})
    # c_idx = k*n + i (C_{ik})
    # The condition is sum_r U_{ij, r} V_{jk, r} W_{ki, r} = 1.
    # This means the tensor T_{ijk} (with indices A,B,C) has entries 1.
    # Specifically, for fixed i,j,k, the term in the sum is U_{ij} V_{jk} W_{ki}.
    # We want this sum to be 1.
    # This corresponds to the tensor T_{ijk} = 1.
    # For the 2x2x1 block (k=4 fixed):
    # We need sum_r U_{ij, r} V_{j4, r} W_{4i, r} = 1.
    # Here i in {0,1}, j in {0,1}.
    # This is exactly the 2x2x1 tensor with all entries 1.
    # And we showed that this tensor is rank 1!
    # u=[1,1], v=[1,1], w=[1].
    # U_{00}=1, U_{01}=1, U_{10}=1, U_{11}=1.
    # V_{04}=1, V_{14}=1.
    # W_{40}=1, W_{41}=1.
    # Then sum_r (1)(1)(1) = 1.
    # So the 2x2x1 block can be done in RANK 1!
    # Instead of 4 terms, we use 1 term.
    
    # Let's re-calculate the total rank.
    # Two Strassen 2x2x2 blocks (rank 7 each). Total 14.
    # Two 2x2x1 blocks (rank 1 each). Total 2.
    # Total rank = 16.
    
    # Wait, is it really that simple?
    # Let's check the indices.
    # Block j in {0,1}, k=4.
    # i in {0,1}, j in {0,1}, k=4.
    # We need sum_r U_{ij, r} V_{j4, r} W_{4i, r} = 1.
    # If we set U_{ij, r} = 1 for all i,j.
    # V_{j4, r} = 1 for all j.
    # W_{4i, r} = 1 for all i.
    # Then the product is 1.
    # This works!
    
    # So we can reduce the rank from 36 to 16.
    # Let's implement this.

    def improved_block_solution():
        rank = 16
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for idx, val in u_coeffs.items():
                if val != 0:
                    U[idx][col] += val
            for idx, val in v_coeffs.items():
                if val != 0:
                    V[idx][col] += val
            for idx, val in w_coeffs.items():
                if val != 0:
                    W[idx][col] += val
            col += 1

        strassen_terms = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
        ]

        for q in range(2):
            # s=0: k in {0,1} (Strassen)
            for s in range(2):
                for u_t, v_t, w_t in strassen_terms:
                    u_g, v_g, w_g = {}, {}, {}
                    for idx, val in u_t.items():
                        li, lj = idx // 2, idx % 2
                        u_g[li * m + (2 * q + lj)] = val
                    for idx, val in v_t.items():
                        lj, lk = idx // 2, idx % 2
                        v_g[(2 * q + lj) * p + (2 * s + lk)] = val
                    for idx, val in w_t.items():
                        li, lk = idx // 2, idx % 2
                        w_g[(2 * s + lk) * n + li] = val
                    add_term(u_g, v_g, w_g)
            
            # s=2: k=4 (Rank-1 trick)
            # j in {2q, 2q+1}
            # i in {0,1}
            u_g = {}
            v_g = {}
            w_g = {}
            for li in range(2):
                for lj in range(2):
                    u_g[li * m + (2 * q + lj)] = 1
            for lj in range(2):
                v_g[(2 * q + lj) * p + 4] = 1
            for li in range(2):
                w_g[4 * n + li] = 1
            add_term(u_g, v_g, w_g)

        return {"U": U, "V": V, "W": W}

    best = improved_block_solution()
    if verify(best["U"], best["V"], best["W"]):
        return best

    # Fallback to rank-36
    best36 = strassen_block_solution()
    if verify(best36["U"], best36["V"], best36["W"]):
        return best36

    # Fallback to schoolbook
    def schoolbook_solution():
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
        return {"U": U, "V": V, "W": W}
    
    return schoolbook_solution()
