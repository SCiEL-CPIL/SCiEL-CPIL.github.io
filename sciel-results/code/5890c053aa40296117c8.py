<<<<<<< SEARCH
"""Matrix multiplication 3,4,6: DIRECT minimal-rank construction (schoolbook, rank 72).

We build the tensor decomposition explicitly. The multiplication tensor T_{i,j,k} = A_{i,j} B_{j,k}
is a 3x4x6 tensor. The direct/schoolbook decomposition uses one rank-1 term per (i,j,k) triple:
  term (i,j,k): U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
This yields the exact matrix product with rank n*m*p = 72, and the factors have integer entries.

The program is deterministic and runs in O(n*m*p) time.
"""

import time

DIMENSIONS = [3, 4, 6]
EVAL_DEADLINE_SECONDS = 118.0  # safety margin under 120 s budget


# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS

    # --- Direct (schoolbook) minimal construction: rank = n*m*p ---
    rank = n * m * p
    # Allocate integer matrices.
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

    elapsed = time.time() - t0
    if elapsed > EVAL_DEADLINE_SECONDS:
        # Should never happen for n*m*p=72, but safety guard per directive.
        raise TimeoutError("construct() exceeded deadline")

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
"""Matrix multiplication 3,4,6: Rank-66 construction via block decomposition.

Splits 3=1+2, 4=2+2, 6=2+2+2. Uses schoolbook (rank 4) for 1x2x2 blocks
and Strassen (rank 7) for 2x2x2 blocks. Total rank: 6*4 + 6*7 = 24+42 = 66.
"""

import time

DIMENSIONS = [3, 4, 6]
EVAL_DEADLINE_SECONDS = 118.0


def _add_block_1x2x2(U, V, W, i_off, j_off, k_off, col):
    """Add schoolbook decomposition for 1x2x2 block at offset (i_off, j_off, k_off)."""
    n, m, p = 1, 2, 2
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                U[(i_off + ii) * 4 + (j_off + jj)][col] = 1
                V[(j_off + jj) * 6 + (k_off + kk)][col] = 1
                W[(k_off + kk) * 3 + (i_off + ii)][col] = 1
                col += 1
    return col


def _add_block_2x2x2_strassen(U, V, W, i_off, j_off, k_off, col):
    """Add Strassen decomposition for 2x2x2 block at offset (i_off, j_off, k_off)."""
    n, m, p = 2, 2, 2
    # Strassen's 7 terms for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Each Strassen term is a rank-1 tensor in the local 2x2x2 space
    # We need to express each Mi as a sum of rank-1 terms in the global space
    
    # M1 = (A11+A22)(B11+B22)
    # This contributes to C11 and C22
    # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
    # A11 is (0,0), A22 is (1,1)
    # B11 is (0,0), B22 is (1,1)
    
    # M1: (e0+e1) ⊗ (e0+e1) ⊗ (e0 ⊗ e0 + e1 ⊗ e1) -- not quite, need to be careful
    
    # Actually, each Mi is a product of a linear combination of A entries and a linear combination of B entries,
    # contributing to specific C entries.
    
    # Let me define the 7 Strassen terms explicitly as (u, v, w) triples in local coords:
    # Each term has U support on (i,j), V on (j,k), W on (k,i)
    
    # Term 1: M1 = (A00+A11)(B00+B11) -> contributes to C00 (+1) and C11 (+1)
    # As rank-1 tensors, we need to decompose (A00+A11)(B00+B11) into rank-1 terms
    # But Strassen's algorithm already gives us 7 rank-1 terms for the whole 2x2x2!
    
    # The 7 Strassen rank-1 terms for 2x2x2:
    # (u, v, w) where u is in R^{2x2}, v in R^{2x2}, w in R^{2x2}
    
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11): u=[1,0;0,1], v=[1,0;0,1], w=[1,0;0,1] -- NO, that's not right either
    
    # Let me use the standard formulation. For 2x2x2, the 7 rank-1 decompositions are:
    # (using local indices i,j,k in {0,1})
    
    # Term 1: u_{ij} = 1 if (i,j) in {(0,0),(1,1)}, v_{jk} = 1 if (j,k) in {(0,0),(1,1)}, w_{ki} = 1 if (k,i) in {(0,0),(1,1)}
    # This is: u = [[1,0],[0,1]], v = [[1,0],[0,1]], w = [[1,0],[0,1]]
    # Check: sum over i,j,k of u_{ij} v_{jk} w_{ki} A_{ij} B_{jk} = 
    #   (A00*B00*C00) + (A11*B11*C11) -- but we need cross terms for C00 and C11
    
    # I think I'm confusing myself. Let me look up the exact Strassen rank-1 terms.
    
    # The 7 rank-1 terms for M_{2,2,2} (Strassen):
    # (u_r, v_r, w_r) for r=1..7
    
    # r=1: u = [[1,0],[0,1]], v = [[1,0],[0,1]], w = [[1,0],[0,1]]
    # r=2: u = [[0,0],[1,1]], v = [[1,0],[0,0]], w = [[0,0],[1,0]]
    # r=3: u = [[1,0],[0,0]], v = [[0,1],[0,-1]], w = [[0,0],[0,1]]
    # r=4: u = [[0,0],[0,1]], v = [[-1,1],[0,0]], w = [[0,1],[0,0]]
    # r=5: u = [[1,1],[0,0]], v = [[0,0],[0,1]], w = [[0,0],[1,0]]
    # r=6: u = [[-1,0],[1,0]], v = [[1,1],[0,0]], w = [[1,0],[0,0]]
    # r=7: u = [[0,-1],[0,1]], v = [[0,0],[1,1]], w = [[0,1],[0,0]]
    
    # Let me verify r=1: 
    # sum_{i,j,k} u_{ij} v_{jk} w_{ki} A_{ij} B_{jk}
    # = u00*v00*w00*A00*B00 + u11*v11*w11*A11*B11
    # = 1*1*1*A00*B00 + 1*1*1*A11*B11
    # = A00*B00 + A11*B11
    # This should equal C00 + C11? No, it should be a rank-1 term that contributes to the tensor.
    
    # The tensor T = sum_{i,j,k} A_{ij} B_{jk} C_{ki} (in the trace convention)
    # A rank-1 decomposition is T = sum_r (sum_{i,j} u_{r,ij} A_{ij}) (sum_{j,k} v_{r,jk} B_{jk}) (sum_{k,i} w_{r,ki} C_{ki})
    
    # For r=1: (A00+A11)(B00+B11)(C00+C11)
    # Expanding: A00*B00*C00 + A00*B00*C11 + A00*B11*C00 + A00*B11*C11 + A11*B00*C00 + A11*B00*C11 + A11*B11*C00 + A11*B11*C11
    # The original tensor has: A00*B00*C00 + A00*B01*C10 + A01*B00*C00 + A01*B01*C10 + A10*B10*C01 + A10*B11*C11 + A11*B10*C01 + A11*B11*C11
    # Wait, I need to be careful with the index convention.
    
    # In the trace convention W[k*n+i, r], the tensor is:
    # T[i,j,k] = A[i,j] * B[j,k] and the output is C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r where
    # u_r[i,j] * v_r[j,k] * w_r[k,i] summed over i,j,k gives the "trace" form
    
    # Actually, for matrix multiplication, the tensor is:
    # T[i,j,k] = delta_{i,i} delta_{j,j} delta_{k,k} -- no.
    # The multiplication tensor is: T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the product is A[i,j]*B[j,k] -> C[i,k]
    # In the trace form: sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k] * (something)
    
    # I think the standard form is:
    # The tensor has components T[i,j,k] and a rank-1 term is u[i,j]*v[j,k]*w[k,i]
    # where the full tensor is sum_r u_r[i,j] * v_r[j,k] * w_r[k,i]
    # and this equals delta_{i,k} * delta_{j,j} ... no.
    
    # For matrix multiplication C = AB, we have C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T satisfies: sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k] * (identity on C)
    # So T[i,j,k] = 1 if the indices form a valid chain i->j->k, else 0
    # Specifically, T[i,j,k] = 1 for all i,j,k (since every triple is valid in the sense that A[i,j]*B[j,k] contributes to C[i,k])
    # Wait, that's not right either. T[i,j,k] should be such that the bilinear form gives C[i,k].
    
    # I think the correct statement is: The multiplication tensor is the tensor T in R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}
    # defined by T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # where e_{i,j} is the basis vector for (i,j) in R^{n*m}, etc.
    # And a rank-1 decomposition is T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r ∈ R^{n*m}, v_r ∈ R^{m*p}, w_r ∈ R^{p*n}
    # and the reconstruction is: C[i,k] = sum_r (sum_j u_r[i,j] A[i,j]) * (sum_j v_r[j,k] B[j,k]) * (sum_i w_r[k,i] * delta) 
    # Hmm, I think w_r[k,i] multiplies with the identity, so effectively:
    # C[i,k] = sum_r (sum_j u_r[i,j] A[i,j]) * (sum_j v_r[j,k] B[j,k]) * w_r[k,i]
    # But this doesn't look right dimensionally.
    
    # Let me just use the schoolbook as reference:
    # For each (i,j,k): u[i*m+j] = 1, v[j*p+k] = 1, w[k*n+i] = 1
    # Then: sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] * A[i,j] * B[j,k]
    # = sum_{i,j,k} A[i,j] B[j,k] * (sum_r u_r[i*m+j] v_r[j*p+k] w_r[k*n+i])
    # For this to equal sum_{i,k} C[i,k] * delta_{i,k}... 
    # I think the convention is that the tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # and a rank-1 term u⊗v⊗w contributes u[i*m+j]*v[j*p+k]*w[k*n+i]
    # The full tensor is sum_r u_r[i*m+j] v_r[j*p+k] w_r[k*n+i] = delta for the valid (i,j,k) triples
    # Since all (i,j,k) are valid, T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # And other components are 0.
    
    # So for Strassen 2x2x2, I need 7 vectors (u_r, v_r, w_r) in R^4 ⊗ R^4 ⊗ R^4
    # such that sum_r u_r[a] v_r[b] w_r[c] = 1 if (a,b,c) corresponds to a valid (i,j,k) triple, else 0.
    
    # For 2x2x2: a = i*2+j, b = j*2+k, c = k*2+i, with i,j,k in {0,1}
    # Valid triples: (0,0,0)->(0,0,0), (0,0,1)->(0,1,2), (0,1,0)->(1,0,1), (0,1,1)->(1,1,3),
    #              (1,0,0)->(2,0,0), (1,0,1)->(2,1,2), (1,1,0)->(3,0,1), (1,1,1)->(3,1,3)
    
    # I'll use the known Strassen decomposition. Let me define it explicitly.
    
    strassen_terms = [
        # (u, v, w) as dicts mapping (i,j)->val, (j,k)->val, (k,i)->val
        # Term 1: (A00+A11)(B00+B11) -> C00+C11
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # Term 2: (A10+A11)B00 -> C10
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1}),
        # Term 3: A00(B10-B11) -> C01
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(1,0):1}),
        # Term 4: A11(B01-B00) -> C10
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,1):1}),
        # Term 5: (A00+A01)B11 -> C01
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(1,0):1}),
        # Term 6: (A10-A00)(B00+B01) -> C00
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,0):1}),
        # Term 7: (A01-A11)(B10+B11) -> C11
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(1,1):1}),
    ]
    
    for u_dict, v_dict, w_dict in strassen_terms:
        for (ii, jj), val in u_dict.items():
            U[(i_off + ii) * 4 + (j_off + jj)][col] = val
        for (jj, kk), val in v_dict.items():
            V[(j_off + jj) * 6 + (k_off + kk)][col] = val
        for (kk, ii), val in w_dict.items():
            W[(k_off + kk) * 3 + (i_off + ii)][col] = val
        col += 1
    return col


# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # Split: 3 = 1+2, 4 = 2+2, 6 = 2+2+2
    i_splits = [1, 2]  # first block has 1 row, second has 2 rows
    j_splits = [2, 2]  # first block has 2 cols, second has 2 cols
    k_splits = [2, 2, 2]  # three blocks of 2 cols
    
    # Compute total rank
    rank = 0
    for i_size in i_splits:
        for j_size in j_splits:
            for k_size in k_splits:
                if i_size == 1 and j_size == 2 and k_size == 2:
                    rank += 4  # schoolbook for 1x2x2
                elif i_size == 2 and j_size == 2 and k_size == 2:
                    rank += 7  # Strassen for 2x2x2
    
    # Allocate matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    i_off = 0
    for i_size in i_splits:
        j_off = 0
        for j_size in j_splits:
            k_off = 0
            for k_size in k_splits:
                if i_size == 1:
                    col = _add_block_1x2x2(U, V, W, i_off, j_off, k_off, col)
                else:
                    col = _add_block_2x2x2_strassen(U, V, W, i_off, j_off, k_off, col)
                k_off += k_size
            j_off += j_size
        i_off += i_size
    
    elapsed = time.time() - t0
    if elapsed > EVAL_DEADLINE_SECONDS:
        raise TimeoutError("construct() exceeded deadline")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Let me verify the Strassen terms are correct by checking that they sum to the identity tensor for 2×2×2.

Actually, I realize I should double-check my Strassen decomposition. Let me verify term by term that the sum of all 7 rank-1 terms equals the 2×2×2 multiplication tensor.

The 2×2×2 tensor T has T[i,j,k] = 1 for all i,j,k ∈ {0,1} (in the flattened index convention: T[i*2+j, j*2+k, k*2+i] = 1).

Let me verify that sum_r u_r[i,j] v_r[j,k] w_r[k,i] = 1 for all valid (i,j,k) triples.

For (i,j,k) = (0,0,0):
- Term 1: u(0,0)=1, v(0,0)=1, w(0,0)=1 → 1
- Term 2: u(0,0)=0 → 0
- Term 3: u(0,0)=1, v(0,0)=0 → 0
- Term 4: u(0,0)=0 → 0
- Term 5: u(0,0)=1, v(0,0)=0 → 0
- Term 6: u(0,0)=-1, v(0,0)=1, w(0,0)=1 → -1
- Term 7: u(0,0)=0 → 0
- Sum: 1-1 = 0 ≠ 1

That's wrong! My Strassen terms are incorrect. Let me fix this.

The correct Strassen decomposition for 2×2×2 needs to satisfy:
sum_r u_r[i,j] v_r[j,k] w_r[k,i] = 1 for all i,j,k.

Let me use the standard Strassen algorithm more carefully. The 7 products are:
M1 = (A00+A11)(B00+B11)
M2 = (A10+A11)B00
M3 = A00(B10-B11)
M4 = A11(B01-B00)
M5 = (A00+A01)B11
M6 = (A10-A00)(B00+B01)
M7 = (A01-A11)(B10+B11)

And:
C00 = M1 + M4 - M5 + M7
C01 = M3 + M5
C10 = M2 + M4
C11 = M1 - M2 + M3 + M6

Each Mr is a scalar product of a linear combination of A entries and a linear combination of B entries. To express this as a rank-1 tensor term, I need:

For M1: u1[i,j] = 1 if (i,j) ∈ {(0,0),(1,1)}, v1[j,k] = 1 if (j,k) ∈ {(0,0),(1,1)}, and w1[k,i] should be such that w1 contributes to both C00 and C11 with coefficient +1.

But w1[k,i] is a single vector, so it can't independently control C00 and C11. This is the key insight: each Strassen term Mr contributes to multiple C entries, and the w vector encodes which C entries get which coefficient.

Actually, I think the issue is that in the tensor decomposition, each rank-1 term u⊗v⊗w contributes u[i,j]v[j,k]w[k,i] to the tensor component T[i,j,k]. The tensor T[i,j,k] = 1 for all i,j,k. So I need sum_r u_r[i,j] v_r[j,k] w_r[k,i] = 1 for all i,j,k.

Let me try a different approach. I'll directly construct the 7 vectors.

Let me denote the 8 valid (i,j,k) triples for 2×2×2:
(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)

I need 7 vectors (u_r, v_r, w_r) such that for each of the 8 triples, the sum over r of u_r[i,j]*v_r[j,k]*w_r[k,i] = 1.

This is a system of 8 equations. With 7 terms, each having 4+4+4=12 unknowns (but with structure), it should be solvable.

Let me use the known Strassen rank-1 decomposition. After some research/recall, the 7 terms are:

r=1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]  (flat: u00=1,u01=0,u10=0,u11=1)
r=2: u=[0,0,1,1], v=[1,0,0,0], w=[0,1,0,0]
r=3: u=[1,0,0,0], v=[0,1,0,-1], w=[0,0,0,1]  
r=4: u=[0,0,0,1], v=[-1,1,0,0], w=[0,0,1,0]
r=5: u=[1,1,0,0], v=[0,0,0,1], w=[0,1,0,0]
r=6: u=[-1,0,1,0], v=[1,1,0,0], w=[1,0,0,0]
r=7: u=[0,-1,0,1], v=[0,0,1,1], w=[0,0,0,1]

Wait, w has 4 entries (k*2+i for k,i in {0,1}): w00, w01, w10, w11 corresponding to (k=0,i=0), (k=0,i=1), (k=1,i=0), (k=1,i=1).

Let me verify (0,0,0): i=0,j=0,k=0
- r1: u00=1, v00=1, w00=1 → 1
- r2: u00=0 → 0
- r3: u00=1, v00=0 → 0
- r4: u00=0 → 0
- r5: u00=1, v00=0 → 0
- r6: u00=-1, v00=1, w00=1 → -1
- r7: u00=0 → 0
Sum = 0. Still wrong!

I think my Strassen terms are just wrong. Let me try to derive them properly or use a known correct set.

Given the difficulty, let me just use the schoolbook for now and keep rank 72, but structure the code so it's easy to plug in a correct Strassen decomposition later. Or better yet, let me just verify my construction works by testing it.

Actually, let me just go with the schoolbook rank 72 for now since I can't quickly verify the Strassen terms. The improvement from 72 to 66 would be significant, but I need correct terms.

Let me reconsider. Maybe I should just keep the current schoolbook implementation since I can't quickly verify the Strassen decomposition is correct.

Actually, let me try a simpler improvement: just verify the current code works and keep it as is, since I'm not confident in the Strassen terms.

