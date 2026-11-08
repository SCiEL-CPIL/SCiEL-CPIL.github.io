"""Matrix multiplication 4x5 by 5x6: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    DEADLINE = 145.0

    n, m, p = 4, 5, 6

    # We will build a list of terms: (u_idx, v_idx, w_idx, u_val, v_val, w_val)
    # Where:
    # U shape: (n*m, rank) = (20, rank)
    # V shape: (m*p, rank) = (30, rank)
    # W shape: (p*n, rank) = (24, rank)
    # 
    # Indexing:
    # A is 4x5 (i=0..3, j=0..4). U index = i*m + j = i*5 + j
    # B is 5x6 (j=0..4, k=0..5). V index = j*p + k = j*6 + k
    # C is 4x6 (i=0..3, k=0..5). W index = k*n + i = k*4 + i (trace convention W[k*n+i])
    #
    # The tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # must equal delta_{ij} for the multiplication tensor?
    # Wait, the matrix multiplication tensor T_{ijk} such that C_{ik} = sum_j A_{ij} B_{jk}
    # The tensor is T_{ijk} where A_{ij}, B_{jk}, C_{ik}.
    # The decomposition is T_{ijk} = sum_r U_{ij, r} V_{jk, r} W_{ik, r}.
    # Note: The problem statement says W[k*n+i, r]. This corresponds to the C_{ik} index being (k, i)?
    # Let's check the standard convention. Usually C = A B.
    # C_{i, k} = sum_j A_{i, j} B_{j, k}.
    # The tensor indices are (i, j, k).
    # The factors are usually U_{ij}, V_{jk}, W_{ik}.
    # Here W is indexed by [k*n + i]. So the row index of W is k*n + i.
    # This means W_{r, col} where col = k*n + i.
    # So the term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # This matches T_{ijk} = sum_r U_{ij, r} V_{jk, r} W_{ik, r} if we map W_{ik, r} to W[k*n+i, r].
    # Yes.

    terms = []  # (u_idx, v_idx, w_idx, val_u, val_v, val_w)

    def add_term(u_idx, v_idx, w_idx, val_u=1, val_v=1, val_w=1):
        if val_u == 0 or val_v == 0 or val_w == 0:
            return
        terms.append((u_idx, v_idx, w_idx, val_u, val_v, val_w))

    # Strategy:
    # Split A (4x5) into A1 (4x4) and A2 (4x1).
    # Split B (5x6) into B1 (4x6) and B2 (1x6).
    # C = A1 B1 + A2 B2.
    #
    # A2 is 4x1. B2 is 1x6.
    # A2 B2 is rank 1 (outer product of a 4-vector and a 6-vector).
    # A2 has non-zero only at j=4. B2 has non-zero only at j=4.
    # C_{i,k} = A_{i,4} * B_{4,k}.
    # This is one term:
    # U[i*5+4, r] = 1 for all i? No, U is for A.
    # Term r: U_{i,4, r} = 1? No, U_{ij, r}.
    # For this term to produce C_{ik} = A_{i4} B_{4k}, we need:
    # sum_r U_{ij, r} V_{jk, r} W_{ik, r}
    # We want a single rank-1 term that is non-zero only when j=4.
    # Let r be the index for this term.
    # U_{i4, r} = 1 for all i. (A_{i4} coefficient)
    # V_{4k, r} = 1 for all k. (B_{4k} coefficient)
    # W_{ik, r} = 1 for all i,k.
    # Then sum_r ... = 1 * 1 * 1 = 1 for the specific indices?
    # Wait. The term is U_{i4, r} V_{4k, r} W_{ik, r}.
    # If we set U_{i4, r} = 1, V_{4k, r} = 1, W_{ik, r} = 1.
    # Then for any i, k, the contribution is 1.
    # But we only want this when the input is at j=4.
    # The tensor identity is:
    # T_{ijk} = sum_r U_{ij, r} V_{jk, r} W_{ik, r}
    # We want T_{i, 4, k} = 1 (if we view it as the identity map for that slice?)
    # Actually, the "multiplication tensor" T_{ijk} is 1 if the indices correspond to a valid path?
    # No, the tensor is the bilinear map.
    # T_{ijk} is the coefficient of A_{ij} B_{jk} in C_{ik}.
    # T_{ijk} = 1 if the indices are consistent? No.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor T has entries T_{ijk}.
    # The decomposition is T_{ijk} = sum_r U_{ij, r} V_{jk, r} W_{ik, r}.
    # Here, T_{ijk} is 1 if the "path" i->j->k is used?
    # No, T_{ijk} is a scalar in the tensor space.
    # The standard basis tensor is e_i \otimes e_j \otimes e_k.
    # The multiplication tensor is sum_{i,j,k} e_i \otimes e_j \otimes e_k ? No.
    # It is sum_{i,j,k} C_{ik}^{(ij)} e_i \otimes e_j \otimes e_k ?
    # Let's stick to the algebraic definition.
    # We want to compute C = A B.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # This is a trilinear form? No, bilinear in A and B.
    # The tensor is in A^* \otimes B^* \otimes C.
    # Entry T_{ijk} corresponds to the coefficient of A_{ij} B_{jk} in C_{ik}.
    # T_{ijk} = 1 if the indices are such that it contributes to C_{ik}.
    # Actually, T is just the identity on the "middle" index?
    # No, T_{ijk} = \delta_{...}?
    # Let's look at the schoolbook rank-120 solution.
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1 for r = i*m*p + j*p + k?
    # In the current code:
    # column = i*m*p + j*p + k?
    # for i in range(n):
    #   for j in range(m):
    #     for k in range(p):
    #       U[i*m+j][column] = 1
    #       V[j*p+k][column] = 1
    #       W[k*n+i][column] = 1
    #       column += 1
    # This implies that for each (i, j, k), there is a term r.
    # Contribution to T_{i', j', k'}:
    # sum_r U_{i'j', r} V_{j'k', r} W_{i'k', r} ?
    # Wait, W index is k*n+i.
    # So W_{r, col} where col = k*n+i.
    # The term for a specific (i,j,k) sets:
    # U_{ij, r} = 1
    # V_{jk, r} = 1
    # W_{ki, r} = 1 (since index is k*n+i)
    # For other (i', j', k'), the product is 0 unless (i',j')=(i,j) and (j',k')=(j,k) and (k',i')=(k,i).
    # This implies i'=i, j'=j, k'=k.
    # So T_{ijk} gets 1 if i=i, j=j, k=k.
    # So the tensor represented is sum_{i,j,k} e_i \otimes e_j \otimes e_k ?
    # No, the tensor T is such that T(e_i, e_j, e_k) = 1?
    # Let's check: C = A B.
    # <T, A \otimes B> = sum_{i,j,k} T_{ijk} A_{ij} B_{jk} ?
    # If T_{ijk} = 1 for all i,j,k, then sum A_{ij} B_{jk} = sum_i,k C_{ik}. That's the trace, not the matrix.
    # The correct tensor for matrix multiplication is:
    # T = sum_{i,j,k} e_i^* \otimes e_j^* \otimes (e_j \otimes e_k)^* ?
    # Let's look at the factors again.
    # U_{ij, r} V_{jk, r} W_{ik, r}.
    # If we set U_{ij, r} = 1, V_{jk, r} = 1, W_{ik, r} = 1.
    # Then for inputs A, B:
    # sum_r U_{ij, r} A_{ij} V_{jk, r} B_{jk} W_{ik, r} ?
    # The bilinear form is B(A, B) = sum_{i,j,k} (sum_r U_{ij, r} V_{jk, r} W_{ik, r}) A_{ij} B_{jk} ?
    # No, the decomposition of the *tensor* T is T = sum_r U_r \otimes V_r \otimes W_r.
    # Then T(A, B) = sum_r <U_r, A> <V_r, B> W_r.
    # Here U_r is a matrix of size n x m. V_r is m x p. W_r is p x n?
    # W_r is indexed by k*n+i. So W_r is a vector of size p*n.
    # This corresponds to a matrix of size p x n? Or n x p?
    # Index k*n+i. k is row (0..p-1), i is col (0..n-1).
    # So W_r is a p x n matrix.
    # Then T(A, B) = sum_r (U_r * A) * (V_r * B)? No.
    # <U_r, A> = sum_{ij} U_{ij, r} A_{ij}.
    # <V_r, B> = sum_{jk} V_{jk, r} B_{jk}.
    # Result is a matrix W_r scaled by the scalar.
    # So C = sum_r (sum_{ij} U_{ij, r} A_{ij}) (sum_{jk} V_{jk, r} B_{jk}) W_r.
    # We want C_{ik} = sum_j A_{ij} B_{jk}.
    # This is equivalent to the tensor T_{ijk} such that:
    # sum_{i,j,k} T_{ijk} A_{ij} B_{jk} = sum_{i,k} C_{ik} D_{ik} for any D?
    # No, C is the output.
    # The tensor T is in U^* \otimes V^* \otimes W (where W is the output space).
    # T_{ijk} is the coefficient.
    # We need T(A, B) = C.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # So T_{ijk} should be 1 if the indices match the "flow" i->j->k?
    # Specifically, T_{ijk} = 1 for all i, j, k?
    # Let's test.
    # sum_r U_{ij, r} V_{jk, r} W_{ik, r} = T_{ijk}.
    # If T_{ijk} = 1 for all i,j,k, then:
    # sum_{i,j,k} 1 * A_{ij} B_{jk} = (sum A) (sum B). This is a scalar.
    # But C is a matrix.
    # Ah, the output W is indexed by (k, i).
    # So the term W_{ik, r} (or W_{k,i, r}) contributes to C_{k, i}?
    # C = A B. C is n x p.
    # Indices of C are (i, k) with i in 0..n-1, k in 0..p-1.
    # The output vector W is of size p*n.
    # Index k*n + i.
    # This flattens a p x n matrix.
    # So the output is a p x n matrix?
    # But C = A B is n x p.
    # Maybe the convention is C^T? Or C is stored column-major?
    # If W is p x n, and index is k*n + i, then W[k, i] is the element.
    # If we want C_{i, k} = sum_j A_{i, j} B_{j, k}, then C_{i, k} corresponds to W[k, i] if we transpose.
    # Or maybe the "trace convention" implies something else.
    # Let's look at the schoolbook code again.
    # U[i*m+j] = 1. (A_{ij})
    # V[j*p+k] = 1. (B_{jk})
    # W[k*n+i] = 1.
    # Term r corresponds to (i, j, k).
    # Contribution to output index (k, i) is 1.
    # So the output matrix M is p x n.
    # M_{k, i} = A_{i, j} B_{j, k} ?
    # Summing over j: M_{k, i} = sum_j A_{i, j} B_{j, k}.
    # This is (A B)_{i, k}.
    # So M_{k, i} = C_{i, k}.
    # So the output is the transpose of C, or C stored with k as row and i as col.
    # This is consistent.
    #
    # So the tensor T_{ijk} is 1 for all i, j, k?
    # Let's check.
    # T_{ijk} = sum_r U_{ij, r} V_{jk, r} W_{ik, r} (where W_{ik, r} is the component at index k*n+i).
    # If we use schoolbook, for a fixed (i,j,k), U_{ij, r}=1, V_{jk, r}=1, W_{ki, r}=1.
    # Then T_{i', j', k'} = 1 if (i', j', k') = (i, j, k).
    # So T_{ijk} = 1 for all i, j, k.
    #
    # So the task is to decompose the tensor T where T_{ijk} = 1 for all i, j, k.
    # Wait, that's the all-ones tensor.
    # Rank of all-ones tensor is 1.
    # But schoolbook uses rank 120.
    # Why?
    # Because the factors U, V, W are not just scalars, they are matrices.
    # The tensor T is in (R^{nm})^* \otimes (R^{mp})^* \otimes R^{pn}.
    # The "all ones" tensor in this space is sum_{i,j,k} e_i \otimes e_j \otimes e_k ?
    # No.
    # The matrix multiplication tensor is:
    # T = sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i} ? (where e_{i,j} is basis of A, etc.)
    # Let's verify.
    # <T, A \otimes B> = sum_{i,j,k} A_{ij} B_{jk} e_{k,i} ?
    # The result is a vector in R^{pn}.
    # The k,i component is sum_j A_{ij} B_{jk}.
    # This is exactly C_{ik}.
    # So the tensor is T = sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ki}.
    # (Using e_{ki} to match the W index k*n+i).
    #
    # So we need to decompose T = sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ki}.
    #
    # Now, back to the Strassen decomposition.
    # We split A (4x5) into A1 (4x4) and A2 (4x1).
    # B (5x6) into B1 (4x6) and B2 (1x6).
    #
    # A2 is 4x1. B2 is 1x6.
    # A2 B2 contribution:
    # A2_{i, 0} B2_{0, k}. (Using local indices for the 1-wide block).
    # Global indices: A_{i, 4} B_{4, k}.
    # Tensor term: e_{i, 4} \otimes e_{4, k} \otimes e_{k, i}.
    # This can be done with rank 1?
    # We need U_{i, 4, r} = 1, V_{4, k, r} = 1, W_{k, i, r} = 1.
    # This works for all i, k.
    # So 1 term for the (A2, B2) block.
    #
    # Now A1 (4x4) * B1 (4x6).
    # B1 is 4x6. Split B1 into B11 (4x4) and B12 (4x2).
    #
    # Block 1: A1 (4x4) * B11 (4x4).
    # Use Strassen for 4x4x4.
    # Rank of 4x4x4 is 49 (using 2x2 Strassen recursion).
    #
    # Block 2: A1 (4x4) * B12 (4x2).
    # Split A1 into four 2x2 blocks: A11, A12, A21, A22.
    # Split B12 into two 2x2 blocks: B121 (top 2 rows), B122 (bottom 2 rows).
    #
    # C12 (4x2) = [A11 B121 + A12 B122; A21 B121 + A22 B122].
    # This requires 4 multiplications of 2x2 matrices.
    # Each 2x2 multiplication has rank 7 (Strassen).
    # Total rank 28.
    #
    # Total rank = 1 (residual) + 49 (4x4x4) + 28 (4x2 tail) = 78.
    #
    # We need to implement the generation of terms for:
    # 1. 2x2 Strassen block.
    # 2. 4x4 Strassen block (composed of 2x2 blocks).
    #
    # Let's define the 2x2 Strassen terms.
    # Input A (2x2), B (2x2).
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]].
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    #
    # We need to map these to global indices.
    # Let's write a helper function to add a 2x2 Strassen block.
    # Parameters:
    # i_off: row offset in A
    # j_off: col offset in A (and row offset in B)
    # k_off: col offset in B
    #
    # This function will add 7 terms to the `terms` list.

    def add_strassen_2x2(i_off, j_off, k_off):
        # A block is A[i_off:i_off+2, j_off:j_off+2]
        # B block is B[j_off:j_off+2, k_off:k_off+2]
        # C block is C[i_off:i_off+2, k_off:k_off+2]
        
        # Indices for A: i in {0,1} -> i_off+i, j in {0,1} -> j_off+j
        # Indices for B: j in {0,1} -> j_off+j, k in {0,1} -> k_off+k
        # Indices for C: i in {0,1} -> i_off+i, k in {0,1} -> k_off+k
        
        # Term 1: M1 = (a+d)(e+h)
        # U: A_{i0,j0}, A_{i1,j1}
        # V: B_{j0,k0}, B_{j1,k1}
        # W: C_{i0,k0}, C_{i1,k0}, C_{i0,k1}, C_{i1,k1}
        
        # We need to expand the products.
        # M1 = (A_{i0,j0} + A_{i1,j1}) * (B_{j0,k0} + B_{j1,k1})
        # This is a rank-1 term in the 2x2 space, but we are adding it to the global tensor.
        # The term in the decomposition is U \otimes V \otimes W.
        # U has non-zeros at (i0, j0) and (i1, j1).
        # V has non-zeros at (j0, k0) and (j1, k1).
        # W has non-zeros at (i0, k0), (i1, k0), (i0, k1), (i1, k1).
        
        # Let's create the term.
        # We need to be careful with the signs and combinations.
        
        # It is easier to just add the 7 scalar terms corresponding to the Strassen factors.
        # But each factor is a linear combination of A and B elements.
        # So each "Strassen term" corresponds to 1 rank in the 2x2 tensor.
        # Since the 2x2 tensor has rank 7, we add 7 terms to the global list.
        
        # Term M1:
        # U part: 1 at (i_off, j_off), 1 at (i_off+1, j_off+1).
        # V part: 1 at (j_off, k_off), 1 at (j_off+1, k_off+1).
        # W part: 1 at (i_off, k_off), 1 at (i_off+1, k_off), 1 at (i_off, k_off+1), 1 at (i_off+1, k_off+1).
        # Wait, the W part comes from the expansion of C.
        # C00 = M1 + ...
        # C01 = ...
        # The W matrix for this rank-1 term must be such that when multiplied by the scalar (M1), it produces the correct part of C.
        # But M1 is not a scalar in the global tensor, it's a value computed from A and B.
        # In the tensor decomposition T = sum U_r \otimes V_r \otimes W_r:
        # U_r is a vector in A-space. V_r in B-space. W_r in C-space.
        # For the 2x2 block, U_r is a vector in R^4.
        # For M1, U_r has 1s at positions (0,0) and (1,1).
        # V_r has 1s at positions (0,0) and (1,1).
        # W_r must be the matrix that M1 multiplies.
        # M1 contributes to C00, C01, C10, C11?
        # C00 += M1
        # C01 += M1? No.
        # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
        # M1 appears in C00 (coeff 1) and C11 (coeff 1).
        # So W_r has 1 at (0,0) and 1 at (1,1).
        
        # So for M1:
        # U: [(i_off, j_off, 1), (i_off+1, j_off+1, 1)]
        # V: [(j_off, k_off, 1), (j_off+1, k_off+1, 1)]
        # W: [(i_off, k_off, 1), (i_off+1, k_off+1, 1)]
        
        add_term(i_off*m + j_off, j_off*p + k_off, k_off*n + i_off, 1, 1, 1)
        add_term(i_off*m + j_off, j_off*p + k_off, (k_off+1)*n + (i_off+1), 1, 1, 1)
        add_term((i_off+1)*m + (j_off+1), (j_off+1)*p + (k_off+1), k_off*n + i_off, 1, 1, 1)
        add_term((i_off+1)*m + (j_off+1), (j_off+1)*p + (k_off+1), (k_off+1)*n + (i_off+1), 1, 1, 1)
        # Wait, this adds 4 terms for M1.
        # But M1 is ONE rank-1 term in the 2x2 decomposition.
        # The U, V, W vectors are fixed.
        # U = e00 + e11.
        # V = e00 + e11.
        # W = e00 + e11.
        # The term is U \otimes V \otimes W.
        # This is ONE term in the global list, but U, V, W are sparse vectors.
        # My `terms` list stores (u_idx, v_idx, w_idx, val_u, val_v, val_w).
        # This format implies that each entry is a "point" in the tensor product.
        # This is not how rank-1 terms work.
        # A rank-1 term is U \otimes V \otimes W.
        # If U is sparse, we can represent it by its non-zero entries.
        # But the rank is determined by the number of such products.
        # If I add 4 entries to `terms`, does that count as 4 to the rank?
        # The evaluator likely sums the non-zero columns or counts the terms?
        # "score is the rank".
        # Usually, rank is the number of terms in the decomposition.
        # If I store the decomposition as a list of (u, v, w) vectors, the rank is len(list).
        # But the output format is matrices U, V, W.
        # The rank is the number of columns (if the decomposition is full rank).
        # So I should build the matrices U, V, W directly.
        # Each column r corresponds to one rank-1 term.
        # U[:, r] is the vector u_r.
        #
        # So I should NOT use a list of scalar entries.
        # I should build the matrices.
        
        # Let's restart the logic.
        # I will create arrays U, V, W of shape (n*m, R), (m*p, R), (p*n, R).
        # I will fill them column by column.
        
        pass # This function needs to be rewritten to work with matrices.

    # Let's change the approach.
    # We will build the matrices directly.
    
    # We need to know the total rank in advance to allocate memory?
    # Or we can use lists of lists and convert at the end.
    # Let's use lists of lists.
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_rank1_term(u_vec, v_vec, w_vec):
        # u_vec: dict of {idx: val}
        # v_vec: dict of {idx: val}
        # w_vec: dict of {idx: val}
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)

    def add_strassen_2x2(i_off, j_off, k_off):
        # M1: (a+d)(e+h)
        u1 = {(i_off)*m + j_off: 1, (i_off+1)*m + (j_off+1): 1}
        v1 = {j_off*p + k_off: 1, (j_off+1)*p + (k_off+1): 1}
        w1 = {k_off*n + i_off: 1, (k_off+1)*n + (i_off+1): 1}
        add_rank1_term(u1, v1, w1)
        
        # M2: (c+d)e
        u2 = {(i_off+1)*m + j_off: 1, (i_off+1)*m + (j_off+1): 1}
        v2 = {j_off*p + k_off: 1}
        w2 = {(k_off)*n + (i_off+1): 1} # C10 += M2
        add_rank1_term(u2, v2, w2)
        
        # M3: a(f-h)
        u3 = {i_off*m + j_off: 1}
        v3 = {j_off*p + (k_off+1): 1, (j_off+1)*p + (k_off+1): -1} # f is B[0,1], h is B[1,1]
        # B indices: j_off + 0, k_off + 1 (f); j_off + 1, k_off + 1 (h).
        w3 = {k_off*n + i_off: 1, (k_off+1)*n + i_off: 1} # C00 += M3, C01 += M3
        add_rank1_term(u3, v3, w3)
        
        # M4: d(g-e)
        u4 = {(i_off+1)*m + (j_off+1): 1}
        v4 = {(j_off+1)*p + k_off: 1, j_off*p + k_off: -1} # g is B[1,0], e is B[0,0]
        w4 = {k_off*n + i_off: 1, (k_off)*n + (i_off+1): 1} # C00 += M4, C10 += M4
        add_rank1_term(u4, v4, w4)
        
        # M5: (a+b)h
        u5 = {i_off*m + j_off: 1, i_off*m + (j_off+1): 1}
        v5 = {(j_off+1)*p + (k_off+1): 1}
        w5 = {k_off*n + i_off: -1, (k_off+1)*n + i_off: 1} # C00 -= M5, C01 += M5
        add_rank1_term(u5, v5, w5)
        
        # M6: (c-a)(e+f)
        u6 = {(i_off+1)*m + j_off: 1, i_off*m + j_off: -1}
        v6 = {j_off*p + k_off: 1, j_off*p + (k_off+1): 1}
        w6 = {(k_off+1)*n + (i_off+1): 1} # C11 += M6
        add_rank1_term(u6, v6, w6)
        
        # M7: (b-d)(g+h)
        u7 = {i_off*m + (j_off+1): 1, (i_off+1)*m + (j_off+1): -1}
        v7 = {(j_off+1)*p + k_off: 1, (j_off+1)*p + (k_off+1): 1}
        w7 = {k_off*n + i_off: 1, (k_off+1)*n + (i_off+1): -1} # C00 += M7, C11 -= M7
        add_rank1_term(u7, v7, w7)

    # Now we need to compose 4x4 from 2x2.
    # A1 (4x4) * B11 (4x4).
    # Split A1 into A11, A12, A21, A22 (2x2 each).
    # Split B11 into B111, B112, B211, B212 (2x2 each).
    
    # Strassen 4x4:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each M is a 2x2 matrix multiplication.
    # We can't just call add_strassen_2x2 directly because the inputs are sums/differences of blocks.
    # We need to add the terms for the linear combinations.
    
    # This is getting complex. Let's write a helper to add a "virtual" 2x2 multiplication
    # where A and B are linear combinations of existing blocks.
    
    # Alternatively, we can just manually add the terms for the 4x4 Strassen.
    # But that's 7 terms, each of which is a 2x2 Strassen (7 terms).
    # So 49 terms.
    
    # Let's define the blocks.
    # A11: i_off=0, j_off=0
    # A12: i_off=0, j_off=2
    # A21: i_off=2, j_off=0
    # A22: i_off=2, j_off=2
    
    # B11: j_off=0, k_off=0
    # B12: j_off=0, k_off=2
    # B21: j_off=2, k_off=0
    # B22: j_off=2, k_off=2
    
    # We need a function add_strassen_2x2_linear_combo
    # But it's easier to just add the terms for each M.
    
    # Let's create a helper to add a 2x2 block multiplication where A and B are given as dicts of {offset: coeff}.
    # This is too general. Let's just hardcode the 49 terms.
    
    # Actually, we can just call add_strassen_2x2 for the "effective" blocks if we could.
    # But we can't.
    
    # Let's try a different strategy.
    # Implement the 4x4x4 rank-49 decomposition explicitly.
    
    # We will add terms for M1..M7 of the 4x4 Strassen.
    # Each Mk is a 2x2 product.
    # We will use the `add_strassen_2x2` function? No, because the inputs are sums.
    
    # Let's write a function `add_2x2_product(A_coeff, B_coeff, C_coeff)`
    # Where A_coeff is a dict { (i_off, j_off): coeff }
    # This is still complex.
    
    # Let's just write the 49 terms manually in a loop?
    # No, let's use the property that the 4x4 Strassen is just 7 2x2 Strassen instances
    # with specific input/output mappings.
    
    # We can simulate the 2x2 Strassen by adding terms that correspond to the
    # linear combinations.
    
    # Let's define a helper `add_virtual_2x2_strassen(A_blocks, B_blocks, C_blocks)`
    # A_blocks: list of (i_off, j_off, coeff)
    # B_blocks: list of (j_off, k_off, coeff)
    # C_blocks: list of (i_off, k_off, coeff)
    
    # This would expand to 7 * (num_A * num_B * num_C) terms?
    # No, the Strassen decomposition is linear.
    # If A = sum a_i, B = sum b_j, then the terms for A*B can be built by
    # adding terms for each pair? No, that would be rank num_A * num_B * 7.
    # But the Strassen algorithm works on the whole matrix.
    
    # Actually, the simplest way is to just use the rank-49 decomposition for 4x4x4
    # found in literature or generate it.
    
    # Given the time, let's stick to the 78 rank estimate and implement it carefully.
    
    # I will implement the 4x4x4 part by explicitly adding the 49 terms.
    # I will define the 7 M's and for each M, I will add the 7 Strassen terms
    # for the 2x2 multiplication, but with the A and B blocks being the sums.
    
    # To do this, I need to be able to add a term where U is a sum of basis vectors.
    # My `add_rank1_term` takes dicts, so it supports this.
    
    # Let's define the 7 M's for 4x4.
    
    # M1 = (A11+A22)(B11+B22)
    # A_part = {A11: 1, A22: 1}
    # B_part = {B11: 1, B22: 1}
    # C_part = {C11: 1, C22: 1}
    
    # We need to add the Strassen decomposition of (A11+A22)(B11+B22).
    # This is not a single 2x2 multiplication. It is a product of two 2x2 matrices
    # that are themselves sums of 2x2 blocks.
    
    # Let A_eff = A11 + A22. This is a 4x4 matrix? No, A11 and A22 are 2x2.
    # They are in different positions.
    # So A_eff is a 4x4 matrix with blocks.
    # But the Strassen 2x2 algorithm applies to 2x2 matrices.
    
    # Wait, the 4x4 Strassen recursion splits 4x4 into 2x2 blocks.
    # So M1 is the product of two 2x2 matrices: (A11+A22) and (B11+B22).
    # But A11 and A22 are 2x2 matrices.
    # So (A11+A22) is a 2x2 matrix (in the block sense).
    # So we can apply the 2x2 Strassen algorithm to this "block matrix".
    # But the "elements" of this block matrix are themselves 2x2 matrices.
    # This is the recursive nature.
    
    # So, to add M1:
    # We add the 7 terms of the 2x2 Strassen algorithm, but where the "scalars" are replaced by 2x2 blocks.
    # This means each of the 7 terms becomes 7 terms (one for each element of the 2x2 product).
    # So 7*7 = 49 terms.
    
    # I will write a function `add_strassen_2x2_block(A_blocks, B_blocks, C_blocks)`
    # Where A_blocks is a dict { (i, j): coeff } for the 2x2 block structure.
    # But the offsets are the same.
    
    # Let's just use the `add_strassen_2x2` function but with modified offsets?
    # No, the offsets are for the global indices.
    
    # I think the cleanest way is to just implement the 49 terms for the 4x4x4 part
    # by calling `add_strassen_2x2` 49 times? No.
    
    # Let's assume the 4x4x4 part is rank 49 and just add a placeholder for now
    # to see if the structure works.
    
    # Actually, I can just use the schoolbook for the 4x4x4 part (rank 64)
    # and see if the total rank is better than 120.
    # 1 + 64 + 28 = 93.
    # This is better than 120.
    
    # Let's implement the schoolbook for the 4x4x4 block.
    
    def add_schoolbook_block(i_off, j_off, k_off, size):
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    u_idx = (i_off+i)*m + (j_off+j)
                    v_idx = (j_off+j)*p + (k_off+k)
                    w_idx = (k_off+k)*n + (i_off+i)
                    add_rank1_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})

    # Use schoolbook for 4x4x4.
    add_schoolbook_block(0, 0, 0, 4)
    
    # Use Strassen 2x2 for the 4x2 tail.
    # A1 (4x4) * B12 (4x2).
    # B12 is j_off=0, k_off=4 (since B1 is 4x6, B12 is cols 4,5).
    # Wait, B1 is 4x6. B11 is 4x4 (cols 0-3). B12 is 4x2 (cols 4-5).
    # So k_off for B12 is 4.
    # But B12 has width 2.
    # We split B12 into two 2x2 blocks: B12_top (rows 0-1, cols 4-5) and B12_bot (rows 2-3, cols 4-5).
    # B12_top: j_off=0, k_off=4.
    # B12_bot: j_off=2, k_off=4.
    
    # A1 is 4x4. Split into A11 (0,0), A12 (0,2), A21 (2,0), A22 (2,2).
    
    # C12 (4x2) = A11 B12_top + A12 B12_bot (for top 2 rows)
    #             + A21 B12_top + A22 B12_bot (for bottom 2 rows)
    
    # So we need 4 Strassen 2x2 multiplications.
    
    # 1. A11 * B12_top -> C11_top (i_off=0, k_off=4)
    add_strassen_2x2(0, 0, 4)
    
    # 2. A12 * B12_bot -> C12_top (i_off=0, k_off=4)
    # Wait, A12 is i_off=0, j_off=2.
    # B12_bot is j_off=2, k_off=4.
    # Result is i_off=0, k_off=4.
    add_strassen_2x2(0, 2, 4)
    
    # 3. A21 * B12_top -> C21_top (i_off=2, k_off=4)
    # A21 is i_off=2, j_off=0.
    # B12_top is j_off=0, k_off=4.
    add_strassen_2x2(2, 0, 4)
    
    # 4. A22 * B12_bot -> C22_top (i_off=2, k_off=4)
    # A22 is i_off=2, j_off=2.
    # B12_bot is j_off=2, k_off=4.
    add_strassen_2x2(2, 2, 4)
    
    # Residual: A2 (4x1) * B2 (1x6).
    # A2 is j=4. B2 is j=4.
    # U: i*5+4 = 1 for all i.
    # V: 4*6+k = 1 for all k.
    # W: k*4+i = 1 for all i,k.
    
    u_res = {}
    for i in range(4):
        u_res[i*5+4] = 1
    v_res = {}
    for k in range(6):
        v_res[4*6+k] = 1
    w_res = {}
    for i in range(4):
        for k in range(6):
            w_res[k*4+i] = 1
            
    add_rank1_term(u_res, v_res, w_res)
    
    # Now convert lists to matrices.
    rank = len(U_list)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for r in range(rank):
        for idx, val in U_list[r].items():
            U[idx, r] = val
        for idx, val in V_list[r].items():
            V[idx, r] = val
        for idx, val in W_list[r].items():
            W[idx, r] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
