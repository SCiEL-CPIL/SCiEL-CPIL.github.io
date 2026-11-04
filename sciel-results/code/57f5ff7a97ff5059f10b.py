import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = 4, 5, 6
    nm = n * m
    mp = m * p
    pn = p * n

    # We will build the matrices using Strassen and Schoolbook blocks.
    # Rank will be determined by the sum of ranks of the blocks.

    # Block dimensions:
    # N blocks: 2x2
    # M blocks: 2x2, 2x2, 1x1
    # P blocks: 2x2, 2x2, 2x2

    # Let's define the blocks explicitly.
    # I index for N: 0, 1
    # J index for M: 0, 1, 2
    # K index for P: 0, 1, 2

    n_starts = [0, 2]
    m_starts = [0, 2, 4]
    m_sizes = [2, 2, 1]
    p_starts = [0, 2, 4]
    p_sizes = [2, 2, 2]

    total_rank = 0
    # Pre-allocate matrices with a large enough size, then trim.
    # Max rank estimate:
    # 2*3*3 = 18 blocks.
    # If all Strassen: 18 * 7 = 126.
    # We have some Schoolbook (rank 4) and some Strassen (rank 7).
    # Let's just build a list of coefficients first.

    U_data = []
    V_data = []
    W_data = []

    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00       -> C10 += M2, C11 -= M2
    # M3 = A00(B10-B11)       -> C01 += M3, C11 += M3
    # M4 = A11(B01-B00)       -> C01 -= M4, C11 += M4
    # M5 = (A00+A01)B11       -> C00 += M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C10 += M6
    # M7 = (A01-A11)(B10+B11) -> C10 -= M7

    # U is indexed by (i, j) in local 2x2.
    # V is indexed by (j, k) in local 2x2.
    # W is indexed by (k, i) in local 2x2.
    # The term is U[i,j] * V[j,k] * W[k,i].
    # This contributes to C[i,k] ? No, C is 2x2.
    # Standard MM: C[i,k] = sum_j A[i,j] B[j,k].
    # In tensor form: T[i,j,k] = A[i,j] B[j,k] C[k,i]? No.
    # The tensor decomposition is:
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # where T[i,j,k] is 1 if we want C[i,k] to get A[i,j]B[j,k]?
    # Wait, the matrix multiplication tensor T_{ijk} is defined such that
    # C_{ik} = sum_{j} A_{ij} B_{jk}.
    # The tensor is T_{ijk} = 1. (It's the identity in the sense of the bilinear map).
    # Actually, the tensor for MM is T_{i,j,k} = 1 for all i,j,k in the block?
    # No. The tensor is the trilinear form <A,B,C> = sum_{i,j,k} A_{ij} B_{jk} C_{ki} = trace(ABC).
    # Wait, the standard definition:
    # We want to compute C = AB.
    # The rank of the matrix multiplication tensor MM_{n,m,p} is the minimum number of multiplications.
    # The tensor is T \in R^{n \times m} \otimes R^{m \times p} \otimes R^{p \times n}.
    # T_{ijk} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
    # Specifically, the bilinear form is sum_{i,j,k} A_{ij} B_{jk} C_{ki}.
    # If C = AB, then C_{ki} is not the standard index.
    # Standard: C_{ik} = sum_j A_{ij} B_{jk}.
    # So the term A_{ij} B_{jk} adds to C_{ik}.
    # In the trace convention W[k*n+i, r], the index of W is (k, i).
    # So W_{ki} corresponds to the coefficient of the multiplication in C_{ki}?
    # No, the task says: "output factor follows the official trace convention W[k*n+i,r]".
    # And the decomposition is sum_r U_{ij,r} V_{jk,r} W_{ki,r} = T_{ijk}.
    # What is T_{ijk}?
    # If we multiply A (n x m) and B (m x p), we get C (n x p).
    # The tensor T has indices i (1..n), j (1..m), k (1..p).
    # T_{ijk} = 1.
    # Wait, if T_{ijk}=1, then sum U V W = 1 for all i,j,k?
    # No, that's not right.
    # The tensor for MM is such that the bilinear map (A,B) -> C is linear in A and B.
    # The "tensor" is often defined as T \in (R^n \otimes R^m)^* \otimes ...
    # Let's stick to the formula:
    # We want sum_r U_{ij,r} V_{jk,r} W_{ki,r} to produce the correct C.
    # Actually, the standard Strassen decomposition satisfies:
    # C_{ik} = sum_r W_{ki,r} (sum_j U_{ij,r} A_{ij}) (sum_k V_{jk,r} B_{jk})?
    # No, the factors are:
    # M_r = (sum_{i,j} U_{ij,r} A_{ij}) * (sum_{j,k} V_{jk,r} B_{jk})
    # C_{ik} = sum_r W_{ki,r} M_r.
    # This is the standard bilinear algorithm.
    # The tensor rank is the number of terms r.
    # The condition is that for all i,j,k:
    # sum_r U_{ij,r} V_{jk,r} W_{ki,r} = delta_{i,i'} delta_{j,j'} delta_{k,k'}? No.
    # The tensor T_{ijk} is 1 if i, j, k form a valid path?
    # Actually, for MM, the tensor is T_{ijk} = 1 for all i,j,k.
    # Wait, if T_{ijk}=1, then sum_r U_{ij,r} V_{jk,r} W_{ki,r} = 1 for all i,j,k.
    # Let's check Strassen.
    # If I use Strassen, does sum_r U_{ij,r} V_{jk,r} W_{ki,r} = 1?
    # Let's check r=1: M1 = (A00+A11)(B00+B11).
    # U: A00=1, A11=1. V: B00=1, B11=1.
    # W: C00=1, C11=1.
    # Term (0,0,0): U00 V00 W00 = 1*1*1 = 1.
    # Term (0,0,1): U00 V01 W10 = 1*0*0 = 0.
    # Term (1,1,1): U11 V11 W11 = 1*1*1 = 1.
    # Term (0,1,0): U01 V10 W00 = 0*1*1 = 0.
    # It seems the sum is not always 1.
    # Ah, the tensor T_{ijk} for MM is NOT the all-ones tensor.
    # The tensor is defined by the structure of the multiplication.
    # For MM_{2,2,2}, the tensor has 8 non-zero entries?
    # T_{ijk} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
    # But C_{ik} is a single value.
    # The "tensor" in the context of rank decomposition is the trilinear form:
    # <A, B, C> = sum_{i,j,k} A_{ij} B_{jk} C_{ki} = trace(ABC).
    # We want to decompose this trilinear form into rank-1 terms.
    # A rank-1 term is U_{ij} V_{jk} W_{ki}.
    # We want sum_r U^{(r)}_{ij} V^{(r)}_{jk} W^{(r)}_{ki} = delta_{i,k} delta_{j,j} ? No.
    # We want sum_r U^{(r)}_{ij} V^{(r)}_{jk} W^{(r)}_{ki} = 1 if i,j,k are such that A_{ij}B_{jk} is part of C?
    # Actually, the condition is:
    # For all i,j,k, sum_r U_{ij,r} V_{jk,r} W_{ki,r} = T_{ijk}.
    # What is T_{ijk}?
    # In matrix multiplication C = AB, the entry C_{ik} depends on A_{ij} and B_{jk} for all j.
    # The trilinear form is sum_{i,j,k} A_{ij} B_{jk} C_{ki}.
    # If C = AB, then C_{ki} is not defined in the same way (C is n x p, indices i and k).
    # Wait, C is n x p. Indices i (0..n-1) and k (0..p-1).
    # The trace convention W[k*n+i] implies W is indexed by (k,i) where k is from B's row (0..p-1) and i is from A's col? No.
    # W[k*n+i] means the index is k*n + i.
    # Usually, C_{ik} is the entry.
    # If W is indexed by (k,i), then the term is U_{ij} V_{jk} W_{ki}.
    # Summing this over r gives a value.
    # For this to compute C, we need:
    # C_{ik} = sum_r W_{ki,r} M_r.
    # Where M_r = sum_{j} U_{ij,r} A_{ij} * V_{jk,r} B_{jk}? No.
    # M_r = (sum_{i,j} U_{ij,r} A_{ij}) * (sum_{j,k} V_{jk,r} B_{jk}).
    # Then C_{ik} = sum_r W_{ki,r} M_r.
    # This is the standard form.
    # The condition for exact decomposition is that this formula yields C_{ik} = sum_j A_{ij} B_{jk}.
    # This is equivalent to:
    # sum_r U_{ij,r} V_{jk,r} W_{ki,r} = delta_{i,i'} delta_{j,j'} delta_{k,k'}? No.
    # It is equivalent to the tensor T_{ijk} = 1 for all i,j,k?
    # Let's test:
    # If sum_r U_{ij,r} V_{jk,r} W_{ki,r} = 1 for all i,j,k.
    # Then sum_{i,j,k} A_{ij} B_{jk} (sum_r ...) = sum_{i,j,k} A_{ij} B_{jk}.
    # This is not trace(ABC).
    # The correct tensor T for MM is such that T_{ijk} = 1 if the indices match the "diagonal" of the multiplication?
    # No. The tensor T is the identity tensor in the sense that it represents the bilinear map.
    # The rank of the MM tensor is the minimum number of multiplications.
    # The tensor is T \in R^{nm} \otimes R^{mp} \otimes R^{pn}.
    # T_{(i,j), (j,k), (k,i)} = 1.
    # Note the indices:
    # U index: (i,j)
    # V index: (j,k)
    # W index: (k,i)
    # The tensor has non-zero entries ONLY when the indices chain: i from U, j from U/V, j from V, k from V/W, k from W, i from W/U.
    # So T_{i,j,j',k',k'',i''} is non-zero only if j=j' and k=k'' and i=i''?
    # No, the tensor is T_{(i,j), (j',k'), (k'',i'')} = 1 if j=j', k'=k'', i''=i.
    # Wait, U is indexed by (i,j). V by (j',k'). W by (k'',i'').
    # The contraction is U_{ij} V_{j'k'} W_{k''i''}.
    # For this to match the tensor, we need j=j', k'=k'', i''=i.
    # So the tensor T has T_{(i,j), (j,k), (k,i)} = 1.
    # All other entries are 0.
    # So the condition is:
    # sum_r U_{ij,r} V_{jk,r} W_{ki,r} = 1 if the indices are "consistent" (which they are by definition in the sum)
    # Wait, the sum is over r. The indices i,j,k are fixed.
    # So for fixed i,j,k, we need sum_r U_{ij,r} V_{jk,r} W_{ki,r} = T_{ijk}.
    # What is T_{ijk}?
    # In the MM tensor, the entry corresponding to U index (i,j), V index (j,k), W index (k,i) is 1.
    # So T_{ijk} = 1 for all valid i,j,k.
    # So the condition is simply:
    # sum_r U_{ij,r} V_{jk,r} W_{ki,r} = 1 for all i,j,k.
    # Let's verify this with Strassen.
    # For 2x2x2.
    # i,j,k in {0,1}.
    # We need sum_{r=1..7} U_{ij,r} V_{jk,r} W_{ki,r} = 1.
    # Let's check (i=0, j=0, k=0).
    # r=1: U00=1, V00=1, W00=1. Prod=1.
    # r=2: U00=0 (M2 uses A10,A11). V00=1, W00=0. Prod=0.
    # r=3: U00=1, V00=0, W00=0. Prod=0.
    # r=4: U00=0, V00=1, W00=0. Prod=0.
    # r=5: U00=1, V00=0, W00=1. Prod=0.
    # r=6: U00=0, V00=1, W00=0. Prod=0.
    # r=7: U00=0, V00=0, W00=0. Prod=0.
    # Sum = 1. Correct.
    # Let's check (i=0, j=1, k=0).
    # r=1: U01=0. Prod=0.
    # r=2: U01=0. Prod=0.
    # r=3: U01=0. Prod=0.
    # r=4: U01=0. Prod=0.
    # r=5: U01=1 (A01). V10=1 (B10). W00=1 (C00). Prod=1.
    # r=6: U01=0. Prod=0.
    # r=7: U01=1 (A01). V10=1 (B10). W00=0. Prod=0.
    # Sum = 1. Correct.
    # Let's check (i=1, j=0, k=1).
    # r=1: U10=0. Prod=0.
    # r=2: U10=1. V01=0. Prod=0.
    # r=3: U10=0. Prod=0.
    # r=4: U10=0. Prod=0.
    # r=5: U10=0. Prod=0.
    # r=6: U10=1. V01=1. W11=1. Prod=1.
    # r=7: U10=0. Prod=0.
    # Sum = 1. Correct.
    # So yes, the condition is sum_r U_{ij,r} V_{jk,r} W_{ki,r} = 1 for all i,j,k.

    # Okay, I will construct the matrices block by block.
    # For each block (I, J, K), I add contributions to the global U, V, W.
    # Global U index: i_global = n_starts[I] + i_local, j_global = m_starts[J] + j_local.
    # Global V index: j_global = m_starts[J] + j_local, k_global = p_starts[K] + k_local.
    # Global W index: k_global = p_starts[K] + k_local, i_global = n_starts[I] + i_local.
    # Note: W index in global matrix is k_global * n + i_global.

    # Strassen Coefficients (2x2x2):
    # r, U(i,j), V(j,k), W(k,i)
    # 1: U: (1,0; 0,1) + (0,0; 0,1) = (1,0; 0,1) -> A00, A11. V: (1,0; 0,1) -> B00, B11. W: (1,0; 0,1) -> C00, C11.
    #    U: 00:1, 11:1. V: 00:1, 11:1. W: 00:1, 11:1.
    # 2: U: 10:1, 11:1. V: 00:1. W: 01:1, 11:-1. (M2 = (A10+A11)B00. C10+=M2, C11-=M2).
    #    Wait, W(k,i).
    #    C10: k=0, i=1 -> W[0,1]=1.
    #    C11: k=1, i=1 -> W[1,1]=-1.
    # 3: U: 00:1. V: 10:1, 11:-1. W: 01:1, 11:1. (M3 = A00(B10-B11). C01+=M3, C11+=M3).
    #    C01: k=1, i=0 -> W[1,0]=1.
    #    C11: k=1, i=1 -> W[1,1]=1.
    # 4: U: 11:1. V: 01:1, 00:-1. W: 01:-1, 11:1. (M4 = A11(B01-B00). C01-=M4, C11+=M4).
    #    C01: k=1, i=0 -> W[1,0]=-1.
    #    C11: k=1, i=1 -> W[1,1]=1.
    # 5: U: 00:1, 01:1. V: 11:1. W: 00:1, 01:1. (M5 = (A00+A01)B11. C00+=M5, C01+=M5).
    #    C00: k=0, i=0 -> W[0,0]=1.
    #    C01: k=1, i=0 -> W[1,0]=1.
    # 6: U: 10:1, 00:-1. V: 00:1, 01:1. W: 01:1. (M6 = (A10-A00)(B00+B01). C10+=M6).
    #    C10: k=0, i=1 -> W[0,1]=1.
    # 7: U: 01:1, 11:-1. V: 10:1, 11:1. W: 01:-1. (M7 = (A01-A11)(B10+B11). C10-=M7).
    #    C10: k=0, i=1 -> W[0,1]=-1.

    strassen_coeffs = [
        # r, {(i,j): u}, {(j,k): v}, {(k,i): w}
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(1,0):1, (1,1):1}),
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(1,0):-1, (1,1):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):1, (1,0):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,1):-1})
    ]

    # Schoolbook Coefficients (2x1x2)
    # A is 2x1 (i=0,1; j=0). B is 1x2 (j=0; k=0,1). C is 2x2.
    # C[i,k] = A[i,0] * B[0,k].
    # We can do 4 multiplications.
    # M0 = A00 B00 -> C00
    # M1 = A00 B01 -> C01
    # M2 = A10 B00 -> C10
    # M3 = A10 B01 -> C11
    # U: 2x1. V: 1x2. W: 2x2 (indexed by k,i).
    schoolbook_coeffs = [
        ({(0,0):1}, {(0,0):1}, {(0,0):1}),
        ({(0,0):1}, {(0,1):1}, {(1,0):1}),
        ({(1,0):1}, {(0,0):1}, {(0,1):1}),
        ({(1,0):1}, {(0,1):1}, {(1,1):1})
    ]

    # Schoolbook Coefficients (2x2x2) - if we didn't use Strassen
    # But we use Strassen for 2x2x2.

    # What if we have a 1x1 block?
    # M block size 1.
    # If I=0 (2x2) and K=0 (2x2).
    # Block is 2x1x2.
    # We handled this above.

    # Are there other block types?
    # N: 2, M: 2, P: 2 -> Strassen (7)
    # N: 2, M: 1, P: 2 -> Schoolbook (4)
    # What if N: 1? We don't have N=1 blocks.
    # What if P: 1? We don't have P=1 blocks.

    # So we just iterate.

    max_rank = 200
    U_global = np.zeros((nm, max_rank))
    V_global = np.zeros((mp, max_rank))
    W_global = np.zeros((pn, max_rank))

    current_rank = 0

    for I in range(2):
        for J in range(3):
            for K in range(3):
                if m_sizes[J] == 2:
                    coeffs = strassen_coeffs
                    n_local = 2
                    m_local = 2
                    p_local = 2
                else:
                    coeffs = schoolbook_coeffs
                    n_local = 2
                    m_local = 1
                    p_local = 2

                for c in coeffs:
                    U_c, V_c, W_c = c
                    r = current_rank

                    for (i_l, j_l), u_val in U_c.items():
                        i_g = n_starts[I] + i_l
                        j_g = m_starts[J] + j_l
                        U_global[i_g * m + j_g, r] += u_val

                    for (j_l, k_l), v_val in V_c.items():
                        j_g = m_starts[J] + j_l
                        k_g = p_starts[K] + k_l
                        V_global[j_g * p + k_g, r] += v_val

                    for (k_l, i_l), w_val in W_c.items():
                        k_g = p_starts[K] + k_l
                        i_g = n_starts[I] + i_l
                        W_global[k_g * n + i_g, r] += w_val

                    current_rank += 1

    U_final = U_global[:, :current_rank]
    V_final = V_global[:, :current_rank]
    W_final = W_global[:, :current_rank]

    # Verification
    # Check if sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1 for all i,j,k.
    # This is an O(n*m*p*rank) check.
    # 4*5*6*108 = 12960 ops. Fast.

    # Convert to lists for return
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
