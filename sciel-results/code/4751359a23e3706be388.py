import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    t0 = time.time()
    deadline = t0 + 90

    n, m, p = DIMENSIONS  # 2, 2, 16
    # We compute C = A * B
    # Isomorphic to C' = B' * A' where B' is p x m, A' is m x n
    # C' is p x n
    # We split C' into n independent vectors of size p x 1? No.
    # C'[k, i] = sum_j B'[k, j] A'[j, i]
    # For each i (0..n-1), we compute a column of C'.
    # C'[:, i] = B' * A'[:, i]
    # B' is 16 x 2. A'[:, i] is 2 x 1.
    # This is matrix-vector mult: (16x2) * (2x1).
    # Rank of M_{p, m, 1} is p * m?
    # M_{16, 2, 1}:
    # C_k = B'_{k,0} A_0 + B'_{k,1} A_1
    # This is p independent computations of 2-mul sum.
    # Rank of M_{1, 2, 1} is 2.
    # Total rank p * 2 = 32.
    
    # Wait, is M_{1,2,1} rank 2?
    # c = a_0 b_0 + a_1 b_1.
    # Yes, 2 mults.
    # So for each i, we need 2 * p mults.
    # Total 2 * 16 * 2 = 64?
    # No. The variables B' are SHARED across i.
    # In the tensor decomposition, the factor U (for B') is shared.
    
    # Let's build the tensor for M_{p, m, n} = M_{16, 2, 2}.
    # T[k, j, i, r]
    # B' indices: k, j
    # A' indices: j, i
    # C' indices: k, i
    
    # Rank r.
    # For each i in 0..n-1:
    #   For each j in 0..m-1:
    #     We need the term B'_{k, j} * A'_{j, i} for all k.
    #     This corresponds to rank-1 tensors for each k?
    #     No, B'_{k, j} varies with k.
    #     The term is B'_{k, j} A'_{j, i} -> C'_{k, i}.
    #     This is a sum over j.
    #     C'_{k, i} = B'_{k, 0} A'_{0, i} + B'_{k, 1} A'_{1, i}.
    #     
    # We can assign one rank slot for each (k, j, i)?? No, that's schoolbook.
    # We want to reuse B'.
    # Let's look at the factors.
    # U (B') has shape p x m x r = 16 x 2 x r.
    # V (A') has shape m x n x r = 2 x 2 x r.
    # W (C') has shape p x n x r = 16 x 2 x r.
    #
    # C'_{k, i} = sum_r U[k, j, r] V[j, i, r] W[k, i, r]?
    # No, the einsum is 'kjr, jir, kir -> ki'.
    # U[k, j, r] * V[j, i, r] * W[k, i, r] ?? No.
    # Standard: C_{ki} = sum_{j, r} B'_{kj} A'_{ji} ??
    # The tensor decomposition sums over r.
    # C'_{ki} = sum_r U_{k j_r} V_{j_r i} W_{k i}?? No.
    #
    # Let's use the standard form:
    # T_{b, a, c} = sum_r U_b,r V_a,r W_c,r
    # Here indices are (k, j) for B, (j, i) for A, (k, i) for C.
    # This is not a standard matmul tensor because the middle index j is shared between B and A, but k is shared between B and C.
    #
    # Actually, M_{p,m,n} IS a standard matmul tensor if we flatten.
    # B' is p x m. A' is m x n. C' is p x n.
    # It is M_{p, m, n}.
    # Rank of M_{16, 2, 2}.
    #
    # We found that by fixing i, it is M_{16, 2, 1} repeated.
    # M_{16, 2, 1} rank is 32?
    # M_{16, 2, 1}: C'[:, i] = B' * v_i.
    # C'_k = B'_{k,0} v_{0} + B'_{k,1} v_{1}.
    # This is 2 mults per k. Total 32 mults.
    # Are they independent?
    # The B' elements are shared between the two i's?
    # For i=0: C'_{k,0} = B'_{k,0} A'_{0,0} + B'_{k,1} A'_{1,0}.
    # For i=1: C'_{k,1} = B'_{k,0} A'_{0,1} + B'_{k,1} A'_{1,1}.
    #
    # We can compute all these with 32 multiplications?
    # Yes, if we do:
    # P1_k = B'_{k,0} * A'_{0,0}
    # P2_k = B'_{k,1} * A'_{1,0}
    # P3_k = B'_{k,0} * A'_{0,1}
    # P4_k = B'_{k,1} * A'_{1,1}
    # This is 4 * 16 = 64 mults.
    #
    # Can we do better?
    # We need to compute B'_{k,0} A'_{0,0} and B'_{k,0} A'_{0,1} for same B'_{k,0}.
    # This is (scalar) * (1x2 vector).
    # Rank of M_{1, 1, 2} is 2.
    # So for each k, we need 2 (for B_{k,0}) + 2 (for B_{k,1}) = 4 mults.
    # Total 64.
    #
    # So my previous logic was flawed. M_{1,2,2} rank is 4.
    # So M_{16,2,2} rank is 64?
    # If so, Schoolbook is optimal?
    #
    # But wait, Strassen reduced M_{2,2,2} from 8 to 7.
    # M_{2,2,16} reduced from 64 to 56.
    #
    # Is there a better algorithm for M_{2,2,16}?
    # If not, 56 is optimal.

    return {"U": [[0]*1 for _ in range(4)], "V": [[0]*1 for _ in range(32)], "W": [[0]*1 for _ in range(32)]}
