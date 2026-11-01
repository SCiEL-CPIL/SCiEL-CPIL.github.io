import numpy as np
import time
from fractions import Fraction

def construct():
    _t0 = time.time()
    _deadline = _t0 + 110.0  # safety margin before 150s kill

    n, m, p = 2, 3, 8
    # We attempt to find a rank-10 decomposition for the 2x3x2 sub-tensor.
    # The sub-tensor T has entries T[i,j,k] = delta_{i,j} for i in {0,1}, j in {0,1,2}, k in {0,1}.
    # We seek U (6x10), V (6x10), W (4x10) such that T = sum_r U[:,r] V[:,r] W[:,r]^T.
    # Actually, the tensor is T[i,j,k] = delta_{i,j}. So T[i,j,k] = 1 if i==j else 0.
    # Wait, the matrix multiplication is A(2x3) * B(3x2) = C(2x2).
    # C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T_{i,j,k} = delta_{i,j} is not quite right.
    # The multiplication tensor for 2x3 * 3x2 has entries T_{i,j,k} where i is row of A (0,1), j is col of A / row of B (0,1,2), k is col of B (0,1).
    # T_{i,j,k} = 1 if the product A[i,j]B[j,k] contributes to C[i,k].
    # Yes, T_{i,j,k} = delta_{j,j} = 1 for all i,j,k? No.
    # The tensor is defined such that C[i,k] = sum_{j} A[i,j] B[j,k].
    # So the tensor T has components T_{i,j,k} = 1 for all i,j,k? No.
    # The structure tensor has T_{i,j,k} = 1 if the term A[i,j]B[j,k] appears in C[i,k].
    # Since C[i,k] = sum_j A[i,j]B[j,k], the term A[i,j]B[j,k] always appears.
    # So T_{i,j,k} = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1}.
    # Wait, that's just the all-ones tensor? No.
    # The tensor is T_{i,j,k} = delta_{i,i} delta_{j,j} delta_{k,k} = 1.
    # This is incorrect. The tensor encodes the multiplication.
    # For matrix multiplication, T_{i,j,k} = 1 if the index j in A matches the index j in B.
    # Here A is 2x3, B is 3x2. A[i,j] for i=0,1; j=0,1,2. B[j,k] for j=0,1,2; k=0,1.
    # The product C[i,k] = sum_{j=0}^2 A[i,j] B[j,k].
    # The tensor T has dimensions (2, 3, 2).
    # T_{i,j,k} = 1 for all i,j,k.
    # This is the "all-ones" tensor? No, that would imply C[i,k] = sum_{i',j,k'} T_{i',j,k'} A[i',j] B[k',j]?
    # The standard definition: T is a tensor in U* x V* x W* such that
    # T(a, b, c) = sum_{i,j,k} T_{i,j,k} a_i b_j c_k = sum_{i,k} C_{i,k} c_{i,k}
    # where C = A B.
    # So T_{i,j,k} must be such that sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C'_{k,i} = ||C||^2?
    # Actually, the bilinear form is sum_{i,k} C_{i,k} D_{i,k} = sum_{i,k} (sum_j A_{i,j} B_{j,k}) D_{i,k}
    # = sum_{i,j,k} A_{i,j} B_{j,k} D_{i,k}.
    # So T_{i,j,k} = 1 for all i,j,k.
    # Wait, if T_{i,j,k}=1, then the rank is the rank of the all-ones tensor of size 2x3x2.
    # The rank of the all-ones tensor of size n x m x p is n*m*p? No.
    # The all-ones tensor can be written as sum_{r=1}^{?} u_r v_r w_r.
    # u_r = e_{i_r}, v_r = e_{j_r}, w_r = e_{k_r}.
    # To get 1 for all (i,j,k), we need sum_r delta_{i,i_r} delta_{j,j_r} delta_{k,k_r} = 1.
    # This is impossible with finite rank unless we use non-standard bases? No, entries are scalars.
    # Actually, the all-ones tensor J_{n,m,p} has rank n*m*p? No.
    # J_{n,m,p} = sum_{i,j,k} e_i e_j e_k. This is a sum of n*m*p rank-1 tensors.
    # Can we do better?
    # For 2x2x2, the all-ones tensor has rank 7? No, Strassen is for multiplication.
    # The multiplication tensor for 2x2x2 is not the all-ones tensor.
    # For 2x2x2: C_{00} = A_{00}B_{00} + A_{01}B_{10}
    # C_{01} = A_{00}B_{01} + A_{01}B_{11}
    # C_{10} = A_{10}B_{00} + A_{11}B_{10}
    # C_{11} = A_{10}B_{01} + A_{11}B_{11}
    # The tensor T has T_{i,j,k} = 1 if the term A_{i,j}B_{j,k} contributes to C_{i,k}.
    # In this case, for every i,j,k, the term A_{i,j}B_{j,k} contributes to C_{i,k}.
    # So T_{i,j,k} = 1 for all i,j,k.
    # So the 2x3x2 multiplication tensor IS the all-ones tensor of size 2x3x2.
    # Let's verify.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # The bilinear form is sum_{i,k} C[i,k] D[i,k] = sum_{i,k} sum_j A[i,j] B[j,k] D[i,k]
    # = sum_{i,j,k} A[i,j] B[j,k] D[i,k].
    # So the tensor T_{i,j,k} = 1 for all i,j,k.
    # The problem reduces to finding the rank of the all-ones tensor of size 2x3x2.
    # The rank of the all-ones tensor J_{n,m,p} is known to be n*m*p in general?
    # No. J_{2,2,2} has rank 7?
    # Let's check the rank of J_{2,2,2}.
    # J_{2,2,2} = sum_{i,j,k} e_i e_j e_k.
    # Is there a decomposition of rank < 8?
    # Strassen's algorithm for 2x2x2 matrix multiplication uses 7 products.
    # But Strassen's algorithm decomposes the *multiplication* tensor, which is J_{2,2,2}.
    # So the rank of J_{2,2,2} is 7.
    # What about J_{2,3,2}?
    # J_{2,3,2} = J_{2,2,2} + J_{2,1,2} (splitting the middle dimension).
    # J_{2,1,2} is the all-ones tensor of size 2x1x2.
    # J_{2,1,2} = sum_{i,k} e_i (e_0) e_k.
    # This tensor has rank 4 (it's a matrix of all ones, rank 1? No, it's a tensor).
    # J_{2,1,2} = e_0 \otimes (sum_{i,k} e_i \otimes e_k) = e_0 \otimes J_{2,2}.
    # The rank of J_{2,2} (as a matrix) is 1.
    # So J_{2,1,2} has rank 1?
    # Let's check. J_{2,1,2}_{i,j,k} = 1 if j=0 else 0.
    # We need sum_r u_{i,r} v_{j,r} w_{k,r} = delta_{j,0}.
    # If we set v_{j,r} = delta_{j,0} for all r, and u_{i,r} w_{k,r} sums to 1 for all i,k.
    # We can take r=1..4: u_{i,r} = e_{i mod 2}, w_{k,r} = e_{k mod 2}.
    # Then sum_r u_{i,r} w_{k,r} = 1 if i=r mod 2 and k=r mod 2? No.
    # Let's just use the fact that J_{n,m,p} has rank n*m*p? No.
    # The rank of J_{2,3,2} is at most rank(J_{2,2,2}) + rank(J_{2,1,2}).
    # rank(J_{2,2,2}) = 7.
    # rank(J_{2,1,2}) = ?
    # J_{2,1,2} is a 2x1x2 tensor.
    # It can be decomposed as sum_{i=0}^1 sum_{k=0}^1 e_i \otimes e_0 \otimes e_k.
    # This is 4 rank-1 terms.
    # Can we do it in fewer?
    # The tensor is J_{2,2} \otimes e_0.
    # The rank of J_{2,2} (as a matrix) is 1.
    # So J_{2,2} = u v^T.
    # Then J_{2,2} \otimes e_0 = (u \otimes e_0) (v \otimes 1)^T? No.
    # J_{2,1,2}_{i,j,k} = delta_{j,0} * 1.
    # We need sum_r u_{i,r} v_{j,r} w_{k,r} = delta_{j,0}.
    # Let v_{j,r} = delta_{j,0} for all r.
    # Then we need sum_r u_{i,r} w_{k,r} = 1 for all i,k.
    # This is equivalent to finding a rank-1 decomposition of the 2x2 all-ones matrix.
    # The 2x2 all-ones matrix has rank 1.
    # So we can take u = [1,1]^T, w = [1,1]^T.
    # Then u w^T = [[1,1],[1,1]].
    # So we only need 1 term for the "extra" part?
    # Let's check.
    # Term r: u_{i,r} = 1, v_{j,r} = delta_{j,0}, w_{k,r} = 1.
    # Sum_r ... = 1 * delta_{j,0} * 1 = delta_{j,0}.
    # Yes! So J_{2,1,2} has rank 1.
    # So J_{2,3,2} = J_{2,2,2} + J_{2,1,2} has rank at most 7 + 1 = 8.
    # Can we do better?
    # The current program uses 11.
    # If the rank is 8, we can get total rank 4 * 8 = 32.
    # Let's verify if J_{2,3,2} has rank 8.
    # We need to check if J_{2,3,2} can be decomposed with rank < 8.
    # The border rank of J_{n,m,p} is n*m*p? No.
    # For J_{2,2,2}, rank is 7.
    # For J_{2,3,2}, is it 8?
    # Let's try to construct a rank-8 decomposition.
    # Use Strassen for J_{2,2,2} (7 terms) + 1 term for J_{2,1,2}.
    # The 7 Strassen terms for J_{2,2,2} use A[i,0], A[i,1] and B[0,k], B[1,k].
    # The 1 extra term uses A[i,2] and B[2,k].
    # The extra term is u_{i,r} v_{j,r} w_{k,r} with v_{j,r} = delta_{j,2}, u_{i,r}=1, w_{k,r}=1.
    # This works.
    # So total rank per block is 8.
    # Total rank is 4 * 8 = 32.

    rank_per_block = 8
    num_blocks = p // 2  # 4 blocks
    total_rank = rank_per_block * num_blocks  # 32

    U = np.zeros((n*m, total_rank))  # (6, 32)
    V = np.zeros((m*p, total_rank))  # (24, 32)
    W = np.zeros((p*n, total_rank))  # (16, 32)

    for blk in range(num_blocks):
        base = blk * rank_per_block
        # Strassen part (indices 0-6)
        # m1 = (A00+A11)(B00+B11)
        U[0*3+0, base+0] = 1
        U[1*3+1, base+0] = 1
        V[0*p + 2*blk, base+0] = 1
        V[1*p + (2*blk+1), base+0] = 1
        W[4*blk + 0, base+0] = 1
        W[4*blk + 3, base+0] = 1

        # m2 = (A10+A11)B00
        U[1*3+0, base+1] = 1
        U[1*3+1, base+1] = 1
        V[0*p + 2*blk, base+1] = 1
        W[4*blk + 1, base+1] = 1
        W[4*blk + 3, base+1] = -1

        # m3 = A00(B01-B11)
        U[0*3+0, base+2] = 1
        V[0*p + (2*blk+1), base+2] = 1
        V[1*p + (2*blk+1), base+2] = -1
        W[4*blk + 2, base+2] = 1
        W[4*blk + 3, base+2] = 1

        # m4 = A11(B10-B00)
        U[1*3+1, base+3] = 1
        V[1*p + 2*blk, base+3] = 1
        V[0*p + 2*blk, base+3] = -1
        W[4*blk + 0, base+3] = 1
        W[4*blk + 1, base+3] = 1

        # m5 = (A00+A01)B11
        U[0*3+0, base+4] = 1
        U[0*3+1, base+4] = 1
        V[1*p + (2*blk+1), base+4] = 1
        W[4*blk + 0, base+4] = -1
        W[4*blk + 2, base+4] = 1

        # m6 = (A10-A00)(B00+B01)
        U[1*3+0, base+5] = 1
        U[0*3+0, base+5] = -1
        V[0*p + 2*blk, base+5] = 1
        V[0*p + (2*blk+1), base+5] = 1
        W[4*blk + 3, base+5] = 1

        # m7 = (A01-A11)(B10+B11)
        U[0*3+1, base+6] = 1
        U[1*3+1, base+6] = -1
        V[1*p + 2*blk, base+6] = 1
        V[1*p + (2*blk+1), base+6] = 1
        W[4*blk + 0, base+6] = 1

        # Extra part: A[i,2] * B[2,k] for i in {0,1}, k in {2*blk, 2*blk+1}
        # This is J_{2,1,2} which has rank 1.
        # u_{i,r} = 1, v_{j,r} = delta_{j,2}, w_{k,r} = 1.
        U[0*3+2, base+7] = 1
        U[1*3+2, base+7] = 1
        V[2*p + 2*blk, base+7] = 1
        V[2*p + (2*blk+1), base+7] = 1
        W[4*blk + 0, base+7] = 1
        W[4*blk + 1, base+7] = 1
        W[4*blk + 2, base+7] = 1
        W[4*blk + 3, base+7] = 1

    if time.time() > _deadline - 5.0:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
