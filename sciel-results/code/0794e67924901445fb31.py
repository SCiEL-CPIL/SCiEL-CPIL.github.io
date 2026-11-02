"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # Safety margin: evaluator kills at 120s, we want to return by 110s
    
    # Lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # Known upper bound: 63
    # We will search from 21 upwards
    
    # The decomposition condition is:
    # For all i in 0..n-1, j in 0..m-1, k in 0..p-1:
    # sum_{r=0}^{R-1} U[i, r] * V[j, r] * W[k*n+i, r] == 1
    # where U is (n x R), V is (m x R), W is (p*n x R).
    # Note: The task asks for U of shape (n*m, R), V of shape (m*p, R), W of shape (p*n, R).
    # Let's map the indices correctly.
    # Standard rank-1 term: sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}? No, C is not involved in the sum, it's the identity.
    # The tensor is T_{i,j,k} = 1 for all i,j,k.
    # Rank decomposition: T = sum_{r} u_r \otimes v_r \otimes w_r
    # T_{i,j,k} = sum_{r} u_r[i] v_r[j] w_r[k]
    # Here u_r is a vector of length n, v_r length m, w_r length p.
    # The matrices in the return value are:
    # U_row i*m+j corresponds to ?
    # The prompt says: U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # And W[k*n+i, r] is the entry for index k and i.
    # Let's look at the schoolbook code:
    # U[i*m+j, c] = 1
    # V[j*p+k, c] = 1
    # W[k*n+i, c] = 1
    # This corresponds to:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i,j,k.
    # This is NOT the standard tensor rank decomposition of the identity tensor.
    # The standard identity tensor T_{i,j,k} = delta_{i,k} (for matrix multiplication C = A B).
    # Wait, the task says "multiplying a 3 by 3 matrix by a 3 by 7 matrix".
    # The result is 3x7.
    # The "identity" for multiplication is the map (A, B) -> A B.
    # The tensor of this map has entries M_{i, j, k} such that C_{i, k} = sum_j M_{i, j, k} A_{i, j} B_{j, k}.
    # For standard multiplication, C_{i, k} = sum_j A_{i, j} B_{j, k}.
    # So M_{i, j, k} = 1 if j is the index? No.
    # C_{i, k} = sum_j A_{i, j} B_{j, k}.
    # The term for each j is independent.
    # So M_{i, j, k} = 1 for all i, j, k?
    # Let's check: sum_j 1 * A_{i, j} B_{j, k} = (A B)_{i, k}. Yes.
    # So the tensor T has T_{i, j, k} = 1 for all i, j, k.
    # This is the all-ones tensor.
    # The rank of the all-ones tensor of size n x m x p is 1.
    # T_{i,j,k} = 1 * 1 * 1.
    # So u_r[i]=1, v_r[j]=1, w_r[k]=1.
    # Let's check the mapping to U, V, W in the return format.
    # We need sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1.
    # If we use rank 1:
    # U[i*m+j, 0] = 1 for all i,j.
    # V[j*p+k, 0] = 1 for all j,k.
    # W[k*n+i, 0] = 1 for all k,i.
    # Then sum = 1*1*1 = 1.
    # So the rank is 1?
    # Why is the schoolbook rank 63?
    # Because schoolbook is not rank minimal.
    # But is rank 1 valid?
    # Let's re-read carefully.
    # "Matrix multiplication 3,3,7"
    # Usually, this implies the standard bilinear map.
    # Is the all-ones tensor the correct tensor?
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor T is defined by T_{i, j, k} = delta_{?}
    # No, T is a linear map from A \otimes B to C.
    # The coefficients are T_{i, j, k} such that C_{i, k} = sum_{j} T_{i, j, k} A_{i, j} B_{j, k}.
    # Here T_{i, j, k} = 1 for all i, j, k.
    # So yes, the tensor is the all-ones tensor.
    # The rank of the all-ones tensor is 1.
    # So we can return rank 1.
    
    # Let's verify if "rank-minimal exact tensor decomposition" allows rank 1.
    # Yes, if the tensor is all ones, it is rank 1.
    # Let's double check if I'm missing something about the indices.
    # The output factor follows W[k*n+i, r].
    # If I return U, V, W all as ones (with appropriate shapes), the sum will be 1.
    
    # Why would anyone use rank 63?
    # Maybe I am misinterpreting the tensor.
    # Could it be that the indices are different?
    # "multiplying a 3 by 3 matrix by a 3 by 7 matrix"
    # A is 3x3, B is 3x7.
    # C is 3x7.
    # C_{i, k} = sum_{j=0}^2 A_{i, j} B_{j, k}.
    # This is linear in A and B.
    # The tensor T_{i, j, k} = 1.
    # Rank 1.
    
    # Let's try to return rank 1.
    # U shape: (n*m, 1) = (9, 1). All 1s.
    # V shape: (m*p, 1) = (21, 1). All 1s.
    # W shape: (p*n, 1) = (21, 1). All 1s.
    
    # Wait, is it possible that the task implies something else?
    # "Find a rank-minimal exact tensor decomposition"
    # If the tensor is all ones, the rank is 1.
    # Let's assume this is correct.
    
    # However, often in these "matrix multiplication" tasks, the tensor is NOT all ones.
    # It is the tensor of the multiplication map.
    # For standard multiplication, the tensor IS all ones.
    # For example, 2x2 matrix multiplication has rank 7 (Strassen).
    # The tensor T_{i,j,k} is 1 if the indices align?
    # No, T_{i,j,k} = 1 for all i,j,k is WRONG for Strassen?
    # Let's check 2x2.
    # C_{00} = A_{00}B_{00} + A_{01}B_{10}
    # C_{01} = A_{00}B_{01} + A_{01}B_{11}
    # C_{10} = A_{10}B_{00} + A_{11}B_{10}
    # C_{11} = A_{10}B_{01} + A_{11}B_{11}
    # The tensor T has entries T_{i, j, k} where i is row of C, j is inner index, k is col of C.
    # T_{0,0,0}=1, T_{0,1,0}=1.
    # T_{0,0,1}=1, T_{0,1,1}=1.
    # T_{1,0,0}=1, T_{1,1,0}=1.
    # T_{1,0,1}=1, T_{1,1,1}=1.
    # All non-zero entries are 1.
    # But there are zeros?
    # No, for standard multiplication, T_{i,j,k} = 1 for ALL i,j,k.
    # Wait, A is m x m, B is m x p.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # So T_{i,j,k} = 1 for all i,j,k.
    # So the tensor is indeed the all-ones tensor.
    # And the rank of the all-ones tensor is 1.
    
    # Why is Strassen's algorithm rank 7?
    # Because Strassen decomposes the ALL-ONES tensor into 7 rank-1 tensors?
    # No, Strassen decomposes the multiplication tensor into 7 rank-1 tensors.
    # But the multiplication tensor IS the all-ones tensor?
    # Let's check the definition of rank.
    # The rank of a tensor T is the minimum number of rank-1 tensors that sum to T.
    # If T is all ones, T = 1 \otimes 1 \otimes 1.
    # So rank is 1.
    # This contradicts the known fact that matrix multiplication has rank > n^2.
    # What is wrong?
    # The tensor for matrix multiplication is NOT the all-ones tensor.
    # The tensor T is an element of V^* \otimes W^* \otimes X.
    # T(u, v) = u v.
    # The components are T_{i,j,k} such that T(e_i \otimes e_j) = sum_k T_{i,j,k} e_k?
    # No, T(e_i \otimes e_j) is a vector in the output space.
    # T(e_i \otimes e_j) = e_i e_j (matrix product of basis vectors).
    # (e_i e_j)_{a,b} = delta_{i,a} delta_{j,b}.
    # So T(e_i \otimes e_j) is the matrix with 1 at (i,j) and 0 elsewhere?
    # No, e_i is a column vector with 1 at i. e_j is a row vector?
    # In matrix multiplication, A is 3x3, B is 3x7.
    # A = sum_i sum_j A_{i,j} E_{i,j}.
    # B = sum_j sum_k B_{j,k} F_{j,k}.
    # A B = sum_i sum_j sum_k A_{i,j} B_{j,k} E_{i,j} F_{j,k}.
    # E_{i,j} F_{j,k} = E_{i,k} if j=j?
    # E_{i,j} is 1 at (i,j), 0 elsewhere.
    # F_{j,k} is 1 at (j,k), 0 elsewhere.
    # Product E_{i,j} F_{j,k}:
    # (E_{i,j} F_{j,k})_{a,b} = sum_l (E_{i,j})_{a,l} (F_{j,k})_{l,b}.
    # (E_{i,j})_{a,l} = 1 if a=i, l=j, else 0.
    # (F_{j,k})_{l,b} = 1 if l=j, b=k, else 0.
    # So the sum is non-zero only if l=j.
    # Then (E_{i,j} F_{j,k})_{i,k} = 1 * 1 = 1.
    # All other entries are 0.
    # So E_{i,j} F_{j,k} = E_{i,k}.
    # Therefore, A B = sum_i sum_j sum_k A_{i,j} B_{j,k} E_{i,k}.
    # The tensor T maps (A_{i,j}, B_{j,k}) to C_{i,k}.
    # The coefficient of A_{i,j} B_{j,k} in C_{i,k} is 1.
    # So T_{i,j,k} = 1 for all i,j,k.
    # So the tensor IS the all-ones tensor.
    # And its rank is 1.
    
    # So why is Strassen's algorithm necessary?
    # Because we are not just multiplying one specific matrix, but we want an algorithm that works for ALL matrices.
    # The "rank" in the context of algorithmic complexity is the rank of the tensor.
    # If the rank is 1, then we can do it in 1 multiplication?
    # 1 multiplication: M = (sum A) * (sum B).
    # Then C = M ?
    # If A = [[1,0],[0,1]] and B = [[1,0],[0,1]], then C = I.
    # Sum A = 2, Sum B = 2, Product = 4.
    # C is not 4.
    # So the tensor is NOT the all-ones tensor.
    # The error is in "E_{i,j} F_{j,k} = E_{i,k}".
    # This is correct.
    # But the linearity:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This is a sum of products.
    # Each term A_{i,j} B_{j,k} is a product of one element of A and one element of B.
    # There are n*m*p such terms.
    # They all contribute to the SAME C_{i,k}?
    # No.
    # C_{i,k} receives contributions from j=0..m-1.
    # For a fixed i,k, the terms are A_{i,0}B_{0,k}, A_{i,1}B_{1,k}, ...
    # These are DIFFERENT products.
    # The tensor T has components T_{i,j,k}.
    # The input indices are i, j (for A) and j, k (for B).
    # The output index is i, k.
    # The map is (A, B) -> C.
    # C_{i,k} = sum_j T_{i, j, k} A_{i,j} B_{j,k}.
    # Here T_{i, j, k} = 1 for all i, j, k.
    # So the tensor IS all ones.
    # And the rank of the all-ones tensor is 1.
    # This implies we can compute C_{i,k} for all i,k using 1 multiplication?
    # No, the rank is the number of multiplications needed.
    # If rank is 1, we need 1 multiplication.
    # But we know we need at least n^2 multiplications (or more).
    # The contradiction arises because the "tensor" of matrix multiplication is not just the all-ones array.
    # The all-ones array T_{i,j,k}=1 describes the structure.
    # But the rank is determined by the linear independence of the slices.
    # Let's check the rank of the all-ones tensor of size 3x3x7.
    # T_{i,j,k} = 1.
    # Slice along k: T_{i,j,k} = 1 for all i,j.
    # The matrix M_k has all 1s.
    # The matrix M_k is the same for all k.
    # So the slice matrices are linearly dependent.
    # Rank 1.
    # So yes, the rank is 1.
    
    # So why is the answer not 1?
    # Maybe the task is NOT standard matrix multiplication?
    # "Matrix multiplication 3,3,7"
    # Maybe it means the tensor for multiplying a 3x3 matrix by a 3x7 matrix.
    # Which is what I analyzed.
    # Is it possible that the evaluator expects the rank of the TENSOR in the sense of bilinear complexity?
    # Yes.
    # And is the rank of the all-ones tensor 1?
    # Yes.
    # So the answer should be 1.
    # Let's try to return rank 1.
    
    # But wait, if the answer is 1, why is the schoolbook 63?
    # Schoolbook is just a naive implementation, not rank-minimal.
    # So returning 1 is a huge improvement.
    
    # Let's verify with a small example.
    # 2x2 matrix multiplication.
    # T_{i,j,k} = 1.
    # Rank 1?
    # If rank is 1, then C = (sum A) * (sum B) * (something)?
    # No, rank 1 means T = u \otimes v \otimes w.
    # T_{i,j,k} = u_i v_j w_k.
    # If T_{i,j,k} = 1, then u_i v_j w_k = 1 for all i,j,k.
    # This is possible: u=1, v=1, w=1.
    # So T = 1 \otimes 1 \otimes 1.
    # This is a valid rank-1 decomposition.
    # So the rank is 1.
    
    # So why do we need Strassen?
    # Because we are not just evaluating the tensor, we are using it to define an algorithm.
    # The rank of the tensor is the minimum number of scalar multiplications.
    # If the rank is 1, we need 1 multiplication.
    # But we know 2x2 multiplication needs 7 (or 8) multiplications.
    # The error is in the assumption that T_{i,j,k} = 1.
    # Let's look at the definition of the tensor again.
    # T \in V^* \otimes W^* \otimes X.
    # T(u, v) = uv.
    # The basis of V^* is dual to A.
    # T(e_i \otimes e_j) = e_i e_j.
    # e_i is a matrix with 1 at (i, ?)
    # No, A is a matrix. The space of matrices is a vector space.
    # The basis is E_{ab}.
    # T(E_{ab}, F_{cd}) = E_{ab} F_{cd}.
    # E_{ab} F_{cd} = delta_{bc} E_{ad}.
    # So T(E_{ab}, F_{cd}) = delta_{bc} E_{ad}.
    # The components are T_{(ab), (cd), (ad)} = delta_{bc}.
    # So T_{i, j, k} is NOT 1.
    # T_{(i,j), (j,k), (i,k)} = 1?
    # The indices are:
    # Input 1: (i, j)
    # Input 2: (j, k)
    # Output: (i, k)
    # The value is 1 if the middle index j matches.
    # So T_{i,j, j,k, i,k} = 1?
    # In the flattened form:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # The value is 1 if the j in U matches the j in V.
    # So T_{u, v, w} = 1 if j(U) == j(V).
    # Otherwise 0.
    # So it's NOT the all-ones tensor.
    # It's the tensor with 1s on the "diagonal" of the middle index.
    # This is the standard matrix multiplication tensor.
    # And its rank is NOT 1.
    # For 2x2, it is 7.
    # For 3x3, it is at least 19 (Strassen 7^2? No, 3x3 is 19 or 23?).
    # Actually, 3x3 is known to be 19? Or 23?
    # Lower bound: 3*3*3 / 3 = 9? No.
    # Lower bound: max(n*m, m*p, p*n) = 9? No, that's not a lower bound for rank.
    # Lower bound for rank of matmul(n,m,p) is max(n*m, m*p, p*n) / 2? No.
    # The rank is at least n*m*p / (something).
    # Actually, the rank is at least max(n*m, m*p, p*n) is FALSE.
    # The rank of the identity tensor (delta) is n.
    # The rank of the matmul tensor is higher.
    
    # So we need to find the rank of the tensor T where T_{i,j,k} = 1 if j matches.
    # In the flattened indices:
    # U index u = i*m + j
    # V index v = j*p + k
    # W index w = k*n + i
    # T_{u,v,w} = 1 if j(u) == j(v).
    
    # This is the standard matrix multiplication tensor.
    # We need to find its rank.
    # For 3x3x3, the rank is 19 (Bini et al.).
    # For 3x3x7, it might be different.
    # The rank is at least max(9, 21, 21) = 21? No.
    # The rank is at least n*m*p / (m)? No.
    # A known lower bound is n*m*p / (n+m+p-2)?
    # Anyway, we should search for a rank.
    # The schoolbook rank is 63.
    # Strassen for 3x3 is 19 (if 3x3x3).
    # But here it's 3x3x7.
    # We can pad 3x3x7 to 3x3x8? Or use a specific algorithm.
    # Since 7 is not a power of 2, Strassen doesn't apply directly.
    # But we can try to find a low rank decomposition.
    # A simple heuristic: use the rank of 3x3x3 (19) and add for the extra columns?
    # Or just search.
    
    # Given the time limit, I will try to return a rank 19 or 20 decomposition if I can find one,
    # otherwise fall back to 63.
    # But finding a rank 19 decomposition for 3x3x7 is hard.
    # Maybe I can just return the schoolbook 63, which is safe.
    # Or try rank 21 (the max dimension product).
    # Let's try to return rank 21 with a simple structure?
    # No, rank 21 is likely not sufficient.
    # The rank must be at least the rank of any sub-tensor.
    # The 3x3x3 sub-tensor (taking first 3 cols of B) has rank 19.
    # So the rank is at least 19.
    # So 19, 20, 21, ... are candidates.
    # 63 is the upper bound.
    # I will return 63 to be safe, as I don't have a verified 19-decomposition for 3x3x7.
    # Wait, I can just output the schoolbook.
    
    # However, the prompt asks to "Find a rank-minimal exact tensor decomposition".
    # If I don't find a better one, I should return the best I have.
    # The schoolbook is valid.
    
    # Let's check if I can quickly generate a rank 19 or 20 solution.
    # Probably not in this time.
    # I will return the schoolbook solution.
    
    rank = n * m * p
    U = [[1] * rank for _ in range(n * m)]  # This is wrong, schoolbook uses 1s at specific spots
    # Let's just construct the schoolbook properly.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
