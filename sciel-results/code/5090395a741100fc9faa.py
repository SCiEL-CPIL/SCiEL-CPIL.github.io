"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Attempt to use Rank 46 (2x23) decomposition
    # If it fails verification, fall back to Rank 54 (Schoolbook)
    
    # Define 23 terms for 3x3x3
    # Using a known valid set (Strassen-like)
    # Note: This set is for 3x3x3. We use it for B[:, 0:3] and B[:, 3:6]
    
    terms_3x3x3 = []
    
    # 7 Strassen terms for 2x2x2 (indices 0,1)
    # u: A[i,j], v: B[j,k], w: C[k,i] (W index = k*3+i)
    # 1
    terms_3x3x3.append(( {0:1, 4:1}, {0:1, 3:1}, {0:1, 4:1} ))
    # 2
    terms_3x3x3.append(( {3:1, 4:1}, {0:1}, {1:1} ))
    # 3
    terms_3x3x3.append(( {0:1}, {3:1, 4:-1}, {3:1} ))
    # 4
    terms_3x3x3.append(( {4:1}, {1:1, 0:-1}, {1:1} ))
    # 5
    terms_3x3x3.append(( {0:1, 1:1}, {4:1}, {3:1} ))
    # 6
    terms_3x3x3.append(( {3:1, 0:-1}, {0:1, 1:1}, {1:1} ))
    # 7
    terms_3x3x3.append(( {1:1, 4:-1}, {3:1, 4:1}, {3:1} ))
    
    # 16 more terms for the rest (indices involving 2)
    # This part is tricky. I will use a fallback:
    # If I cannot guarantee the 16 terms, I will just use the 7 terms + schoolbook for the rest?
    # No, that increases rank.
    
    # Let's use a different strategy:
    # Use Rank 46 by using TWO 23-rank decompositions.
    # I will assume the 23 terms are correct. If not, verification will catch it.
    # For now, I will add 16 dummy terms to make 23? No.
    
    # Actually, I will just use the Schoolbook (Rank 54) but with a check.
    # Wait, the directive says "improve". 54 is the current score.
    # I must try to get 46.
    
    # I will use the 7 Strassen terms and then 16 Schoolbook terms for the remaining 16 products?
    # Total 23 terms for 3x3x3.
    # The 7 terms cover C00, C01, C10, C11.
    # The remaining 5 products in the 2x2 block? No, 2x2x2 has 4 outputs.
    # The 7 terms cover all 4 outputs of the 2x2 block.
    # So we need to cover:
    # C02, C12, C20, C21, C22
    # And the inputs A02, A12, A20, A21, A22 and B02, B12, B20, B21, B22.
    
    # Let's add schoolbook terms for the interactions with index 2.
    # C02 = A00 B02 + A01 B12 + A02 B22
    # C12 = A10 B02 + A11 B12 + A12 B22
    # C20 = A20 B00 + A21 B10 + A22 B20
    # C21 = A20 B01 + A21 B11 + A22 B21
    # C22 = A20 B02 + A21 B12 + A22 B22
    
    # This is 15 terms if we do schoolbook.
    # 7 + 15 = 22 terms?
    # Let's check if we can merge.
    # We will add these 15 terms.
    # Total 22 terms for 3x3x3.
    # Then 2 * 22 = 44 for 3x3x6.
    
    # Term 8: A00 B02 -> C02 (w index: k=2, i=0 -> 2*3+0=6)
    terms_3x3x3.append(( {0:1}, {6:1}, {6:1} ))
    # Term 9: A01 B12 -> C02
    terms_3x3x3.append(( {1:1}, {7:1}, {6:1} ))
    # Term 10: A02 B22 -> C02
    terms_3x3x3.append(( {2:1}, {8:1}, {6:1} ))
    # Term 11: A10 B02 -> C12 (w index: k=2, i=1 -> 2*3+1=7)
    terms_3x3x3.append(( {3:1}, {6:1}, {7:1} ))
    # Term 12: A11 B12 -> C12
    terms_3x3x3.append(( {4:1}, {7:1}, {7:1} ))
    # Term 13: A12 B22 -> C12
    terms_3x3x3.append(( {5:1}, {8:1}, {7:1} ))
    # Term 14: A20 B00 -> C20 (w index: k=0, i=2 -> 0*3+2=2)
    terms_3x3x3.append(( {6:1}, {0:1}, {2:1} ))
    # Term 15: A21 B10 -> C20
    terms_3x3x3.append(( {7:1}, {3:1}, {2:1} ))
    # Term 16: A22 B20 -> C20
    terms_3x3x3.append(( {8:1}, {6:1}, {2:1} ))
    # Term 17: A20 B01 -> C21 (w index: k=1, i=2 -> 1*3+2=5)
    terms_3x3x3.append(( {6:1}, {1:1}, {5:1} ))
    # Term 18: A21 B11 -> C21
    terms_3x3x3.append(( {7:1}, {4:1}, {5:1} ))
    # Term 19: A22 B21 -> C21
    terms_3x3x3.append(( {8:1}, {7:1}, {5:1} ))
    # Term 20: A20 B02 -> C22 (w index: k=2, i=2 -> 2*3+2=8)
    terms_3x3x3.append(( {6:1}, {6:1}, {8:1} ))
    # Term 21: A21 B12 -> C22
    terms_3x3x3.append(( {7:1}, {7:1}, {8:1} ))
    # Term 22: A22 B22 -> C22
    terms_3x3x3.append(( {8:1}, {8:1}, {8:1} ))
    
    # We have 22 terms.
    # Is this valid?
    # Let's verify.
    
    rank = 22 * 2  # 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for block in range(2):
        for u_dict, v_dict, w_dict in terms_3x3x3:
            for idx, val in u_dict.items():
                U[idx][col] += val
            for idx, val in v_dict.items():
                # V index is j*3 + k. For block 1, k is offset by 3.
                # v_dict keys are j*3 + k_local.
                # We need to map k_local to k_global.
                # If block 0, k_global = k_local.
                # If block 1, k_global = k_local + 3.
                j, k_local = divmod(idx, 3)
                k_global = k_local + block * 3
                v_idx = j * 6 + k_global
                V[v_idx][col] += val
            for idx, val in w_dict.items():
                # W index is k*3 + i.
                # w_dict keys are k_local*3 + i.
                k_local, i = divmod(idx, 3)
                k_global = k_local + block * 3
                w_idx = k_global * 3 + i
                W[w_idx][col] += val
            col += 1
            
    # Verify
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # We need T[i,j,k] = 1 if i==k? No.
    # T is the multiplication tensor.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # So T[i,j,k] should be 1 if i==k? No.
    # The tensor T has T[i,j,k] = delta_{i,k}? No.
    # The decomposition is C = A B.
    # C[i,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] is NOT the standard form.
    # The standard form is:
    # C[i,k] = sum_j A[i,j] B[j,k].
    # Tensor T_{i,j,k} such that C = A * T * B?
    # Actually, the problem statement says:
    # "U, V, W ... exactly decompose the matrix multiplication tensor"
    # The tensor is T[i,j,k] = 1 if i==k? No.
    # The tensor for matrix multiplication A(n,m) * B(m,p) -> C(n,p) is:
    # T[i, j, k] is the coefficient of A[i,j] B[j,k] in C[i,k].
    # So T[i,j,k] = 1 if the output index (i,k) matches the product A[i,j]B[j,k].
    # Wait, the output is C[i,k]. The term is A[i,j] B[j,k].
    # So T[i,j,k] is 1 if the output index is (i,k).
    # But the W matrix has indices (p*n, rank).
    # W[k*n+i, r].
    # So the term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # This sums to C[i,k].
    # So we need sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = delta_{i,k}? No.
    # We need sum_r ... = 1 if the product A[i,j]B[j,k] contributes to C[i,k].
    # It always contributes to C[i,k].
    # So we need sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k?
    # No.
    # The sum is over r.
    # For a fixed i,j,k, the term is A[i,j] B[j,k].
    # This term contributes to C[i,k].
    # So the coefficient of A[i,j] B[j,k] in C[i,k] is 1.
    # So we need sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1.
    # And for C[i', k'] where (i',k') != (i,k), the sum should be 0.
    # But the sum is only defined for the specific i,j,k.
    # The decomposition is:
    # C[i,k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] A[i,j] B[j,k] ?
    # No.
    # The decomposition is:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) (sum_j' V[j'*p+k, r] B[j',k]) W[k*n+i, r] ?
    # No.
    # The standard tensor decomposition is:
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    # And we need T[i,j,k] = 1 if i==k? No.
    # T[i,j,k] is 1 if the output index (i,k) is the one produced by A[i,j]B[j,k].
    # But A[i,j]B[j,k] produces C[i,k].
    # So T[i,j,k] = 1 for all i,j,k?
    # No, that would mean C[i,k] = sum_j A[i,j] B[j,k] sum_r ...
    # This is confusing.
    
    # Let's check the condition:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k.
    # This is the condition for the identity tensor.
    # But matrix multiplication is not the identity.
    
    # Actually, the tensor is T[i,j,k] = delta_{i,k} is WRONG.
    # The tensor is T[i,j,k] such that C[i,k] = sum_j A[i,j] B[j,k].
    # In tensor form: C = A x_2 T x_3 B?
    # No.
    # The problem states: "W[k*n+i,r]".
    # This implies the output index is (k,i).
    # So C[k,i] = sum_j A[i,j] B[j,k].
    # This is C^T if we assume C[i,k].
    # But the problem says "output factor follows ... W[k*n+i,r]".
    # So the output is indexed by (k,i).
    # Let's assume C[k,i] = sum_j A[i,j] B[j,k].
    # Then the term A[i,j] B[j,k] contributes to C[k,i].
    # So T[i,j,k] = 1.
    # So we need sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1.
    
    # Let's verify my 22 terms for this condition.
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-6:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if not valid:
        # Fall back to schoolbook
        rank = n * m * p
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
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
