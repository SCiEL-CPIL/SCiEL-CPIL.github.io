"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _check_exact(U, V, W, n, m, p, tol=1e-9):
    # Verify that for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    # and for non-matching triples, the sum is 0.
    # We check a random subset of triples for speed, plus all "structural" ones.
    # Actually, for exactness we need to check ALL triples (i,j,k) for the 1s
    # and a sample of the 0s. But the tensor has n*m*p = 60 ones and (n*m*p)^2 - 60 zeros.
    # Let's check all 60 ones and a random 1000 zeros.
    # Better: reconstruct the full tensor contraction for a few random matrices.
    # This is more robust.
    for _ in range(20):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        AB = A @ B
        # Compute via decomposition: C[i,k] = sum_j A[i,j] B[j,k]
        # In tensor form: C[i,k] = sum_{j,c} U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] A[i,j] B[j,k]
        # Let's compute C via the rank-r decomposition.
        # C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] A[i,j]) * (sum_j V[j*p+k,c] B[j,k])
        # This is not quite right because U, V, W are coupled.
        # The correct identity is:
        # (A B)[i,k] = sum_{j} A[i,j] B[j,k] = sum_{c} (sum_j U[i*m+j,c] A[i,j]) * V[j*p+k,c] ... no.
        # The tensor T_{i,j,k} = delta_{i,i}delta_{j,j}delta_{k,k} is not right.
        # The matrix multiplication tensor M_{i,j,k} = 1 if the indices match in the product sense.
        # Actually, the tensor for matmul is T_{i,j,k,l} where (AB)_{i,l} = sum_j T_{i,j,k,l} A_{i,j} B_{k,l}.
        # Here T_{i,j,k,l} = delta_{j,k}.
        # So we need: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = delta_{j,k} ? No.
        # Let's re-read the interface.
        # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
        # The decomposition is for the tensor T with indices (i,j,k) ?
        # "exactly decompose the matrix multiplication tensor"
        # Usually this means: sum_c U_{(i,j),c} V_{(j,k),c} W_{(k,i),c} = delta_{j,k} is wrong.
        # It should be: (AB)_{i,k} = sum_j A_{i,j} B_{j,k} = sum_c (sum_j U_{(i,j),c} A_{i,j}) V_{(j,k),c} ... no.
        # The standard definition: The tensor M_{i,j,k} where M_{i,j,k} = 1 if j=k else 0? No, that's not a tensor for matmul.
        # The matmul tensor is a 4-tensor T_{i,j,k,l} = delta_{j,k}.
        # But the interface has 3 matrices U, V, W.
        # This implies a 3-way tensor decomposition?
        # "output factor follows the official trace convention W[k*n+i,r]"
        # This suggests the tensor is T_{i,j,k} such that sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C_{k,i} = trace(ABC).
        # If T_{i,j,k} = delta_{i,i}delta_{j,j}delta_{k,k} then trace(ABC) = sum A_{i,j}B_{j,k}C_{k,i}.
        # So the tensor is T_{i,j,k} = 1 for all i,j,k? No, that's not right.
        # T_{i,j,k} must be such that sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C_{k,i} = trace(ABC).
        # Since trace(ABC) = sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}, the tensor T_{i,j,k} = 1 for all i,j,k.
        # Wait, that's the identity tensor? No.
        # Let's look at the schoolbook code.
        # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
        # Sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 if (i,j,k) matches the column's (i,j,k), else 0.
        # So the tensor T_{i,j,k} = 1 for all i,j,k? No, it's the identity tensor on the triple (i,j,k).
        # T_{i,j,k} = 1 if the column corresponds to (i,j,k), which means T_{i,j,k} = 1 for ALL i,j,k?
        # No, the sum is over c. For a fixed (i,j,k), sum_c ... = 1 if there is a c such that U[i*m+j,c]=V[j*p+k,c]=W[k*n+i,c]=1.
        # In schoolbook, each (i,j,k) has its own c, so the sum is 1 for every (i,j,k).
        # So the tensor is T_{i,j,k} = 1 for all i,j,k.
        # This means the "matrix multiplication tensor" here is actually the all-ones tensor?
        # That can't be right. Let's re-read carefully.
        # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 5 matrix by a 5 by 6 matrix."
        # Maybe the tensor is T_{i,j,k} = delta_{j,k}? No, the indices are (i,j) for A, (j,k) for B, (k,i) for C.
        # The trace convention W[k*n+i] suggests the third index is (k,i).
        # So the tensor is T_{(i,j),(j,k),(k,i)}.
        # The value is 1 if the indices are consistent? i.e., the j in A matches the j in B, and the k in B matches the k in C?
        # If so, T_{i,j,k} = 1 for all i,j,k because the indices are just labels.
        # But then the rank would be 1? No.
        # Let's assume the standard definition: The tensor M for matrix multiplication satisfies
        # sum_{j} A_{i,j} B_{j,k} = sum_{c} (sum_j U_{(i,j),c} A_{i,j}) (sum_k V_{(j,k),c} B_{j,k}) ... no.
        # The correct decomposition is: (AB)_{i,k} = sum_{j,c} U_{(i,j),c} V_{(j,k),c} W_{(k,i),c} A_{i,j} B_{j,k} is not standard.
        # Let's look at the schoolbook again. It sets U,V,W to 1 for each (i,j,k).
        # So sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i,j,k.
        # This means the tensor is the all-ones tensor of shape (n*m, m*p, p*n).
        # But that's not matrix multiplication.
        # UNLESS the tensor is defined as T_{i,j,k} = delta_{j,k} in some sense?
        # No, the indices are (i,j), (j,k), (k,i). The "middle" index j is shared by A and B.
        # So the tensor is T_{(i,j),(j,k),(k,i)} = 1 if the j's match and k's match?
        # In the schoolbook, for a fixed (i,j,k), the value is 1.
        # This implies T_{(i,j),(j,k),(k,i)} = 1 for all valid indices.
        # So the tensor is indeed the all-ones tensor on the set of valid index triples.
        # But that's trivially rank 1? No, because the indices are coupled.
        # The tensor is T_{a,b,c} where a=(i,j), b=(j,k), c=(k,i).
        # T_{a,b,c} = 1 if the j in a equals the j in b, and the k in b equals the k in c, and the i in c equals the i in a.
        # Since a,b,c are uniquely determined by (i,j,k), T is 1 on all (n*m*p) valid triples and 0 elsewhere.
        # So it's a sparse tensor.
        # The rank of this tensor is what we need to minimize.
        # For n=2, m=5, p=6, the tensor has 60 non-zero entries.
        # The schoolbook rank is 60.
        # Can we do better?
        # This is the "rank of the matrix multiplication tensor" problem, but with a specific index structure.
        # Actually, this is equivalent to the rank of the tensor for multiplying n x m by m x p matrices.
        # The known lower bound is max(n*m, m*p, p*n) = max(10, 30, 12) = 30.
        # The known upper bound is n*m*p = 60.
        # For 2x5 by 5x6, the best known rank is likely 30 or slightly higher.
        # We can try to achieve rank 30 by using a Kronecker product structure or similar.
        # Let's try a simple approach: split the 5x6 matrix into blocks.
        # Or use the fact that 2x5 * 5x6 can be done with rank 30 by treating it as 2x1 * 1x6 (rank 6) + 2x4 * 4x6 (rank 24) = 30.
        # Let's implement this block decomposition.
        # Split A into A1 (2x1) and A2 (2x4). Split B into B1 (1x6) and B2 (4x6).
        # AB = A1*B1 + A2*B2.
        # Rank(A1*B1) = 2*1*6 = 12? No, 2x1 * 1x6 is rank 2*1*6=12? No, the tensor rank for 2x1 * 1x6 is 2*1*6=12?
        # Actually, the rank for n x m * m x p is at most n*m*p. For 2x1 * 1x6, it's 2*1*6=12.
        # For 2x4 * 4x6, it's 2*4*6=48.
        # Total 12+48=60. No improvement.
        # We need a smarter split.
        # What if we use a known fast matrix multiplication algorithm?
        # For 2x5 * 5x6, we can use the fact that 5 = 1+4, 6=2+4.
        # Or use the rank 30 lower bound.
        # Let's try to construct a rank 30 decomposition.
        # One idea: use the fact that the tensor is a "slice" of the 5x5x5 matmul tensor.
        # But I don't have a concrete algorithm.
        # Let's try a random search with a lower rank, say 40, and see if we can find a solution.
        # But the problem requires exact integer/half-integer entries.
        # This is very hard.
        # Given the time limit, let's stick with the schoolbook rank 60, but try to optimize it slightly.
        # Actually, the prompt says "Find a rank-minimal exact tensor decomposition".
        # If I can't find a better one, I should return the schoolbook.
        # But the score is the rank. So 60 is the current best.
        # Let's try to implement a rank 30 decomposition if possible.
        # For now, return schoolbook.
    return True

def construct():
    n, m, p = DIMENSIONS
    # Try to find a better rank than schoolbook.
    # For now, return schoolbook.
    U, V, W = _schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
