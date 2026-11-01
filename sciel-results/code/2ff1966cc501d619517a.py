"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-35 exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    
    Strategy:
    The tensor for C = A B where A is 2x2 and B is 2x10 can be viewed as
    5 independent 2x2x2 multiplication problems, one for each pair of columns in B.
    
    Total Rank = 5 * 7 = 35.
    
    We use the standard Strassen algorithm for each 2x2 block.
    """
    n, m, p = DIMENSIONS
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    # Strassen coefficients for the 7 products
    # A inputs: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1] -> indices 0,1,2,3 in U
    # B inputs: e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1] -> indices relative to block in V
    
    strassen_A = [
        [1, 0, 0, 1],  # P1: (a+d)
        [0, 0, 1, 1],  # P2: (c+d)
        [1, 0, 0, 0],  # P3: a
        [0, 0, 0, 1],  # P4: d
        [1, 1, 0, 0],  # P5: (a+b)
        [-1, 0, 1, 0], # P6: (c-a)
        [0, 1, 0, -1]  # P7: (b-d)
    ]
    
    strassen_B = [
        [1, 0, 0, 1],  # P1: (e+h)
        [1, 0, 0, 0],  # P2: e
        [0, 1, 0, -1], # P3: (f-h)
        [-1, 0, 1, 0], # P4: (g-e)
        [0, 0, 0, 1],  # P5: h
        [1, 1, 0, 0],  # P6: (e+f)
        [0, 0, 1, 1]   # P7: (g+h)
    ]
    
    # W contributions for each product to the local 2x2 result C
    # Local C indices: 0->C00, 1->C01, 2->C10, 3->C11
    # M00 = P1 + P4 - P5 + P7
    # M01 = P3 + P5
    # M10 = P2 + P4
    # M11 = P1 - P2 + P3 + P6
    
    w_contrib = [
        [0] * 4 for _ in range(7)
    ]
    
    # P1
    w_contrib[0][0] = 1  # C00
    w_contrib[0][3] = 1  # C11
    
    # P2
    w_contrib[1][2] = 1  # C10
    w_contrib[1][3] = -1 # C11
    
    # P3
    w_contrib[2][1] = 1  # C01
    w_contrib[2][3] = 1  # C11
    
    # P4
    w_contrib[3][0] = 1  # C00
    w_contrib[3][2] = 1  # C10
    
    # P5
    w_contrib[4][0] = -1 # C00
    w_contrib[4][1] = 1  # C01
    
    # P6
    w_contrib[5][3] = 1  # C11
    
    # P7
    w_contrib[6][0] = 1  # C00

    # Fill U, V, W
    for b in range(5):
        for r in range(7):
            col = b * 7 + r
            
            # Fill U
            U[0][col] = strassen_A[r][0]
            U[1][col] = strassen_A[r][1]
            U[2][col] = strassen_A[r][2]
            U[3][col] = strassen_A[r][3]
            
            # Fill V
            # V index: j*p + k
            # e -> j=0, k=2*b
            # f -> j=0, k=2*b+1
            # g -> j=1, k=2*b
            # h -> j=1, k=2*b+1
            k0 = 2 * b
            k1 = 2 * b + 1
            
            V[0 * p + k0][col] = strassen_B[r][0]
            V[0 * p + k1][col] = strassen_B[r][1]
            V[1 * p + k0][col] = strassen_B[r][2]
            V[1 * p + k1][col] = strassen_B[r][3]
            
            # Fill W
            # W index: k*n + i
            # C00 -> i=0, k=2*b -> idx (2*b)*2 + 0
            # C01 -> i=0, k=2*b+1 -> idx (2*b+1)*2 + 0
            # C10 -> i=1, k=2*b -> idx (2*b)*2 + 1
            # C11 -> i=1, k=2*b+1 -> idx (2*b+1)*2 + 1
            
            W[(2 * b) * 2 + 0][col] = w_contrib[r][0]
            W[(2 * b + 1) * 2 + 0][col] = w_contrib[r][1]
            W[(2 * b) * 2 + 1][col] = w_contrib[r][2]
            W[(2 * b + 1) * 2 + 1][col] = w_contrib[r][3]
            
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W):
    """
    Verifies that the tensor decomposition exactly computes matrix multiplication.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # Convert to numpy for speed, but use exact arithmetic via fractions if needed?
    # Given small size, numpy float is fine for verification if we check near-zero.
    # But the task says "exactly decompose". The entries are integers.
    # Let's use numpy.
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The tensor T is defined by T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    # We want T[i, j, k] = 1 if (i, j, k) corresponds to a non-zero entry in A*B, else 0.
    # A*B: C[i, k] = sum_j A[i, j] B[j, k]
    # So T[i, j, k] should be 1 if the term A[i, j] B[j, k] appears, which is always true for valid i,j,k.
    # Wait, the tensor for matrix multiplication is T_{ijk} = delta_{jk} ? No.
    # The multiplication tensor T satisfies:
    # sum_{j} A_{ij} B_{jk} = sum_{j} A_{ij} B_{jk}
    # The tensor is T_{ijk} = 1 if the indices match the structure?
    # Actually, the tensor is defined such that:
    # C_{ik} = sum_{j} A_{ij} B_{jk}
    # This is a bilinear map. The rank is the minimum number of scalar multiplications.
    # The condition for a decomposition {U_r, V_r, W_r} is:
    # For all i, j, k:
    # sum_{r} U_{i, r} V_{j, r} W_{k, r} = 1 if the term A_{ij} B_{jk} is part of the product?
    # No, the standard definition is:
    # T_{ijk} = 1 if j is the "connecting" index?
    # Let's check the definition in the prompt or standard literature.
    # "Matrix multiplication tensor" T_{n,m,p} has entries T_{ijk} = 1 if i,j,k are valid?
    # Actually, T_{ijk} = 1 if the indices correspond to a valid term in the sum.
    # For C = AB, C_{ik} = sum_j A_{ij} B_{jk}.
    # So the tensor is T_{ijk} = 1 if the indices (i, j, k) form a valid term.
    # i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # All such triples are valid terms. So T_{ijk} = 1 for all valid i, j, k.
    # Wait, that's not right. If T_{ijk}=1 for all, then sum_r U_{ir} V_{jr} W_{kr} = 1 for all i,j,k.
    # This implies U V^T W^T is the all-ones tensor?
    # Let's re-read carefully.
    # "exactly decompose the matrix multiplication tensor"
    # The tensor M_{n,m,p} is defined by M_{ijk} = 1 if the indices are compatible?
    # In the context of rank of matrix multiplication, the tensor is:
    # T_{ijk} = 1 if i, j, k are such that A_{ij} B_{jk} contributes to C_{ik}.
    # Since every A_{ij} B_{jk} contributes to C_{ik}, T_{ijk} = 1 for all i, j, k.
    # Is that correct?
    # Let's check Strassen.
    # For 2x2x2, rank 7.
    # If T_{ijk}=1 for all 8 indices, then we need sum_r U_{ir} V_{jr} W_{kr} = 1 for all 8 (i,j,k).
    # This is a system of 8 equations.
    # Let's test with the current construction.
    
    # Let's compute the tensor from the decomposition and compare to the target tensor.
    # Target tensor: T_target[i, j, k] = 1 for all i, j, k?
    # Let's verify with a simple case.
    # If A = [[1,0],[0,0]], B = [[1,0],[0,0]], C = [[1,0],[0,0]].
    # Term A[0,0]B[0,0] = 1*1 = 1.
    # Term A[0,1]B[1,0] = 0*0 = 0.
    # ...
    # The tensor decomposition computes the bilinear form.
    # The condition is:
    # sum_{j} A_{ij} B_{jk} = sum_{r} (sum_{j} U_{ir} A_{ij}?) No.
    # The decomposition is:
    # C_{ik} = sum_{r} (sum_{j} U_{ir} A_{ij}?) No, U is indexed by i (and j is implicit in the input vector?).
    # The interface says:
    # U: (n*m, rank). Index i*m + j.
    # V: (m*p, rank). Index j*p + k.
    # W: (p*n, rank). Index k*n + i.
    
    # So the computation is:
    # For each r:
    #   scalar_r = sum_{i,j} U_{i*m+j, r} A_{ij} * sum_{j',k} V_{j'*p+k, r} B_{j'k} ?
    #   No, that's not how it works.
    #   The standard tensor decomposition for matrix multiplication is:
    #   C_{ik} = sum_{r} W_{k*n+i, r} * (sum_{j} U_{i*m+j, r} A_{ij}) * (sum_{j'} V_{j'*p+k, r} B_{j'k}) ?
    #   No, that would be rank 1 tensors in the output space.
    
    # Let's look at the structure of Strassen.
    # P1 = (a+d)(e+h).
    # U_{P1} has a=1, d=1. So sum_{i,j} U_{i*m+j, P1} A_{ij} = a + d.
    # V_{P1} has e=1, h=1. So sum_{j,k} V_{j*p+k, P1} B_{jk} = e + h.
    # W_{P1} has C00=1, C11=1.
    # So C_{00} gets +1 * (a+d) * (e+h).
    # C_{11} gets +1 * (a+d) * (e+h).
    #
    # So the formula is:
    # C_{ik} = sum_{r} W_{k*n+i, r} * (sum_{j} U_{i*m+j, r} A_{ij}) * (sum_{j'} V_{j'*p+k, r} B_{j'k})
    # This is NOT correct because U and V indices mix i,j and j',k.
    # The "scalar multiplication" is between a linear combination of A entries and a linear combination of B entries.
    # Let u_r(A) = sum_{i,j} U_{i*m+j, r} A_{ij}
    # Let v_r(B) = sum_{j,k} V_{j*p+k, r} B_{jk}
    # Then the contribution to C_{ik} is W_{k*n+i, r} * u_r(A) * v_r(B).
    #
    # So we need:
    # C_{ik} = sum_{j} A_{ij} B_{jk} = sum_{r} W_{k*n+i, r} * u_r(A) * v_r(B)
    #
    # This must hold for all A, B.
    # This is equivalent to:
    # sum_{r} W_{k*n+i, r} U_{i'*m+j, r} V_{j'*p+k', r} = delta_{i,i'} delta_{j,j'} delta_{k,k'} ?
    # No.
    # Let's expand the RHS:
    # sum_{r} W_{k*n+i, r} (sum_{i',j} U_{i'*m+j, r} A_{i'j}) (sum_{j',k'} V_{j'*p+k', r} B_{j'k'})
    # = sum_{i',j,j',k'} A_{i'j} B_{j'k'} (sum_{r} W_{k*n+i, r} U_{i'*m+j, r} V_{j'*p+k', r})
    #
    # We want this to equal sum_{j} A_{ij} B_{jk}.
    # So we need:
    # sum_{r} W_{k*n+i, r} U_{i'*m+j, r} V_{j'*p+k', r} = 1 if (i',j,j',k') = (i,j,j,k) i.e. i'=i, j'=j, k'=k and j=j' ?
    # Wait, the sum over j in C_{ik} = sum_j A_{ij} B_{jk} involves the same j for A and B.
    # So we need the coefficient of A_{i'j} B_{j'k'} in the expansion to be 1 if i'=i, k'=k, and j=j', and 0 otherwise.
    # i.e. sum_{r} W_{k*n+i, r} U_{i'*m+j, r} V_{j'*p+k', r} = delta_{i,i'} delta_{k,k'} delta_{j,j'}
    
    # So the verification condition is:
    # For all i, i', j, j', k, k':
    # sum_{r} W_{k*n+i, r} U_{i'*m+j, r} V_{j'*p+k', r} == 1 if i==i', j==j', k==k' else 0.
    
    # Let's implement this check.
    
    # Dimensions:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We can form the tensor T_verify[i, j, k, i', j', k'] = sum_r ...
    # But that's huge.
    # Instead, for each i, i', k, k', we can check the matrix M_{j, j'} = sum_r W_{k*n+i, r} U_{i'*m+j, r} V_{j'*p+k', r}
    # This matrix should be the identity matrix I_{m} if i==i' and k==k', and zero matrix otherwise.
    
    for i in range(n):
        for i_prime in range(n):
            for k in range(p):
                for k_prime in range(p):
                    # Compute M[j, j_prime]
                    M = np.zeros((m, m))
                    for r in range(rank):
                        w_val = W[k * n + i][r]
                        if w_val == 0:
                            continue
                        for j in range(m):
                            u_val = U[i_prime * m + j][r]
                            if u_val == 0:
                                continue
                            for j_prime in range(m):
                                v_val = V[j_prime * p + k_prime][r]
                                if v_val == 0:
                                    continue
                                M[j, j_prime] += w_val * u_val * v_val
                    
                    # Check M
                    if i == i_prime and k == k_prime:
                        if not np.allclose(M, np.eye(m)):
                            return False
                    else:
                        if not np.allclose(M, np.zeros((m, m))):
                            return False
    return True

# EVOLVE-BLOCK-END
