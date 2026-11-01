"""Matrix multiplication 2x3x6 rank-minimal decomposition."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_2x3x3_rank7():
    """
    Construct a rank-7 decomposition for 2x3x3 matrix multiplication.
    
    The multiplication tensor for C = A*B where A is 2x3, B is 3x3, C is 2x3:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    U shape: (6, 7)  -- indices (i*m+j, r)
    V shape: (9, 7)  -- indices (j*p+k, r)
    W shape: (6, 7)  -- indices (k*n+i, r)
    
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k'*n+i', r] = delta_{i,i'}*delta_{k,k'}
    """
    n, m, p = 2, 3, 3
    rank = 7
    
    # We need 7 terms. Let me construct them using a systematic approach.
    # 
    # Key insight: For 2x3x3, we can use the following structure:
    # - 3 terms for row 0 of A (i=0): each handles one column of B
    # - 3 terms for row 1 of A (i=1): each handles one column of B
    # - 1 extra term to handle the cross-terms
    #
    # Actually, let me use a known construction. The rank of 2x3x3 is 7.
    # 
    # Here's a valid rank-7 decomposition:
    # Term 0: A[0,0]*B[0,0] -> C[0,0]
    #   U[0,0]=1, V[0,0]=1, W[0,0]=1
    # Term 1: A[0,1]*B[1,0] -> C[0,0]
    #   U[1,1]=1, V[3,1]=1, W[0,1]=1
    # Term 2: A[0,2]*B[2,0] -> C[0,0]
    #   U[2,2]=1, V[6,2]=1, W[0,2]=1
    # Term 3: A[0,0]*B[0,1] -> C[0,1]
    #   U[0,3]=1, V[1,3]=1, W[3,3]=1
    # Term 4: A[0,1]*B[1,1] -> C[0,1]
    #   U[1,4]=1, V[4,4]=1, W[3,4]=1
    # Term 5: A[0,2]*B[2,1] -> C[0,1]
    #   U[2,5]=1, V[7,5]=1, W[3,5]=1
    # Term 6: A[1,0]*B[0,2] -> C[1,2]
    #   U[3,6]=1, V[2,6]=1, W[6,6]=1
    #
    # Wait, this only gives 7 terms but doesn't cover all 18 products.
    # I need to be more systematic.
    
    # Let me use a different approach: solve for the decomposition directly.
    # 
    # For 2x3x3, the schoolbook has 18 terms (one for each i,j,k triple).
    # We need to find 7 rank-1 tensors that sum to the multiplication tensor.
    #
    # Let me try a structured approach based on known results.
    # The 2x3x3 tensor rank is 7, and one known decomposition uses:
    
    U = np.zeros((6, rank))
    V = np.zeros((9, rank))
    W = np.zeros((6, rank))
    
    # Indices: U[i*3+j, r], V[j*3+k, r], W[k*2+i, r]
    
    # Term 0: contributes to C[0,0] from A[0,0]*B[0,0]
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # Term 1: contributes to C[0,0] from A[0,1]*B[1,0]
    U[1, 1] = 1; V[3, 1] = 1; W[0, 1] = 1
    # Term 2: contributes to C[0,0] from A[0,2]*B[2,0]
    U[2, 2] = 1; V[6, 2] = 1; W[0, 2] = 1
    # Term 3: contributes to C[0,1] from A[0,0]*B[0,1]
    U[0, 3] = 1; V[1, 3] = 1; W[2, 3] = 1
    # Term 4: contributes to C[0,1] from A[0,1]*B[1,1]
    U[1, 4] = 1; V[4, 4] = 1; W[2, 4] = 1
    # Term 5: contributes to C[0,1] from A[0,2]*B[2,1]
    U[2, 5] = 1; V[7, 5] = 1; W[2, 5] = 1
    # Term 6: contributes to C[1,2] from A[1,0]*B[0,2]
    U[3, 6] = 1; V[2, 6] = 1; W[4, 6] = 1
    
    # Let me verify this... it's clearly incomplete. This approach won't work
    # with just 7 naive terms.
    
    # Let me try a completely different approach: use numerical optimization
    # to find a rank-7 decomposition.
    
    return None  # Fallback to schoolbook

def _find_2x3x3_decomposition():
    """
    Find a rank-7 decomposition for 2x3x3 matrix multiplication.
    
    Returns (U, V, W) numpy arrays with shapes (6,7), (9,7), (6,7) or None.
    """
    n, m, p = 2, 3, 3
    rank = 7
    
    # Construct a rank-7 decomposition using the following strategy:
    # Use terms that exploit the structure to cover multiple products at once.
    
    U = np.zeros((6, rank))
    V = np.zeros((9, rank))
    W = np.zeros((6, rank))
    
    # Term 0: Handle C[0,0] from all 3 products
    # A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0] -> C[0,0]
    # This requires: u[0]=1, u[1]=1, u[2]=1 (for i=0, j=0,1,2)
    #               v[0]=1, v[3]=1, v[6]=1 (for j=0,1,2, k=0)
    #               w[0]=1 (for k=0, i=0)
    U[0, 0] = 1; U[1, 0] = 1; U[2, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1; V[6, 0] = 1
    W[0, 0] = 1
    
    # But this also contributes to other C entries! Let's check:
    # For C[0,1]: u[0]*v[1]*w[2] + u[1]*v[4]*w[2] + u[2]*v[7]*w[2] = 1*0*0 + 1*0*0 + 1*0*0 = 0 OK
    # For C[0,2]: similar = 0 OK
    # For C[1,0]: u[3]*v[0]*w[0] + u[4]*v[3]*w[0] + u[5]*v[6]*w[0] = 0*1*1 + 0*1*1 + 0*1*1 = 0 OK
    # Good, term 0 only contributes to C[0,0].
    
    # Term 1: Handle C[0,1] from all 3 products
    U[0, 1] = 1; U[1, 1] = 1; U[2, 1] = 1
    V[1, 1] = 1; V[4, 1] = 1; V[7, 1] = 1
    W[2, 1] = 1
    
    # Term 2: Handle C[0,2] from all 3 products
    U[0, 2] = 1; U[1, 2] = 1; U[2, 2] = 1
    V[2, 2] = 1; V[5, 2] = 1; V[8, 2] = 1
    W[4, 2] = 1
    
    # Term 3: Handle C[1,0] from all 3 products
    U[3, 3] = 1; U[4, 3] = 1; U[5, 3] = 1
    V[0, 3] = 1; V[3, 3] = 1; V[6, 3] = 1
    W[1, 3] = 1
    
    # Term 4: Handle C[1,1] from all 3 products
    U[3, 4] = 1; U[4, 4] = 1; U[5, 4] = 1
    V[1, 4] = 1; V[4, 4] = 1; V[7, 4] = 1
    W[3, 4] = 1
    
    # Term 5: Handle C[1,2] from all 3 products
    U[3, 5] = 1; U[4, 5] = 1; U[5, 5] = 1
    V[2, 5] = 1; V[5, 5] = 1; V[8, 5] = 1
    W[5, 5] = 1
    
    # That's only 6 terms! We need 7. But wait, let me verify if 6 terms suffice.
    # If 6 terms work, then rank <= 6, contradicting the known rank of 7.
    # So there must be an error in my construction.
    
    # Let me verify term 0 more carefully:
    # Contribution to C[i',k'] = sum_j U[i'*3+j, 0] * V[j*3+k', 0] * W[k'*2+i', 0]
    # For i'=0, k'=0: sum_j U[j,0]*V[j*3,0]*W[0,0] = U[0,0]*V[0,0]*W[0,0] + U[1,0]*V[3,0]*W[0,0] + U[2,0]*V[6,0]*W[0,0]
    # = 1*1*1 + 1*1*1 + 1*1*1 = 3, but we need 1!
    
    # Ah, I see the problem. Each term contributes 3 to C[0,0], not 1.
    # I need to scale: use 1/3 for each.
    
    # But the problem says entries should be integer or half-integer. 1/3 is not allowed!
    # So this approach doesn't work with the given constraints.
    
    return None

def _build_2x3x6_rank12():
    """
    Construct a rank-12 decomposition for 2x3x6 matrix multiplication.
    
    Strategy: Use the structure that 6 = 2*3, and exploit the fact that we can
    use a rank-6 decomposition for 2x3x3 and combine two copies with appropriate
    scaling to get rank 12 for 2x3x6.
    
    Actually, let me think about this more carefully. The schoolbook rank is 36.
    We want to find a lower rank decomposition.
    
    Key insight: For 2x3x6, we can view the output as two 2x3 blocks. If we can
    find a rank-6 decomposition for each 2x3x3 block that doesn't interfere with
    the other, we'd get rank 12.
    
    Let me construct a rank-6 decomposition for 2x3x3 first, then combine.
    
    For 2x3x3, the rank is known to be 7, but let me see if I can get a valid
    rank-6 decomposition that works in the context of 2x3x6.
    
    Actually, the known rank of 2x3x3 is 7, so I can't get rank 6 for 2x3x3 alone.
    But for 2x3x6, maybe I can do better than 2*7=14.
    
    Let me try a different approach: use the fact that 2x3x6 = 2x3x(2*3) and
    exploit the tensor product structure.
    
    The multiplication tensor for 2x3x6 is the Kronecker product of the 2x3x3
    tensor and the 1x1x2 identity tensor. So its rank is at most rank(2x3x3)*rank(1x1x2) = 7*2 = 14.
    
    But maybe I can do better. Let me try to construct a rank-12 decomposition directly.
    
    Actually, let me just use the Kronecker product construction with rank 14,
    and see if I can verify it works.
    """
    n, m, p = 2, 3, 6
    rank = 14
    
    # First, get a rank-7 decomposition for 2x3x3
    # I'll use a known valid construction
    
    U233 = np.zeros((6, 7))
    V233 = np.zeros((9, 7))
    W233 = np.zeros((6, 7))
    
    # Here's a valid rank-7 decomposition for 2x3x3:
    # The idea is to use 3 terms for each row of the output, plus 1 extra term.
    
    # Terms for C[0,0], C[0,1], C[0,2] (row 0 of output)
    # Term 0: A[0,0]*B[0,0] -> C[0,0]
    U233[0, 0] = 1; V233[0, 0] = 1; W233[0, 0] = 1
    # Term 1: A[0,1]*B[1,0] -> C[0,0]  
    U233[1, 1] = 1; V233[3, 1] = 1; W233[0, 1] = 1
    # Term 2: A[0,2]*B[2,0] -> C[0,0]
    U233[2, 2] = 1; V233[6, 2] = 1; W233[0, 2] = 1
    
    # Term 3: A[0,0]*B[0,1] -> C[0,1]
    U233[0, 3] = 1; V233[1, 3] = 1; W233[2, 3] = 1
    # Term 4: A[0,1]*B[1,1] -> C[0,1]
    U233[1, 4] = 1; V233[4, 4] = 1; W233[2, 4] = 1
    # Term 5: A[0,2]*B[2,1] -> C[0,1]
    U233[2, 5] = 1; V233[7, 5] = 1; W233[2, 5] = 1
    
    # Term 6: A[0,0]*B[0,2] -> C[0,2]
    U233[0, 6] = 1; V233[2, 6] = 1; W233[4, 6] = 1
    
    # Wait, this only covers C[0,*] entries. I need to also cover C[1,*].
    # This construction is incomplete. Let me try a different approach.
    
    # Let me use the fact that for 2x3x3, I can use the following valid rank-7 decomposition:
    # This is based on the Strassen-like algorithm adapted for 2x3x3.
    
    # Actually, let me just use a simpler approach: use 2 copies of a rank-7 
    # decomposition for 2x3x3, giving rank 14 for 2x3x6.
    
    # But first I need a valid rank-7 decomposition for 2x3x3.
    
    # Let me construct one properly. The multiplication tensor for 2x3x3 has
    # 18 entries. I need 7 rank-1 tensors that sum to it.
    
    # Here's a valid construction:
    # Use the following 7 terms:
    
    U233 = np.zeros((6, 7))
    V233 = np.zeros((9, 7))
    W233 = np.zeros((6, 7))
    
    # Term 0: (A[0,0]+A[0,1]+A[0,2]) * (B[0,0]+B[1,0]+B[2,0]) -> contributes to C[0,0]
    # But this also contributes to other entries. Let me be more careful.
    
    # Actually, let me use a known valid rank-7 decomposition from the literature.
    # For 2x3x3, one valid decomposition is:
    
    # I'll use the following approach: solve the system of equations.
    # The constraint is:
    # sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*2+i', r] = delta_{i,i'} for all i,j,k,i'
    
    # This is a system of 2*3*3*2 = 36 equations (but many are redundant).
    # Actually, it's 2*3 = 6 independent equations for each (j,k) pair, so 18 equations total.
    # Wait, let me think again.
    
    # The tensor T[i,j,k] = sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*2+i, r] should equal 1 for all i,j,k.
    # That's 2*3*3 = 18 equations.
    
    # And we have 6*7 + 9*7 + 6*7 = 42 + 63 + 42 = 147 variables, but with scaling invariance.
    
    # Let me try to construct a valid decomposition by solving the system.
    
    # Actually, let me just use a simple valid construction:
    
    # For 2x3x3, use the following 7 terms:
    
    # Term 0: A[0,0]*B[0,0] -> C[0,0]
    U233[0, 0] = 1; V233[0, 0] = 1; W233[0, 0] = 1
    
    # Term 1: A[0,1]*B[1,0] -> C[0,0]
    U233[1, 1] = 1; V233[3, 1] = 1; W233[0, 1] = 1
    
    # Term 2: A[0,2]*B[2,0] -> C[0,0]
    U233[2, 2] = 1; V233[6, 2] = 1; W233[0, 2] = 1
    
    # Term 3: A[1,0]*B[0,1] -> C[1,1]
    U233[3, 3] = 1; V233[1, 3] = 1; W233[3, 3] = 1
    
    # Term 4: A[1,1]*B[1,1] -> C[1,1]
    U233[4, 4] = 1; V233[4, 4] = 1; W233[3, 4] = 1
    
    # Term 5: A[1,2]*B[2,1] -> C[1,1]
    U233[5, 5] = 1; V233[7, 5] = 1; W233[3, 5] = 1
    
    # Term 6: A[0,0]*B[0,2] + A[1,0]*B[0,0] -> C[0,2] + C[1,0]
    # This is tricky. Let me use a different term.
    
    # Term 6: A[0,0]*B[0,2] -> C[0,2]
    U233[0, 6] = 1; V233[2, 6] = 1; W233[4, 6] = 1
    
    # But this doesn't cover C[1,0], C[1,2], C[0,1], C[0,2] properly.
    
    # I'm going in circles. Let me just use the schoolbook rank 36 and return that.
    
    return None

def construct():
    """
    Construct rank-minimal decomposition for 2x3x6 matrix multiplication.
    """
    # Try to find a rank-14 decomposition using Kronecker product of 2x3x3 (rank 7) and 1x1x2 (rank 2)
    # This gives rank 14, which is better than schoolbook rank 36.
    
    n, m, p = 2, 3, 6
    rank = 14
    
    # First, construct a rank-7 decomposition for 2x3x3
    # I'll use a valid construction based on known results
    
    U233 = np.zeros((6, 7))
    V233 = np.zeros((9, 7))
    W233 = np.zeros((6, 7))
    
    # Here's a valid rank-7 decomposition for 2x3x3:
    # Based on the structure of the problem, I'll use the following terms:
    
    # The key insight is that we can use 3 terms for the first row of output,
    # 3 terms for the second row of output, and 1 extra term to handle cross-terms.
    
    # Let me use a systematic construction:
    
    # Term 0: (A[0,0]+A[0,1]+A[0,2]) * (B[0,0]+B[1,0]+B[2,0]) * (C[0,0] term)
    # This is getting complicated. Let me just use the schoolbook and return rank 36.
    
    # Actually, let me try a different approach: use the fact that 2x3x6 can be
    # decomposed using 12 terms by exploiting the structure.
    
    # For now, let me just return the schoolbook rank 36.
    
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
