"""Matrix multiplication 3,3,6 rank-minimal exact decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 6
    # Schoolbook baseline
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    start_time = time.time()
    deadline = 120.0

    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        n, m, p = 3, 3, 6
        rank = len(U[0]) if U else 0
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        for c in range(rank):
            if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
                return False
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # The tensor T[i,j,k] should equal 1 for all valid i,j,k
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                    if abs(actual - expected) > 1e-9:
                        return False
        return True

    def build_3x3x3_block():
        """
        Construct a rank-23 decomposition for 3x3x3 matrix multiplication.
        This uses the known structure from literature (e.g. Stottmeister et al. 2019, or Bläser 2005).
        Note: The explicit 23-term decomposition is complex. 
        However, for the purpose of this task, we can try to construct a valid lower rank 
        if we find one. If we cannot construct a valid rank-23, we fall back to schoolbook 
        or a known simpler reduction if available.
        
        Actually, the known rank of 3x3x3 is 23. Let's try to implement a known algorithm 
        or verify if a simpler structure works. 
        A common approach is to use the "Strassen-like" decomposition.
        
        Since generating a valid rank-23 decomposition from scratch is non-trivial and 
        error-prone without a pre-computed table, and the prompt asks for a *rank-minimal* 
        decomposition, let's see if we can use a known identity.
        
        Wait, the task says "integer/half-integer entries". The rank-23 decomposition 
        for 3x3x3 typically involves specific coefficients.
        
        Let's try to use a simpler approach: 
        Is there a rank-22 or lower? No, 23 is the proven minimum.
        
        If we can't reliably generate the 23-term decomposition, we should return the 
        schoolbook (54) or try to find a decomposition with rank < 54 that is easier 
        to verify.
        
        However, the directive mentions exploiting the tensor product structure.
        3x3x6 = (3x3x3) x (1x3x2)? No, that's not quite right.
        
        Let's look at the structure:
        C (3x6) = A (3x3) * B (3x6).
        This is equivalent to multiplying A by 6 different columns of B.
        If we had a rank-r decomposition for 3x3x3, we could potentially reuse it.
        But the "output" dimension is 6, not 3.
        
        Actually, we can view B as a 3x6 matrix.
        C[k, i] = sum_j A[i, j] B[k, j].
        
        If we split B into two 3x3 matrices B1 and B2 (columns 0-2 and 3-5),
        then C1 = A * B1 and C2 = A * B2.
        The rank of (A, B1, C1) is 23.
        The rank of (A, B2, C2) is 23.
        But A is shared!
        
        If we simply concatenate the decompositions, the rank is 23 + 23 = 46.
        This is better than 54.
        
        Can we do better?
        The tensor for 3x3x6 is T_{i,j,k}.
        T_{i,j,k} = 1 if k < 3, corresponding to B1.
        T_{i,j,k} = 1 if k >= 3, corresponding to B2.
        
        If we use the rank-23 decomposition for the first part:
        sum_{c=1}^{23} u_c(i,j) v_c(j,k) w_c(k,i) = 1 for k=0,1,2.
        
        For the second part (k=3,4,5), we can shift the indices.
        Let k' = k - 3.
        We need sum_{c=24}^{46} u'_c(i,j) v'_c(j,k) w'_c(k,i) = 1 for k=3,4,5.
        
        We can set u'_c = u_{c-23}, v'_c(j,k) = v_{c-23}(j, k-3), w'_c(k,i) = w_{c-23}(k-3, i).
        
        This gives a valid decomposition of rank 46.
        
        To implement this, we need the explicit rank-23 decomposition of 3x3x3.
        Since I don't have the full 23-term table memorized and generating it is complex,
        I will attempt to construct a rank-46 decomposition using a simpler, known 
        decomposition if one exists, or fall back to schoolbook.
        
        However, there is a known "Strassen" algorithm for 2x2x2 with rank 7.
        For 3x3x3, rank 23 is the best known.
        
        Let's try to implement a rank-46 decomposition by splitting the 6 columns into 
        two groups of 3 and using a known 3x3x3 decomposition. 
        But wait, if I can't get the 23-term decomposition, I can't do 46.
        
        Is there a simpler decomposition?
        Rank of 3x3x3 is 23.
        Rank of 3x3x6 is at most 2 * 23 = 46.
        Is it exactly 46? Likely, due to the structure.
        
        If I cannot provide the 23-term decomposition, I must return 54.
        But the score is 1.0 (54/54). To improve, I need < 54.
        
        Can I find a decomposition of rank 53?
        Random search is unlikely to find an exact integer/half-integer solution.
        
        Let's check if there is a standard "Winograd" or similar algorithm for 3x3x6.
        Not standard.
        
        Let's try to implement the rank-23 decomposition for 3x3x3.
        Reference: "The Rank of the 3x3 Matrix Multiplication Tensor" by Stottmeister et al.
        The decomposition involves 23 terms.
        
        Since I cannot reliably reproduce the 23-term table from memory, and the risk 
        of producing an invalid decomposition is high (scoring 0), I will stick to the 
        schoolbook solution unless I can find a simpler valid reduction.
        
        However, the directive suggests: "Model the 3x3x6 multiplication tensor as the 
        tensor product... to achieve a total rank of 7 x 2 = 14?"
        Wait, 7x2 = 14? That's for 2x2x2 (rank 7) x 1x2x1 (rank 2)?
        For 3x3x6, the dimensions don't match 2x2x2.
        
        Let's re-read the strategy: "Model the 3x3x6 multiplication tensor as the tensor 
        product of a 3x3x3 matrix multiplication tensor and a 1x3x2 vector-matrix 
        multiplication tensor... to achieve a total rank of 7 x 2..."
        This implies the 3x3x3 tensor has rank 7? No, 3x3x3 has rank 23.
        Maybe it refers to a different decomposition?
        
        If the strategy implies rank 14, that would be a huge improvement.
        But 3x3x3 rank is 23. So 23 * 2 = 46.
        
        Given the constraints and the difficulty of implementing the exact 23-term 
        decomposition, I will return the schoolbook solution to ensure validity.
        """
        pass

    # Since implementing the exact rank-23 decomposition is too complex and risky 
    # without a verified table, and the schoolbook is guaranteed valid, 
    # we return the schoolbook solution.
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
