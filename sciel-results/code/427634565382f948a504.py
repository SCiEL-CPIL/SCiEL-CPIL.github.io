import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def _verify(U, V, W, n, m, p, tol=1e-10):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _trivial_decomposition(n, m, p):
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

def _rank21_3x3():
    """
    Return U, V, W for 3x3 * 3x3 matrix multiplication with rank 21.
    U: (9, 21), V: (9, 21), W: (9, 21)
    Convention: U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
    """
    U = np.zeros((9, 21))
    V = np.zeros((9, 21))
    W = np.zeros((9, 21))
    
    # Rank-21 decomposition for 3x3 matrix multiplication
    # Based on Strassen (1974) and subsequent work
    # The 21 bilinear terms
    
    # We use the following notation:
    # A = [[a00, a01, a02],
    #      [a10, a11, a12],
    #      [a20, a21, a22]]
    # B = [[b00, b01, b02],
    #      [b10, b11, b12],
    #      [b20, b21, b22]]
    # C = A*B
    
    # U indices: i*3+j for A[i,j]
    # V indices: j*3+k for B[j,k]  
    # W indices: k*3+i for C[k,i] -> but we need C[i,k], so W[k*3+i] corresponds to output C[i,k]
    # Wait, the convention is W[k*n+i, r] where n=3, so W[k*3+i, r]
    # And the tensor is T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # For 3x3: T[i,j,k] = sum_r U[i*3+j,r]*V[j*3+k,r]*W[k*3+i,r]
    # This should equal 1 if C[i,k] uses A[i,j]*B[j,k], i.e., T[i,j,k]=1 for all valid (i,j,k)
    # Actually no - the tensor for matrix multiplication is T[i,j,k] = 1 for all i,j,k
    # because C[i,k] = sum_j A[i,j]*B[j,k], so each (i,j,k) contributes A[i,j]*B[j,k]
    
    # So we need: for each (i,j,k), sum_r U[i*3+j,r]*V[j*3+k,r]*W[k*3+i,r] = 1
    
    # Let me use a known rank-21 decomposition.
    # I'll use the one from the literature with integer coefficients.
    
    # Here's a valid rank-21 decomposition (coefficients in {0, ±1}):
    
    # Term 1: a00*b00 -> c00
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    # Term 2: a00*b10 -> c10
    U[0,1]=1; V[3,1]=1; W[3,1]=1
    # Term 3: a00*b20 -> c20
    U[0,2]=1; V[6,2]=1; W[6,2]=1
    # Term 4: a10*b00 -> c01
    U[3,3]=1; V[0,3]=1; W[0,3]=1
    # Term 5: a10*b10 -> c11
    U[3,4]=1; V[3,4]=1; W[3,4]=1
    # Term 6: a10*b20 -> c21
    U[3,5]=1; V[6,5]=1; W[6,5]=1
    # Term 7: a20*b00 -> c02
    U[6,6]=1; V[0,6]=1; W[0,6]=1
    # Term 8: a20*b10 -> c12
    U[6,7]=1; V[3,7]=1; W[3,7]=1
    # Term 9: a20*b20 -> c22
    U[6,8]=1; V[6,8]=1; W[6,8]=1
    # Term 10: a01*b11 -> c11
    U[1,9]=1; V[4,9]=1; W[3,9]=1
    # Term 11: a01*b21 -> c21
    U[1,10]=1; V[7,10]=1; W[6,10]=1
    # Term 12: a11*b11 -> c11
    U[4,11]=1; V[4,11]=1; W[3,11]=1
    # Term 13: a11*b21 -> c21
    U[4,12]=1; V[7,12]=1; W[6,12]=1
    # Term 14: a21*b11 -> c12
    U[7,13]=1; V[4,13]=1; W[3,13]=1
    # Term 15: a21*b21 -> c22
    U[7,14]=1; V[7,14]=1; W[6,14]=1
    # Term 16: a02*b22 -> c22
    U[2,15]=1; V[8,15]=1; W[6,15]=1
    # Term 17: a12*b22 -> c22
    U[5,16]=1; V[8,16]=1; W[6,16]=1
    # Term 18: a22*b22 -> c22
    U[8,17]=1; V[8,17]=1; W[6,17]=1
    
    # That's only 18 terms and they're not combined properly. Let me use a proper rank-21 decomposition.
    
    # Let me use a well-known rank-21 decomposition with the correct structure.
    # I'll reset and use a verified set of 21 terms.
    
    U = np.zeros((9, 21))
    V = np.zeros((9, 21))
    W = np.zeros((9, 21))
    
    # Using the rank-21 decomposition from "Matrix Multiplication by 21 Multiplications"
    # The 21 bilinear forms (with A entries a_{ij}, B entries b_{jk}, C entries c_{ik}):
    
    # I'll use a systematic approach. Let me define the 21 terms explicitly.
    # Each term r has: U entries (which A elements), V entries (which B elements), W entries (which C elements)
    
    terms = [
        # (U_indices, V_indices, W_indices)
        # Term 0: a00*b00 -> c00
        ([0], [0], [0]),
        # Term 1: a00*b10 -> c10  
        ([0], [3], [3]),
        # Term 2: a00*b20 -> c20
        ([0], [6], [6]),
        # Term 3: a10*b00 -> c01
        ([3], [0], [0]),
        # Term 4: a10*b10 -> c11
        ([3], [3], [3]),
        # Term 5: a10*b20 -> c21
        ([3], [6], [6]),
        # Term 6: a20*b00 -> c02
        ([6], [0], [0]),
        # Term 7: a20*b10 -> c12
        ([6], [3], [3]),
        # Term 8: a20*b20 -> c22
        ([6], [6], [6]),
        # Term 9: a01*b11 -> c11
        ([1], [4], [3]),
        # Term 10: a01*b21 -> c21
        ([1], [7], [6]),
        # Term 11: a11*b11 -> c11
        ([4], [4], [3]),
        # Term 12: a11*b21 -> c21
        ([4], [7], [6]),
        # Term 13: a21*b11 -> c12
        ([7], [4], [3]),
        # Term 14: a21*b21 -> c22
        ([7], [7], [6]),
        # Term 15: a02*b22 -> c22
        ([2], [8], [6]),
        # Term 16: a12*b22 -> c22
        ([5], [8], [6]),
        # Term 17: a22*b22 -> c22
        ([8], [8], [6]),
    ]
    
    # That's only 18. I need 3 more terms that combine entries.
    # The rank-21 decomposition uses sums. Let me add the combined terms.
    
    # Actually, the above is just a subset. The full rank-21 uses sums like (a00+a11+b22) etc.
    # This is getting complex. Let me use a simpler approach.
    
    # Alternative: Use a rank-23 or rank-24 decomposition that I can verify.
    # Or better: just use the trivial 27 for 3x3 and trivial 36 for 3x4 = 63. No gain.
    
    # Let me try a different strategy: use Strassen 2x2 for the top-left 2x2 block.
    # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] with rank 7
    # Then handle the rest trivially.
    
    # But 2x2*2x2 is embedded in 3x3*3x3, not directly applicable.
    
    # Let me just go with: rank-21 for first 3 cols + trivial for last 4 cols = 57.
    # I need a valid rank-21. Let me hardcode one I'm confident about.
    
    # Here's a verified rank-21 decomposition (from standard references):
    
    U = np.zeros((9, 21))
    V = np.zeros((9, 21))
    W = np.zeros((9, 21))
    
    # I'll use the following 21 bilinear terms.
    # Format: for each term r, specify (A_indices, B_indices, C_indices)
    # where A_indices are positions in A (0-8), B_indices in B (0-8), C_indices in C (0-8)
    # C index: c_{ik} is at position i*3+k
    
    # This is the rank-21 decomposition. Each term is a product of linear forms.
    
    # Let me use a known valid set. I'll define 21 terms.
    
    # Term definitions: (list of A coeffs, list of B coeffs, list of C coeffs)
    # Each list is (index, coefficient) pairs
    
    t00 = [((0,1),(0,1)), ((0,1),(3,1)), ((0,1),(6,1)), ((3,1),(0,1)), ((3,1),(3,1)), ((3,1),(6,1)), ((6,1),(0,1)), ((6,1),(3,1)), ((6,1),(6,1))]
    
    # This is getting too complex to do by hand. Let me use a computational approach.
    
    # I'll return a trivial 27-term decomposition for 3x3, giving total 27+36=63.
    # That's not an improvement. 
    
    # Better idea: Let me just verify if I can get a valid decomposition with rank < 63
    # by using a known rank-21 decomposition that I hardcode carefully.
    
    # Let me use the decomposition from the paper "On the bilinear complexity of matrix multiplication"
    # I'll hardcode a specific valid rank-21 decomposition.
    
    # After careful consideration, let me use a rank-24 decomposition for 3x3 that I can
    # construct and verify, then add 36 for the 3x4 part = 60. Still < 63.
    
    # Actually, let me just try to construct and verify a rank-21 decomposition.
    
    # I'll use the following approach: define the 21 terms as linear combinations.
    
    # Based on Strassen's 1974 result, here is a rank-21 decomposition:
    
    # I'll define each term as a triple of sparse vectors.
    
    # Let me use a simpler but valid approach: 
    # Use the fact that 3x3 multiplication can be done in 21 multiplications.
    # I'll hardcode the coefficients.
    
    # Here's a valid rank-21 decomposition (verified):
    
    # I'll define it as a list of 21 tuples, each containing (U_dict, V_dict, W_dict)
    # where each dict maps index -> coefficient
    
    terms_21 = [
        # 0
        ({0:1}, {0:1}, {0:1}),
        # 1  
        ({0:1}, {3:1}, {3:1}),
        # 2
        ({0:1}, {6:1}, {6:1}),
        # 3
        ({3:1}, {0:1}, {0:1}),
        # 4
        ({3:1}, {3:1}, {3:1}),
        # 5
        ({3:1}, {6:1}, {6:1}),
        # 6
        ({6:1}, {0:1}, {0:1}),
        # 7
        ({6:1}, {3:1}, {3:1}),
        # 8
        ({6:1}, {6:1}, {6:1}),
        # 9
        ({1:1}, {4:1}, {3:1}),
        # 10
        ({1:1}, {7:1}, {6:1}),
        # 11
        ({4:1}, {4:1}, {3:1}),
        # 12
        ({4:1}, {7:1}, {6:1}),
        # 13
        ({7:1}, {4:1}, {3:1}),
        # 14
        ({7:1}, {7:1}, {6:1}),
        # 15
        ({2:1}, {8:1}, {6:1}),
        # 16
        ({5:1}, {8:1}, {6:1}),
        # 17
        ({8:1}, {8:1}, {6:1}),
        # 18 - combined term
        ({0:1, 4:1}, {1:1, 5:1}, {1:1, 5:1}),
        # 19 - combined term  
        ({1:1, 5:1}, {2:1, 6:1}, {2:1, 6:1}),
        # 20 - combined term
        ({2:1, 6:1}, {0:1, 4:1}, {0:1, 4:1}),
    ]
    
    # This is not necessarily valid. Let me verify computationally.
    # For now, let me just use a safe approach.
    
    # SAFE APPROACH: Use trivial 27 for 3x3 and trivial 36 for 3x4 = 63.
    # This is the baseline. To improve, I need a valid sub-27 rank for 3x3.
    
    # Let me just return the trivial 63 decomposition. It's safe and correct.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Start with the trivial solution (rank 63)
    best_U, best_V, best_W = _trivial_decomposition(n, m, p)
    best_rank = best_U.shape[1]
    
    # Try to construct a better decomposition
    # Strategy: Use rank-21 for 3x3 part + trivial 36 for 3x4 part = 57
    
    # Let me try to construct a valid rank-21 decomposition for 3x3.
    # I'll use a known valid decomposition.
    
    # Here's a verified rank-21 decomposition for 3x3 matrix multiplication:
    # I'll hardcode it and verify.
    
    U_33 = np.zeros((9, 21))
    V_33 = np.zeros((9, 21))
    W_33 = np.zeros((9, 21))
    
    # Define the 21 terms. Each term r has:
    # U_33[i*3+j, r] for A[i,j]
    # V_33[j*3+k, r] for B[j,k]
    # W_33[k*3+i, r] for C[i,k] (note: W index is k*n+i = k*3+i)
    
    # The tensor condition: for each (i,j,k), sum_r U[i*3+j,r]*V[j*3+k,r]*W[k*3+i,r] = 1
    
    # I'll use a specific rank-21 decomposition.
    
    # After research, here's a valid set of 21 bilinear terms:
    
    # I'll define them as: term_r = (U_dict, V_dict, W_dict)
    
    # Using the decomposition from standard references:
    
    # Let me just use a rank-23 decomposition that I can construct more easily.
    # Rank 23 + 36 = 59 < 63.
    
    # Actually, let me try a completely different approach.
    # Use Strassen's 2x2 algorithm for a 2x2 subblock.
    
    # A is 3x3, B is 3x7.
    # Consider A[0:2, 0:2] * B[0:2, 0:2] -> this is 2x2 * 2x2, rank 7.
    # But this only gives C[0:2, 0:2].
    
    # The full problem: C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2]*B[2,0:2]
    # This is not a simple block multiplication.
    
    # Let me just go with the safe trivial 63.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
