import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 3, 4
    
    # Try to find a rank-20 decomposition
    # Known: rank(M_{2,3,4}) = 20
    # Construction: use a specific rank-20 tensor decomposition
    
    # Index conventions:
    # A is 2x3: A[i,j] -> index i*3+j in U (i in {0,1}, j in {0,1,2})
    #   A[0,0]=0, A[0,1]=1, A[0,2]=2
    #   A[1,0]=3, A[1,1]=4, A[1,2]=5
    # B is 3x4: B[j,k] -> index j*4+k in V (j in {0,1,2}, k in {0,1,2,3})
    #   B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3
    #   B[1,0]=4, B[1,1]=5, B[1,2]=6, B[1,3]=7
    #   B[2,0]=8, B[2,1]=9, B[2,2]=10, B[2,3]=11
    # C is 2x4: C[i,k] -> index k*2+i in W (trace convention W[k*n+i])
    #   C[0,0]=0, C[0,1]=2, C[0,2]=4, C[0,3]=6
    #   C[1,0]=1, C[1,1]=3, C[1,2]=5, C[1,3]=7
    
    rank = 20
    U = [[0] * rank for _ in range(n * m)]  # 6 x 20
    V = [[0] * rank for _ in range(m * p)]  # 12 x 20
    W = [[0] * rank for _ in range(p * n)]  # 8 x 20
    
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val
    
    # Rank-20 construction for M_{2,3,4}
    # Based on the known optimal decomposition
    
    # Terms 0-6: Strassen-like for the "left" 2x2x2 block (A[:,0:2] * B[0:2,:])
    # But we need to handle the 3rd column of A and 3rd row of B specially
    
    # Let me use a known rank-20 construction:
    # The idea is to partition differently and share terms
    
    # A[0,0]=0, A[0,1]=1, A[0,2]=2
    # A[1,0]=3, A[1,1]=4, A[1,2]=5
    # B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3
    # B[1,0]=4, B[1,1]=5, B[1,2]=6, B[1,3]=7
    # B[2,0]=8, B[2,1]=9, B[2,2]=10, B[2,3]=11
    
    # C[0,0]=0, C[0,1]=2, C[0,2]=4, C[0,3]=6
    # C[1,0]=1, C[1,1]=3, C[1,2]=5, C[1,3]=7
    
    # Use a construction that achieves rank 20
    # This is based on the explicit rank-20 decomposition
    
    # Term 0: (A00+A11)(B00+B11) -> C00+C11
    add_term([(0,1),(4,1)], [(0,1),(5,1)], [(0,1),(3,1)], 0)
    
    # Term 1: (A10+A11)B00 -> C10-C11
    add_term([(3,1),(4,1)], [(0,1)], [(1,1),(3,-1)], 1)
    
    # Term 2: A00(B01-B11) -> C01+C11
    add_term([(0,1)], [(1,1),(5,-1)], [(2,1),(3,1)], 2)
    
    # Term 3: A11(B10-B00) -> C00+C10
    add_term([(4,1)], [(4,1),(0,-1)], [(0,1),(1,1)], 3)
    
    # Term 4: (A00+A01)B11 -> -C00+C01
    add_term([(0,1),(1,1)], [(5,1)], [(0,-1),(2,1)], 4)
    
    # Term 5: (A10-A00)(B00+B01) -> C11
    add_term([(3,1),(0,-1)], [(0,1),(1,1)], [(3,1)], 5)
    
    # Term 6: (A01-A11)(B10+B11) -> C00
    add_term([(1,1),(4,-1)], [(4,1),(5,1)], [(0,1)], 6)
    
    # Terms 7-13: Handle the right part B[:,2:4] with Strassen-like
    # But share with the left part where possible
    
    # Term 7: (A00+A11)(B02+B13) -> C02+C13
    add_term([(0,1),(4,1)], [(2,1),(7,1)], [(4,1),(7,1)], 7)
    
    # Term 8: (A10+A11)B02 -> C12-C13
    add_term([(3,1),(4,1)], [(2,1)], [(5,1),(7,-1)], 8)
    
    # Term 9: A00(B03-B13) -> C03+C13
    add_term([(0,1)], [(3,1),(7,-1)], [(6,1),(7,1)], 9)
    
    # Term 10: A11(B12-B02) -> C02+C12
    add_term([(4,1)], [(6,1),(2,-1)], [(4,1),(5,1)], 10)
    
    # Term 11: (A00+A01)B13 -> -C02+C03
    add_term([(0,1),(1,1)], [(7,1)], [(4,-1),(6,1)], 11)
    
    # Term 12: (A10-A00)(B02+B03) -> C13
    add_term([(3,1),(0,-1)], [(2,1),(3,1)], [(7,1)], 12)
    
    # Term 13: (A01-A11)(B12+B13) -> C02
    add_term([(1,1),(4,-1)], [(6,1),(7,1)], [(4,1)], 13)
    
    # Terms 14-19: Handle A[:,2] * B[2,:] (the cross terms)
    # A[0,2]=2, A[1,2]=5
    # B[2,0]=8, B[2,1]=9, B[2,2]=10, B[2,3]=11
    
    # We need 6 more terms (14-19) for 8 cross products, so we need to combine
    # A02*B20 -> C00, A02*B21 -> C01, A02*B22 -> C02, A02*B23 -> C03
    # A12*B20 -> C10, A12*B21 -> C11, A12*B22 -> C12, A12*B23 -> C13
    
    # Combine pairs:
    # Term 14: A02*B20 -> C00
    add_term([(2,1)], [(8,1)], [(0,1)], 14)
    
    # Term 15: A02*B21 -> C01
    add_term([(2,1)], [(9,1)], [(2,1)], 15)
    
    # Term 16: A12*B20 -> C10
    add_term([(5,1)], [(8,1)], [(1,1)], 16)
    
    # Term 17: A12*B21 -> C11
    add_term([(5,1)], [(9,1)], [(3,1)], 17)
    
    # Term 18: A02*B22 -> C02
    add_term([(2,1)], [(10,1)], [(4,1)], 18)
    
    # Term 19: A02*B23 -> C03
    add_term([(2,1)], [(11,1)], [(6,1)], 19)
    
    # But wait, I'm missing A12*B22 -> C12 and A12*B23 -> C13
    # I only have 20 terms total, and I've used all 20
    # Let me check if I can combine some of the cross terms
    
    # Actually, I need to reconsider. Let me verify the current count:
    # Terms 0-6: 7 terms (left Strassen)
    # Terms 7-13: 7 terms (right Strassen)  
    # Terms 14-19: 6 terms (cross)
    # Total: 20 terms
    # But I need 8 cross products, so I'm missing 2
    
    # Let me try a different approach: modify the Strassen blocks to absorb some cross terms
    # Or use a different partition entirely
    
    # Let me try a known rank-20 construction from the literature
    # The key is to use a non-trivial partition
    
    # Reset and try again with a different construction
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Use a construction based on the "bilinear complexity" approach
    # For M_{2,3,4}, rank = 20
    
    # I'll use a construction that treats the problem as:
    # Split B into B_left (3x2) and B_right (3x2)
    # But use shared Strassen terms
    
    # Actually, let me just verify: can I do it with 20 terms by being smarter?
    
    # The issue is that the cross terms A[:,2]*B[2,:] give 8 independent products
    # But some of these might be combinable with the Strassen terms
    
    # Let me try a completely different rank-20 construction
    
    # Based on explicit constructions in the literature:
    
    # Term 0: A00*B00 -> C00
    add_term([(0,1)], [(0,1)], [(0,1)], 0)
    # Term 1: A00*B01 -> C01  
    add_term([(0,1)], [(1,1)], [(2,1)], 1)
    # Term 2: A00*B02 -> C02
    add_term([(0,1)], [(2,1)], [(4,1)], 2)
    # Term 3: A00*B03 -> C03
    add_term([(0,1)], [(3,1)], [(6,1)], 3)
    # Term 4: A01*B10 -> C00
    add_term([(1,1)], [(4,1)], [(0,1)], 4)
    # Term 5: A01*B11 -> C01
    add_term([(1,1)], [(5,1)], [(2,1)], 5)
    # Term 6: A01*B12 -> C02
    add_term([(1,1)], [(6,1)], [(4,1)], 6)
    # Term 7: A01*B13 -> C03
    add_term([(1,1)], [(7,1)], [(6,1)], 7)
    # Term 8: A02*B20 -> C00
    add_term([(2,1)], [(8,1)], [(0,1)], 8)
    # Term 9: A02*B21 -> C01
    add_term([(2,1)], [(9,1)], [(2,1)], 9)
    # Term 10: A02*B22 -> C02
    add_term([(2,1)], [(10,1)], [(4,1)], 10)
    # Term 11: A02*B23 -> C03
    add_term([(2,1)], [(11,1)], [(6,1)], 11)
    # Term 12: A10*B00 -> C10
    add_term([(3,1)], [(0,1)], [(1,1)], 12)
    # Term 13: A10*B01 -> C11
    add_term([(3,1)], [(1,1)], [(3,1)], 13)
    # Term 14: A10*B02 -> C12
    add_term([(3,1)], [(2,1)], [(5,1)], 14)
    # Term 15: A10*B03 -> C13
    add_term([(3,1)], [(3,1)], [(7,1)], 15)
    # Term 16: A11*B10 -> C10
    add_term([(4,1)], [(4,1)], [(1,1)], 16)
    # Term 17: A11*B11 -> C11
    add_term([(4,1)], [(5,1)], [(3,1)], 17)
    # Term 18: A11*B12 -> C12
    add_term([(4,1)], [(6,1)], [(5,1)], 18)
    # Term 19: A11*B13 -> C13
    add_term([(4,1)], [(7,1)], [(7,1)], 19)
    
    # But this is just the direct 24-term multiplication with 4 terms missing
    # I'm missing: A12*B20, A12*B21, A12*B22, A12*B23
    # So this doesn't work - I only have 20 terms but need 24 for direct
    
    # The rank-20 construction must use Strassen-like savings
    # Let me go back to the 22-term solution and see if I can reduce by 2
    
    # Actually, I realize I should just return the working 22-term solution
    # since finding a valid rank-20 construction is non-trivial
    
    # Let me rebuild the known-good 22-term solution
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
    
    # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)
    
    # Cross terms: A[0:2, 2:3] * B[2:3, 0:4]
    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
    add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
    add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
    add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)
    
    # Verify half-integer entries
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if not (val == int(val) or val * 2 == int(val * 2)):
                    raise ValueError("Non-half-integer entry")
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
