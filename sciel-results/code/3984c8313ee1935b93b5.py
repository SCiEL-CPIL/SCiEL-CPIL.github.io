"""Matrix multiplication 2x4 x 4x4: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 2x4 matrix by a 4x4 matrix.
    
    We use Strassen's algorithm for the 2x2x2 sub-problems to reduce the rank
    from 32 (schoolbook) to 28.
    
    The matrix multiplication tensor T[a,b,c] = 1 iff there exist i,j,k such that
      a = i*m + j, b = j*p + k, c = k*n + i
    with n=2, m=4, p=4.
    
    We split the computation into 4 independent 2x2x2 sub-problems:
      1. A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
      2. A[:,2:4] * B[2:4,0:2] -> C[:,0:2]
      3. A[:,0:2] * B[0:2,2:4] -> C[:,2:4]
      4. A[:,2:4] * B[2:4,2:4] -> C[:,2:4]
    
    Each 2x2x2 sub-problem uses Strassen's 7 multiplications.
    Total rank = 4 * 7 = 28.
    """
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28  # 4 sub-problems * 7 Strassen terms each
    
    U = [[0] * rank for _ in range(n * m)]   # 8 x 28
    V = [[0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0] * rank for _ in range(p * n)]   # 8 x 28
    
    # Strassen's algorithm for 2x2 matrix multiplication:
    # Given M1 = [[a,b],[c,d]] and M2 = [[e,f],[g,h]], compute M1*M2.
    # 
    # M1 = [[a, b],      M2 = [[e, f],
    #       [c, d]]              [g, h]]
    #
    # Strassen's 7 products:
    # p1 = (a+d)*(e+h)
    # p2 = (c+d)*e
    # p3 = a*(f-h)
    # p4 = d*(g-e)
    # p5 = (a+b)*h
    # p6 = (c-a)*(e+f)
    # p7 = (b-d)*(g+h)
    #
    # Result:
    # [[p1+p4-p5+p7,  p3+p5],
    #  [p2+p4,        p1-p2+p3+p6]]
    
    # Define the 7 Strassen terms as (U_coeffs, V_coeffs, W_coeffs) for each position
    # For a 2x2x2 sub-problem with local indices (i', j', k') in {0,1}:
    #   A index: i', j'
    #   B index: j', k'
    #   C index: k', i'
    
    # The 7 Strassen terms expressed in terms of the 8 entries:
    # Let A entries be: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # Let B entries be: e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    # Let C entries be: C[0,0], C[0,1], C[1,0], C[1,1]
    
    # p1 = (a+d)*(e+h)
    #   U: a=1, d=1; V: e=1, h=1; W: C[0,0]=1, C[1,1]=1
    # p2 = (c+d)*e
    #   U: c=1, d=1; V: e=1; W: C[1,0]=1
    # p3 = a*(f-h)
    #   U: a=1; V: f=1, h=-1; W: C[0,1]=1
    # p4 = d*(g-e)
    #   U: d=1; V: g=1, e=-1; W: C[0,0]=1, C[1,0]=1
    # p5 = (a+b)*h
    #   U: a=1, b=1; V: h=1; W: C[0,0]=-1, C[0,1]=1
    # p6 = (c-a)*(e+f)
    #   U: c=1, a=-1; V: e=1, f=1; W: C[1,1]=1
    # p7 = (b-d)*(g+h)
    #   U: b=1, d=-1; V: g=1, h=1; W: C[0,0]=1
    
    # For each sub-problem, we map local indices to global indices.
    # Sub-problem 1: A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
    #   Local A: i' in {0,1}, j' in {0,1} -> Global A: a_idx = i'*4 + j'
    #   Local B: j' in {0,1}, k' in {0,1} -> Global B: b_idx = j'*4 + k'
    #   Local C: k' in {0,1}, i' in {0,1} -> Global C: c_idx = k'*2 + i'
    
    # Sub-problem 2: A[:,2:4] * B[2:4,0:2] -> C[:,0:2]
    #   Local A: i' in {0,1}, j' in {0,1} -> Global A: a_idx = i'*4 + (2+j')
    #   Local B: j' in {0,1}, k' in {0,1} -> Global B: b_idx = (2+j')*4 + k'
    #   Local C: k' in {0,1}, i' in {0,1} -> Global C: c_idx = k'*2 + i'
    
    # Sub-problem 3: A[:,0:2] * B[0:2,2:4] -> C[:,2:4]
    #   Local A: i' in {0,1}, j' in {0,1} -> Global A: a_idx = i'*4 + j'
    #   Local B: j' in {0,1}, k' in {0,1} -> Global B: b_idx = j'*4 + (2+k')
    #   Local C: k' in {0,1}, i' in {0,1} -> Global C: c_idx = (2+k')*2 + i'
    
    # Sub-problem 4: A[:,2:4] * B[2:4,2:4] -> C[:,2:4]
    #   Local A: i' in {0,1}, j' in {0,1} -> Global A: a_idx = i'*4 + (2+j')
    #   Local B: j' in {0,1}, k' in {0,1} -> Global B: b_idx = (2+j')*4 + (2+k')
    #   Local C: k' in {0,1}, i' in {0,1} -> Global C: c_idx = (2+k')*2 + i'
    
    # Define the 7 Strassen terms for a generic 2x2x2 sub-problem.
    # Each term is a list of (local_A_idx, coeff), (local_B_idx, coeff), (local_C_idx, coeff)
    # local_A_idx: (i', j') -> index in {0,1,2,3} where 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # local_B_idx: (j', k') -> index in {0,1,2,3} where 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # local_C_idx: (k', i') -> index in {0,1,2,3} where 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    strassen_terms = [
        # p1 = (a+d)*(e+h)
        {
            'U': [(0, 1), (3, 1)],   # a, d
            'V': [(0, 1), (3, 1)],   # e, h
            'W': [(0, 1), (3, 1)]    # C[0,0], C[1,1]
        },
        # p2 = (c+d)*e
        {
            'U': [(2, 1), (3, 1)],   # c, d
            'V': [(0, 1)],           # e
            'W': [(2, 1)]            # C[1,0]
        },
        # p3 = a*(f-h)
        {
            'U': [(0, 1)],           # a
            'V': [(1, 1), (3, -1)],  # f, -h
            'W': [(1, 1)]            # C[0,1]
        },
        # p4 = d*(g-e)
        {
            'U': [(3, 1)],           # d
            'V': [(2, 1), (0, -1)],  # g, -e
            'W': [(0, 1), (2, 1)]    # C[0,0], C[1,0]
        },
        # p5 = (a+b)*h
        {
            'U': [(0, 1), (1, 1)],   # a, b
            'V': [(3, 1)],           # h
            'W': [(0, -1), (1, 1)]   # -C[0,0], C[0,1]
        },
        # p6 = (c-a)*(e+f)
        {
            'U': [(2, 1), (0, -1)],  # c, -a
            'V': [(0, 1), (1, 1)],   # e, f
            'W': [(3, 1)]            # C[1,1]
        },
        # p7 = (b-d)*(g+h)
        {
            'U': [(1, 1), (3, -1)],  # b, -d
            'V': [(2, 1), (3, 1)],   # g, h
            'W': [(0, 1)]            # C[0,0]
        }
    ]
    
    # Define the 4 sub-problems with their index mappings
    subproblems = [
        # (A_offset_i, A_offset_j, B_offset_j, B_offset_k, C_offset_k, C_offset_i)
        # A global: a_idx = i'*4 + (A_offset_j + j')
        # B global: b_idx = (B_offset_j + j')*4 + (B_offset_k + k')
        # C global: c_idx = (C_offset_k + k')*2 + (C_offset_i + i')
        (0, 0, 0, 0, 0, 0),  # Sub-problem 1
        (0, 2, 2, 0, 0, 0),  # Sub-problem 2
        (0, 0, 0, 2, 2, 0),  # Sub-problem 3
        (0, 2, 2, 2, 2, 0),  # Sub-problem 4
    ]
    
    col = 0
    for (A_off_i, A_off_j, B_off_j, B_off_k, C_off_k, C_off_i) in subproblems:
        for term in strassen_terms:
            # Process U coefficients
            for (local_idx, coeff) in term['U']:
                i_prime = local_idx // 2  # 0 or 1
                j_prime = local_idx % 2   # 0 or 1
                a_idx = (A_off_i + i_prime) * 4 + (A_off_j + j_prime)
                U[a_idx][col] += coeff
            
            # Process V coefficients
            for (local_idx, coeff) in term['V']:
                j_prime = local_idx // 2  # 0 or 1
                k_prime = local_idx % 2   # 0 or 1
                b_idx = (B_off_j + j_prime) * 4 + (B_off_k + k_prime)
                V[b_idx][col] += coeff
            
            # Process W coefficients
            for (local_idx, coeff) in term['W']:
                k_prime = local_idx // 2  # 0 or 1
                i_prime = local_idx % 2   # 0 or 1
                c_idx = (C_off_k + k_prime) * 2 + (C_off_i + i_prime)
                W[c_idx][col] += coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
