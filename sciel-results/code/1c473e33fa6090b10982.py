import numpy as np
import itertools

D = 11

def construct():
    vectors = []
    
    # 1. Generate 240 roots of E8 in the first 8 dimensions
    # E8 roots are:
    # 112 vectors of type (+/-1, +/-1, 0, ..., 0) (D8 roots)
    # 128 vectors of type (+/-1/2, ..., +/-1/2) with even number of minuses (and sum 0? no, just parity)
    # Actually, the 128 half-integer roots of E8 are vectors in R^8 with all entries +/-1/2 such that the number of -1/2 is even.
    # Let's verify norm: 8 * (1/4) = 2. Correct.
    
    # D8 roots
    for i, j in itertools.combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                vectors.append(v)
                
    # E8 half-integer roots
    # All entries are +/- 0.5. Number of -0.5 is even.
    # Total vectors: 2^8 = 256. Half have even number of minuses?
    # Sum of (-1)^k where k is num minuses.
    # We want k even.
    # There are 128 such vectors.
    for signs in itertools.product([1, -1], repeat=8):
        if sum(1 for s in signs if s == -1) % 2 == 0:
            v = [0] * D
            for i in range(8):
                v[i] = signs[i] * 0.5
            # Convert to integer? The task says "integer vectors".
            # The task says "Construct nonzero integer vectors".
            # Half-integers are NOT integer vectors.
            # So I CANNOT use half-integer vectors.
            
    # Re-read task: "Construct nonzero integer vectors in R^11"
    # This means all components must be integers.
    # So I cannot use the half-integer roots of E8 or E9.
    # I am limited to vectors with integer coordinates.
    
    # The seed uses (1, 1, 0, ...) which are integers.
    # The max norm is sqrt(2).
    # Min distance is sqrt(2).
    
    # If I use only integer vectors, the possible norms are sqrt(k) for k = 1, 2, ...
    # If max norm is sqrt(2), all vectors must have norm squared <= 2.
    # Integer vectors with norm squared <= 2:
    # 1. (1, 0, ..., 0) type: norm 1.
    # 2. (1, 1, 0, ..., 0) type: norm sqrt(2).
    
    # If I include norm 1 vectors, the max norm is still sqrt(2) (if I have any norm sqrt(2) vectors).
    # But min distance must be >= sqrt(2).
    # Distance between (1,0,...) and (1,1,0,...) is 1.
    # 1 < sqrt(2). Invalid.
    # So I cannot mix norm 1 and norm sqrt(2) vectors if they are close.
    
    # If I use ONLY norm 1 vectors:
    # Max norm = 1.
    # Min distance must be >= 1.
    # Vectors like (1,0,...), (0,1,...), etc.
    # Distance between (1,0) and (0,1) is sqrt(2) >= 1.
    # Distance between (1,0) and (-1,0) is 2 >= 1.
    # So I can take all 2*11 = 22 vectors of type (e_i) and (-e_i).
    # Score 22. Worse than 220.
    
    # If I use ONLY norm sqrt(2) vectors:
    # These are (1, 1, 0, ...), (1, -1, 0, ...).
    # This is exactly the seed. Score 220.
    
    # Can I have max norm > sqrt(2)?
    # If I use vectors of norm squared 3, e.g., (1, 1, 1, 0, ...).
    # Max norm = sqrt(3).
    # Min distance must be >= sqrt(3).
    # Can I pack more vectors?
    # Total number of integer vectors with norm squared <= 3:
    # Norm 1: 22
    # Norm 2: 220
    # Norm 3: 11 * 4 * 2 = 88? (1,1,1,0..) permutations.
    #   11 choices for pos, 10 for pos2, 9 for pos3? No, 3 distinct non-zeros.
    #   Permutations of (1,1,1,0..) with signs.
    #   Choose 3 indices: C(11,3) = 165.
    #   Signs: 2^3 = 8.
    #   Total 165 * 8 = 1320.
    #   But we need min distance >= sqrt(3).
    
    # Let's try to construct a set with max norm squared 3.
    # If I take all 1320 vectors of norm sqrt(3), is min distance >= sqrt(3)?
    # Distance between (1,1,1,0..) and (1,1,-1,0..) is sqrt(0+0+4)=2 >= sqrt(3).
    # Distance between (1,1,1,0..) and (1,1,0,1..) is sqrt(0+0+1+1)=sqrt(2) < sqrt(3).
    # So I cannot take all.
    
    # The seed score is 220.
    # If I can't beat 220 with integer vectors, I should return the seed.
    
    return np.array(vectors, dtype=np.int64)
