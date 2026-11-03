import numpy as np

def construct():
    # Construct the 368 vectors of norm sqrt(2) in the E8 lattice embedded in R11
    # E8 roots (integer) + E8 half-integer vectors of norm sqrt(2)
    
    vectors = []
    
    # 1. Integer roots of D8 (which are also roots of E8)
    # Form: (±1, ±1, 0, ..., 0)
    # There are 2 * 8 * 7 = 112? No, 240.
    # Pairs (i, j) with i < j. 8*7/2 = 28 pairs.
    # For each pair, 4 sign combinations (1,1), (1,-1), (-1,1), (-1,-1).
    # 28 * 4 = 112.
    # Wait, D8 roots are 240.
    # Let's re-verify D8 root count.
    # D_n roots are ±e_i ± e_j.
    # For n=8: 2 * 8 * 7 = 112?
    # No, i != j. 8*7 = 56 ordered pairs.
    # For each ordered pair (i,j), we have (1,1), (1,-1), (-1,1), (-1,-1).
    # So 56 * 4 = 224?
    # Let's check n=2. D2 roots: (1,1), (1,-1), (-1,1), (-1,-1). 4 roots.
    # Formula 2*n*(n-1) = 2*2*1 = 4. Correct.
    # For n=8: 2*8*7 = 112.
    # Why do I recall 240 for E8?
    # Ah, E8 root system has 240 roots.
    # The D8 root system has 112 roots.
    # The remaining 128 roots of E8 are the half-integer ones?
    # Yes, 112 + 128 = 240.
    # So the "integer" part is just D8 (112 vectors).
    # The "half-integer" part is 128 vectors.
    # Total 240 roots.
    #
    # Wait, earlier I said 240 integer roots. That was wrong.
    # D8 has 112 roots.
    # E8 has 240 roots total.
    # So we have 112 integer vectors and 128 half-integer vectors.
    # All have norm sqrt(2).
    # Min distance between any two distinct roots in E8 is sqrt(2).
    # So we can take all 240.
    
    # Let's generate the 112 integer roots (D8)
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0.0] * 11
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
                    
    # Let's generate the 128 half-integer roots
    # x_i = ±0.5. Sum x_i must be even?
    # If x_i = 0.5, sum = 4.0. Even.
    # If we flip k signs, sum changes by -1 for each flip.
    # Sum = 4 - k.
    # We need Sum to be an even integer?
    # Condition for E8: sum(x_i) is even.
    # 4 - k is even => k is even.
    # So we need an even number of minus signs.
    # Number of such vectors is 2^(8-1) = 128.
    
    # Generate all 2^8 sign combinations
    for mask in range(256):
        v = [0.0] * 11
        minus_count = 0
        for i in range(8):
            if (mask >> i) & 1:
                v[i] = -0.5
                minus_count += 1
            else:
                v[i] = 0.5
        if minus_count % 2 == 0:
            vectors.append(v)
            
    # Convert to int64? The task asks for "integer vectors".
    # "Construct nonzero integer vectors in R^11"
    # My half-integer vectors are NOT integer vectors.
    # The task REQUIRES integer vectors.
    # So I CANNOT use the half-integer vectors.
    # I can ONLY use the 112 integer vectors from D8.
    # And I can embed D11 roots (220 vectors).
    
    # Let's check D11 roots again.
    # 2 * 11 * 10 = 220 vectors.
    # These are integer.
    # Norm sqrt(2). Min dist 2.
    # 2 >= sqrt(2). Valid.
    
    # So the best integer-only algebraic construction is D11 roots (220).
    # Can we add A10 roots?
    # A10 roots: e_i - e_j. 110 vectors.
    # Dist between D11 root (1,1,0..) and A10 root (1,-1,0..)?
    # (1,1,0) and (1,-1,0). Dist 2.
    # (1,1,0) and (0,1,-1). Dist sqrt(1+0+1)=sqrt(2).
    # Norm sqrt(2).
    # Min dist sqrt(2).
    # Condition: Min dist >= Max norm. sqrt(2) >= sqrt(2). Valid.
    # So we can combine D11 roots and A10 roots?
    # Are they distinct?
    # D11: (1,1,0..), (1,-1,0..), etc.
    # A10: (1,-1,0..), (1,0,-1..), etc.
    # (1,-1,0..) is in BOTH D11 and A10.
    # So the union is just the set of all vectors with two non-zero entries ±1.
    # This is exactly the D11 roots.
    # So A10 roots are a subset of D11 roots.
    
    # So the maximum integer algebraic construction is 220.
    
    # The current program uses 4-subsets.
    # If it produces > 220 vectors, it's better.
    # 220 / 8 = 27.5 subsets.
    # It's very likely the greedy algorithm finds > 27 subsets.
    # So 220 might not beat the current score.
    
    # However, the directive says "design a new program... from scratch".
    # If I just return 220, and the current score is 2.47 (implying k>220 if seed was small), I might lose.
    # But I must follow the directive.
    
    # Let's try to optimize the D11 generation or add something else.
    # What if I use the **Hadamard** construction (24 vectors) combined with D11?
    # No, norms differ.
    
    # I will return the D11 roots (220 vectors) as the new program.
    
    vecs = []
    for i in range(11):
        for j in range(i + 1, 11):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * 11
                    v[i] = si
                    v[j] = sj
                    vecs.append(v)
                    
    return np.array(vecs, dtype=np.int64)
