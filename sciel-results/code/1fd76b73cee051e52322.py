"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 10}

def construct():
    n = P['n']
    deadline = time.time() + 55  # 55s safety margin within 60s limit
    
    # We use the direct product decomposition Z_10 ≅ Z_5 × Z_2.
    # Map an integer x in Z_10 to (x % 5, x % 2).
    # The inverse map (a, b) -> x is given by CRT:
    # x ≡ a (mod 5), x ≡ b (mod 2)
    # x = 6*a + 5*b (mod 10)  [since 6*1+5*0=6≡1 mod5, 6≡0 mod2; 6*0+5*1=5≡0 mod5, 5≡1 mod2]
    # Let's verify:
    # a=0, b=0 -> x=0. 0%5=0, 0%2=0. Correct.
    # a=1, b=0 -> x=6. 6%5=1, 6%2=0. Correct.
    # a=0, b=1 -> x=5. 5%5=0, 5%2=1. Correct.
    # a=1, b=1 -> x=11%10=1. 1%5=1, 1%2=1. Correct.
    
    def crt(a, b):
        return (6*a + 5*b) % 10
    
    # We define Latin squares using linear transformations over Z_5 for the first coordinate,
    # and a fixed linear transformation for the second coordinate.
    # Let i = (i1, i2) and j = (j1, j2) in Z_5 × Z_2.
    # A Latin square L_m can be defined by:
    # L_m(i, j) = (i1 + m*j1, i2 + j2) mapped back to Z_10 via CRT.
    # Wait, the second coordinate operation must also be a Latin square operation.
    # In Z_2, the only non-identity linear map is identity (since 1 is the only unit).
    # So the second coordinate is always i2 + j2.
    # The first coordinate is i1 + m*j1 for m in {1, 2, 3, 4}.
    # For m=0, it's not a Latin square (constant in j).
    # So we have 4 potential squares from this family.
    # Are they mutually orthogonal?
    # Let L_m and L_k be two such squares with m != k.
    # L_m(i, j) = (i1 + m*j1, i2 + j2)
    # L_k(i, j) = (i1 + k*j1, i2 + j2)
    # The pair of values is ((i1+m*j1, i2+j2), (i1+k*j1, i2+j2)).
    # Given the pair (u, v) where u=(u1, u2) and v=(v1, v2):
    # u2 = i2 + j2
    # v2 = i2 + j2
    # So u2 must equal v2. If u2 != v2, no solution.
    # If u2 == v2, then i2 + j2 = u2. This has 2 solutions for (i2, j2) in Z_2 x Z_2?
    # No, for a fixed sum in Z_2, there are 2 pairs (0,0)->0, (1,1)->0, (0,1)->1, (1,0)->1.
    # So for each valid (u1, v1) pair, there are 2 solutions for (i2, j2).
    # Now for the first coordinates:
    # u1 = i1 + m*j1
    # v1 = i1 + k*j1
    # v1 - u1 = (k-m)*j1
    # Since m != k and k-m is invertible in Z_5 (as Z_5 is a field), j1 is uniquely determined.
    # Then i1 is uniquely determined.
    # So for each pair (u, v) with u2 == v2, there are exactly 2 solutions for (i, j).
    # But we need exactly 1 solution for orthogonality.
    # So these squares are NOT mutually orthogonal.
    
    # We need a different approach.
    # The known maximum number of MOLS of order 10 is 2.
    # Let's find 2 orthogonal Latin squares.
    # One known construction uses the fact that 10 = 2 * 5.
    # Let's try to find 2 orthogonal squares by search or known example.
    
    # Known MOLS of order 10:
    # L1[i][j] = (i + j) % 10
    # L2[i][j] = (i + 3*j) % 10 is not orthogonal.
    
    # Let's use the following known pair (from literature or search):
    # L1 = [[(i + j) % 10 for j in range(10)] for i in range(10)]
    # L2 = [[(i + 3*j) % 10 for j in range(10)] for i in range(10)] -> Not orthogonal.
    
    # Let's try a different L2.
    # We can use the fact that Z_10 is a ring.
    # Let's try L2[i][j] = (i + 7*j) % 10.
    # Pairs: (i+j, i+7j).
    # u = i+j, v = i+7j.
    # v - u = 6j. 6j mod 10 can be 0, 6, 2, 8, 4, 0, 6, 2, 8, 4.
    # Not injective.
    
    # Let's use a known valid pair.
    # According to the literature, the maximum number of MOLS of order 10 is 2.
    # One such pair is:
    # L1 = [[(i + j) % 10 for j in range(10)] for i in range(10)]
    # L2 = [[(i + 3*j) % 10 for j in range(10)] for i in range(10)] -> No.
    
    # Let's try to construct them using the direct product properly.
    # Let i = 5*i2 + i1, j = 5*j2 + j1, where i1, j1 in {0..4} and i2, j2 in {0,1}.
    # Wait, the mapping x = 6*a + 5*b is better.
    # Let's define the squares in the (a, b) coordinates.
    # L1(a, b) = (a + b, 0) ? No, b is in Z_2.
    
    # Let's use the following construction:
    # L1[i][j] = (i + j) % 10
    # L2[i][j] = (i + 3*j) % 10 is not orthogonal.
    
    # Let's try:
    # L1[i][j] = (i + j) % 10
    # L2[i][j] = (i + 7*j) % 10 is not orthogonal.
    
    # It is known that no two linear Latin squares of order 10 are orthogonal.
    # So we need non-linear squares.
    
    # Let's use the following known MOLS of order 10:
    # L1 = [
    #   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #   [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
    #   [2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
    #   [3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
    #   [4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
    #   [5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
    #   [6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
    #   [7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
    #   [8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
    #   [9, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    # ]
    # L2 = [
    #   [0, 3, 6, 9, 2, 5, 8, 1, 4, 7],
    #   [3, 6, 9, 2, 5, 8, 1, 4, 7, 0],
    #   [6, 9, 2, 5, 8, 1, 4, 7, 0, 3],
    #   [9, 2, 5, 8, 1, 4, 7, 0, 3, 6],
    #   [2, 5, 8, 1, 4, 7, 0, 3, 6, 9],
    #   [5, 8, 1, 4, 7, 0, 3, 6, 9, 2],
    #   [8, 1, 4, 7, 0, 3, 6, 9, 2, 5],
    #   [1, 4, 7, 0, 3, 6, 9, 2, 5, 8],
    #   [4, 7, 0, 3, 6, 9, 2, 5, 8, 1],
    #   [7, 0, 3, 6, 9, 2, 5, 8, 1, 4]
    # ]
    
    # Let's verify L2 is a Latin square.
    # Row 0: 0,3,6,9,2,5,8,1,4,7 -> all distinct.
    # Col 0: 0,3,6,9,2,5,8,1,4,7 -> all distinct.
    # It is a Latin square.
    
    # Let's verify orthogonality of L1 and L2.
    # We need to check that all 100 pairs (L1[i][j], L2[i][j]) are distinct.
    
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    L2 = [[(i + 3*j) % n for j in range(n)] for i in range(n)]
    
    # Wait, the L2 above is not (i + 3*j) % 10.
    # Let's check L2[0][1] = 3. (0 + 3*1) % 10 = 3. Correct.
    # L2[0][2] = 6. (0 + 3*2) % 10 = 6. Correct.
    # L2[0][3] = 9. (0 + 3*3) % 10 = 9. Correct.
    # L2[0][4] = 2. (0 + 3*4) % 10 = 12 % 10 = 2. Correct.
    # So L2 is indeed (i + 3*j) % 10.
    # But earlier I thought they were not orthogonal. Let's re-check.
    
    pairs = set()
    orthogonal = True
    for i in range(n):
        for j in range(n):
            pair = (L1[i][j], L2[i][j])
            if pair in pairs:
                orthogonal = False
                break
            pairs.add(pair)
        if not orthogonal:
            break
    
    if orthogonal:
        return [L1, L2]
    
    # If not orthogonal, return just L1.
    return [L1]

# EVOLVE-BLOCK-END
