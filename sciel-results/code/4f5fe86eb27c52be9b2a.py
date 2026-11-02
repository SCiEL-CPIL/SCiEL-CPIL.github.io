import time

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    
    # We'll build U (4 x rank), V (18 x rank), W (18 x rank)
    # U index: i*m+j = i*2+j  (a=0, b=1, c=2, d=3)
    # V index: j*p+k = j*9+k  (B[0,k]=k, B[1,k]=9+k)
    # W index: k*n+i = k*2+i  (C[0,k]=2k, C[1,k]=2k+1)
    
    terms = []  # Each term is (U_dict, V_dict, W_dict)
    
    # 2x2x3 decomposition: rank 9
    # For columns k0, k1, k2 (consecutive):
    # B[0,k0]=k0, B[0,k1]=k1, B[0,k2]=k2
    # B[1,k0]=9+k0, B[1,k1]=9+k1, B[1,k2]=9+k2
    # C[0,k0]=2k0, C[1,k0]=2k0+1, C[0,k1]=2k1, C[1,k1]=2k1+1, C[0,k2]=2k2, C[1,k2]=2k2+1
    #
    # Decomposition (9 terms):
    # M1=(a+d)(B0k0+B1k0) -> C[0,k0]+, C[1,k1]+
    # M2=(c+d)B0k0 -> C[1,k0]+, C[1,k1]-
    # M3=a(B0k1-B1k1) -> C[0,k1]+, C[1,k1]+
    # M4=d(B1k0-B0k0) -> C[0,k0]+, C[1,k0]+
    # M5=(a+b)B1k1 -> C[0,k0]-, C[0,k1]+
    # M6=(c-a)(B0k0+B0k1) -> C[1,k1]+
    # M7=(b-d)(B1k0+B1k1) -> C[0,k0]+
    # M8=a*B0k2 -> C[0,k2]
    # M9=d*B1k2 -> C[1,k2]
    # Plus we need b*B1k2 and c*B0k2... that's 11 terms.
    
    # Let me try a different rank-9 approach:
    # Actually for 2x2x3, minimum rank is 9. Let me use:
    # M1=(a+d)(e+h) -> C[1,0]+, C[0,1]+  [note: e=B[0,0], h=B[1,0]]
    # Wait, I need to be more careful.
    
    # Let me use a known rank-9 decomposition for 2x2x3:
    # A=[[a,b],[c,d]], B=[[e,f,g],[h,i,j]], C=[[ae+bh,af+bi,ag+bj],[ce+dh,cf+di,cg+dj]]
    #
    # M1=(a+d)(e+h) -> C[0,0]+, C[1,1]+  ... this is getting confused.
    
    # Let me just use 3 Strassen blocks of rank 7 for columns (0,1),(2,3),(4,5) 
    # and schoolbook for (6,7,8) with rank 6, total = 3*7+6 = 27
    # But schoolbook for 2x2x3 is rank 6 (each of 6 C entries needs one term if we can't share)
    # Actually schoolbook for 2x2x3: C has 6 entries, each is sum of 2 products, so 12 terms.
    # That's worse.
    
    # Let me try: 3 blocks of (2 columns Strassen rank 7) + 1 block of (3 columns, rank 9)
    # = 3*7 + 9 = 30. Still not as good as 32.
    
    # Or: find a rank-9 decomposition for 2x2x3 and use 3 of them = 27.
    
    # Here's a rank-9 decomposition for 2x2x3:
    # Columns: k0, k1, k2
    # e=B[0,k0], f=B[0,k1], g=B[0,k2], h=B[1,k0], i=B[1,k1], j=B[1,k2]
    #
    # M1=(a+d)(e+h) -> C[0,k0]+, C[1,k1]+
    # M2=(c+d)e -> C[1,k0]+, C[1,k1]-
    # M3=a(f-h) -> C[0,k1]+, C[1,k1]+  
    # M4=d(h-e) -> C[0,k0]+, C[1,k0]+
    # M5=(a+b)i -> C[0,k0]-, C[0,k1]+
    # M6=(c-a)(e+f) -> C[1,k1]+
    # M7=(b-d)(h+i) -> C[0,k0]+
    # M8=a*g -> C[0,k2]
    # M9=c*g + d*j -> C[1,k2]  ... but this needs two U entries for one term
    
    # The issue: for column k2, we need:
    # C[0,k2] = a*g + b*j
    # C[1,k2] = c*g + d*j
    # This needs at least 4 terms in schoolbook, or we can share.
    
    # Let me try: M8=(a+b)(g+j) -> C[0,k2]+, C[1,k2]+
    # M9=(c-a)g -> C[1,k2]+, C[0,k2]-  ... no
    # M9=(c-a)g -> C[1,k2]+, and we need to fix C[0,k2]
    
    # C[0,k2] = a*g + b*j
    # C[1,k2] = c*g + d*j
    #
    # M8=(a+b)(g+j) = a*g+a*j+b*g+b*j -> C[0,k2]+, C[1,k2]+ (partial)
    # M9=(c-a)g = c*g-a*g -> C[1,k2]+, C[0,k2]- (partial)
    # M10=(d-b)j = d*j-b*j -> C[1,k2]+, C[0,k2]- (partial)
    #
    # Sum: C[0,k2]: (a*g+a*j+b*g+b*j) + (-a*g) + (-b*j) = a*j+b*g ... not right
    #
    # Let me try:
    # M8=a*(g+j) -> C[0,k2]+, C[1,k2] no...
    #
    # I'll use schoolbook for the 3rd column: 4 terms
    # So each 2x2x3 block: 7 (Strassen for first 2 cols) + 4 (schoolbook for 3rd col) = 11
    # 3 blocks: 33. Worse.
    
    # Let me try a completely different grouping.
    # Use Strassen on all 9 columns by pairing them cleverly.
    # 9 columns: pair (0,1),(2,3),(4,5),(6,7) with Strassen (4*7=28) + col 8 schoolbook (4) = 32.
    # That's the current solution.
    
    # Can I do better? What if I use a rank-8 decomposition for 2x2x3?
    # The rank of 2x2x3 is known to be 9 (not 8).
    
    # So the best I can do with 3 blocks of 2x2x3 is 3*9=27 if I find rank-9 decomp.
    # Let me carefully construct rank-9 for 2x2x3.
    
    # After careful analysis, here's a rank-9 decomposition:
    # M1=(a+d)(e+h) -> C[0,k0]+, C[1,k1]+
    # M2=(c+d)e -> C[1,k0]+, C[1,k1]-
    # M3=a(f-h) -> C[0,k1]+, C[1,k1]+
    # M4=d(h-e) -> C[0,k0]+, C[1,k0]+
    # M5=(a+b)i -> C[0,k0]-, C[0,k1]+
    # M6=(c-a)(e+f) -> C[1,k1]+
    # M7=(b-d)(h+i) -> C[0,k0]+
    # M8=(a+b)(g+j) -> C[0,k2]+, C[1,k2]+
    # M9=(c-a)g + (d-b)j -> C[1,k2]+, C[0,k2]-
    #
    # Let me verify M8+M9 for column k2:
    # M8: (a+b)(g+j) = ag+aj+bg+bj
    # M9: (c-a)g+(d-b)j = cg-ag+dj-bj
    # Sum: ag+aj+bg+bj+cg-ag+dj-bj = aj+bg+cg+dj
    # C[0,k2] should be: ag+bj
    # C[1,k2] should be: cg+dj
    # Sum gives: C[0,k2]+C[1,k2] = aj+bg+cg+dj ... not separable.
    
    # This approach doesn't work cleanly. Let me just stick with the current rank-32 solution
    # and try to optimize within that family.
    
    # Actually, let me try: 4 Strassen blocks for (0,1),(2,3),(4,5),(6,7) = 28 terms
    # For column 8: use 2 terms instead of 4 by sharing:
    # C[0,8] = a*B[0,8] + b*B[1,8]
    # C[1,8] = c*B[0,8] + d*B[1,8]
    # M1=(a+b)(B[0,8]+B[1,8]) -> C[0,8]+, C[1,8]+
    # M2=(c-a)B[0,8] -> C[1,8]+, C[0,8]-
    # M3=(d-b)B[1,8] -> C[1,8]+, C[0,8]-
    # Check: C[0,8]: (a+b)(B08+B18) - (c-a)B08 - (d-b)B18
    #   = aB08+aB18+bB08+bB18 - cB08+aB08 - dB18+bB18
    #   = (2a-b-c)B08 + (a+2b-d)B18 ... not right.
    
    # M1=(a+b)(B[0,8]+B[1,8]) -> C[0,8]+, C[1,8]+
    # M2=(c-a)B[0,8] -> C[1,8]+, C[0,8]-
    # M3=(d-b)B[1,8] -> C[1,8]+, C[0,8]-
    # C[0,8] = M1 - M2 - M3 = (a+b)(B08+B18) - (c-a)B08 - (d-b)B18
    #   = aB08+aB18+bB08+bB18 - cB08+aB08 - dB18+bB18
    #   = (2a+b-c)B08 + (a+2b-d)B18
    # Should be: aB08+bB18. Not matching.
    
    # I'll keep the current rank-32 solution. It's already optimal for this approach.
    
    terms = []
    
    # Strassen blocks for k-pairs: (0,1), (2,3), (4,5), (6,7)
    for block_idx in range(4):
        k0 = 2 * block_idx
        k1 = k0 + 1
        
        # a=A[0,0]=0, b=A[0,1]=1, c=A[1,0]=2, d=A[1,1]=3
        # e=B[0,k0], f=B[0,k1], g=B[1,k0], h=B[1,k1]
        # V indices: e=k0, f=k1, g=9+k0, h=9+k1
        # W indices: C[0,k0]=2*k0, C[1,k0]=2*k0+1, C[0,k1]=2*k1, C[1,k1]=2*k1+1
        
        # M1 = (a+d)(e+h) -> C[0,k0] +1, C[1,k1] +1
        U1 = {0: 1, 3: 1}
        V1 = {k0: 1, 9+k1: 1}
        W1 = {2*k0: 1, 2*k1+1: 1}
        terms.append((U1, V1, W1))
        
        # M2 = (c+d)e -> C[1,k0] +1, C[1,k1] -1
        U2 = {2: 1, 3: 1}
        V2 = {k0: 1}
        W2 = {2*k0+1: 1, 2*k1+1: -1}
        terms.append((U2, V2, W2))
        
        # M3 = a(f-h) -> C[0,k1] +1, C[1,k1] +1
        U3 = {0: 1}
        V3 = {k1: 1, 9+k1: -1}
        W3 = {2*k1: 1, 2*k1+1: 1}
        terms.append((U3, V3, W3))
        
        # M4 = d(g-e) -> C[0,k0] +1, C[1,k0] +1
        U4 = {3: 1}
        V4 = {9+k0: 1, k0: -1}
        W4 = {2*k0: 1, 2*k0+1: 1}
        terms.append((U4, V4, W4))
        
        # M5 = (a+b)h -> C[0,k0] -1, C[0,k1] +1
        U5 = {0: 1, 1: 1}
        V5 = {9+k1: 1}
        W5 = {2*k0: -1, 2*k1: 1}
        terms.append((U5, V5, W5))
        
        # M6 = (c-a)(e+f) -> C[1,k1] +1
        U6 = {2: 1, 0: -1}
        V6 = {k0: 1, k1: 1}
        W6 = {2*k1+1: 1}
        terms.append((U6, V6, W6))
        
        # M7 = (b-d)(g+h) -> C[0,k0] +1
        U7 = {1: 1, 3: -1}
        V7 = {9+k0: 1, 9+k1: 1}
        W7 = {2*k0: 1}
        terms.append((U7, V7, W7))
    
    # Schoolbook for k=8
    k = 8
    # S1 = A[0,0]*B[0,8] -> C[0,8]
    terms.append(({0: 1}, {k: 1}, {2*k: 1}))
    # S2 = A[0,1]*B[1,8] -> C[0,8]
    terms.append(({1: 1}, {9+k: 1}, {2*k: 1}))
    # S3 = A[1,0]*B[0,8] -> C[1,8]
    terms.append(({2: 1}, {k: 1}, {2*k+1: 1}))
    # S4 = A[1,1]*B[1,8] -> C[1,8]
    terms.append(({3: 1}, {9+k: 1}, {2*k+1: 1}))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(18)]
    W = [[0] * rank for _ in range(18)]
    
    for t, (Ud, Vd, Wd) in enumerate(terms):
        for idx, val in Ud.items():
            U[idx][t] = val
        for idx, val in Vd.items():
            V[idx][t] = val
        for idx, val in Wd.items():
            W[idx][t] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
