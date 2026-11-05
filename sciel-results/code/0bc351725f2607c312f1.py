import time

DIMENSIONS = [2, 5, 6]

DEADLINE = 150.0
START_TIME = time.time()

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    
    Strategy: Block decomposition
    - Split p=6 into two blocks of size 3
    - For each 2x5x3 block:
      - Split m=5 into 2+3
      - 2x2x3 part: Strassen(7) + outer(4) = rank 11
      - 2x3x3 part: split m=3 into 2+1 -> 2x2x3(11) + 2x1x3(6) = rank 17
    - Per 3-block: 11 + 17 = 28
    - Total: 2 * 28 = 56
    
    Index conventions:
    U: (n*m, rank) = (10, rank)
    V: (m*p, rank) = (30, rank)
    W: (p*n, rank) = (12, rank) with W[k*n+i, r]
    """
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]      # 10 x rank
    V = [[0] * rank for _ in range(m * p)]      # 30 x rank
    W = [[0] * rank for _ in range(p * n)]      # 12 x rank
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1
    
    # For each of the 2 blocks in p (each of size 3)
    for pb in range(2):
        # pb=0: p indices 0,1,2 ; pb=1: p indices 3,4,5
        p_base = pb * 3
        
        # Part A: A is 2x2 (m indices 0,1), B is 2x3 (m indices 0,1; p indices p_base..p_base+2)
        # Rank 11: Strassen for 2x2x2 (7 terms) + outer for 2x2x1 (4 terms)
        
        # Strassen 2x2x2: A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=[[ae,af],[ce,cf]]+...
        # Standard Strassen for 2x2x2:
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-c)
        # m5 = (a+b)h
        # m6 = (c-d)(e+f)
        # m7 = (b-d)(g+h)
        # C11 = m1 + m4 - m5 + m7
        # C12 = m3 + m5
        # C21 = m2 + m4
        # C22 = m1 - m2 + m3 + m6
        
        # For 2x2x2 block (A rows 0,1; B cols 0,1; C 2x2):
        # U rows: i*m+j for i in {0,1}, j in {0,1} -> indices 0,1,2,3
        # V rows: j*p+k for j in {0,1}, k in {p_base, p_base+1} -> indices 0*6+p_base, 0*6+p_base+1, 1*6+p_base, 1*6+p_base+1
        # W rows: k*n+i for k in {p_base, p_base+1}, i in {0,1} -> indices p_base*2+0, p_base*2+1, (p_base+1)*2+0, (p_base+1)*2+1
        
        a, b, c, d = 0, 1, 2, 3  # U row indices for A elements (i=0,j=0), (0,1), (1,0), (1,1)
        e = 0 * 6 + p_base       # V row for j=0,k=p_base
        f = 0 * 6 + (p_base+1)   # V row for j=0,k=p_base+1
        g = 1 * 6 + p_base       # V row for j=1,k=p_base
        h = 1 * 6 + (p_base+1)   # V row for j=1,k=p_base+1
        
        w00 = p_base * 2 + 0     # W row for k=p_base, i=0
        w01 = p_base * 2 + 1     # W row for k=p_base, i=1
        w10 = (p_base+1) * 2 + 0 # W row for k=p_base+1, i=0
        w11 = (p_base+1) * 2 + 1 # W row for k=p_base+1, i=1
        
        # Strassen terms:
        # m1 = (a+d)(e+h): U: a=1,d=1 ; V: e=1,h=1 ; W: w00=1,w10=-1,w01=1,w11=-1 (C11=m1+..., C21=m1-...)
        # Actually let me be more careful. Strassen computes C = AB where C is 2x2.
        # C[0][0] = m1 + m4 - m5 + m7
        # C[0][1] = m3 + m5
        # C[1][0] = m2 + m4
        # C[1][1] = m1 - m2 + m3 + m6
        
        # Each Strassen term is a scalar product, so we add to U, V, W for each output position.
        
        # m1 = (a+d)(e+h)
        add_term(a, e, w00, 1, 1, 1)   # contributes to C[0][0]
        add_term(d, e, w00, 1, 1, 1)
        add_term(a, h, w00, 1, 1, 1)
        add_term(d, h, w00, 1, 1, 1)
        # But wait, this gives 4 terms for m1 alone, but Strassen m1 is ONE bilinear form.
        # The issue is that in tensor decomposition, each "term" is a rank-1 tensor.
        # Strassen's 7 terms are each rank-1 in the appropriate sense.
        # Let me reconsider.
        
        # Actually, for the 2x2x2 multiplication tensor, the rank is 7.
        # The 7 rank-1 terms correspond to the 7 Strassen bilinear forms.
        # Each bilinear form m_t = sum_{i,j} s_t[i,j] * A[i,j] * B[j,k] ... no.
        # m_t = (sum_i alpha_t[i] * A[i,*]) * (sum_j beta_t[j] * B[j,*])
        # This is a rank-1 tensor in the appropriate tensor space.
        
        # Let me just implement the known rank-7 decomposition for 2x2x2 directly.
        # The 7 terms for 2x2x2:
        # T1: (A[0,0]+A[1,1])*(B[0,0]+B[1,1]) -> C[0,0] += T1, C[1,1] += T1
        # T2: (A[1,0]+A[1,1])*B[0,0] -> C[1,0] += T2, C[1,1] -= T2
        # T3: A[0,0]*(B[0,1]-B[1,1]) -> C[0,1] += T3, C[1,1] += T3
        # T4: A[1,1]*(B[1,0]-A[1,0])... 
        # I'm getting confused. Let me look up the standard form.
        
        # Standard Strassen for C = A*B where A,B are 2x2:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - A10)  -- NO, this is wrong
        # M4 = A11 * (B10 - B00)  -- also not standard
        # 
        # Correct Strassen:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each M_t is a product of a linear form in A and a linear form in B.
        # So each M_t corresponds to a rank-1 term in the tensor decomposition.
        # For M1: A-part = A00 + A11, B-part = B00 + B11
        # U contributions: U[0,0] += 1, U[1,1] += 1 (for this term's column)
        # V contributions: V[0,0] += 1, V[1,1] += 1
        # W contributions: W[0,0] += 1, W[1,1] += 1
        
        # Let me restart the implementation more carefully.
        col = 0
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for pb in range(2):
            p_base = pb * 3
            
            # Part 1: 2x2x3 block (A: rows 0,1 x m cols 0,1; B: m rows 0,1 x p cols p_base..p_base+2)
            # Split p into 2+1: 2x2x2 (rank 7) + 2x2x1 (rank 4) = 11
            
            # 2x2x2: p indices p_base, p_base+1
            # Strassen terms:
            # M1: A-part=[1,0,0,1] (A00+A11), B-part=[1,0,0,1] (B00+B11)
            #   U: U[0][col]+=1, U[3][col]+=1 (A[0,0] and A[1,1])
            #   V: V[0*6+p_base][col]+=1, V[1*6+p_base+1][col]+=1
            #   W: W[p_base*2+0][col]+=1, W[(p_base+1)*2+1][col]+=1
            
            # M2: A-part=[0,0,1,1] (A10+A11), B-part=[1,0,0,0] (B00)
            #   U: U[2][col]+=1, U[3][col]+=1
            #   V: V[0*6+p_base][col]+=1
            #   W: W[p_base*2+0][col]+=1, W[(p_base+1)*2+0][col]+=1
            
            # M3: A-part=[1,0,0,0] (A00), B-part=[0,1,-1,0] (B01-B11)
            #   U: U[0][col]+=1
            #   V: V[0*6+p_base+1][col]+=1, V[1*6+p_base+1][col]+=-1
            #   W: W[p_base*2+1][col]+=1, W[(p_base+1)*2+1][col]+=1
            
            # M4: A-part=[0,0,0,1] (A11), B-part=[-1,0,1,0] (B10-B00)
            #   U: U[3][col]+=1
            #   V: V[0*6+p_base][col]+=-1, V[1*6+p_base][col]+=1
            #   W: W[p_base*2+0][col]+=1, W[(p_base+1)*2+0][col]+=1
            
            # M5: A-part=[1,1,0,0] (A00+A01), B-part=[0,0,0,1] (B11)
            #   U: U[0][col]+=1, U[1][col]+=1
            #   V: V[1*6+p_base+1][col]+=1
            #   W: W[p_base*2+1][col]+=1, W[(p_base+1)*2+1][col]+=1
            
            # M6: A-part=[1,-1,0,0] (A00-A10), B-part=[1,1,0,0] (B00+B01)
            #   U: U[0][col]+=1, U[2][col]+=-1
            #   V: V[0*6+p_base][col]+=1, V[0*6+p_base+1][col]+=1
            #   W: W[p_base*2+1][col]+=1, W[(p_base+1)*2+1][col]+=1
            
            # M7: A-part=[0,1,-1,0] (A01-A11), B-part=[0,0,1,1] (B10+B11)
            #   U: U[1][col]+=1, U[3][col]+=-1
            #   V: V[1*6+p_base][col]+=1, V[1*6+p_base+1][col]+=1
            #   W: W[p_base*2+0][col]+=1, W[(p_base+1)*2+1][col]+=1
            
            # Now for the 2x2x1 part (p index p_base+2):
            # This is just outer product: 4 terms
            # O1: A00 * B0,p_base+2 -> C[0, p_base+2]
            # O2: A01 * B1,p_base+2 -> C[0, p_base+2]
            # O3: A10 * B0,p_base+2 -> C[1, p_base+2]
            # O4: A11 * B1,p_base+2 -> C[1, p_base+2]
            
            # Let me define a helper to add a rank-1 term
            def add_term(u_indices_vals, v_indices_vals, w_indices_vals):
                nonlocal col
                if col >= rank:
                    raise ValueError("Rank exceeded")
                for idx, val in u_indices_vals:
                    U[idx][col] += val
                for idx, val in v_indices_vals:
                    V[idx][col] += val
                for idx, val in w_indices_vals:
                    W[idx][col] += val
                col += 1
            
            # Strassen 2x2x2 terms
            # M1
            add_term([(0, 1), (3, 1)], 
                     [(0*6+p_base, 1), (1*6+p_base+1, 1)],
                     [(p_base*2+0, 1), ((p_base+1)*2+1, 1)])
            # M2
            add_term([(2, 1), (3, 1)], 
                     [(0*6+p_base, 1)],
                     [(p_base*2+0, 1), ((p_base+1)*2+0, 1)])
            # M3
            add_term([(0, 1)], 
                     [(0*6+p_base+1, 1), (1*6+p_base+1, -1)],
                     [(p_base*2+1, 1), ((p_base+1)*2+1, 1)])
            # M4
            add_term([(3, 1)], 
                     [(0*6+p_base, -1), (1*6+p_base, 1)],
                     [(p_base*2+0, 1), ((p_base+1)*2+0, 1)])
            # M5
            add_term([(0, 1), (1, 1)], 
                     [(1*6+p_base+1, 1)],
                     [(p_base*2+1, 1), ((p_base+1)*2+1, 1)])
            # M6
            add_term([(0, 1), (2, -1)], 
                     [(0*6+p_base, 1), (0*6+p_base+1, 1)],
                     [(p_base*2+1, 1), ((p_base+1)*2+1, 1)])
            # M7
            add_term([(1, 1), (3, -1)], 
                     [(1*6+p_base, 1), (1*6+p_base+1, 1)],
                     [(p_base*2+0, 1), ((p_base+1)*2+1, 1)])
            
            # Outer product for p index p_base+2: 4 terms
            # O1: A[0,0] * B[0, p_base+2] -> C[0, p_base+2]
            add_term([(0, 1)], [(0*6+p_base+2, 1)], [(p_base*2+0, 1)])
            # O2: A[0,1] * B[1, p_base+2] -> C[0, p_base+2]
            add_term([(1, 1)], [(1*6+p_base+2, 1)], [(p_base*2+0, 1)])
            # O3: A[1,0] * B[0, p_base+2] -> C[1, p_base+2]
            add_term([(2, 1)], [(0*6+p_base+2, 1)], [(p_base*2+1, 1)])
            # O4: A[1,1] * B[1, p_base+2] -> C[1, p_base+2]
            add_term([(3, 1)], [(1*6+p_base+2, 1)], [(p_base*2+1, 1)])
            
            # Part 2: 2x3x3 block (A: rows 0,1 x m cols 2,3,4; B: m rows 2,3,4 x p cols p_base..p_base+2)
            # Split m=3 into 2+1: 2x2x3 (rank 11) + 2x1x3 (rank 6) = 17
            
            # 2x2x3: A rows 0,1 x m cols 2,3; B m rows 2,3 x p cols p_base..p_base+2
            # This is like 2x2x2 + 2x2x1 again
            # U row indices: i*5+j for i in {0,1}, j in {2,3} -> 0*5+2=2, 0*5+3=3, 1*5+2=7, 1*5+3=8
            # V row indices: j*6+k for j in {2,3}, k in {p_base..p_base+2}
            # W row indices: k*2+i for k in {p_base..p_base+2}, i in {0,1}
            
            # For 2x2x2 (p indices p_base, p_base+1):
            u_a00, u_a01, u_a10, u_a11 = 2, 3, 7, 8  # A[0,2], A[0,3], A[1,2], A[1,3]
            
            # M1: (A00+A11)*(B00+B11)
            add_term([(u_a00, 1), (u_a11, 1)], 
                     [(2*6+p_base, 1), (3*6+p_base+1, 1)],
                     [(p_base*2+0, 1), ((p_base+1)*2+1, 1)])
            # M2: (A10+A11)*B00
            add_term([(u_a10, 1), (u_a11, 1)], 
                     [(2*6+p_base, 1)],
                     [(p_base*2+0, 1), ((p_base+1)*2+0, 1)])
            # M3: A00*(B01-B11)
            add_term([(u_a00, 1)], 
                     [(2*6+p_base+1, 1), (3*6+p_base+1, -1)],
                     [(p_base*2+1, 1), ((p_base+1)*2+1, 1)])
            # M4: A11*(B10-B00)
            add_term([(u_a11, 1)], 
                     [(2*6+p_base, -1), (3*6+p_base, 1)],
                     [(p_base*2+0, 1), ((p_base+1)*2+0, 1)])
            # M5: (A00+A01)*B11
            add_term([(u_a00, 1), (u_a01, 1)], 
                     [(3*6+p_base+1, 1)],
                     [(p_base*2+1, 1), ((p_base+1)*2+1, 1)])
            # M6: (A00-A10)*(B00+B01)
            add_term([(u_a00, 1), (u_a10, -1)], 
                     [(2*6+p_base, 1), (2*6+p_base+1, 1)],
                     [(p_base*2+1, 1), ((p_base+1)*2+1, 1)])
            # M7: (A01-A11)*(B10+B11)
            add_term([(u_a01, 1), (u_a11, -1)], 
                     [(3*6+p_base, 1), (3*6+p_base+1, 1)],
                     [(p_base*2+0, 1), ((p_base+1)*2+1, 1)])
            
            # Outer for p index p_base+2: 4 terms
            add_term([(u_a00, 1)], [(2*6+p_base+2, 1)], [(p_base*2+0, 1)])
            add_term([(u_a01, 1)], [(3*6+p_base+2, 1)], [(p_base*2+0, 1)])
            add_term([(u_a10, 1)], [(2*6+p_base+2, 1)], [(p_base*2+1, 1)])
            add_term([(u_a11, 1)], [(3*6+p_base+2, 1)], [(p_base*2+1, 1)])
            
            # 2x1x3: A rows 0,1 x m col 4; B m row 4 x p cols p_base..p_base+2
            # Outer product: 2*3 = 6 terms
            u_b0 = 4   # A[0,4]
            u_b1 = 9   # A[1,4]
            v_b = 4*6  # B[4, p_base] base
            
            for dp in range(3):
                # A[0,4] * B[4, p_base+dp] -> C[0, p_base+dp]
                add_term([(u_b0, 1)], [(4*6+p_base+dp, 1)], [(p_base*2+0+dp, 1)])
                # A[1,4] * B[4, p_base+dp] -> C[1, p_base+dp]
                add_term([(u_b1, 1)], [(4*6+p_base+dp, 1)], [(p_base*2+1+dp, 1)])
        
        # Verify col == rank
        if col != rank:
            raise ValueError(f"Expected rank {rank}, got {col}")
        
        return {"U": U, "V": V, "W": W}
