import numpy as np
import time

def construct():
    start_time = time.time()
    n, m, p = 4, 5, 5
    
    # Split: i = {0,1},{2,3}; j = {0,1},{2,3},{4}; k = {0,1},{2,3},{4}
    i_groups = [[0,1], [2,3]]
    j_groups = [[0,1], [2,3], [4]]
    k_groups = [[0,1], [2,3], [4]]
    
    # Collect all rank-1 terms as (U_entries, V_entries, W_entries)
    # Each term is a dict: {(i,j): val} for U, {(j,k): val} for V, {(k,i): val} for W
    terms = []
    
    def add_schoolbook(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U = {(i,j): 1}
                    V = {(j,k): 1}
                    W = {(k,i): 1}
                    terms.append((U, V, W))
    
    def add_strassen_2x2x2(r, c, l):
        """Strassen for 2x2x2 with rows r=[r0,r1], cols/rows c=[c0,c1], cols l=[l0,l1]"""
        r0, r1 = r
        c0, c1 = c
        l0, l1 = l
        
        # 7 Strassen products
        # P1 = (A[r0,c0]+A[r1,c0])*(B[c0,l0]+B[c0,l1])
        # Wait, I need to be careful about which elements go where.
        # In the 2x2 sub-problem:
        # A_sub = [[A[r0,c0], A[r0,c1]], [A[r1,c0], A[r1,c1]]]
        # B_sub = [[B[c0,l0], B[c0,l1]], [B[c1,l0], B[c1,l1]]]
        # C_sub[i,l] = sum_j A_sub[i,j]*B_sub[j,l]
        # C_sub[0,0] = A[r0,c0]*B[c0,l0] + A[r0,c1]*B[c1,l0]
        # etc.
        
        # Strassen products (verified):
        # P1 = (a+d)(e+h) where a=A[r0,c0], d=A[r1,c1], e=B[c0,l0], h=B[c1,l1]
        # P2 = (c+d)e where c=A[r1,c0], d=A[r1,c1], e=B[c0,l0]
        # P3 = a(g-h) where a=A[r0,c0], g=B[c1,l0], h=B[c1,l1]
        # P4 = d(f-e) where d=A[r1,c1], f=B[c0,l1], e=B[c0,l0]
        # P5 = (a+b)h where a=A[r0,c0], b=A[r0,c1], h=B[c1,l1]
        # P6 = (c-a)(e+f) where c=A[r1,c0], a=A[r0,c0], e=B[c0,l0], f=B[c0,l1]
        # P7 = (b-d)(g+h) where b=A[r0,c1], d=A[r1,c1], g=B[c1,l0], h=B[c1,l1]
        
        # C[0,0] = P1 - P5 + P7
        # C[0,1] = P3 + P5
        # C[1,0] = P2 + P4 - P6
        # C[1,1] = P1 - P3 + P4
        
        # Let me verify C[0,0] = P1 - P5 + P7:
        # P1 = (a+d)(e+h) = ae+ah+de+dh
        # P5 = (a+b)h = ah+bh
        # P7 = (b-d)(g+h) = bg+bh-dg-dh
        # P1-P5+P7 = ae+ah+de+dh-ah-bh+bg+bh-dg-dh = ae+de+bg-dg
        # C[0,0] should be a*e + b*g (since C[0,0] = A[r0,c0]*B[c0,l0] + A[r0,c1]*B[c1,l0])
        # ae+de+bg-dg vs ae+bg. Not equal unless d(e-g)=0.
        
        # So this formula is WRONG. Let me try the other variant.
        
        # Try: P1=(a+b)(e+f), P2=(c+d)e, P3=a(f-g), P4=d(h-g), P5=(a+b)h, P6=(c-a)(e+f), P7=(b-d)(g+h)
        # C[0,0] = P1-M5+M7
        # P1 = (a+b)(e+f) = ae+af+be+bf
        # P5 = (a+b)h = ah+bh
        # P7 = (b-d)(g+h) = bg+bh-dg-dh
        # P1-P5+P7 = ae+af+be+bf-ah-bh+bg+bh-dg-dh = ae+af+be+bf-ah+bg-dg-dh
        # C[0,0] = ae+cg. Still not matching.
        
        # I think the correct Strassen is:
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B21-B22)
        # P4 = A22(B12-B11)
        # P5 = (A11+A12)B22
        # P6 = (A21-A11)(B11+B12)
        # P7 = (A12-A22)(B21+B22)
        # C11 = P1-P5+P7
        # C12 = P3+P5
        # C21 = P2+P4-P6
        # C22 = P1-P3+P4
        
        # Where A11=a, A12=b, A21=c, A22=d, B11=e, B12=f, B21=g, B22=h
        # C11 = ae+cg, C12 = af+dh, C21 = ce+dg, C22 = cf+dh
        
        # Let me verify C12 = P3+P5:
        # P3 = a(g-h), P5 = (a+b)h
        # P3+P5 = ag-ah+ah+bh = ag+bh
        # C12 = af+dh. Not equal!
        
        # I'm clearly getting the Strassen formula wrong. Let me just use schoolbook
        # for all blocks and get rank 100. At least it's correct.
        
        # Actually, let me just use schoolbook for the 2x2x2 blocks too.
        add_schoolbook([r0,r1], [c0,c1], [l0,l1])
    
    # Build all terms
    for ig in i_groups:
        for jg in j_groups:
            for kg in k_groups:
                if len(ig)==2 and len(jg)==2 and len(kg)==2:
                    add_strassen_2x2x2(ig, jg, kg)
                else:
                    add_schoolbook(ig, jg, kg)
    
    total_rank = len(terms)
    
    # Build U, V, W
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    for col, (U_d, V_d, W_d) in enumerate(terms):
        for (i,j), val in U_d.items():
            U[i*m+j, col] = val
        for (j,k), val in V_d.items():
            V[j*p+k, col] = val
        for (k,i), val in W_d.items():
            W[k*n+i, col] = val
    
    # Verify
    # T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Should be 1 for all i,j,k
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for c in range(total_rank))
                if val != 1:
                    max_err = max(max_err, abs(val-1))
    
    if max_err > 0:
        print(f"VERIFICATION FAILED! max_err={max_err}")
        # Fall back to pure schoolbook
        rank = n*m*p
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        total_rank = rank
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
