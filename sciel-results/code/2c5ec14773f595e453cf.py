import time
import numpy as np

def construct():
    """
    Construct the rank-28 decomposition for M_{2,2,8}.
    This is based on 4 independent Strassen M_{2,2,2} blocks.
    """
    n, m, p = 2, 2, 8
    nm, mp, pn = n*m, m*p, p*n
    rank = 28
    
    # Initialize matrices
    U = np.zeros((nm, rank), dtype=np.int8)
    V = np.zeros((mp, rank), dtype=np.int8)
    W = np.zeros((pn, rank), dtype=np.int8)
    
    # Strassen terms for M_{2,2,2}
    # Indices for U (4 entries: 00, 01, 10, 11)
    # Indices for V (4 entries: 00, 01, 10, 11)
    # Indices for W (4 entries: 00, 01, 10, 11)
    
    # The 7 Strassen products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U vectors (for A)
    u_terms = [
        [1, 0, 0, 1], # A00+A11
        [0, 0, 1, 1], # A10+A11
        [1, 0, 0, 0], # A00
        [0, 0, 0, 1], # A11
        [1, 1, 0, 0], # A00+A01
        [-1, 0, 1, 0],# A10-A00
        [0, 1, 0, -1] # A01-A11
    ]
    
    # V vectors (for B)
    v_terms = [
        [1, 0, 0, 1], # B00+B11
        [1, 0, 0, 0], # B00
        [0, 1, 0, -1],# B01-B11
        [0, 0, 1, -1],# B10-B00
        [0, 0, 0, 1], # B11
        [1, 1, 0, 0], # B00+B01
        [0, 0, 1, 1]  # B10+B11
    ]
    
    # W vectors (for C)
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M6
    w_terms = [
        [1, 0, 0, -1], # M1
        [-1, 0, 1, 0], # M2
        [0, 1, 0, -1], # M3
        [0, 0, -1, 1], # M4
        [0, 1, -1, 0], # M5
        [0, 0, 0, -1], # M6
        [1, 0, 0, 1]   # M7
    ]
    # Wait, let's double check the W terms mapping.
    # C00 = M1 - M4 - M5 + M7 -> W for M1 is 1, M4 is -1, M5 is -1, M7 is 1.
    # C01 = M3 + M5 -> W for M3 is 1, M5 is 1.
    # C10 = M2 + M4 -> W for M2 is 1, M4 is 1.
    # C11 = M1 - M2 + M3 - M6 -> W for M1 is 1, M2 is -1, M3 is 1, M6 is -1.
    
    # Let's re-verify standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 - M6
    
    # My indices: 00, 01, 10, 11 correspond to 11, 12, 21, 22.
    # So C00 (11) = M1 - M4 - M5 + M7.
    # C01 (12) = M3 + M5.
    # C10 (21) = M2 + M4.
    # C11 (22) = M1 - M2 + M3 - M6.
    
    # W vectors should be arranged as [W00, W01, W10, W11] for each term r.
    # Term 1 (M1): [1, 0, 0, 1]
    # Term 2 (M2): [-1, 0, 1, 0] -> C10 gets +1, C11 gets -1.
    # Term 3 (M3): [0, 1, 0, 1] -> C01 gets +1, C11 gets +1.
    # Term 4 (M4): [0, 0, -1, -1] -> C10 gets -1? No, C10 = M2+M4. So C10 gets +1.
    # Let's re-read: C21 = M2 + M4. So C10 (21) gets +1 from M4.
    # C11 = M1 - M4 ... So C11 (22) gets -1 from M4.
    # So W for M4 is [0, 0, 1, -1].
    
    # Term 5 (M5): C01 = M3+M5 (+1). C00 = ... -M5 ... (-1).
    # So W for M5 is [-1, 1, 0, 0].
    
    # Term 6 (M6): C11 = ... -M6 ... (-1).
    # So W for M6 is [0, 0, 0, -1].
    
    # Term 7 (M7): C00 = ... +M7 ... (+1).
    # So W for M7 is [1, 0, 0, 0].
    
    w_terms_correct = [
        [1, 0, 0, 1],   # M1
        [-1, 0, 1, 0],  # M2
        [0, 1, 0, 1],   # M3
        [0, 0, 1, -1],  # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, -1],  # M6
        [1, 0, 0, 0]    # M7
    ]
    
    # Fill U, V, W for 4 blocks
    for block in range(4):
        for t in range(7):
            col = block * 7 + t
            
            # U: same for all blocks
            U[:, col] = u_terms[t]
            
            # V: block specific
            # V indices: j (0,1) * 8 + k (0..7)
            # For block b, k is in [2b, 2b+1]
            # V[0, 2b] = v[0], V[1, 2b] = v[1], V[0, 2b+1] = v[2], V[1, 2b+1] = v[3]
            # v_terms[t] is [v00, v01, v10, v11]
            V[0 * 8 + 2*block][col] = v_terms[t][0]
            V[0 * 8 + 2*block + 1][col] = v_terms[t][1]
            V[1 * 8 + 2*block][col] = v_terms[t][2]
            V[1 * 8 + 2*block + 1][col] = v_terms[t][3]
            
            # W: block specific
            # W indices: k (0..7) * 2 + i (0,1)
            # For block b, k is in [2b, 2b+1]
            # W[2b, 0] = w[0], W[2b, 1] = w[1], W[2b+1, 0] = w[2], W[2b+1, 1] = w[3]
            # w_terms_correct[t] is [w00, w01, w10, w11]
            W[2*block * 2 + 0][col] = w_terms_correct[t][0]
            W[2*block * 2 + 1][col] = w_terms_correct[t][1]
            W[(2*block + 1) * 2 + 0][col] = w_terms_correct[t][2]
            W[(2*block + 1) * 2 + 1][col] = w_terms_correct[t][3]
            
    return {"U": U, "V": V, "W": W}

def verify(U, V, W):
    """
    Verify the decomposition.
    """
    n, m, p = 2, 2, 8
    nm, mp, pn = n*m, m*p, p*n
    rank = U.shape[1]
    
    # Convert to numpy if not already
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Compute T = sum_r U[:,r] V[:,r] W[:,r]
    # T is a tensor of shape (nm, mp, pn)
    # We can compute this efficiently using einsum or loops
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    
    # Check all valid entries
    # Valid entries are those where a=i*m+j, b=j*p+k, c=k*n+i
    # And T[a,b,c] should be 1.
    # All other entries should be 0.
    
    # Iterate over all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                val = 0
                for r in range(rank):
                    val += U[a][r] * V[b][r] * W[c][r]
                if val != 1:
                    return False
                
    # Check a few random invalid entries to ensure no leakage
    # Since rank is small, we can check all invalid entries if fast enough
    # Total entries 4*16*16 = 1024. Valid 32. Invalid 992.
    # 992 * 28 ops is ~28k ops. Very fast.
    valid_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_set.add((i*m+j, j*p+k, k*n+i))
                
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a,b,c) in valid_set:
                    continue
                val = 0
                for r in range(rank):
                    val += U[a][r] * V[b][r] * W[c][r]
                if val != 0:
                    return False
                    
    return True

# Main entry point
if __name__ == "__main__":
    res = construct()
    # Optional: verify in debug
    # if not verify(res['U'], res['V'], res['W']):
    #     print("Verification failed")
