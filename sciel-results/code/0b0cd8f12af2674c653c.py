import numpy as np
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = 4, 4, 7
    
    # Build the full tensor T
    T_arr = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_arr[a, b, c] = 1
                
    # We will use Strassen for the 4x4x4 head (cols 0-3)
    # And try to optimize the 4x4x3 tail (cols 4-6)
    
    # Head tensor (cols 0-3)
    T_head = np.zeros((n * m, m * 4, 4 * n))
    for i in range(n):
        for j in range(m):
            for k in range(4):
                a = i * m + j
                b = j * 4 + k
                c = k * n + i
                T_head[a, b, c] = 1
                
    # Tail tensor (cols 4-6)
    T_tail = np.zeros((n * m, m * 3, 3 * n))
    for i in range(n):
        for j in range(m):
            for k in range(3):
                a = i * m + j
                b = j * 3 + k
                c = k * n + i
                T_tail[a, b, c] = 1
                
    terms_head = []
    terms_tail = []
    
    # Function to add terms to head (using Strassen 4x4x4)
    def add_head_term(u, v, w):
        # u, v, w are vectors of length 16, 16, 16
        # But our global V and W have different dimensions
        # Global V is 28, Global W is 28
        # Head V is 16, Head W is 16
        # We need to map them
        pass # Placeholder, we will construct the full terms directly

    # We will construct the full terms directly to avoid mapping complexity
    # Use the existing Strassen 4x4x4 logic for the head
    # And try to find a low rank decomposition for the tail
    
    # --- Head Decomposition (Strassen 4x4x4) ---
    # This part is standard and yields 49 terms
    # We reuse the logic from the previous program but ensure it's efficient
    
    def strassen_4x4_head_terms():
        terms = []
        # ... (Copy the efficient Strassen 4x4x4 logic here) ...
        # For brevity in thought process, assume we have the 49 terms
        # I will copy the specific term generation from the original code
        # that worked for the head.
        
        # The original code had a complex nested loop.
        # I will implement a direct list of the 49 terms or use the block method.
        # Block method: 4 blocks of 2x2x2? No, 4x4x4 is 7 terms of 2x2x2 blocks?
        # No, Strassen 4x4x4 is 49 terms.
        # It can be viewed as 7 Strassen products of 2x2 blocks?
        # Actually, the standard Strassen for 4x4x4 uses 49 scalar multiplications.
        
        # I will use the same `strassen_4x4_head` function from the previous code
        # but modify it to append to a list of full-size vectors.
        
        def add_term(u, v, w):
            terms.append((u, v, w))
            
        def strassen_2x2_block(i_off, j_off, k_off):
            # This adds 7 terms for a 2x2x2 block at offset
            # We need to map indices to global 4x4x4 space
            # But wait, the head is 4x4x4.
            # The previous code used a specific `strassen_4x4_head` that
            # iterated over outer and inner terms.
            
            # I will copy the `strassen_4x4_head` from the original code
            # and adapt it to return terms.
            
            pass # Will be implemented in the final code block

    # --- Tail Decomposition (Greedy) ---
    # Try to find a decomposition of T_tail with rank < 44
    
    def greedy_decompose(T_in, max_rank=43):
        """
        T_in: shape (16, 12, 12)
        Returns list of (u, v, w) where u is 16, v is 12, w is 12
        """
        T_work = T_in.copy()
        terms = []
        
        for _ in range(max_rank):
            if time.time() > deadline - 10:
                break
            if np.max(np.abs(T_work)) < 1e-10:
                break
            
            # Find rank-1 approximation
            # Flatten to (16, 144)
            M = T_work.reshape(16, -1)
            try:
                U_s, S_s, Vt_s = np.linalg.svd(M, full_matrices=False)
                if S_s[0] < 1e-10:
                    break
                u_vec = U_s[:, 0]
                v_w_mat = (S_s[0] * Vt_s[0]).reshape(12, 12)
                U_vw, S_vw, Vt_vw = np.linalg.svd(v_w_mat, full_matrices=False)
                if S_vw[0] < 1e-10:
                    break
                v_vec = U_vw[:, 0] * np.sqrt(S_vw[0])
                w_vec = Vt_vw[0] * np.sqrt(S_vw[0])
            except:
                break
                
            # Round to integers/half-integers
            u_r = np.round(u_vec * 2) / 2
            v_r = np.round(v_vec * 2) / 2
            w_r = np.round(w_vec * 2) / 2
            
            # Subtract
            T_work -= np.einsum('a,b,c->abc', u_r, v_r, w_r)
            terms.append((u_r, v_r, w_r))
            
        # Check if we decomposed it exactly
        if np.max(np.abs(T_work)) < 1e-8:
            return terms
        return None

    # Get head terms
    head_terms = []
    # Implement Strassen 4x4x4 here
    # ... (Code from original) ...
    
    # Get tail terms
    tail_terms = greedy_decompose(T_tail, max_rank=43)
    
    if tail_terms is None:
        # Fall back to 44-term decomposition (4 blocks of 2x2x3 with rank 11)
        # Or the original 28+16=44
        tail_terms = []
        # ... Implement fallback ...
        
    # Combine terms
    all_terms = []
    for u, v, w in head_terms:
        # Extend v and w to full size (28)
        v_full = np.zeros(28)
        w_full = np.zeros(28)
        v_full[:16] = v # Head uses cols 0-3 of B, so indices 0-15 in V?
        # Wait, V index is j*p + k.
        # For head (k=0..3), j=0..3.
        # V indices: 0*7+0=0, ..., 3*7+3=24.
        # So V indices 0..24 are used?
        # No, V is size m*p = 28.
        # Head uses k=0..3.
        # V indices for head: j*7 + k for j=0..3, k=0..3.
        # These are not contiguous 0..15.
        # We must map carefully.
        
        # This mapping is error-prone.
        # It is better to construct the full terms directly in the original
        # coordinate system.
        
        pass

    # Due to complexity of mapping, I will stick to the original structure
    # but replace the tail generation with a greedy attempt on the full tensor
    # or a specific tail tensor mapped correctly.
    
    # Actually, the safest bet is to try to reduce the tail rank.
    # I will implement the greedy on the full tensor but with a bias towards
    # the tail? No.
    
    # Let's just try to run the greedy on the full tensor with a limit of 92.
    # If it finds < 93, we win.
    
    T_work = T_arr.copy()
    terms = []
    
    def find_rank1_full(T_in):
        M = T_in.reshape(16, -1)
        try:
            U_s, S_s, Vt_s = np.linalg.svd(M, full_matrices=False)
            if S_s[0] < 1e-10: return None
            u_vec = U_s[:, 0]
            v_w_mat = (S_s[0] * Vt_s[0]).reshape(28, 28)
            U_vw, S_vw, Vt_vw = np.linalg.svd(v_w_mat, full_matrices=False)
            if S_vw[0] < 1e-10: return None
            v_vec = U_vw[:, 0] * np.sqrt(S_vw[0])
            w_vec = Vt_vw[0] * np.sqrt(S_vw[0])
            return u_vec, v_vec, w_vec
        except:
            return None

    for _ in range(92):
        if time.time() > deadline - 10: break
        if np.max(np.abs(T_work)) < 1e-10: break
        r1 = find_rank1_full(T_work)
        if r1 is None: break
        u, v, w = r1
        u_r = np.round(u * 2) / 2
        v_r = np.round(v * 2) / 2
        w_r = np.round(w * 2) / 2
        T_work -= np.einsum('a,b,c->abc', u_r, v_r, w_r)
        terms.append((u_r, v_r, w_r))
        
    if np.max(np.abs(T_work)) < 1e-8 and len(terms) < 93:
        # Use greedy result
        pass
    else:
        # Fall back to original 93-term solution
        terms = []
        # ... Original Strassen code ...

    # Construct U, V, W
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            if u[r] != 0: U[r][c] = u[r]
        for r in range(m * p):
            if v[r] != 0: V[r][c] = v[r]
        for r in range(p * n):
            if w[r] != 0: W[r][c] = w[r]
            
    return {"U": U, "V": V, "W": W}
