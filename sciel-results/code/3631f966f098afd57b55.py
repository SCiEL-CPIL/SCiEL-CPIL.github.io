import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # We will build the decomposition block by block.
    # Rank estimate: 74.
    # We allocate a buffer of size 84 to be safe, then trim?
    # Or just allocate 84 and use a counter.
    
    # Using a list of terms (u_idx, v_idx, w_idx, val) is easier to manage,
    # then convert to matrices.
    # But the interface requires matrices.
    
    # Let's use a dynamic list of columns.
    # Each column is a rank-1 term.
    # We will store U, V, W as lists of lists (sparse-ish) or just build the final arrays.
    
    # Max possible rank is 84.
    U = [[0] * 84 for _ in range(nm)]
    V = [[0] * 84 for _ in range(mp)]
    W = [[0] * 84 for _ in range(pn)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        if col >= 84:
            return
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1

    def add_strassen(i_off, j_off, k_off):
        """
        Adds Strassen decomposition for a 2x2x2 block.
        i_off: offset for row in A (0 or 1)
        j_off: offset for col in A / row in B (0 or 1)
        k_off: offset for col in B (0 or 1)
        """
        # Indices for A (2x2)
        # A00: (i_off, j_off) -> idx i_off*m + j_off
        # A01: (i_off, j_off+1)
        # A10: (i_off+1, j_off)
        # A11: (i_off+1, j_off+1)
        
        # Indices for B (2x2)
        # B00: (j_off, k_off) -> idx j_off*p + k_off
        # B01: (j_off, k_off+1)
        # B10: (j_off+1, k_off)
        # B11: (j_off+1, k_off+1)
        
        # Indices for C (2x2)
        # C00: (k_off, i_off) -> idx k_off*n + i_off
        # C01: (k_off+1, i_off)
        # C10: (k_off, i_off+1)
        # C11: (k_off+1, i_off+1)
        
        # Strassen Multiplications P1..P7
        
        # P1 = (A00+A11)(B00+B11)
        u1 = [(i_off*m + j_off, 1), ((i_off+1)*m + (j_off+1), 1)]
        v1 = [(j_off*p + k_off, 1), ((j_off+1)*p + (k_off+1), 1)]
        w1 = [(k_off*n + i_off, 1), ((k_off+1)*n + i_off, 1), (k_off*n + (i_off+1), 1), ((k_off+1)*n + (i_off+1), 1)]
        
        # Wait, W for P1 contributes to all 4 C entries?
        # C00 = P1 - P2 - P4 + P7
        # C01 = P3 + P5
        # C10 = P6 + P2
        # C11 = P1 - P5 + P6 - P3
        
        # So for each P, we add the term to U and V, and then add the weighted W terms.
        # It is more efficient to add the full rank-1 tensor for each P.
        
        # P1
        # U part: A00 + A11
        # V part: B00 + B11
        # W part: C00 + C01 + C10 + C11 ? No.
        # The term is u v^T w^T.
        # The contribution to C is (u . A) (v . B) w.
        # So we add:
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C01=1, C10=1, C11=1
        # This creates a rank-1 tensor that adds 1 to ALL C entries?
        # No, the tensor is sum_r u_r v_r w_r.
        # If we set W to have 1s in all 4 positions, then the term contributes to all 4 C entries.
        # Then we subtract other terms to cancel out where needed.
        
        # Let's define the 7 rank-1 tensors.
        
        def add_rank1(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            if col >= 84: return
            for idx, val in u_coeffs:
                U[idx][col] += val
            for idx, val in v_coeffs:
                V[idx][col] += val
            for idx, val in w_coeffs:
                W[idx][col] += val
            col += 1

        # P1: (A00+A11)(B00+B11)
        # Contributes to C00, C01, C10, C11 with +1?
        # No, in Strassen, the output is linear combination of Ps.
        # The decomposition is sum P_r u_r v_r w_r.
        # The w_r vector determines WHICH C entries are affected.
        
        # Correct Strassen Decomposition:
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A01(B11-B01)
        # P4 = A11(B10-B00)
        # P5 = (A00+A01)B11
        # P6 = (A10-A00)(B00+B01)
        # P7 = (A01-A11)(B10+B11)
        
        # C00 = P1 - P2 - P4 + P7
        # C01 = P3 + P5
        # C10 = P6 + P2
        # C11 = P1 - P5 + P6 - P3
        
        # This means:
        # Term 1 (P1):
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1  (Wait, P1 contributes to C00 and C11)
        # Let's check C00: P1 is +1. C11: P1 is +1.
        # C01: P1 is 0. C10: P1 is 0.
        # So W for P1: C00=1, C11=1.
        
        # Term 2 (P2):
        # U: A10=1, A11=1
        # V: B00=1
        # W: C00=-1, C10=1
        
        # Term 3 (P3):
        # U: A01=1
        # V: B11=1, B01=-1
        # W: C01=1, C11=-1
        
        # Term 4 (P4):
        # U: A11=1
        # V: B10=1, B00=-1
        # W: C00=-1
        
        # Term 5 (P5):
        # U: A00=1, A01=1
        # V: B11=1
        # W: C01=1, C11=-1
        
        # Term 6 (P6):
        # U: A10=1, A00=-1
        # V: B00=1, B01=1
        # W: C10=1, C11=1
        
        # Term 7 (P7):
        # U: A01=1, A11=-1
        # V: B10=1, B11=1
        # W: C00=1, C11=-1
        
        # Let's verify C11: P1(1) - P5(1) + P6(1) - P3(1) = 0?
        # Formula: C11 = P1 - P5 + P6 - P3.
        # My W for P1: C11=1.
        # My W for P5: C11=-1.
        # My W for P6: C11=1.
        # My W for P3: C11=-1.
        # Sum: 1 - (-1) + 1 - (-1) = 4? No.
        # The term is sum P_r * (u_r v_r w_r).
        # The value at C11 is sum P_r * W_{C11, r}.
        # P1: W=1.
        # P5: W=-1.
        # P6: W=1.
        # P3: W=-1.
        # Sum = P1 - P5 + P6 - P3. Correct.
        
        # Indices:
        a00 = i_off*m + j_off
        a01 = i_off*m + (j_off+1)
        a10 = (i_off+1)*m + j_off
        a11 = (i_off+1)*m + (j_off+1)
        
        b00 = j_off*p + k_off
        b01 = j_off*p + (k_off+1)
        b10 = (j_off+1)*p + k_off
        b11 = (j_off+1)*p + (k_off+1)
        
        c00 = k_off*n + i_off
        c01 = k_off*n + (i_off+1)
        c10 = (k_off+1)*n + i_off
        c11 = (k_off+1)*n + (i_off+1)
        
        # Add 7 terms
        add_rank1([(a00,1),(a11,1)], [(b00,1),(b11,1)], [(c00,1),(c11,1)])
        add_rank1([(a10,1),(a11,1)], [(b00,1)], [(c00,-1),(c10,1)])
        add_rank1([(a01,1)], [(b11,1),(b01,-1)], [(c01,1),(c11,-1)])
        add_rank1([(a11,1)], [(b10,1),(b00,-1)], [(c00,-1)])
        add_rank1([(a00,1),(a01,1)], [(b11,1)], [(c01,1),(c11,-1)])
        add_rank1([(a10,1),(a00,-1)], [(b00,1),(b01,1)], [(c10,1),(c11,1)])
        add_rank1([(a01,1),(a11,-1)], [(b10,1),(b11,1)], [(c00,1),(c11,-1)])

    def add_schoolbook_1xn(i_off, j_off, k_off, n_size, m_size, p_size):
        """
        Adds schoolbook decomposition for a 1 x m_size x p_size block.
        A is 1 x m_size. B is m_size x p_size. C is p_size x 1.
        """
        for j in range(m_size):
            for k in range(p_size):
                u_idx = (i_off)*m + (j_off + j)
                v_idx = (j_off + j)*p + (k_off + k)
                w_idx = (k_off + k)*n + i_off
                add_term(u_idx, v_idx, w_idx, 1)

    def add_schoolbook_2x2x1(i_off, j_off, k_off):
        """
        Adds schoolbook decomposition for a 2 x 2 x 1 block.
        Rank 4? Or 2?
        M_{2,2,1} rank is 2.
        C0 = A00 B00 + A01 B10
        C1 = A10 B00 + A11 B10
        This is rank 2?
        P1 = (A00+A10) B00
        P2 = (A01+A11) B10
        C0 = P1 - A10 B00 + A01 B10 ? No.
        
        Let's just use schoolbook rank 4 for simplicity if not sure.
        Actually, M_{2,2,1} is rank 2.
        P1 = (A00+A10) B00
        P2 = (A01+A11) B10
        C0 = P1 - A10 B00 + A01 B10?
        No, C0 = A00 B00 + A01 B10.
        C1 = A10 B00 + A11 B10.
        
        Let's use rank 4 schoolbook to be safe and simple.
        """
        for i in range(2):
            for j in range(2):
                for k in range(1):
                    u_idx = (i_off+i)*m + (j_off+j)
                    v_idx = (j_off+j)*p + (k_off+k)
                    w_idx = (k_off+k)*n + (i_off+i)
                    add_term(u_idx, v_idx, w_idx, 1)

    # Decomposition Plan:
    # 1. Block (0, 0, 0) : 1x4x7? No.
    # We split n=3 into 1 + 2.
    # Block 1: i in [0], j in [0..3], k in [0..6]. (1x4x7)
    # Block 2: i in [1..2], j in [0..3], k in [0..6]. (2x4x7)
    
    # Block 1: 1x4x7
    # Use schoolbook. Rank 28.
    add_schoolbook_1xn(0, 0, 0, 1, 4, 7)
    
    # Block 2: 2x4x7
    # Split j into [0..1] and [2..3].
    # Sub-block 2a: i in [1..2], j in [0..1], k in [0..6]. (2x2x7)
    # Sub-block 2b: i in [1..2], j in [2..3], k in [0..6]. (2x2x7)
    
    # Sub-block 2a: 2x2x7
    # Split k into [0..1], [2..3], [4..5], [6].
    # 2a-1: 2x2x2 (Strassen, Rank 7)
    # 2a-2: 2x2x2 (Strassen, Rank 7)
    # 2a-3: 2x2x2 (Strassen, Rank 7)
    # 2a-4: 2x2x1 (Schoolbook Rank 4, or Opt Rank 2)
    
    add_strassen(1, 0, 0) # i_off=1, j_off=0, k_off=0
    add_strassen(1, 0, 2) # i_off=1, j_off=0, k_off=2
    add_strassen(1, 0, 4) # i_off=1, j_off=0, k_off=4
    add_schoolbook_2x2x1(1, 0, 6) # i_off=1, j_off=0, k_off=6
    
    # Sub-block 2b: 2x2x7
    # 2b-1: 2x2x2
    # 2b-2: 2x2x2
    # 2b-3: 2x2x2
    # 2b-4: 2x2x1
    
    add_strassen(1, 2, 0)
    add_strassen(1, 2, 2)
    add_strassen(1, 2, 4)
    add_schoolbook_2x2x1(1, 2, 6)
    
    # Total Rank:
    # Block 1: 28
    # 2a: 7+7+7+4 = 25
    # 2b: 7+7+7+4 = 25
    # Total: 28 + 25 + 25 = 78.
    
    # Trim unused columns
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
