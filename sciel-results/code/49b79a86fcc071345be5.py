import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140.0

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_map, v_map, w_map):
        u = np.zeros(n * m, dtype=float)
        v = np.zeros(m * p, dtype=float)
        w = np.zeros(p * n, dtype=float)
        for (i, j), val in u_map.items():
            u[i * m + j] = val
        for (j, k), val in v_map.items():
            v[j * p + k] = val
        for (k, i), val in w_map.items():
            w[k * n + i] = val
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)

    def strassen_2x2(i0, i1, j0, j1, k0, k1):
        """
        Adds 7 terms for Strassen multiplication of 2x2 blocks.
        A rows i0, i1; cols j0, j1.
        B rows j0, j1; cols k0, k1.
        C rows k0, k1; cols i0, i1.
        """
        # M1 = (A00+A11)(B00+B11)
        add_term({(i0,j0):1, (i1,j1):1}, {(j0,k0):1, (j1,k1):1}, {(k0,i0):1, (k1,i1):1})
        # M2 = (A10+A11)B00
        add_term({(i1,j0):1, (i1,j1):1}, {(j0,k0):1}, {(k0,i1):1, (k1,i1):-1})
        # M3 = A00(B01-B11)
        add_term({(i0,j0):1}, {(j0,k1):1, (j1,k1):-1}, {(k1,i0):1, (k1,i1):1})
        # M4 = A11(B10-B00)
        add_term({(i1,j1):1}, {(j1,k0):1, (j0,k0):-1}, {(k0,i0):1, (k0,i1):1})
        # M5 = (A00+A01)B11
        add_term({(i0,j0):1, (i0,j1):1}, {(j1,k1):1}, {(k0,i0):-1, (k1,i0):1})
        # M6 = (A10-A00)(B00+B01)
        add_term({(i1,j0):1, (i0,j0):-1}, {(j0,k0):1, (j0,k1):1}, {(k1,i1):1})
        # M7 = (A01-A11)(B10+B11)
        add_term({(i0,j1):1, (i1,j1):-1}, {(j1,k0):1, (j1,k1):1}, {(k0,i0):1})

    def schoolbook_block(i_range, j_range, k_range):
        """
        Adds terms for schoolbook multiplication of blocks.
        A: i in i_range, j in j_range
        B: j in j_range, k in k_range
        C: k in k_range, i in i_range
        """
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    add_term({(i,j):1}, {(j,k):1}, {(k,i):1})

    # Split indices into top (0,1) and bottom (2,3,4)
    # Top indices: 0, 1
    # Bottom indices: 2, 3, 4
    
    # We need to compute C = A * B
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # We split the sum over j into j in {0,1} and j in {2,3,4}
    # But we also split i and k to apply Strassen where possible.
    
    # Block 1: i in {0,1}, k in {0,1} (C_22)
    # C_22 = A_22 B_22 + A_23 B_32
    # A_22: i in {0,1}, j in {0,1}
    # B_22: j in {0,1}, k in {0,1}
    # A_23: i in {0,1}, j in {2,3,4}
    # B_32: j in {2,3,4}, k in {0,1}
    
    # A_22 B_22: Strassen
    strassen_2x2(0, 1, 0, 1, 0, 1)
    
    # A_23 B_32:
    # Split j in {2,3,4} into {2,3} and {4}
    # A_23_a: i in {0,1}, j in {2,3}
    # B_32_a: j in {2,3}, k in {0,1}
    # A_23_b: i in {0,1}, j in {4}
    # B_32_b: j in {4}, k in {0,1}
    
    # A_23_a B_32_a: 2x2 * 2x2 -> Strassen
    strassen_2x2(0, 1, 2, 3, 0, 1)
    
    # A_23_b B_32_b: 2x1 * 1x2 -> Schoolbook (2*1*2 = 4 terms)
    # i in {0,1}, j=4, k in {0,1}
    for i in [0, 1]:
        for k in [0, 1]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})
            
    # Note: A_23_a * B_32_b and A_23_b * B_32_a are zero because j indices don't match.

    # Block 2: i in {0,1}, k in {2,3,4} (C_23)
    # C_23 = A_22 B_23 + A_23 B_33
    # A_22: i in {0,1}, j in {0,1}
    # B_23: j in {0,1}, k in {2,3,4}
    # A_23: i in {0,1}, j in {2,3,4}
    # B_33: j in {2,3,4}, k in {2,3,4}
    
    # A_22 B_23:
    # Split k in {2,3,4} into {2,3} and {4}
    # B_23_a: j in {0,1}, k in {2,3}
    # B_23_b: j in {0,1}, k in {4}
    
    # A_22 B_23_a: 2x2 * 2x2 -> Strassen
    strassen_2x2(0, 1, 0, 1, 2, 3)
    
    # A_22 B_23_b: 2x2 * 2x1 -> Schoolbook (2*2*1 = 4 terms)
    for i in [0, 1]:
        for j in [0, 1]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})

    # A_23 B_33:
    # Split j in {2,3,4} into {2,3} and {4}
    # Split k in {2,3,4} into {2,3} and {4}
    
    # A_23_a (j in {2,3}) * B_33_a (j in {2,3}, k in {2,3}): 2x2 * 2x2 -> Strassen
    strassen_2x2(0, 1, 2, 3, 2, 3)
    
    # A_23_a (j in {2,3}) * B_33_b (j in {2,3}, k in {4}): 2x2 * 2x1 -> Schoolbook (4 terms)
    for i in [0, 1]:
        for j in [2, 3]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
            
    # A_23_b (j in {4}) * B_33_b (j in {4}, k in {4}): 2x1 * 1x1 -> Schoolbook (2 terms)
    for i in [0, 1]:
        add_term({(i,4):1}, {(4,4):1}, {(4,i):1})
        
    # A_23_b (j in {4}) * B_33_a (j in {2,3}, k in {2,3}): Mismatch (j=4 vs j=2,3)
    # A_23_a (j in {2,3}) * B_33_c? No, B_33_b is k=4.
    # What about A_23_b (j=4) * B_33_k23?
    # B_33 has j in {2,3,4}.
    # If we take B_33 part with j=4, k in {2,3}:
    # A_23_b (j=4) * B_33_c (j=4, k in {2,3}): 2x1 * 1x2 -> Schoolbook (4 terms)
    for i in [0, 1]:
        for k in [2, 3]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})

    # Block 3: i in {2,3,4}, k in {0,1} (C_32)
    # C_32 = A_32 B_22 + A_33 B_32
    # A_32: i in {2,3,4}, j in {0,1}
    # B_22: j in {0,1}, k in {0,1}
    # A_33: i in {2,3,4}, j in {2,3,4}
    # B_32: j in {2,3,4}, k in {0,1}
    
    # A_32 B_22:
    # Split i in {2,3,4} into {2,3} and {4}
    # A_32_a: i in {2,3}, j in {0,1}
    # A_32_b: i in {4}, j in {0,1}
    
    # A_32_a B_22: 2x2 * 2x2 -> Strassen
    strassen_2x2(2, 3, 0, 1, 0, 1)
    
    # A_32_b B_22: 1x2 * 2x2 -> Schoolbook (4 terms)
    for j in [0, 1]:
        for k in [0, 1]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})

    # A_33 B_32:
    # Split i in {2,3,4} into {2,3} and {4}
    # Split j in {2,3,4} into {2,3} and {4}
    
    # A_33_a (i in {2,3}, j in {2,3}) * B_32_a (j in {2,3}, k in {0,1}): 2x2 * 2x2 -> Strassen
    strassen_2x2(2, 3, 2, 3, 0, 1)
    
    # A_33_a (i in {2,3}, j in {2,3}) * B_32_b (j in {4}, k in {0,1}): Mismatch
    # A_33_b (i in {2,3}, j in {4}) * B_32_a (j in {2,3}, k in {0,1}): Mismatch
    # A_33_b (i in {2,3}, j in {4}) * B_32_b (j in {4}, k in {0,1}): 2x1 * 1x2 -> Schoolbook (4 terms)
    for i in [2, 3]:
        for k in [0, 1]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})
            
    # A_33_c (i in {4}, j in {2,3}) * B_32_a (j in {2,3}, k in {0,1}): 1x2 * 2x2 -> Schoolbook (4 terms)
    for j in [2, 3]:
        for k in [0, 1]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
            
    # A_33_d (i in {4}, j in {4}) * B_32_b (j in {4}, k in {0,1}): 1x1 * 1x2 -> Schoolbook (2 terms)
    for k in [0, 1]:
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})

    # Block 4: i in {2,3,4}, k in {2,3,4} (C_33)
    # C_33 = A_32 B_23 + A_33 B_33
    # A_32: i in {2,3,4}, j in {0,1}
    # B_23: j in {0,1}, k in {2,3,4}
    # A_33: i in {2,3,4}, j in {2,3,4}
    # B_33: j in {2,3,4}, k in {2,3,4}
    
    # A_32 B_23:
    # Split i in {2,3,4} into {2,3} and {4}
    # Split k in {2,3,4} into {2,3} and {4}
    
    # A_32_a (i in {2,3}) * B_23_a (k in {2,3}): 2x2 * 2x2 -> Strassen
    strassen_2x2(2, 3, 0, 1, 2, 3)
    
    # A_32_a (i in {2,3}) * B_23_b (k in {4}): 2x2 * 2x1 -> Schoolbook (4 terms)
    for i in [2, 3]:
        for j in [0, 1]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
            
    # A_32_b (i in {4}) * B_23_a (k in {2,3}): 1x2 * 2x2 -> Schoolbook (4 terms)
    for j in [0, 1]:
        for k in [2, 3]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
            
    # A_32_b (i in {4}) * B_23_b (k in {4}): 1x2 * 2x1 -> Schoolbook (2 terms)
    for j in [0, 1]:
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})

    # A_33 B_33:
    # Split i, j, k into {2,3} and {4}
    
    # A_33_a (i in {2,3}, j in {2,3}) * B_33_a (j in {2,3}, k in {2,3}): 2x2 * 2x2 -> Strassen
    strassen_2x2(2, 3, 2, 3, 2, 3)
    
    # A_33_a (i in {2,3}, j in {2,3}) * B_33_b (j in {2,3}, k in {4}): 2x2 * 2x1 -> Schoolbook (4 terms)
    for i in [2, 3]:
        for j in [2, 3]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
            
    # A_33_b (i in {2,3}, j in {4}) * B_33_b (j in {4}, k in {4}): 2x1 * 1x1 -> Schoolbook (2 terms)
    for i in [2, 3]:
        add_term({(i,4):1}, {(4,4):1}, {(4,i):1})
        
    # A_33_c (i in {4}, j in {2,3}) * B_33_a (j in {2,3}, k in {2,3}): 1x2 * 2x2 -> Schoolbook (4 terms)
    for j in [2, 3]:
        for k in [2, 3]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
            
    # A_33_c (i in {4}, j in {2,3}) * B_33_b (j in {2,3}, k in {4}): 1x2 * 2x1 -> Schoolbook (2 terms)
    for j in [2, 3]:
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})
        
    # A_33_d (i in {4}, j in {4}) * B_33_b (j in {4}, k in {4}): 1x1 * 1x1 -> Schoolbook (1 term)
    add_term({(4,4):1}, {(4,4):1}, {(4,4):1})

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
