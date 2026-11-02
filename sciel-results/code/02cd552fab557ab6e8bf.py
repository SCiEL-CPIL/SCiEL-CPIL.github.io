"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Construct a rank-42 decomposition for 2x2 * 2x12 matrix multiplication.
    This uses the Strassen algorithm for the 2x2x2 core, repeated for each pair of columns.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    num_pairs = p // 2  # 6
    rank = num_pairs * 7  # 42
    
    # Initialize factor matrices
    # U: (n*m, rank) = (4, 42)
    # V: (m*p, rank) = (24, 42)
    # W: (p*n, rank) = (24, 42)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 7 products for 2x2 * 2x2
    # Indices: A[i,j], B[j,k], C[i,k] with i,j,k in {0,1}
    # U index: i*m + j
    # V index: j*p + k (but within the 2x2 block, k is 0 or 1)
    # W index: k*n + i
    
    # Define the Strassen factors
    # Each product s has:
    # u_s: vector of length 4 (A entries)
    # v_s: vector of length 2 (B entries for this block)
    # w_s: vector of length 2 (C entries for this block)
    
    # Standard Strassen:
    # P1 = (A00 + A11) * (B00 + B11)
    # P2 = (A10 + A11) * B00
    # P3 = A00 * (B01 - B11)
    # P4 = A11 * (B10 - B00)
    # P5 = (A00 + A01) * B11
    # P6 = (A10 - A00) * (B00 + B01)
    # P7 = (A01 - A11) * (B10 + B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # We define the non-zero entries for each factor.
    # Format: (index, value)
    
    strassen_factors = [
        # P1
        {
            'u': [(0, 1), (3, 1)],  # A00, A11
            'v': [(0, 1), (1, 1)],  # B00, B11
            'w': [(0, 1), (3, 1)]   # C00, C11 (indices: 0*2+0=0, 1*2+1=3)
        },
        # P2
        {
            'u': [(2, 1), (3, 1)],  # A10, A11
            'v': [(0, 1)],          # B00
            'w': [(2, 1), (3, -1)]  # C10, C11 (indices: 0*2+1=2, 1*2+1=3) -> Wait.
            # C10 is index 0*2+1 = 2. C11 is index 1*2+1 = 3.
            # Formula: C10 = P2 + P4. So P2 contributes +1 to C10.
            # C11 = P1 - P2 + P3 + P6. So P2 contributes -1 to C11.
        },
        # P3
        {
            'u': [(0, 1)],          # A00
            'v': [(1, 1), (3, -1)], # B01, B11 -> indices in 2x2 block: B[0,1]=1, B[1,1]=3? No.
            # In the 2x2 block, B indices are 0 (B00), 1 (B01), 2 (B10), 3 (B11) if flattened row-major?
            # My V index definition: j*p + k.
            # In the block, j is 0 or 1. k is 0 or 1.
            # So local indices: 0 (j=0,k=0), 1 (j=0,k=1), 2 (j=1,k=0), 3 (j=1,k=1).
            # B01 is j=0, k=1 -> index 1.
            # B11 is j=1, k=1 -> index 3.
            'w': [(1, 1), (3, 1)]   # C01, C11?
            # C01 = P3 + P5. P3 contributes +1 to C01.
            # C11 = ... + P3 ... P3 contributes +1 to C11.
            # C01 index: k=1, i=0 -> 1*2+0 = 2?
            # Wait, W index is k*n + i.
            # C[i,k]. i in {0,1}, k in {0,1}.
            # C00: i=0,k=0 -> 0*2+0=0.
            # C01: i=0,k=1 -> 1*2+0=2.
            # C10: i=1,k=0 -> 0*2+1=1.
            # C11: i=1,k=1 -> 1*2+1=3.
            # Let's re-verify Strassen C formulas.
            # C00 = P1 + P4 - P5 + P7
            # C01 = P3 + P5
            # C10 = P2 + P4
            # C11 = P1 - P2 + P3 + P6
        },
    ]
    
    # Let's just hardcode the non-zero entries for U, V, W for each of the 7 products.
    # This is safer than deriving it on the fly.
    
    # Product 1: (A00+A11)(B00+B11)
    # U: A00(0), A11(3)
    # V: B00(0), B11(3)
    # W: C00(0), C11(3)
    
    # Product 2: (A10+A11)B00
    # U: A10(2), A11(3)
    # V: B00(0)
    # W: C10(1), C11(3) with coeff -1 for C11
    
    # Product 3: A00(B01-B11)
    # U: A00(0)
    # V: B01(1), B11(3) with coeff -1
    # W: C01(2), C11(3)
    
    # Product 4: A11(B10-B00)
    # U: A11(3)
    # V: B10(2), B00(0) with coeff -1
    # W: C00(0), C10(1)
    
    # Product 5: (A00+A01)B11
    # U: A00(0), A01(1)
    # V: B11(3)
    # W: C01(2), C11(3) with coeff -1 for C01?
    # C01 = P3 + P5. So +1.
    # C11 = ... - P5 ... So -1.
    
    # Product 6: (A10-A00)(B00+B01)
    # U: A10(2), A00(0) with coeff -1
    # V: B00(0), B01(1)
    # W: C11(3)
    
    # Product 7: (A01-A11)(B10+B11)
    # U: A01(1), A11(3) with coeff -1
    # V: B10(2), B11(3)
    # W: C00(0), C10(1) with coeff -1 for C00?
    # C00 = ... + P7. So +1.
    # C10 = ... no P7.
    # Wait, C10 = P2 + P4.
    # C00 = P1 + P4 - P5 + P7.
    # So P7 contributes to C00.
    
    # Let's use a verified list of tuples.
    # (u_indices, v_indices, w_indices)
    # Indices are local to the 2x2 block.
    
    strassen_terms = [
        # 1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # 2
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)]),
        # 3
        ([(0, 1)], [(1, 1), (3, -1)], [(2, 1), (3, 1)]),
        # 4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        # 5
        ([(0, 1), (1, 1)], [(3, 1)], [(2, 1), (3, -1)]),
        # 6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # 7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    col = 0
    for t in range(num_pairs):
        if time.time() > deadline:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        base_k = 2 * t  # Start index for k in B and C
        
        for s in range(7):
            u_entries, v_entries, w_entries = strassen_terms[s]
            
            # Fill U
            for idx, coeff in u_entries:
                U[idx, col] = coeff
                
            # Fill V
            # Global V index: j * p + k
            # Local v index: j * 2 + k_local (0 or 1)
            # k_global = base_k + k_local
            for idx, coeff in v_entries:
                j = idx // 2
                k_local = idx % 2
                global_idx = j * p + (base_k + k_local)
                V[global_idx, col] = coeff
                
            # Fill W
            # Global W index: k * n + i
            # Local w index: k_local * 2 + i
            # k_global = base_k + k_local
            for idx, coeff in w_entries:
                k_local = idx // 2
                i = idx % 2
                global_idx = (base_k + k_local) * n + i
                W[global_idx, col] = coeff
                
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
