"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    
    Uses structured algebraic ansatz with Strassen-like splitting.
    """
    n, m, p = 2, 3, 3
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    def verify_decomp(U, V, W, tol=1e-10):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        rank = U.shape[1]
        if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1) > tol:
                        return False
        return True

    # Schoolbook baseline
    U_sb = np.zeros((6, 18), dtype=int)
    V_sb = np.zeros((9, 18), dtype=int)
    W_sb = np.zeros((6, 18), dtype=int)
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U_sb[i*3+j, col] = 1
                V_sb[j*3+k, col] = 1
                W_sb[k*2+i, col] = 1
                col += 1
    best_decomp = (U_sb, V_sb, W_sb)
    best_rank = 18
    
    # STRASSEN-LIKE ALGEBRAIC ANSATZ
    # Key idea: Split B into overlapping 2x2 blocks and use linear combinations
    # For 2x3 * 3x3, we can view this as computing two rows of C
    
    # Strategy: Use the fact that with only 2 rows in A, we can create
    # shared intermediate products
    
    # Let A = [a00 a01 a02; a10 a11 a12]
    # Let B = [b00 b01 b02; b10 b11 b12; b20 b21 b22]
    
    # Define linear combinations of A rows:
    # s0 = a00, s1 = a01, s2 = a02, s3 = a10, s4 = a11, s5 = a12
    # s6 = a00 + a10, s7 = a01 + a11, s8 = a02 + a12
    # s9 = a00 + a01, s10 = a01 + a02, s11 = a10 + a11, s12 = a11 + a12
    
    # Define linear combinations of B columns:
    # t0 = b00, t1 = b01, t2 = b02, t3 = b10, t4 = b11, t5 = b12
    # t6 = b20, t7 = b21, t8 = b22
    # t9 = b00 + b10, t10 = b01 + b11, t11 = b02 + b12
    # t12 = b10 + b20, t13 = b11 + b21, t14 = b12 + b22
    # t15 = b00 + b01, t16 = b01 + b02, t17 = b10 + b11, t18 = b11 + b12
    # t19 = b20 + b21, t20 = b21 + b22
    
    # Define linear combinations of C entries:
    # r0 = c00, r1 = c01, r2 = c02, r3 = c10, r4 = c11, r5 = c12
    # r6 = c00 + c10, r7 = c01 + c11, r8 = c02 + c12
    # r9 = c00 + c01, r10 = c01 + c02, r11 = c10 + c11, r12 = c11 + c12
    
    # Try rank-16 construction with specific pattern:
    # Products of form: (linear combo of A) * (linear combo of B) -> (linear combo of C)
    
    def try_strassen_ansatz(rank=16):
        """Try structured Strassen-like ansatz for given rank."""
        if time.time() > deadline:
            return None
        
        # Define basis vectors for A combinations (6 entries: a00,a01,a02,a10,a11,a12)
        A_combos = [
            [1,0,0,0,0,0],  # a00
            [0,1,0,0,0,0],  # a01
            [0,0,1,0,0,0],  # a02
            [0,0,0,1,0,0],  # a10
            [0,0,0,0,1,0],  # a11
            [0,0,0,0,0,1],  # a12
            [1,0,0,1,0,0],  # a00+a10
            [0,1,0,0,1,0],  # a01+a11
            [0,0,1,0,0,1],  # a02+a12
            [1,1,0,0,0,0],  # a00+a01
            [0,1,1,0,0,0],  # a01+a02
            [0,0,0,1,1,0],  # a10+a11
            [0,0,0,0,1,1],  # a11+a12
            [1,0,0,0,1,0],  # a00+a11
            [0,1,0,0,0,1],  # a01+a12
            [0,0,1,1,0,0],  # a02+a10
            [1,1,0,1,0,0],  # a00+a01+a10
            [0,1,1,0,1,0],  # a01+a02+a11
            [0,0,1,0,0,1],  # a02+a12
            [1,0,0,0,0,1],  # a00+a12
        ]
        
        # Define basis vectors for B combinations (9 entries: b00..b22)
        B_combos = [
            [1,0,0,0,0,0,0,0,0],  # b00
            [0,1,0,0,0,0,0,0,0],  # b01
            [0,0,1,0,0,0,0,0,0],  # b02
            [0,0,0,1,0,0,0,0,0],  # b10
            [0,0,0,0,1,0,0,0,0],  # b11
            [0,0,0,0,0,1,0,0,0],  # b12
            [0,0,0,0,0,0,1,0,0],  # b20
            [0,0,0,0,0,0,0,1,0],  # b21
            [0,0,0,0,0,0,0,0,1],  # b22
            [1,0,0,1,0,0,0,0,0],  # b00+b10
            [0,1,0,0,1,0,0,0,0],  # b01+b11
            [0,0,1,0,0,1,0,0,0],  # b02+b12
            [0,0,0,1,0,0,1,0,0],  # b10+b20
            [0,0,0,0,1,0,0,1,0],  # b11+b21
            [0,0,0,0,0,1,0,0,1],  # b12+b22
            [1,0,0,0,1,0,0,0,0],  # b00+b11
            [0,1,0,0,0,1,0,0,0],  # b01+b12
            [0,0,0,1,0,0,0,1,0],  # b10+b21
            [0,0,0,0,1,0,0,0,1],  # b11+b22
            [1,0,0,0,0,0,1,0,0],  # b00+b20
            [0,1,0,0,0,0,0,1,0],  # b01+b21
            [0,0,1,0,0,0,0,0,1],  # b02+b22
            [1,1,0,0,0,0,0,0,0],  # b00+b01
            [0,1,1,0,0,0,0,0,0],  # b01+b02
            [0,0,0,1,1,0,0,0,0],  # b10+b11
            [0,0,0,0,1,1,0,0,0],  # b11+b12
            [0,0,0,0,0,0,1,1,0],  # b20+b21
            [0,0,0,0,0,0,0,1,1],  # b21+b22
            [1,0,0,0,0,0,0,1,0],  # b00+b21
            [0,1,0,0,0,0,1,0,0],  # b01+b20
        ]
        
        # Define basis vectors for C combinations (6 entries: c00..c12)
        C_combos = [
            [1,0,0,0,0,0],  # c00
            [0,1,0,0,0,0],  # c01
            [0,0,1,0,0,0],  # c02
            [0,0,0,1,0,0],  # c10
            [0,0,0,0,1,0],  # c11
            [0,0,0,0,0,1],  # c12
            [1,0,0,1,0,0],  # c00+c10
            [0,1,0,0,1,0],  # c01+c11
            [0,0,1,0,0,1],  # c02+c12
            [1,1,0,0,0,0],  # c00+c01
            [0,1,1,0,0,0],  # c01+c02
            [0,0,0,1,1,0],  # c10+c11
            [0,0,0,0,1,1],  # c11+c12
            [1,0,0,0,1,0],  # c00+c11
            [0,1,0,0,0,1],  # c01+c12
            [0,0,1,1,0,0],  # c02+c10
            [1,1,0,1,0,0],  # c00+c01+c10
            [0,1,1,0,1,0],  # c01+c02+c11
            [1,0,0,0,0,1],  # c00+c12
            [0,0,1,0,0,1],  # c02+c12
        ]
        
        # For each triple (A_combo, B_combo, C_combo), check if it's valid
        # A_combo * B_combo should equal C_combo when applied to the tensor
        
        valid_triples = []
        for ai, a_combo in enumerate(A_combos):
            for bi, b_combo in enumerate(B_combos):
                for ci, c_combo in enumerate(C_combos):
                    # Check: sum_j a_combo[j] * b_combo[j'] = c_combo contribution
                    # Actually, we need: for all i,j,k:
                    # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
                    # This is complex, so let's just try all combinations and verify
                    
                    # Create U, V, W for this single term
                    U_single = np.zeros((6, 1))
                    V_single = np.zeros((9, 1))
                    W_single = np.zeros((6, 1))
                    
                    U_single[:, 0] = a_combo
                    V_single[:, 0] = b_combo
                    W_single[:, 0] = c_combo
                    
                    # Check if this term contributes correctly
                    # For the full tensor, we need sum over all terms
                    # A single term contributes: U[i,j] * V[j,k] * W[k,i] for each (i,j,k)
                    contrib = 0
                    for i in range(2):
                        for j in range(3):
                            for k in range(3):
                                val = a_combo[i*3+j] * b_combo[j*3+k] * c_combo[k*2+i]
                                contrib += val
                    
                    # Store for later combination
                    valid_triples.append((ai, bi, ci, a_combo, b_combo, c_combo))
        
        # Now try to find a subset of size `rank` that sums to the identity tensor
        # Use greedy + local search
        
        import random
        random.seed(42)
        
        # Precompute contribution vectors for each triple
        contrib_vectors = []
        for (ai, bi, ci, a_combo, b_combo, c_combo) in valid_triples:
            vec = np.zeros(18)
            idx = 0
            for i in range(2):
                for j in range(3):
                    for k in range(3):
                        vec[idx] = a_combo[i*3+j] * b_combo[j*3+k] * c_combo[k*2+i]
                        idx += 1
            contrib_vectors.append(vec)
        
        target = np.ones(18)
        
        # Try random subsets
        for attempt in range(5000):
            if time.time() > deadline:
                return None
            
            # Select random subset
            indices = random.sample(range(len(valid_triples)), rank)
            
            # Check if sum of contributions equals target
            total = np.zeros(18)
            for idx in indices:
                total += contrib_vectors[idx]
            
            if np.allclose(total, target, atol=1e-10):
                # Found valid decomposition!
                U = np.zeros((6, rank), dtype=int)
                V = np.zeros((9, rank), dtype=int)
                W = np.zeros((6, rank), dtype=int)
                
                for c, idx in enumerate(indices):
                    ai, bi, ci, a_combo, b_combo, c_combo = valid_triples[idx]
                    U[:, c] = a_combo
                    V[:, c] = b_combo
                    W[:, c] = c_combo
                
                if verify_decomp(U, V, W):
                    return (U, V, W)
        
        return None
    
    # Try rank 16 first
    result = try_strassen_ansatz(rank=16)
    if result is not None:
        best_rank = 16
        best_decomp = result
    elif time.time() < deadline - 20:
        # Try rank 17
        result = try_strassen_ansatz(rank=17)
        if result is not None:
            best_rank = 17
            best_decomp = result
        elif time.time() < deadline - 10:
            # Try rank 15
            result = try_strassen_ansatz(rank=15)
            if result is not None:
                best_rank = 15
                best_decomp = result
    
    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
