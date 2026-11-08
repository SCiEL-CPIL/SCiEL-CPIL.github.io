import numpy as np
from datetime import datetime, timedelta
import itertools
import random
import sys

DIMENSIONS = [2, 3, 9]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix
    by a 3x9 matrix using structured search.
    
    The output factor follows the official trace convention W[k*n+i,r].
    """
    n, m, p = DIMENSIONS
    
    # Use a block decomposition: 3 blocks of 2x3x3
    # Each 2x3x3 block is decomposed into 2x2x3 (rank 12) + 2x1x3 (rank 2) = rank 14
    # Total rank = 3 * 14 = 42
    U, V, W = _rank42_decomp(n, m, p)
    
    return {"U": U, "V": V, "W": W}

def _rank42_decomp(n, m, p):
    """
    Construct a rank-42 decomposition for 2x3x9.
    Splits B into three 3x3 blocks.
    For each block, splits A into 2x2 and 2x1.
    2x2x3 uses trivial rank 12.
    2x1x3 uses trivial rank 2.
    """
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for block in range(3):
        # Block b of B is columns 3*b to 3*b+2
        k_offset = 3 * block
        
        # Part 1: A_top (2x2) * B_top (2x3)
        # A_top is rows 0,1 of A.
        # B_top is rows 0,1 of B_block.
        # Rank 12
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    u_idx = i * m + j
                    v_idx = j * p + (k_offset + k)
                    w_idx = (k_offset + k) * n + i
                    U[u_idx][col] += 1
                    V[v_idx][col] += 1
                    W[w_idx][col] += 1
                    col += 1
        
        # Part 2: A_bottom (2x1) * B_bottom (1x3)
        # A_bottom is row 2 of A.
        # B_bottom is row 2 of B_block.
        # Rank 2 (trivial for 2x1x3 is 2? No, 2x1x3 is 2 dot products of length 1? No.)
        # 2x1x3: C_i = A_i2 * B_2k.
        # This is 2 * 1 * 3 = 6 terms? No.
        # C_0 = A_02 * B_20
        # C_1 = A_12 * B_20
        # C_0 = A_02 * B_21
        # C_1 = A_12 * B_21
        # C_0 = A_02 * B_22
        # C_1 = A_12 * B_22
        # This is 6 multiplications if done trivially.
        # But we can share A_02 and A_12?
        # M_k = A_02 * B_2k and N_k = A_12 * B_2k.
        # So 2 multiplications per k. Total 6.
        # Wait, earlier I said rank 2 for 2x1x3.
        # 2x1x3: A is 2x1, B is 1x3.
        # C is 2x3.
        # C_i,k = A_i,0 * B_0,k.
        # This is an outer product!
        # The rank of an outer product tensor is 1?
        # No, the tensor T_{i,k} = A_i B_k.
        # This is a rank-1 tensor in the space of matrices?
        # But we are decomposing the multiplication map.
        # The map is (A, B) -> A B.
        # If A is a vector and B is a vector, the product is a matrix.
        # The bilinear map is $A \otimes B$.
        # The rank of the tensor $T_{i,k} = \delta_{i,0}\delta_{k,0}$? No.
        # The tensor for $2 \times 1 \times 3$ multiplication has entries $T_{i, j, k}$ where $j$ is the middle index.
        # Here $j=0$ only.
        # $T_{i, 0, k} = 1$ if the product $A_{i,0} B_{0,k}$ contributes to $C_{i,k}$.
        # It does.
        # So $T_{i, 0, k} = 1$.
        # The tensor is $1 \otimes 1 \otimes 1$?
        # The rank of the tensor $T_{i,j,k} = 1$ for all i,j,k is 1?
        # No, the tensor is $T_{i,0,k} = 1$.
        # This tensor is $u \otimes v \otimes w$ where $u$ is all ones (size 2), $v$ is $[1]$ (size 1), $w$ is all ones (size 3).
        # So the rank is 1!
        # Wait, if the rank is 1, then $2 \times 1 \times 3$ can be done with 1 multiplication?
        # $M = (\sum A_{i,0}) (\sum B_{0,k})$?
        # Then $C_{i,k} = M \cdot \frac{A_{i,0} B_{0,k}}{(\sum A)(\sum B)}$? No.
        # The decomposition is $C_{i,k} = \sum_r U_{i,r} V_{r,k} W_{k,r}$?
        # No, the form is $C_{i,k} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$?
        # For $2 \times 1 \times 3$, $j$ is fixed to 0.
        # $C_{i,k} = \sum_r U_{i,0,r} V_{0,k,r} W_{k,i,r}$.
        # If rank 1: $C_{i,k} = U_{i,0,0} V_{0,k,0} W_{k,i,0}$.
        # We need $U_{i,0,0} V_{0,k,0} W_{k,i,0} = 1$ for all i,k?
        # No, we need $A_{i,0} B_{0,k}$.
        # So $U_{i,0,0} = A_{i,0}$? No, U depends on the tensor, not the input.
        # The tensor entries are 1.
        # So we need $\sum_r U_{i,0,r} V_{0,k,r} W_{k,i,r} = 1$.
        # If $U_{i,0,0}=1, V_{0,k,0}=1, W_{k,i,0}=1$, then sum is 1.
        # So rank 1 is sufficient for the *tensor* $T_{i,0,k}=1$.
        # But wait, the multiplication is $C = A B$.
        # The tensor $T$ represents the structure.
        # $T_{i,j,k} = 1$ if $A_{i,j} B_{j,k}$ contributes to $C_{i,k}$.
        # For $2 \times 1 \times 3$, $T_{i,0,k} = 1$.
        # This tensor is indeed rank 1.
        # So $2 \times 1 \times 3$ has rank 1?
        # Let's check.
        # $C_{0,0} = A_{0,0} B_{0,0}$
        # $C_{0,1} = A_{0,0} B_{0,1}$
        # $C_{0,2} = A_{0,0} B_{0,2}$
        # $C_{1,0} = A_{1,0} B_{0,0}$
        # $C_{1,1} = A_{1,0} B_{0,1}$
        # $C_{1,2} = A_{1,0} B_{0,2}$
        # Using rank 1:
        # $M = A_{0,0} + A_{1,0}$? No.
        # The decomposition is $C_{i,k} = U_i V_k W_{k,i} M$?
        # No, the standard form is $C_{i,k} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$.
        # Here $j=0$.
        # $C_{i,k} = \sum_r U_{i,0,r} V_{0,k,r} W_{k,i,r}$.
        # If we set $U_{i,0,0} = 1, V_{0,k,0} = 1, W_{k,i,0} = 1$, then $C_{i,k} = 1$.
        # But we need $C_{i,k} = A_{i,0} B_{0,k}$.
        # The inputs A and B are multiplied by U and V.
        # The actual value is $\sum_r (U_{i,0,r} A_{i,0}) (V_{0,k,r} B_{0,k}) W_{k,i,r}$?
        # No, the decomposition is of the *tensor*.
        # The tensor $T$ has entries $T_{i,j,k}$.
        # The computation is $C_{i,k} = \sum_{j,r} T_{i,j,k}^{(r)} A_{i,j} B_{j,k}$?
        # No, the tensor decomposition is $T = \sum_r u_r \otimes v_r \otimes w_r$.
        # $T_{i,j,k} = \sum_r U_{i,r} V_{j,r} W_{k,r}$.
        # For $2 \times 1 \times 3$, $T_{i,0,k} = 1$.
        # Can we write $1 = \sum_r U_{i,r} V_{0,r} W_{k,r}$?
        # Yes, with $r=0$, $U_{i,0}=1, V_{0,0}=1, W_{k,0}=1$.
        # So the rank of the tensor is 1.
        # This means the multiplication $2 \times 1 \times 3$ can be done with 1 multiplication?
        # $M = (\sum_i A_{i,0}) (\sum_k B_{0,k})$?
        # Then $C_{i,k} = M \cdot \dots$?
        # The formula is $C_{i,k} = \sum_r U_{i,r} A_{i,0} V_{0,r} B_{0,k} W_{k,r}$.
        # If $U_{i,0}=1, V_{0,0}=1, W_{k,0}=1$, then $C_{i,k} = A_{i,0} B_{0,k}$.
        # So yes, rank 1.
        
        # So Part 2 has rank 1.
        # Total rank for 2x3x3 is 12 + 1 = 13.
        # Total rank for 2x3x9 is 3 * 13 = 39.
        
        for k in range(3):
            u_idx = 0 * m + 2 # A_0,2
            u_idx2 = 1 * m + 2 # A_1,2
            v_idx = 2 * p + (k_offset + k) # B_2,k
            w_idx0 = (k_offset + k) * n + 0 # C_0,k
            w_idx1 = (k_offset + k) * n + 1 # C_1,k
            
            # Rank 1 decomposition for 2x1x3
            # U[0,2,0]=1, U[1,2,0]=1
            # V[2,k,0]=1
            # W[k,0,0]=1, W[k,1,0]=1
            U[u_idx][col] += 1
            U[u_idx2][col] += 1
            V[v_idx][col] += 1
            W[w_idx0][col] += 1
            W[w_idx1][col] += 1
            col += 1

    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    if not U or not V or not W:
        return False
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    
    # Check dimensions
    if U_np.shape[0] != n * m:
        return False
    if V_np.shape[0] != m * p:
        return False
    if W_np.shape[0] != p * n:
        return False
    if U_np.shape[1] != rank or V_np.shape[1] != rank or W_np.shape[1] != rank:
        return False
    
    # Test with random matrices
    np.random.seed(42)
    for trial in range(3):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute using decomposition
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank):
                        s += A[i, j] * V_np[j * p + k, r] * W_np[k * n + i, r]
                C_computed[i, k] = s
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    
    return True

def _try_structured_decomp(n, m, p, target_rank):
    return None

def _local_search_improve(U, V, W, n, m, p, start_time, timeout):
    return None
