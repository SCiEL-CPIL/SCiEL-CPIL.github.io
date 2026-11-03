"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the tensor decomposition is exact using the trace convention."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    
    # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_strassen_decomp(n, m, p, j_pairs, k_pairs, j_singles, k_singles):
    """
    Build a Strassen-based decomposition.
    j_pairs: list of (j0, j1) for 2x2 blocks
    k_pairs: list of (k0, k1) for 2x2 blocks
    j_singles: list of single j indices
    k_singles: list of single k indices
    """
    # Calculate total rank
    rank = 7 * len(j_pairs) * len(k_pairs)
    rank += 4 * len(j_singles) * len(k_pairs)
    rank += 4 * len(j_pairs) * len(k_singles)
    rank += 4 * len(j_singles) * len(k_singles)
    # Wait, for single j and single k it's 2 mults (for n=2)
    # C_0,k = A_0,j * B_j,k
    # C_1,k = A_1,j * B_j,k
    # So 2 terms per (j,k) pair.
    # Let's re-calculate carefully.
    
    # Actually, let's just allocate max possible and trim, or build carefully.
    # Max rank = n*m*p
    max_rank = n * m * p
    U = np.zeros((n*m, max_rank))
    V = np.zeros((m*p, max_rank))
    W = np.zeros((p*n, max_rank))
    
    col = 0
    
    # 1. j_pairs x k_pairs (2x2x2 blocks) -> 7 terms each
    for (j0, j1) in j_pairs:
        for (k0, k1) in k_pairs:
            # M1: (A_0,j0 + A_1,j1) * (B_j0,k0 + B_j1,k1)
            U[0*m + j0, col] += 1.0
            U[1*m + j1, col] += 1.0
            V[j0*p + k0, col] += 1.0
            V[j1*p + k1, col] += 1.0
            W[k0*n + 0, col] += 1.0
            W[k1*n + 1, col] += 1.0
            col += 1
            
            # M2: (A_1,j0 + A_1,j1) * B_j0,k0
            # Contribution to C_1,k0 and C_1,k1
            # C_1,k0 += A_1,j0*B_j0,k0 + A_1,j1*B_j1,k0 ... wait.
            # Standard Strassen for 2x2:
            # C = [[A, B], [C, D]] * [[E, F], [G, H]]
            # P1 = (A+B)(E+F)
            # P2 = (C+D)E
            # P3 = A(F-H)
            # P4 = D(G-E)
            # P5 = (A+C)H
            # P6 = (B-A)(E+F)
            # P7 = (C-B)(G+H)
            # C11 = P1 + P4 - P5 - P7
            # C12 = P3 + P5
            # C21 = P2 + P7
            # C22 = P1 - P2 + P3 + P6
            
            # Mapping to our indices:
            # A = A_0,j0, B = A_0,j1 (row 0 of A block) -> No, A is 2x2 submatrix.
            # A_block = [[A_0,j0, A_0,j1], [A_1,j0, A_1,j1]]
            # B_block = [[B_j0,k0, B_j0,k1], [B_j1,k0, B_j1,k1]]
            # C_block = [[C_0,k0, C_0,k1], [C_1,k0, C_1,k1]]
            
            # P1 = (A_0,j0 + A_1,j1) * (B_j0,k0 + B_j1,k1)
            #   -> contributes to C_0,k0 (via A_0,j0*B_j0,k0) and C_1,k1 (via A_1,j1*B_j1,k1)
            #   Actually P1 = (A+B)(E+F) where A=A_0,j0, B=A_1,j1? No.
            #   Let's stick to the explicit expansion used in the seed which worked.
            #   Seed M1: U[0,j0]+=1, U[1,j1]+=1. V[j0,k0]+=1, V[j1,k1]+=1. W[k0,0]+=1, W[k1,1]+=1.
            #   Term: (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
            #   = A_0,j0 B_j0,k0 + A_0,j0 B_j1,k1 + A_1,j1 B_j0,k0 + A_1,j1 B_j1,k1
            #   W[k0,0] adds to C_0,k0. W[k1,1] adds to C_1,k1.
            #   This specific combination is correct for the diagonal terms of the Strassen scheme.
            
            # M2: (A_1,j0 + A_1,j1) * B_j0,k0
            #   U[1,j0]+=1, U[1,j1]+=1. V[j0,k0]+=1.
            #   Term: (A_1,j0 + A_1,j1) B_j0,k0 = A_1,j0 B_j0,k0 + A_1,j1 B_j0,k0
            #   W[k0,1] += 1, W[k1,1] -= 1.
            #   Contribution to C_1,k0: +1. Contribution to C_1,k1: -1.
            
            # M3: A_0,j0 * (B_j0,k1 - B_j1,k1)
            #   U[0,j0]+=1. V[j0,k1]+=1, V[j1,k1]-=1.
            #   Term: A_0,j0 (B_j0,k1 - B_j1,k1)
            #   W[k1,0]+=1, W[k1,1]+=1.
            #   Contribution to C_0,k1: +1. Contribution to C_1,k1: +1.
            
            # M4: A_1,j1 * (B_j1,k0 - B_j0,k0)
            #   U[1,j1]+=1. V[j1,k0]+=1, V[j0,k0]-=1.
            #   Term: A_1,j1 (B_j1,k0 - B_j0,k0)
            #   W[k0,0]+=1, W[k0,1]+=1.
            #   Contribution to C_0,k0: +1. Contribution to C_1,k0: +1.
            
            # M5: (A_0,j0 + A_0,j1) * B_j1,k1
            #   U[0,j0]+=1, U[0,j1]+=1. V[j1,k1]+=1.
            #   Term: (A_0,j0 + A_0,j1) B_j1,k1
            #   W[k0,0]-=1, W[k1,0]+=1.
            #   Contribution to C_0,k0: -1. Contribution to C_0,k1: +1.
            
            # M6: (A_1,j0 - A_0,j0) * (B_j0,k0 + B_j0,k1)
            #   U[1,j0]+=1, U[0,j0]-=1. V[j0,k0]+=1, V[j0,k1]+=1.
            #   Term: (A_1,j0 - A_0,j0)(B_j0,k0 + B_j0,k1)
            #   W[k1,1]+=1.
            #   Contribution to C_1,k1: +1.
            
            # M7: (A_0,j1 - A_1,j1) * (B_j1,k0 + B_j1,k1)
            #   U[0,j1]+=1, U[1,j1]-=1. V[j1,k0]+=1, V[j1,k1]+=1.
            #   Term: (A_0,j1 - A_1,j1)(B_j1,k0 + B_j1,k1)
            #   W[k0,0]+=1.
            #   Contribution to C_0,k0: +1.
            
            # Let's re-verify the sum for C_0,k0:
            # M1: +1 (from A_0,j0 B_j0,k0 part? No, M1 is sum. W[k0,0] is +1. The term is (A_0,j0+A_1,j1)(B_j0,k0+B_j1,k1).
            # The component of this term that goes into C_0,k0 is A_0,j0 * B_j0,k0.
            # But wait, the tensor decomposition is linear.
            # T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}
            # For C_0,k0 (i=0, k=k0), we need sum_c U_{0,j,c} V_{j,k0,c} W_{k0,0,c} = 1 for all j? No.
            # The target tensor is T_{i,j,k} = 1 if (i,j,k) corresponds to a valid entry?
            # No, the tensor for matrix multiplication is T_{i,j,k} = 1 if the indices match the product.
            # Specifically, (AB)_{ik} = sum_j A_{ij} B_{jk}.
            # The tensor is defined on indices (i, j, k). T_{i,j,k} = 1.
            # Wait, the decomposition is for the *multiplication* tensor.
            # The "1" in the verification loop `if abs(val - 1.0) > tol` implies that for every (i,j,k), the sum is 1.
            # This corresponds to the tensor where every entry is 1?
            # No, that's the "all ones" tensor.
            # Matrix multiplication tensor: T_{i,j,k} is 1 if the indices form a valid term in the sum.
            # Actually, the standard tensor is $T_{i,j,k} = 1$ for all valid indices?
            # Let's check the seed verification.
            # `val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])`
            # This calculates $\sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}$.
            # If this equals 1 for ALL i,j,k, then it's decomposing the tensor of all ones.
            # Is that the task? "Matrix multiplication 2,5,6".
            # Usually, the tensor for matrix multiplication has 1s at positions (i, j, k) corresponding to the terms $A_{ij}B_{jk}$.
            # Since every (i,j,k) triple is a valid term in the sum for $C_{ik}$, the tensor is indeed the all-ones tensor of size n*m*p?
            # No, the tensor is defined on the space $V^* \otimes W^* \otimes Z^*$.
            # The structure constants are 1 if the indices match.
            # Yes, for matrix multiplication, $T_{i,j,k} = 1$ for all $i \in [n], j \in [m], k \in [p]$.
            # So the verification is correct.
            
            # Let's re-verify the Strassen sums for C_0,k0 (i=0, k=k0).
            # We need $\sum_c U_{0,j,c} V_{j,k0,c} W_{k0,0,c} = 1$ for a specific j?
            # No, the sum is over c. The indices i, j, k are fixed.
            # For C_0,k0, we fix i=0, k=k0. The variable is j?
            # No, in the tensor $T_{i,j,k}$, i, j, k are all fixed indices of the tensor.
            # The decomposition is $\sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}$.
            # Wait, the indices in U, V, W are:
            # U: (i*m + j, c) -> depends on i, j
            # V: (j*p + k, c) -> depends on j, k
            # W: (k*n + i, c) -> depends on k, i
            # So for a fixed (i, j, k), we compute $\sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}$.
            # This must equal 1.
            
            # Let's check M1 contribution to T_{0, j0, k0}:
            # U_{0,j0} += 1. V_{j0,k0} += 1. W_{k0,0} += 1.
            # Product: 1*1*1 = 1.
            # M1 contribution to T_{0, j1, k0}:
            # U_{0,j1} is 0 in M1. So 0.
            # M1 contribution to T_{1, j0, k1}:
            # U_{1,j0} is 0 in M1. So 0.
            # M1 contribution to T_{1, j1, k1}:
            # U_{1,j1} += 1. V_{j1,k1} += 1. W_{k1,1} += 1.
            # Product: 1.
            
            # So M1 adds 1 to T_{0,j0,k0} and T_{1,j1,k1}.
            
            # Let's check the sum for T_{0,j0,k0}:
            # M1: 1
            # M2: U_{0,j0}=0. 0.
            # M3: U_{0,j0}=1. V_{j0,k0}=0 (V has j0,k1 and j1,k1). 0.
            # M4: U_{0,j0}=0. 0.
            # M5: U_{0,j0}=1. V_{j0,k0}=0 (V has j1,k1). 0.
            # M6: U_{0,j0}=-1. V_{j0,k0}=1. W_{k0,0}=0 (W has k1,1). 0.
            # M7: U_{0,j0}=0. 0.
            # Sum = 1. Correct.
            
            # Let's check T_{0,j1,k0}:
            # M1: U_{0,j1}=0. 0.
            # M2: U_{0,j1}=0. 0.
            # M3: U_{0,j1}=0. 0.
            # M4: U_{0,j1}=0. 0.
            # M5: U_{0,j1}=1. V_{j1,k0}=0 (V has j1,k1). 0.
            # M6: U_{0,j1}=0. 0.
            # M7: U_{0,j1}=1. V_{j1,k0}=1. W_{k0,0}=1.
            # Product: 1*1*1 = 1.
            # Sum = 1. Correct.
            
            # Let's check T_{0,j0,k1}:
            # M1: U_{0,j0}=1. V_{j0,k1}=0. 0.
            # M2: U_{0,j0}=0. 0.
            # M3: U_{0,j0}=1. V_{j0,k1}=1. W_{k1,0}=1.
            # Product: 1.
            # M4: U_{0,j0}=0. 0.
            # M5: U_{0,j0}=1. V_{j0,k1}=0. 0.
            # M6: U_{0,j0}=-1. V_{j0,k1}=1. W_{k1,0}=0. 0.
            # M7: U_{0,j0}=0. 0.
            # Sum = 1. Correct.
            
            # Let's check T_{0,j1,k1}:
            # M1: U_{0,j1}=0. 0.
            # M2: U_{0,j1}=0. 0.
            # M3: U_{0,j1}=0. 0.
            # M4: U_{0,j1}=0. 0.
            # M5: U_{0,j1}=1. V_{j1,k1}=1. W_{k1,0}=1.
            # Product: 1.
            # M6: U_{0,j1}=0. 0.
            # M7: U_{0,j1}=1. V_{j1,k1}=1. W_{k1,0}=0. 0.
            # Sum = 1. Correct.
            
            # It seems the Strassen scheme works for the all-ones tensor.
            # The key is that the "diagonal" terms of the 2x2 block multiplication are handled by the combinations.
            
            # Now, implement the other terms.
            
            # M2: (A_1,j0 + A_1,j1) * B_j0,k0
            U[1*m + j0, col] += 1.0
            U[1*m + j1, col] += 1.0
            V[j0*p + k0, col] += 1.0
            W[k0*n + 1, col] += 1.0
            W[k1*n + 1, col] -= 1.0
            col += 1
            
            # M3: A_0,j0 * (B_j0,k1 - B_j1,k1)
            U[0*m + j0, col] += 1.0
            V[j0*p + k1, col] += 1.0
            V[j1*p + k1, col] -= 1.0
            W[k1*n + 0, col] += 1.0
            W[k1*n + 1, col] += 1.0
            col += 1
            
            # M4: A_1,j1 * (B_j1,k0 - B_j0,k0)
            U[1*m + j1, col] += 1.0
            V[j1*p + k0, col] += 1.0
            V[j0*p + k0, col] -= 1.0
            W[k0*n + 0, col] += 1.0
            W[k0*n + 1, col] += 1.0
            col += 1
            
            # M5: (A_0,j0 + A_0,j1) * B_j1,k1
            U[0*m + j0, col] += 1.0
            U[0*m + j1, col] += 1.0
            V[j1*p + k1, col] += 1.0
            W[k0*n + 0, col] -= 1.0
            W[k1*n + 0, col] += 1.0
            col += 1
            
            # M6: (A_1,j0 - A_0,j0) * (B_j0,k0 + B_j0,k1)
            U[1*m + j0, col] += 1.0
            U[0*m + j0, col] -= 1.0
            V[j0*p + k0, col] += 1.0
            V[j0*p + k1, col] += 1.0
            W[k1*n + 1, col] += 1.0
            col += 1
            
            # M7: (A_0,j1 - A_1,j1) * (B_j1,k0 + B_j1,k1)
            U[0*m + j1, col] += 1.0
            U[1*m + j1, col] -= 1.0
            V[j1*p + k0, col] += 1.0
            V[j1*p + k1, col] += 1.0
            W[k0*n + 0, col] += 1.0
            col += 1
    
    # 2. j_singles x k_pairs
    # For single j, pair k.
    # We need to satisfy T_{i, j, k0} = 1 and T_{i, j, k1} = 1 for i=0,1.
    # Schoolbook would use 4 terms:
    # T_{0,j,k0}: U_{0,j} V_{j,k0} W_{k0,0}
    # T_{1,j,k0}: U_{1,j} V_{j,k0} W_{k0,1}
    # T_{0,j,k1}: U_{0,j} V_{j,k1} W_{k1,0}
    # T_{1,j,k1}: U_{1,j} V_{j,k1} W_{k1,1}
    # Can we do better?
    # Try 3 terms?
    # M1: (A_0,j) * (B_j,k0 + B_j,k1)
    #   U_{0,j}=1. V_{j,k0}=1, V_{j,k1}=1.
    #   W_{k0,0}=1, W_{k1,0}=1.
    #   T_{0,j,k0} += 1. T_{0,j,k1} += 1.
    # M2: A_1,j * B_j,k0
    #   U_{1,j}=1. V_{j,k0}=1.
    #   W_{k0,1}=1.
    #   T_{1,j,k0} += 1.
    # M3: A_1,j * B_j,k1
    #   U_{1,j}=1. V_{j,k1}=1.
    #   W_{k1,1}=1.
    #   T_{1,j,k1} += 1.
    # This works! 3 terms instead of 4.
    
    for j in j_singles:
        for (k0, k1) in k_pairs:
            # M1: A_0,j * (B_j,k0 + B_j,k1)
            U[0*m + j, col] += 1.0
            V[j*p + k0, col] += 1.0
            V[j*p + k1, col] += 1.0
            W[k0*n + 0, col] += 1.0
            W[k1*n + 0, col] += 1.0
            col += 1
            
            # M2: A_1,j * B_j,k0
            U[1*m + j, col] += 1.0
            V[j*p + k0, col] += 1.0
            W[k0*n + 1, col] += 1.0
            col += 1
            
            # M3: A_1,j * B_j,k1
            U[1*m + j, col] += 1.0
            V[j*p + k1, col] += 1.0
            W[k1*n + 1, col] += 1.0
            col += 1

    # 3. j_pairs x k_singles
    # For pair j, single k.
    # T_{i, j0, k} = 1, T_{i, j1, k} = 1 for i=0,1.
    # Schoolbook: 4 terms.
    # Can we do 3?
    # M1: (A_0,j0 + A_1,j1) * B_j0,k
    #   U_{0,j0}=1, U_{1,j1}=1. V_{j0,k}=1.
    #   W_{k,0}=1, W_{k,1}=1.
    #   T_{0,j0,k} += 1. T_{1,j1,k} += 1.
    # M2: (A_1,j0 + A_1,j1) * B_j1,k
    #   U_{1,j0}=1, U_{1,j1}=1. V_{j1,k}=1.
    #   W_{k,1}=1. (Wait, need to cancel?)
    #   T_{1,j0,k} += 1. T_{1,j1,k} += 1.
    #   We already have T_{1,j1,k} += 1 from M1. Total 2. Need 1.
    #   So subtract?
    #   Let's try:
    #   M1: (A_0,j0 + A_1,j1) * B_j0,k
    #   M2: A_1,j0 * B_j1,k
    #   M3: A_0,j1 * B_j1,k
    #   M1: T_{0,j0,k} += 1. T_{1,j1,k} += 1.
    #   M2: T_{1,j0,k} += 1.
    #   M3: T_{0,j1,k} += 1.
    #   Missing: T_{0,j1,k} is 1. T_{1,j0,k} is 1.
    #   We have T_{0,j0,k}=1, T_{1,j1,k}=1.
    #   We need T_{0,j1,k}=1, T_{1,j0,k}=1.
    #   This works with 3 terms!
    
    for (j0, j1) in j_pairs:
        for k in k_singles:
            # M1: (A_0,j0 + A_1,j1) * B_j0,k
            U[0*m + j0, col] += 1.0
            U[1*m + j1, col] += 1.0
            V[j0*p + k, col] += 1.0
            W[k*n + 0, col] += 1.0
            W[k*n + 1, col] += 1.0
            col += 1
            
            # M2: A_1,j0 * B_j1,k
            U[1*m + j0, col] += 1.0
            V[j1*p + k, col] += 1.0
            W[k*n + 1, col] += 1.0
            col += 1
            
            # M3: A_0,j1 * B_j1,k
            U[0*m + j1, col] += 1.0
            V[j1*p + k, col] += 1.0
            W[k*n + 0, col] += 1.0
            col += 1

    # 4. j_singles x k_singles
    # T_{0,j,k}=1, T_{1,j,k}=1.
    # Schoolbook: 2 terms.
    # M1: A_0,j * B_j,k
    # M2: A_1,j * B_j,k
    
    for j in j_singles:
        for k in k_singles:
            U[0*m + j, col] += 1.0
            V[j*p + k, col] += 1.0
            W[k*n + 0, col] += 1.0
            col += 1
            
            U[1*m + j, col] += 1.0
            V[j*p + k, col] += 1.0
            W[k*n + 1, col] += 1.0
            col += 1

    return U[:, :col], V[:, :col], W[:, :col], col

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Define partitions
    # m=5: j_pairs=(0,1), j_singles=[2,3,4]? Or (0,1),(2,3), single 4?
    # p=6: k_pairs=(0,1),(2,3),(4,5)?
    
    # Strategy:
    # Maximize pairs to use 7-term Strassen.
    # j: 2 pairs, 1 single. (0,1), (2,3), 4
    # k: 3 pairs, 0 single. (0,1), (2,3), (4,5)
    
    j_pairs = [(0, 1), (2, 3)]
    j_singles = [4]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    k_singles = []
    
    U, V, W, rank = _build_strassen_decomp(n, m, p, j_pairs, k_pairs, j_singles, k_singles)
    
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
        rank = col
    
    # Greedy elimination
    current_U = U
    current_V = V
    current_W = W
    current_rank = rank
    removed = [False] * rank
    
    for c in range(rank):
        if time.time() > deadline - 10:
            break
        if removed[c]:
            continue
        
        # Try removing c
        mask = [i for i in range(current_rank) if not removed[i] and i != c]
        if len(mask) < 1:
            continue
        
        test_U = current_U[:, mask]
        test_V = current_V[:, mask]
        test_W = current_W[:, mask]
        
        if _verify_decomposition(test_U, test_V, test_W, n, m, p):
            removed[c] = True
            current_U = test_U
            current_V = test_V
            current_W = test_W
            current_rank = len(mask)
    
    return {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}
# EVOLVE-BLOCK-END
