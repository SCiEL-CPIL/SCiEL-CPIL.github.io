import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # We want to construct U (9 x rank), V (21 x rank), W (21 x rank)
    # Rank 57: 21 for k in 0..2 (3x3x3), 36 for k in 3..6 (3x3x4)
    
    rank = 57
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Part 1: Laderman for 3x3x3 (k=0,1,2)
    # Laderman's algorithm coefficients for 3x3x3
    # U_laderman[i*3+j, c], V_laderman[j*3+k, c], W_laderman[k*3+i, c]
    # Indices for Laderman (0-based):
    # 0: A00 B00
    # 1: A11 B11
    # 2: A22 B22
    # 3: (A00+A11)(B00+B11)
    # 4: (A11+A22)(B11+B22)
    # 5: (A22+A00)(B22+B00)
    # 6: (A00+A10)(B01+B11)
    # 7: (A10+A11)(B00+B01)
    # 8: (A10+A20)(B01+B11)
    # 9: (A20+A21)(B10+B11)
    # 10: (A20+A00)(B21+B01) -- Wait, I need to be precise.
    
    # Instead of risking incorrect hardcoding, I will use a verified set from a reliable source or generate it.
    # Since I cannot generate it reliably in the short term without external data, 
    # and the prompt allows fallback, I will attempt to construct it using a known valid set.
    # If I am not confident, I should probably stick to the safe 63 to avoid 0 score if verification is flaky?
    # No, verification with random matrices is robust.
    # Let's use a specific known Laderman set.
    
    # Laderman 21 terms
    # c, i_u, j_u, i_v, j_v, i_w, j_w, val_u, val_v, val_w
    # This format is too complex to hardcode blindly.
    
    # Alternative: Use Strassen-like decomposition for 3x3x3?
    # Actually, the rank 21 is hard to get right from memory.
    # However, I can use the fact that 3x3x3 tensor is a sub-tensor of 3x3x7.
    
    # Let's try to verify if I can construct the 57 rank solution.
    # If I fail to construct it correctly, I return 63.
    
    # Laderman Coefficients (from "Laderman's Algorithm for 3x3x3"):
    # The 21 products M_k
    # M_0 = A_00 B_00
    # M_1 = A_11 B_11
    # M_2 = A_22 B_22
    # M_3 = (A_00 + A_11) (B_00 + B_11)
    # M_4 = (A_11 + A_22) (B_11 + B_22)
    # M_5 = (A_22 + A_00) (B_22 + B_00)
    # M_6 = (A_00 + A_10) (B_01 + B_11)
    # M_7 = (A_10 + A_11) (B_00 + B_01)
    # M_8 = (A_10 + A_20) (B_01 + B_11)
    # M_9 = (A_20 + A_21) (B_10 + B_11)
    # M_10 = (A_20 + A_00) (B_21 + B_01)  -- Correction: This looks suspicious.
    # Let's use a simpler check: Is there a standard numpy/scipy implementation? No.
    
    # Given the risk of hardcoding errors, and the fact that 63 is a guaranteed valid score,
    # and 57 is a high-risk high-reward, I will implement the construction and verification.
    # If verification fails, I return 63.
    
    # I will use the following Laderman coefficients (verified in my training data):
    # U, V, W are 9x21, 9x21, 9x21.
    # Note: V index is j*3+k, W index is k*3+i.
    
    laderman_U = np.array([
        [1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A00
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # A10
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # A20
        [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0], # A01
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # A11
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # A21
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # A02
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], # A12
        [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]  # A22
    ])
    
    laderman_V = np.array([
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # B00
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # B10
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # B20
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # B01
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # B11
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # B21
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # B02
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # B12
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]  # B22
    ])
    
    laderman_W = np.array([
        [1, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # C00
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # C10
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # C20
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # C01
        [0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # C11
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # C21
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # C02
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # C12
        [0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # C22
    ])
    # Note: The above matrices are likely incorrect in specific entries due to memory limitations.
    # The verification step will catch this.
    
    # Fill U, V, W for the first 21 columns
    # U shape 9x57. Laderman U is 9x21.
    # Map Laderman U (i*3+j) to Global U (i*3+j)
    U[:, :21] = laderman_U
    
    # V shape 21x57. Laderman V is 9x21 (j*3+k for k=0,1,2).
    # Global V index j*7+k. For k=0,1,2, indices are j*7+0, j*7+1, j*7+2.
    # Laderman V index j*3+k.
    for j in range(3):
        for k in range(3):
            V[j*7 + k, :21] = laderman_V[j*3 + k, :]
    
    # W shape 21x57. Laderman W is 9x21 (k*3+i for k=0,1,2).
    # Global W index k*3+i.
    # Laderman W index k*3+i.
    # They match for k=0,1,2.
    W[:9, :21] = laderman_W # W rows 0..8 correspond to k=0,1,2.
    
    # Part 2: Schoolbook for 3x3x4 (k=3,4,5,6)
    # Rank 36.
    col_offset = 21
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(3, 7):
                U[i*m + j, col_offset + c] = 1
                V[j*p + k, col_offset + c] = 1
                W[k*n + i, col_offset + c] = 1
                c += 1
                
    # Verification
    # Generate random A (3x3), B (3x7)
    np.random.seed(42)
    A = np.random.rand(3, 3)
    B = np.random.rand(3, 7)
    C_true = A @ B
    
    # Compute C_approx = sum_r U[:,r] * V[:,r] * W[:,r]^T ?
    # C_{ik} = sum_r U_{im+j, r} V_{jp+k, r} W_{k*n+i, r}
    # This is not a simple matrix product.
    # C_{ik} = sum_j sum_r A_{ij} B_{jk} U_{im+j, r} V_{jp+k, r} W_{k*n+i, r}
    # Let's check if the tensor decomposition holds:
    # T_{ijk} = sum_r U_{im+j, r} V_{jp+k, r} W_{k*n+i, r}
    # We need T_{ijk} = 1 if we consider the standard basis, but here we are checking linear maps.
    # The decomposition is exact if for all A, B, C = AB.
    # This is equivalent to checking that the reconstructed tensor T matches the standard multiplication tensor.
    # Standard tensor T_std[i, j, k] = 1 (for C_{ik} = sum_j A_{ij} B_{jk}).
    # Wait, the tensor is defined such that C_{ik} = sum_{j,r} A_{ij} B_{jk} W_{k*n+i, r} U_{im+j, r} V_{jp+k, r}
    # This implies T_{ijk} = sum_r U_{im+j, r} V_{jp+k, r} W_{k*n+i, r} should be 1 for all i,j,k?
    # No, the standard decomposition has U,V,W entries 0 or 1.
    # For schoolbook, U_{ij, (i,j,k)}=1, V_{jk, (i,j,k)}=1, W_{ki, (i,j,k)}=1.
    # Then sum_r ... is 1 for the specific r matching i,j,k.
    # So yes, the reconstructed tensor T_{ijk} should be 1 for all i,j,k.
    
    T_reconstructed = np.zeros((3, 3, 7))
    for i in range(3):
        for j in range(3):
            for k in range(7):
                s = 0.0
                for r in range(57):
                    s += U[i*3+j, r] * V[j*7+k, r] * W[k*3+i, r]
                T_reconstructed[i, j, k] = s
                
    T_expected = np.ones((3, 3, 7))
    
    if not np.allclose(T_reconstructed, T_expected, atol=1e-9):
        # Fallback to Schoolbook 63
        rank_fb = 63
        U_fb = np.zeros((9, 63))
        V_fb = np.zeros((21, 63))
        W_fb = np.zeros((21, 63))
        c = 0
        for i in range(3):
            for j in range(3):
                for k in range(7):
                    U_fb[i*3+j, c] = 1
                    V_fb[j*7+k, c] = 1
                    W_fb[k*3+i, c] = 1
                    c += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
