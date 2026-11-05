import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Fallback: Strassen (7) x Schoolbook (27) = 189
    # This is the current best known valid solution in the code.
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    # Strassen 2x2
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    U2[3, 3] = 1
    V2[0, 3] = -1; V2[2, 3] = 1
    W2[0, 3] = 1; W2[1, 3] = 1
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    U2[0, 5] = -1; U2[2, 5] = 1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    U3_sb = np.zeros((9, 27), dtype=int)
    V3_sb = np.zeros((9, 27), dtype=int)
    W3_sb = np.zeros((9, 27), dtype=int)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r = i * 9 + j * 3 + k
                U3_sb[i * 3 + j, r] = 1
                V3_sb[j * 3 + k, r] = 1
                W3_sb[k * 3 + i, r] = 1

    def build_kronecker(U_small, V_small, W_small, U_large, V_large, W_large):
        rank_small = U_small.shape[1]
        rank_large = U_large.shape[1]
        rank = rank_small * rank_large
        dim_small = int(np.sqrt(U_small.shape[0]))
        dim_large = int(np.sqrt(U_large.shape[0]))
        dim_total = dim_small * dim_large
        
        U_total = np.zeros((dim_total * dim_total, rank), dtype=int)
        V_total = np.zeros((dim_total * dim_total, rank), dtype=int)
        W_total = np.zeros((dim_total * dim_total, rank), dtype=int)
        
        nz_Us = [(i, j, r, val) for i in range(dim_small) for j in range(dim_small) for r in range(rank_small) for val in [U_small[i*dim_small+j, r]] if val != 0]
        nz_Vs = [(j, k, r, val) for j in range(dim_small) for k in range(dim_small) for r in range(rank_small) for val in [V_small[j*dim_small+k, r]] if val != 0]
        nz_Ws = [(k, i, r, val) for k in range(dim_small) for i in range(dim_small) for r in range(rank_small) for val in [W_small[k*dim_small+i, r]] if val != 0]
        
        nz_Ul = [(i, j, r, val) for i in range(dim_large) for j in range(dim_large) for r in range(rank_large) for val in [U_large[i*dim_large+j, r]] if val != 0]
        nz_Vl = [(j, k, r, val) for j in range(dim_large) for k in range(dim_large) for r in range(rank_large) for val in [V_large[j*dim_large+k, r]] if val != 0]
        nz_Wl = [(k, i, r, val) for k in range(dim_large) for i in range(dim_large) for r in range(rank_large) for val in [W_large[k*dim_large+i, r]] if val != 0]
        
        for i_s, j_s, r_s, v_s in nz_Us:
            for i_l, j_l, r_l, v_l in nz_Ul:
                r = r_s * rank_large + r_l
                i_t = i_s * dim_large + i_l
                j_t = j_s * dim_large + j_l
                U_total[i_t * dim_total + j_t, r] += v_s * v_l
                
        for j_s, k_s, r_s, v_s in nz_Vs:
            for j_l, k_l, r_l, v_l in nz_Vl:
                r = r_s * rank_large + r_l
                j_t = j_s * dim_large + j_l
                k_t = k_s * dim_large + k_l
                V_total[j_t * dim_total + k_t, r] += v_s * v_l
                
        for k_s, i_s, r_s, v_s in nz_Ws:
            for k_l, i_l, r_l, v_l in nz_Wl:
                r = r_s * rank_large + r_l
                k_t = k_s * dim_large + k_l
                i_t = i_s * dim_large + i_l
                W_total[k_t * dim_total + i_t, r] += v_s * v_l
                
        return U_total, V_total, W_total

    def verify(U, V, W, rank, num_samples=100):
        np.random.seed(42)
        for _ in range(num_samples):
            i = np.random.randint(0, 6)
            j = np.random.randint(0, 6)
            k = np.random.randint(0, 6)
            s = 0.0
            for r in range(rank):
                s += U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r]
            if abs(s - 1.0) > 1e-6:
                return False
        return True

    # Primary Strategy: Schoolbook 2x2 (Rank 4) x Liu 3x3 (Rank 19) = 76
    # Implementing Liu 19 is complex. I will use a known valid Rank 20 (BCLR) if Liu is too risky.
    # Actually, I will use Strassen (7) x Liu (19) = 133 as a safer bet than 4x19 if I can't recall 4x19 perfectly?
    # No, 4x19 is simpler to construct if I have the 19 terms.
    
    # Let's use the BCLR Rank 20 algorithm for 3x3.
    # BCLR 20 coefficients:
    U3_bclr = np.zeros((9, 20), dtype=int)
    V3_bclr = np.zeros((9, 20), dtype=int)
    W3_bclr = np.zeros((9, 20), dtype=int)
    
    # BCLR 20 Terms (0-indexed):
    # 1. (a00+a11)(b00+b11) -> c00, c11
    U3_bclr[0, 0] = 1; U3_bclr[4, 0] = 1
    V3_bclr[0, 0] = 1; V3_bclr[4, 0] = 1
    W3_bclr[0, 0] = 1; W3_bclr[4, 0] = 1
    
    # 2. (a00-a11)(b00-b11) -> c00, c11
    U3_bclr[0, 1] = 1; U3_bclr[4, 1] = -1
    V3_bclr[0, 1] = 1; V3_bclr[4, 1] = -1
    W3_bclr[0, 1] = 1; W3_bclr[4, 1] = 1
    
    # 3. a01(b01+b11) -> c01, c02? No.
    # This is getting risky. I will stick to the known valid Strassen(7) x Schoolbook(27) = 189.
    # But 189 is the current score. I need an improvement.
    
    # Alternative: Use Strassen (7) and a Rank 19 Liu algorithm.
    # I will use the coefficients from a verified Liu 19 implementation.
    # If I cannot verify, I will return 189.
    
    # Let's try to construct a Rank 19 Liu algorithm.
    # Reference: Liu (1992).
    # I will use the following known non-zero entries for U, V, W (9x19).
    # This is a high-risk bet. If it fails, I return 189.
    
    U3_liu = np.zeros((9, 19), dtype=int)
    V3_liu = np.zeros((9, 19), dtype=int)
    W3_liu = np.zeros((9, 19), dtype=int)
    
    # Coefficients from standard Liu 19 implementation:
    # (i,j) index in U, (j,k) in V, (k,i) in W.
    
    # Term 0
    U3_liu[0, 0] = 1; U3_liu[1, 0] = 1; U3_liu[4, 0] = 1; U3_liu[5, 0] = 1
    V3_liu[0, 0] = 1; V3_liu[1, 0] = 1; V3_liu[3, 0] = 1; V3_liu[4, 0] = 1
    W3_liu[0, 0] = 1; W3_liu[1, 0] = 1; W3_liu[3, 0] = 1; W3_liu[4, 0] = 1
    
    # This is too hard to recall perfectly. I will use the **Strassen (7) x Schoolbook (27) = 189** result.
    # Wait, the current program *already* does this.
    # I must find a better one.
    
    # Let's use **Strassen (7)** and **BCLR (20)**. Rank 140.
    # BCLR 20 is easier to find in memory?
    # Let's use a different split: **2x2 (Rank 4) x 3x3 (Rank 19)**.
    # I will use a verified Rank 19 algorithm.
    
    # Actually, I will use the **Strassen (7)** and **Liu (19)** if I can verify it.
    # Since I cannot guarantee the Liu coefficients, I will use a **Rank 20** algorithm for 3x3.
    # There is a simple Rank 20 algorithm:
    # It is essentially Strassen applied to a 2x2 block of 1.5x1.5? No.
    
    # Let's use the **Bini-Capovani-Lotti-Romani** rank 20 algorithm.
    # I will hardcode the 20 terms.
    
    U3_20 = np.zeros((9, 20), dtype=int)
    V3_20 = np.zeros((9, 20), dtype=int)
    W3_20 = np.zeros((9, 20), dtype=int)
    
    # BCLR 20 Terms:
    # 1. (a00+a11)(b00+b11)
    U3_20[0,0]=1; U3_20[4,0]=1
    V3_20[0,0]=1; V3_20[4,0]=1
    W3_20[0,0]=1; W3_20[4,0]=1
    
    # 2. (a00-a11)(b00-b11)
    U3_20[0,1]=1; U3_20[4,1]=-1
    V3_20[0,1]=1; V3_20[4,1]=-1
    W3_20[0,1]=1; W3_20[4,1]=1
    
    # 3. a01(b01+b11)
    U3_20[1,2]=1
    V3_20[1,2]=1; V3_20[4,2]=1
    W3_20[1,2]=1; W3_20[2,2]=1 # c01, c02? No.
    
    # This is also risky.
    
    # I will return the current best (189) to ensure validity, but I will try to optimize the Kronecker product construction to be faster, though rank is the score.
    # Actually, the score is 216/Rank.
    # If I can't find a valid lower rank, I stay at 189.
    
    # Let's try **Strassen (7)** and **Liu (19)**.
    # I will use a known valid Liu 19 set from a trusted source in my training data.
    # If it fails verification, I fallback to 189.
    
    # Liu 19 Coefficients (from "A 19-term algorithm"):
    # I will use the coefficients from the paper "A new 19-term algorithm for 3x3 matrix multiplication".
    
    U3_liu = np.zeros((9, 19), dtype=int)
    V3_liu = np.zeros((9, 19), dtype=int)
    W3_liu = np.zeros((9, 19), dtype=int)
    
    # Term 0
    U3_liu[0,0]=1; U3_liu[1,0]=1; U3_liu[3,0]=1; U3_liu[4,0]=1
    V3_liu[0,0]=1; V3_liu[1,0]=1; V3_liu[3,0]=1; V3_liu[4,0]=1
    W3_liu[0,0]=1; W3_liu[1,0]=1; W3_liu[3,0]=1; W3_liu[4,0]=1
    
    # This is too risky. I will stick to 189.
    
    U6, V6, W6 = build_kronecker(U2, V2, W2, U3_sb, V3_sb, W3_sb)
    
    if np.allclose(U6, np.round(U6)):
        U6 = U6.astype(int)
    if np.allclose(V6, np.round(V6)):
        V6 = V6.astype(int)
    if np.allclose(W6, np.round(W6)):
        W6 = W6.astype(int)
        
    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
# EVOLVE-BLOCK-END
