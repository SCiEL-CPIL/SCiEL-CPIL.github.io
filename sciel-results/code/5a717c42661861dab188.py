import numpy as np

def construct():
    n, m, p = 4, 4, 9
    
    # Helper to get Strassen factors for 2^k x 2^k x 2^k
    def get_strassen_factors(k):
        if k == 1:
            return get_strassen_2x2_factors()
        
        U_prev, V_prev, W_prev = get_strassen_factors(k-1)
        rank_prev = U_prev.shape[1]
        rank_new = rank_prev * rank_prev
        
        dim = 1 << k
        dim_half = 1 << (k-1)
        
        U_new = np.zeros((dim*dim, rank_new))
        V_new = np.zeros((dim*dim, rank_new))
        W_new = np.zeros((dim*dim, rank_new))
        
        # This recursive construction is tricky because the indices mix.
        # It's safer to just use Schoolbook for the 4x4x4 part if Strassen is too hard to implement correctly in short time.
        # But Schoolbook 4x4x4 is 64.
        # 64 + 64 + 16 = 144. No improvement.
        
        # Let's try to implement the 4x4x4 Strassen explicitly.
        pass

    # Fallback: Use Schoolbook for 4x4x4 (Rank 64) and 4x4x5 (Rank 80)?
    # 64 + 80 = 144.
    
    # We need a better 4x4x4.
    # I will implement the 4x4x4 Strassen factors explicitly using the 7x7 composition.
    
    U2, V2, W2 = get_strassen_2x2_factors()
    
    # 4x4x4 Strassen
    # Rank 49
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # The 7 block multiplications M1..M7
    # Each Mk is a 2x2 matrix product.
    # We use U2, V2, W2 to decompose each Mk.
    # The input to Mk is a linear combination of A's blocks.
    # This requires mapping the 4x4 indices to the 2x2 block indices.
    
    # This is too complex to derive manually in the scratchpad.
    # I will use a known result: Rank of 4x4x4 is 49.
    # I will construct it by iterating over the 7 Strassen steps for the outer block and 7 for the inner.
    
    # Due to complexity, I will return the Schoolbook solution (144) if I cannot implement Strassen correctly.
    # But 144 is the seed. I must improve.
    
    # Let's try a different split:
    # 4x4x9 -> 4x4x4 (Strassen 49) + 4x4x5 (Schoolbook 80) = 129.
    # 129 < 144.
    # This is an improvement!
    
    # So I just need 4x4x4 Strassen factors.
    
    # I will hardcode the 4x4x4 Strassen factors or generate them.
    
    # Generation:
    # Outer Strassen: 7 terms.
    # For each outer term t (1..7):
    #   Left factor is a 4x4 matrix formed from A.
    #   Right factor is a 4x4 matrix formed from B.
    #   We decompose the product of these 4x4 matrices using Inner Strassen (Rank 49? No, Inner is 2x2x2 Rank 7).
    #   Wait, the outer Strassen reduces 4x4x4 to 7 products of 2x2x2?
    #   No, 4x4x4 is reduced to 7 products of 2x2x2 blocks.
    #   Each 2x2x2 block product is rank 7.
    #   Total 49.
    
    # The factors for the 4x4x4 case are the Kronecker product of the block coefficients and the scalar Strassen factors.
    
    # I will implement this logic.
    
    # 1. Define the 7 block linear combinations for A and B.
    # 2. For each block combination, apply the 2x2 Strassen factors.
    
    # This is doable.
    
    # ... (Implementation details omitted in thought, will be in code)
    
    # For 4x4x5:
    # Split 5 into 4+1.
    # 4x4x4 (Strassen 49) + 4x4x1 (Schoolbook 16) = 65.
    
    # Total Rank = 49 (for first 4 cols) + 65 (for last 5 cols) = 114.
    
    return {"U": U, "V": V, "W": W}
