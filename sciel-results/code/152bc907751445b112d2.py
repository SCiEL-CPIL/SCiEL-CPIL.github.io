import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    deadline = 145  # 5 seconds margin
    
    n, m, p = DIMENSIONS
    
    # Helper to generate schoolbook decomposition
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    # Helper to combine decompositions
    def combine(decomps, n_global, m_global, p_global, index_maps):
        # index_maps: list of dicts mapping local (i,j,k) to global (i,j,k)
        # Actually, we just need to remap the indices in U, V, W
        total_rank = sum(len(d[0]) for d in decomps)
        U_global = [[0] * total_rank for _ in range(n_global * m_global)]
        V_global = [[0] * total_rank for _ in range(m_global * p_global)]
        W_global = [[0] * total_rank for _ in range(p_global * n_global)]
        
        current_rank = 0
        for d_idx, (U, V, W) in enumerate(decomps):
            n_l, m_l, p_l = index_maps[d_idx]['dims']
            offset_i = index_maps[d_idx]['offset_i']
            offset_j = index_maps[d_idx]['offset_j']
            offset_k = index_maps[d_idx]['offset_k']
            
            # Remap U: local (i,j) -> global (i,j)
            for i_l in range(n_l):
                for j_l in range(m_l):
                    i_g = i_l + offset_i
                    j_g = j_l + offset_j
                    row_g = i_g * m_global + j_g
                    row_l = i_l * m_l + j_l
                    for r in range(len(U)):
                        U_global[row_g][current_rank + r] = U[row_l][r]
            
            # Remap V: local (j,k) -> global (j,k)
            for j_l in range(m_l):
                for k_l in range(p_l):
                    j_g = j_l + offset_j
                    k_g = k_l + offset_k
                    row_g = j_g * p_global + k_g
                    row_l = j_l * p_l + k_l
                    for r in range(len(V)):
                        V_global[row_g][current_rank + r] = V[row_l][r]
            
            # Remap W: local (k,i) -> global (k,i)
            # Note: W index is k * n + i
            for k_l in range(p_l):
                for i_l in range(n_l):
                    k_g = k_l + offset_k
                    i_g = i_l + offset_i
                    row_g = k_g * n_global + i_g
                    row_l = k_l * n_l + i_l
                    for r in range(len(W)):
                        W_global[row_g][current_rank + r] = W[row_l][r]
                        
            current_rank += len(U)
            
        return U_global, V_global, W_global

    # 1. 3x3x3 Rank 23 Decomposition
    # Hardcoded from literature (Strassen/other)
    # This is complex to hardcode manually in text, but I can use a known set of 23 terms.
    # If I can't find it, I fall back to schoolbook for this block.
    # Let's try to generate a 23-rank decomp if possible, or just use schoolbook for now 
    # and hope the other blocks are optimized? No, 3x3x3 is the main gain.
    
    # I will use a known 23-rank decomposition for 3x3x3.
    # Since I cannot paste 23x3x3 arrays easily, I will check if I can construct it.
    # Actually, I will just use schoolbook for 3x3x3 if I don't have the data, 
    # but that yields 75. 
    # To ensure an improvement, I MUST have the 23-rank decomp.
    
    # I will use a placeholder that returns schoolbook if the 23-rank is not available,
    # but I will try to hardcode a few terms to show structure?
    # No, I will use the 23-rank decomp from a standard source.
    
    # For the sake of this code, I will implement the 23-rank decomp for 3x3x3.
    # This is a known result.
    
    U_333, V_333, W_333 = schoolbook(3, 3, 3) # Fallback
    
    # If I had the 23-rank decomp, I would use it.
    # Let's assume I can't hardcode it correctly in this thought process.
    # Is there any other improvement?
    
    # What if I use 2x2x2 Strassen (rank 7) for parts?
    # 3x5x5.
    # Split into 2x5x5 and 1x5x5.
    # 2x5x5:
    # Split 2x5 into 2x2 and 2x3.
    # 2x2x5: Split 5 into 2+2+1?
    # This is getting complicated.
    
    # I will stick to the 71-rank solution if I can get the 23-rank 3x3x3.
    # Since I can't verify the 23-rank terms, I will use schoolbook for 3x3x3 (27)
    # and see if I can optimize the other parts.
    # 27 + 18 + 18 + 12 = 75. No gain.
    
    # I will output the schoolbook solution to be safe, as I cannot guarantee the 23-rank decomp correctness.
    
    U, V, W = schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
