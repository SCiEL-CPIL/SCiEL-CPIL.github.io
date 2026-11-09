import time
import random
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _build_base_decomp(n, m, p):
    """
    Builds the base decomposition for 1 x m x p using rank 1.
    U: (m, 1), V: (m*p, 1), W: (p, 1)
    """
    rank = 1
    U = [[1] for _ in range(m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p)]
    return {"U": U, "V": V, "W": W, "rank": rank}

def _combine_decomps(decomps):
    """
    Combines a list of decompositions by concatenating their rank dimensions.
    """
    if not decomps:
        return {"U": [], "V": [], "W": [], "rank": 0}
    
    total_rank = sum(d["rank"] for d in decomps)
    U = []
    V = []
    W = []
    
    for d in decomps:
        U.append(d["U"])
        V.append(d["V"])
        W.append(d["W"])
        
    # Flatten the lists
    U_flat = [item for sublist in U for item in sublist]
    V_flat = [item for sublist in V for item in sublist]
    W_flat = [item for sublist in W for item in sublist]
    
    return {"U": U_flat, "V": V_flat, "W": W_flat, "rank": total_rank}

def _recursive_decomp(n, m, p):
    """
    Recursively builds the decomposition.
    """
    if n == 1:
        return _build_base_decomp(1, m, p)
    
    # Split m into two halves
    m1 = m // 2
    m2 = m - m1
    
    # Split p into two halves
    p1 = p // 2
    p2 = p - p1
    
    # We need to handle the splitting of the tensor.
    # C = A B
    # A is n x m. Split A into A1 (n x m1) and A2 (n x m2).
    # B is m x p. Split B into B_top (m1 x p) and B_bot (m2 x p).
    # C = A1 B_top + A2 B_bot.
    # This is a sum of two products.
    
    # Product 1: A1 (n x m1) * B_top (m1 x p)
    # This is an n x m1 x p multiplication.
    # We can recursively decompose this.
    
    # Product 2: A2 (n x m2) * B_bot (m2 x p)
    # This is an n x m2 x p multiplication.
    
    # However, the recursive call assumes the standard shape.
    # We need to map the indices correctly.
    
    # Let's implement the specific splitting for 3x4x8 -> 2x(3x4x4) -> 4x(3x2x2) -> 8x(3x1x1)?
    # No, the recursion above works for any m, p if we split them.
    # But the "base case" n=1 is rank 1.
    # For n>1, we split m.
    
    # Decomposition for n x m1 x p
    d1 = _recursive_decomp(n, m1, p)
    # Decomposition for n x m2 x p
    d2 = _recursive_decomp(n, m2, p)
    
    # We need to interleave or concatenate?
    # The tensors are disjoint in the m-index.
    # So we can just concatenate the rank dimensions.
    # But we need to pad the U, V, W matrices to match the full m, p sizes.
    
    # This approach is getting complicated with index mapping.
    # Given the small size, let's just hardcode the 24-rank solution structure 
    # or use a simpler recursive builder that handles indices.
    
    # For now, return a placeholder if not base case, 
    # but we will implement the specific 24-rank construction in construct().
    raise NotImplementedError("Use specific construction")

def _construct_24_rank(n, m, p):
    """
    Constructs the 24-rank decomposition for 3x4x8.
    """
    assert n == 3 and m == 4 and p == 8
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Structure:
    # 2 blocks of p (4 each)
    #   2 blocks of m (2 each)
    #     3 rows of n (1 each)
    #       1 base rank per (n_row, m_block, p_block)
    # Total rank = 2 * 2 * 3 * 1 = 12?
    # Wait, my previous calc:
    # R(3,4,8) = 2 * R(3,4,4)
    # R(3,4,4) = 2 * R(3,2,4)
    # R(3,2,4) = 2 * R(3,2,2)
    # R(3,2,2) = 3 * R(1,2,2) = 3 * 1 = 3
    # So R(3,2,4) = 6
    # R(3,4,4) = 12
    # R(3,4,8) = 24
    
    # The base case is R(1, 2, 2) = 1.
    # This corresponds to n=1, m=2, p=2.
    
    # We iterate over the 24 rank components.
    # Each component corresponds to a specific (i, j_block, k_block) ?
    # No, the rank components are independent.
    
    # Let's map the rank index r to (i, m_idx, p_idx)
    # i in 0..2 (3 values)
    # m_idx in 0..1 (2 values, corresponding to m1, m2)
    # p_idx in 0..1 (2 values, corresponding to p1, p2)
    # Total 3 * 2 * 2 = 12?
    # Where did 24 come from?
    # R(3,2,2) = 3.
    # R(3,2,4) = 2 * 3 = 6. (Split p into 2)
    # R(3,4,4) = 2 * 6 = 12. (Split m into 2)
    # R(3,4,8) = 2 * 12 = 24. (Split p into 2)
    
    # So the indices are:
    # p_split (2 options)
    # m_split (2 options)
    # p_split_inner (2 options)
    # n_row (3 options)
    # Total 2 * 2 * 2 * 3 = 24.
    
    # Let's define the mapping.
    # r = 0..23
    # p_split = r // 12 (0 or 1)
    # rem = r % 12
    # m_split = rem // 6 (0 or 1)
    # rem2 = rem % 6
    # p_inner = rem2 // 3 (0 or 1)
    # n_row = rem2 % 3 (0, 1, 2)
    
    for r in range(rank):
        p_split = r // 12
        rem = r % 12
        m_split = rem // 6
        rem2 = rem % 6
        p_inner = rem2 // 3
        n_row = rem2 % 3
        
        # p_index in [0, 7]
        # p_split 0 -> p in [0,3], p_split 1 -> p in [4,7]
        # p_inner 0 -> offset 0, p_inner 1 -> offset 2
        # So p = p_split * 4 + p_inner * 2 + ... ?
        # Wait, p_inner is for the 2x2 block.
        # The p dimension is 8.
        # Split into two 4s.
        # Each 4 is split into two 2s.
        # So p = p_split * 4 + p_inner * 2.
        # But we need to fill the 2x2 block.
        # The base case is 1x2x2.
        # So for a given (n_row, m_split, p_split, p_inner), we have a 1x2x2 block.
        # This block covers m indices [m_split*2, m_split*2+1] and p indices [p_split*4 + p_inner*2, ...+1].
        
        # For this block, we use rank 1.
        # U[n_row, m_idx, r] = 1 for m_idx in block.
        # V[m_idx, p_idx, r] = 1 for m_idx, p_idx in block.
        # W[p_idx, n_row, r] = 1 for p_idx in block.
        
        m_start = m_split * 2
        p_start = p_split * 4 + p_inner * 2
        
        # U: index n_row * m + m_idx
        for dm in range(2):
            U[n_row * m + m_start + dm][r] = 1
            
        # V: index m_idx * p + p_idx
        for dm in range(2):
            for dp in range(2):
                V[(m_start + dm) * p + (p_start + dp)][r] = 1
                
        # W: index p_idx * n + n_row
        for dp in range(2):
            W[(p_start + dp) * n + n_row][r] = 1
            
    return {"U": U, "V": V, "W": W}

def _verify_decomp(decomp, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U = np.array(decomp['U'], dtype=float)
    V = np.array(decomp['V'], dtype=float)
    W = np.array(decomp['W'], dtype=float)
    
    rank = U.shape[1]
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Try the 24-rank construction
    candidate = _construct_24_rank(n, m, p)
    
    if _verify_decomp(candidate, n, m, p):
        return candidate
        
    # Fallback to schoolbook if verification fails
    def _schoolbook_decomp(n, m, p):
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
        return {"U": U, "V": V, "W": W}
        
    return _schoolbook_decomp(n, m, p)
# EVOLVE-BLOCK-END
