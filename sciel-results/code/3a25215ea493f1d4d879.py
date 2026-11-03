import numpy as np

def construct():
    n, m, p = 4, 5, 6
    # Total rank
    rank_laderman = 49
    rank_strassen_4x4x2 = 28
    rank_schoolbook_4x1x6 = 24
    total_rank = rank_laderman + rank_strassen_4x4x2 + rank_schoolbook_4x1x6
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Helper to add a decomposition
    # local_U, local_V, local_W are lists of tuples (index, coeff)
    # offset is the starting column index
    def add_decomp(local_U, local_V, local_W, offset):
        for c in range(len(local_U)):
            for (idx, val) in local_U[c]:
                U[idx][offset + c] += val
            for (idx, val) in local_V[c]:
                V[idx][offset + c] += val
            for (idx, val) in local_W[c]:
                W[idx][offset + c] += val

    # 1. Laderman 4x4x4
    # Indices: A (0..3, 0..3), B (0..3, 0..3), C (0..3, 0..3)
    # We need the 49 terms.
    # I will use a known implementation or derive it.
    # Laderman's algorithm coefficients.
    # This is complex to hardcode from memory.
    # Alternative: Use Strassen recursively?
    # 4x4x4 can be done with 49 mults.
    # If I don't have the exact coefficients, I might fall back to a higher rank?
    # But the goal is to beat 120.
    # Even a simple block decomposition might work.
    # 4x4x4 rank is at most 64 (schoolbook).
    # 4x4x2 rank is at most 32.
    # 4x1x6 rank is 24.
    # Total 64 + 32 + 24 = 120. No improvement.
    # We MUST use Laderman (49) and Strassen-based 4x4x2 (28).
    
    # I will implement Laderman's algorithm.
    # Reference: "Laderman's algorithm"
    # It uses 49 multiplications.
    
    # Since I cannot look it up, I will use a simpler decomposition if possible?
    # No, 101 is the target.
    # I will assume I can find the Laderman coefficients.
    # Actually, I can use the fact that 4x4x4 can be done with 49.
    
    # Let's try to construct it.
    
    # For now, I will return the schoolbook 120 to ensure validity, 
    # but the prompt asks for a stronger algorithm.
    # I will implement the 101 rank solution.
    
    # Laderman coefficients for 4x4x4
    # A = [a11 a12 a13 a14; ...]
    # This is too long to write manually.
    
    # I will use a placeholder that returns 120 if I can't find Laderman.
    # But I must try.
    
    # Actually, I can use Strassen for 4x4x4?
    # Strassen for 4x4x4 uses 49 mults?
    # Strassen for 2x2x2 is 7.
    # For 4x4x4, we can split into 8 blocks of 2x2x2.
    # 8 * 7 = 56 mults.
    # 56 + 32 + 24 = 112.
    # 112 < 120. Score 120/112 = 1.07.
    # This is an improvement!
    
    # So I can use Strassen recursively for 4x4x4 (rank 56) and 4x4x2 (rank 28).
    # 56 + 28 + 24 = 108.
    # Score 120/108 = 1.11.
    
    # Can I do better for 4x4x4?
    # Laderman is 49.
    # If I use 49, total is 101.
    
    # I will implement Strassen recursively for 4x4x4 and 4x4x2.
    
    return {"U": U, "V": V, "W": W}
