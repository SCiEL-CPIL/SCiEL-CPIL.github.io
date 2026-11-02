# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 7
    deadline = time.time() + 120

    def build_rank22():
        rank = 22
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Block 1: Cols 0-1 (Rank 7 Strassen)
        # ... (same as before) ...
        
        # Block 2: Cols 2-3 (Rank 7 Strassen)
        # ... (same as before) ...
        
        # Block 3: Cols 4-6 (Rank 8 for 2x2x3)
        # ... (Need coefficients) ...
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # ...
# EVOLVE-BLOCK-END
