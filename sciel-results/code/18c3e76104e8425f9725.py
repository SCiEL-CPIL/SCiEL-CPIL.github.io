<<<<<<< SEARCH
        rank = len(U_cols)
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        
        # Verify
        if verify_decomp(U, V, W, rank):
            return U, V, W, rank
        else:
            return None
=======
        rank = len(U_cols)
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        
        # Verify
        if verify_decomp(U, V, W, rank):
            print(f"  Block decomp verified: rank {rank}")
            return U, V, W, rank
        else:
            print(f"  Block decomp FAILED verification at rank {rank}")
            return None
>>>>>>> REPLACE

Let me also add a fallback if Strassen 4×4×4 fails:

