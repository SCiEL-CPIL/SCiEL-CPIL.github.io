            # Verify
            if _verify_decomposition(U_full, V_full, W_full, T, tol=1e-5):
                # Try to round to half-integers
                U_round = np.round(U_full * 2) / 2
                V_round = np.round(V_full * 2) / 2
                W_round = np.round(W_full * 2) / 2
                
                if _verify_decomposition(U_round, V_round, W_round, T, tol=1e-5):
                    return U_round, V_round, W_round, full_rank
                else:
                    # If rounding fails, return floats? 
                    # The task says "integer/half-integer entries".
                    # If we can't round, it might be invalid.
                    # But if we return floats that are exact, maybe it's okay?
                    # "exactly decompose" suggests exactness.
                    # Let's assume we need to return the rounded version if it works.
                    # If it doesn't work, we might have to discard this solution.
                    pass
