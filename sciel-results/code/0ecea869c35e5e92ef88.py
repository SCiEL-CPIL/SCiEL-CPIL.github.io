            # Update Covariance
            # Rank-one
            if hsig:
                C = (1 - c_1) * C + c_1 * np.outer(y_sigma, y_sigma)
            else:
                C = (1 - c_1) * C + c_1 * (2 * np.outer(pc, pc) - C) # This is for rank-one of pc? No.
                # Actually, if hsig is False, we often don't update C with y_sigma?
                # Standard: C = (1-c1-cmu)C + c1*yy^T + cmu*sum(zz^T)
                # The hsig affects the update of pc, and sometimes the rank-one term.
                # Let's use the simpler combined form which is more stable for small implementations:
                pass
