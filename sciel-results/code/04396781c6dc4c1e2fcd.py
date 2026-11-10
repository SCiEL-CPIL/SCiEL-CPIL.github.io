            rem_vals = rem[indices]
            m = rem_vals % p == 0
            if np.any(m):
                rem_vals[m] //= p
                # Check for higher powers
                # Only the elements that were divisible need to be checked
                # But we don't have the indices of m easily accessible for sub-masking without creating a new array.
                # We can just check the whole array again?
                # For p > sqrt(W), the power is 1.
                # For small p, we might need more.
                
                # Let's do a second pass only if necessary
                m2 = rem_vals % p == 0
                if np.any(m2):
                    rem_vals[m2] //= p
                    m3 = rem_vals % p == 0
                    if np.any(m3):
                        # Rare
                        while np.any(rem_vals % p == 0):
                            rem_vals[rem_vals % p == 0] //= p
