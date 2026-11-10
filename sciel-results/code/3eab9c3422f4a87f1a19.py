            vals = rem[indices]
            while True:
                mask = vals % p == 0
                if not np.any(mask):
                    break
                vals[mask] //= p
            rem[indices] = vals
