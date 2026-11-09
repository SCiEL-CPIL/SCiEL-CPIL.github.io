def priority(vector, n):
                # Convert vector to a hashable int
                # Use a simple polynomial rolling hash
                h = 0
                base = 3
                # To avoid overflow issues in Python (Python handles big ints, so it's fine, but modulo keeps it small)
                # We want a float in [0, 1) or just a comparable float.
                # Let's use a large prime modulus to get good distribution.
                mod = 10**9 + 7
                for x in vector:
                    h = (h * base + x) % mod
                return float(h)
