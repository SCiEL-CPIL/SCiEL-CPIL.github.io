            try:
                k_inv = pow(A, -1, B)
            except ValueError:
                continue
                
            k = (B - k_inv) % B
            n0 = A * k
