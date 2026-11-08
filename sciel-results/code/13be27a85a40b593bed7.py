    def get_decomp(n, m, p):
        # ... base cases ...
        
        best_rank = float('inf')
        best_UVW = None
        
        # Try splitting n
        if n > 1:
            for n1 in range(1, n):
                n2 = n - n1
                U1, V1, W1 = get_decomp(n1, m, p)
                U2, V2, W2 = get_decomp(n2, m, p)
                r1 = U1.shape[1]
                r2 = U2.shape[1]
                total_r = r1 + r2
                if total_r < best_rank:
                    # Construct combined
                    # ...
                    best_rank = total_r
                    # Store U1, V1, W1, U2, V2, W2 and split type
        # Try splitting m
        # Try splitting p
        
        # If no split better than schoolbook?
        # Schoolbook rank is n*m*p.
        # But base cases handle n=1, m=1, p=1.
        # If n,m,p > 1, we must split.
        
        # Return best
