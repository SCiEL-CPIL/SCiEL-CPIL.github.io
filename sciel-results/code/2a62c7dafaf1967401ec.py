    def decompose(n, m, p):
        if n == 2 and m == 2 and p == 2:
            return get_strassen_222()
        if n == 1 or m == 1 or p == 1:
            return get_schoolbook(n, m, p)
            
        best_rank = float('inf')
        best_result = None
        
        # Try splitting m
        if m > 2:
            m1, m2 = 2, m - 2
            U1, V1, W1 = decompose(n, m1, p)
            U2, V2, W2 = decompose(n, m2, p)
            U, V, W = combine_m(U1, V1, W1, U2, V2, W2, n, m1, m2, p)
            rank = len(U[0])
            if rank < best_rank:
                best_rank = rank
                best_result = (U, V, W)
                
        # Try splitting p
        if p > 2:
            p1, p2 = 2, p - 2
            U1, V1, W1 = decompose(n, m, p1)
            U2, V2, W2 = decompose(n, m, p2)
            U, V, W = combine_p(U1, V1, W1, U2, V2, W2, n, m, p1, p2)
            rank = len(U[0])
            if rank < best_rank:
                best_rank = rank
                best_result = (U, V, W)
                
        # Try splitting n
        if n > 2:
            n1, n2 = 2, n - 2
            U1, V1, W1 = decompose(n1, m, p)
            U2, V2, W2 = decompose(n2, m, p)
            U, V, W = combine_n(U1, V1, W1, U2, V2, W2, n1, n2, m, p)
            rank = len(U[0])
            if rank < best_rank:
                best_rank = rank
                best_result = (U, V, W)
                
        if best_result:
            return best_result
        else:
            return get_schoolbook(n, m, p)
