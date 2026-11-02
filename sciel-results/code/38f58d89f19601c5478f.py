<<<<<<< SEARCH
    def verify_fast(U, V, W, n, m, p):
        """Faster verification using numpy."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check valid entries: T[a,b,c] = 1 for valid (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = np.dot(np.dot(U_arr[a], V_arr[b]), W_arr[c])
                    if abs(s - 1) > 1e-9:
                        return False
        
        # Check invalid entries: T[a,b,c] = 0 for invalid combos
        # Only check a subset for speed, then full check if subset passes
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if not (j_a == j_b and k_b == k_c and i_a == i_c):
                        s = np.dot(np.dot(U_arr[a], V_arr[b]), W_arr[c])
                        if abs(s) > 1e-9:
                            return False
        return True
=======
    def verify_fast(U, V, W, n, m, p):
        """Faster verification using numpy."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check valid entries: T[a,b,c] = 1 for valid (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = float(np.dot(np.dot(U_arr[a], V_arr[b]), W_arr[c]))
                    if abs(s - 1) > 1e-9:
                        return False
        
        # Check invalid entries: T[a,b,c] = 0 for invalid combos
        # Only check a subset for speed, then full check if subset passes
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if not (j_a == j_b and k_b == k_c and i_a == i_c):
                        s = float(np.dot(np.dot(U_arr[a], V_arr[b]), W_arr[c]))
                        if abs(s) > 1e-9:
                            return False
        return True
>>>>>>> REPLACE
