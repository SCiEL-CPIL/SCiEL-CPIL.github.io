        b0 = f(t0, np.zeros(n))
        A0 = np.zeros((n, n))
        e = np.zeros(n)
        for i in range(n):
            e[i] = 1.0
            A0[:, i] = f(t0, e) - b0
            e[i] = 0.0
            
        b1 = f(t1, np.zeros(n))
        A1 = np.zeros((n, n))
        for i in range(n):
            e[i] = 1.0
            A1[:, i] = f(t1, e) - b1
            e[i] = 0.0
            
        # Check if A and b are constant
        if np.linalg.norm(A0 - A1) < 1e-10 * (np.linalg.norm(A0) + 1.0) and \
           np.linalg.norm(b0 - b1) < 1e-10 * (np.linalg.norm(b0) + 1.0):
             # Use A0 and b0
             ...
