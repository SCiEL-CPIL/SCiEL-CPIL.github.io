<<<<<<< SEARCH
        # Initial state
        mean = np.zeros(dim)
        sigma = 2.0  # Initial step size, covers the box [-5, 5] reasonably well
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        invsqrtC = np.linalg.inv(np.sqrtm(C))
=======
        # Initial state
        mean = np.zeros(dim)
        sigma = 2.0  # Initial step size, covers the box [-5, 5] reasonably well
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        invsqrtC = np.eye(dim)
>>>>>>> REPLACE
