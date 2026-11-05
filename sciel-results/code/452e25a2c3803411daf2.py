                U = np.random.choice([-1, 0, 1], size=(nn*mm, r))
                V = np.random.choice([-1, 0, 1], size=(mm*pp, r))
                # Ensure no zero columns
                for c in range(r):
                    if np.all(U[:, c] == 0):
                        U[np.random.randint(nn*mm), c] = 1
                    if np.all(V[:, c] == 0):
                        V[np.random.randint(mm*pp), c] = 1
