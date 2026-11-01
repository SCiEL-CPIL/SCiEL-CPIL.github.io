            if len(V5_even) > 0:
                dots = V5_even @ V5_even.T
                np.fill_diagonal(dots, -100)
                max_dot = np.max(dots)
                if max_dot <= 2:
                    return V5_even
