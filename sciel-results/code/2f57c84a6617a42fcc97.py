                if f_e < f_r:
                    simplex[dim] = x_e
                    fvals[dim] = f_e
                else:
                    simplex[dim] = x_r
                    fvals[dim] = f_r
