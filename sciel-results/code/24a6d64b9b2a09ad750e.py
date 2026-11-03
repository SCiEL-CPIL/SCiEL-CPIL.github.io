            rest_arr = np.array(list(rest))
            if len(rest_arr) == 0:
                conflict_with_rest = np.zeros(len(neigh), dtype=bool)
            else:
                # Correctly index the submatrix
                submatrix = Conflict[np.ix_(neigh, rest_arr)]
                conflict_with_rest = submatrix.any(axis=1)
