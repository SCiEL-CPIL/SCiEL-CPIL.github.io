import numpy as np

def heuristics(prizes, distance_matrix, max_len):
        n = len(prizes)
        # d_j0 is distance from each node j to depot 0
        # distance_matrix[j, 0]
        d_to_depot = distance_matrix[:, 0]
        
        # prizes is shape (n,), need to broadcast to (n, n) for rows
        # prizes[None, :] gives shape (1, n)
        
        # denominator: distance_matrix (n, n) + d_to_depot (n,) broadcasted to (n, n) along axis 0
        # We want d[j, 0] for each column j.
        # d_to_depot is a column vector effectively when broadcast?
        # distance_matrix[i, j] + d_to_depot[j]
        # d_to_depot shape (n,). To add to columns of distance_matrix, we need shape (1, n) or (n, 1)?
        # distance_matrix is (n, n).
        # We want to add d_to_depot[j] to element [i, j].
        # So we need d_to_depot to be broadcastable to (n, n) such that it varies by column.
        # d_to_depot[None, :] has shape (1, n). This broadcasts correctly to (n, n) adding to each row the same vector.
        
        denom = distance_matrix + d_to_depot[None, :] + 1e-9
        numer = prizes[None, :] + 1e-9
        
        eta = numer / denom
        
        # Ensure positive
        eta = np.where(eta > 0, eta, 1e-9)
        
        return eta
