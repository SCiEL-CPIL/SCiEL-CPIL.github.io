<<<<<<< SEARCH
    # Apply comparators
    for i, j in net:
        wi = wires[:, i]
        wj = wires[:, j]
        # min goes to i, max goes to j
        wires[:, i] = np.minimum(wi, wj)
        wires[:, j] = np.maximum(wi, wj)
=======
    # Apply comparators
    for i, j in net:
        wi = wires[:, i]
        wj = wires[:, j]
        # min goes to i, max goes to j
        # Compute min and max from original values before updating wires
        min_val = np.minimum(wi, wj)
        max_val = np.maximum(wi, wj)
        wires[:, i] = min_val
        wires[:, j] = max_val
>>>>>>> REPLACE
