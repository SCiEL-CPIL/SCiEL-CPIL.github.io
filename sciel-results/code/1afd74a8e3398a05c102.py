# First: try border rank perturbation analysis on rank-36
# ...
for j in range(min(null_dim, 5)):
    # ...
    R_new = R - 1
    U_reduced = np.delete(U36, remove_col, axis=1)
    # ...
    U_init = U_reduced + scale * dU[:, remove_col:R] if remove_col < R else U_reduced
    # BUG: dU has shape (n*m, R). dU[:, remove_col:R] has shape (n*m, R - remove_col).
    # U_reduced has shape (n*m, R-1).
    # If remove_col != 0, the shapes (n*m, R-1) and (n*m, R-remove_col) do not match for broadcasting.
    # Example: R=36, remove_col=5. U_reduced is (8, 35). dU[:, 5:36] is (8, 31).
    # (8, 35) + (8, 31) -> Broadcast error.
