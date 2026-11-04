def _snap_to_half(arr):
    return np.round(arr * 2) / 2

def _is_valid_half_integer(arr, tol=1e-9):
    # Check if all elements are close to multiples of 0.5
    scaled = arr * 2
    rounded = np.round(scaled)
    return np.allclose(scaled, rounded, atol=tol)
