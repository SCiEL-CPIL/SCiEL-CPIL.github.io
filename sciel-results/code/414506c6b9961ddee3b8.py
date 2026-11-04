def snap_to_half_int(arr):
    # Round to nearest 0.5
    return np.round(arr * 2) / 2

# In construct:
# After solving, check error with snapped values.
# If error is still small, return snapped values.
