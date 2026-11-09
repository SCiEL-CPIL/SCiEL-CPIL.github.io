    # Compare with baseline
    if len(selected_vectors) > baseline_count:
        return np.array(selected_vectors, dtype=np.int64)
    else:
        return np.array(baseline_vectors, dtype=np.int64)
