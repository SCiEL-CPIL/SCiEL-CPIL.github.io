def evaluate_np(points):
    pts = np.array(points, dtype=np.int32)
    # pts shape (30, 2)
    # We need all triples.
    # This is hard to vectorize directly without generating all triples indices.
    # Let's stick to the loop but optimize it.
    # Pre-computed triples list is good.
    pass
