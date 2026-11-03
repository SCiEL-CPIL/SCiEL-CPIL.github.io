<<<<<<< SEARCH
def _compute_coverage(selected, block_triples, n_triples):
    """Compute coverage count for each triple."""
    counts = [0] * n_triples
    for idx in selected:
        for tr_idx in block_triples[idx]:
            counts[tr_idx] += 1
    return counts
=======
def _compute_coverage(selected, block_triples, n_triples):
    """Compute coverage count for each triple."""
    counts = np.zeros(n_triples, dtype=np.int32)
    for idx in selected:
        counts[block_triples[idx]] += 1
    return counts
>>>>>>> REPLACE

