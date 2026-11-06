import numpy as np
import math
import time

_START_TIME = time.time()
_DEADLINE = 265.0  # safety margin before 270s kill

def _elapsed():
    return time.time() - _START_TIME

def _jc_distance(a, b):
    """Compute Jukes-Cantor corrected distance between two DNA sequences."""
    n = len(a)
    if n == 0:
        return 0.0
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    p = diffs / n
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - (4.0 / 3.0) * p)

def _build_distance_matrix(sequences, labels):
    """Build pairwise JC distance matrix using numpy for speed."""
    n = len(labels)
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.array([list(map(lambda x: seq_map[x], s)) for s in sequences.values()], dtype=np.int8)
    
    # Compute pairwise Hamming distances vectorized
    diff = (seqs[:, None, :] != seqs[None, :, :]).sum(axis=2)
    p = diff / diff.shape[1]
    
    # JC Correction
    p_safe = np.clip(p, 0, 0.7499)
    D = -0.75 * np.log(1.0 - 4.0 * p_safe / 3.0)
    np.fill_diagonal(D, 0)
    
    # Map back to labels
    dist = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[(labels[i], labels[j])] = D[i, j]
    
    return dist

def _neighbor_joining(dist, labels):
    """
    Perform Neighbor Joining. Returns a Newick string.
    """
    nodes = list(labels)
    trees = {name: name for name in labels}
    matrix = dist
    
    while len(nodes) > 3:
        if _elapsed() > _DEADLINE:
            break
        
        n = len(nodes)
        total = {a: sum(matrix[(a, b)] for b in nodes if b != a) for a in nodes}
        best = None
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                criterion = (n - 2) * matrix[(a, b)] - total[a] - total[b]
                if best is None or criterion < best[0]:
                    best = (criterion, a, b)
        
        if best is None:
            break
        
        _, a, b = best
        joined = f"({trees[a]},{trees[b]})"
        new = f"n{len(nodes)}_{a}_{b}"
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            matrix[(new, x)] = matrix[(x, new)] = 0.5 * (matrix[(a, x)] + matrix[(b, x)] - matrix[(a, b)])
        nodes = rest + [new]
        trees[new] = joined
    
    return "(" + ",".join(trees[x] for x in nodes) + ");"

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Build distance matrix
    dist = _build_distance_matrix(sequences, labels)
    
    # Run Neighbor Joining
    result = _neighbor_joining(dist, labels)
    
    return result
