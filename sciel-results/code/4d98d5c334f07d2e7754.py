"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    # Ensure adjacency is a numpy array
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit

    # Compute degree
    degree = adj.sum(axis=1)

    # Compute the Laplacian: L = D - A
    # D is diagonal with degrees
    L = np.diag(degree) - adj

    # Compute eigenvalues and eigenvectors of the Laplacian
    # We want the Fiedler vector (second smallest eigenvalue's eigenvector)
    # For efficiency, use eigh which works for symmetric matrices
    try:
        if time.time() < deadline:
            eigvals, eigvecs = np.linalg.eigh(L)
            # Sort by eigenvalue (eigh already returns sorted for symmetric)
            # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
            # eigvecs[:, 1] is the Fiedler vector
            fiedler = eigvecs[:, 1]
            fiedler_abs = np.abs(fiedler)
        else:
            # Fallback: use degree-based scoring
            return -degree.astype(float)
    except Exception:
        return -degree.astype(float)

    # Normalize fiedler values to [0, 1] range for better combination
    fmin = fiedler_abs.min()
    fmax = fiedler_abs.max()
    if fmax > fmin:
        fiedler_norm = (fiedler_abs - fmin) / (fmax - fmin)
    else:
        fiedler_norm = np.zeros_like(fiedler_abs)

    # Also compute eigenvector centrality as a secondary signal
    # Eigenvector centrality: principal eigenvector of adjacency matrix
    try:
        if time.time() < deadline:
            # Use power iteration for speed on large graphs
            # Or just use numpy's eig
            eigvals_adj, eigvecs_adj = np.linalg.eig(adj)
            # Find index of largest eigenvalue
            idx = np.argmax(eigvals_adj)
            evec = eigvecs_adj[:, idx]
            # Take absolute value and normalize
            evec_abs = np.abs(evec)
            emin = evec_abs.min()
            emax = evec_abs.max()
            if emax > emin:
                evec_norm = (evec_abs - emin) / (emax - emin)
            else:
                evec_norm = np.zeros_like(evec_abs)
        else:
            evec_norm = np.zeros_like(degree)
    except Exception:
        evec_norm = np.zeros_like(degree)

    # Normalize degree to [0, 1]
    dmin = degree.min()
    dmax = degree.max()
    if dmax > dmin:
        degree_norm = (degree - dmin) / (dmax - dmin)
    else:
        degree_norm = np.zeros_like(degree)

    # Composite score:
    # We want to select vertices that are:
    # 1. Low degree (easier to keep independent) -> low degree_norm
    # 2. High Fiedler value (peripheral, less entangled) -> high fiedler_norm
    # 3. Low eigenvector centrality (not central) -> low evec_norm
    
    # Weighted combination:
    # score = -w1 * degree_norm + w2 * fiedler_norm - w3 * evec_norm
    w1 = 1.0
    w2 = 0.8
    w3 = 0.3

    score = -w1 * degree_norm + w2 * fiedler_norm - w3 * evec_norm

    return score
# EVOLVE-BLOCK-END
