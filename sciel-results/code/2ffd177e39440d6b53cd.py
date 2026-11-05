"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_decode(adjacency, scores):
    """Fixed greedy decoder: pick vertices in descending score order, skipping neighbours."""
    n = adjacency.shape[0]
    order = np.argsort(-scores, kind='stable')
    picked = np.zeros(n, dtype=bool)
    forbidden = np.zeros(n, dtype=bool)
    count = 0
    for v in order:
        if forbidden[v]:
            continue
        picked[v] = True
        count += 1
        # Mark neighbours as forbidden
        neighbors = np.where(adjacency[v])[0]
        forbidden[neighbors] = True
    return count


def priority(adjacency):
    """
    Compute vertex scores for the Maximum Independent Set greedy decoder.
    Uses a combination of local degree and a global centrality measure
    (e.g., eigenvector centrality via power iteration) to better approximate
    which vertices to pick.
    """
    deadline = time.time() + 110  # Safety margin before 120s kill
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]

    if n == 0:
        return np.array([])

    degree = adjacency.sum(axis=1).astype(float)

    # --- Eigenvector centrality via power iteration ---
    # A[v, u] = 1 if edge. Eigenvector centrality: x ~ A^T x, but for undirected A = A^T.
    # Normalize to avoid dominance by large degrees.
    max_deg = degree.max()
    if max_deg > 0:
        scaled = adjacency.astype(float) / max_deg
    else:
        scaled = adjacency.astype(float)

    # Power iteration for largest eigenvector of scaled adjacency
    x = np.ones(n, dtype=float)
    x /= np.linalg.norm(x)
    for _ in range(50):
        if time.time() > deadline:
            break
        x_new = scaled.T @ x  # same as scaled @ x for symmetric
        norm = np.linalg.norm(x_new)
        if norm < 1e-12:
            break
        x_new /= norm
        if np.linalg.norm(x_new - x) < 1e-8:
            x = x_new
            break
        x = x_new

    centrality = x  # higher centrality = more "important" vertex

    # --- Composite score ---
    # We want low-degree vertices to be preferred (they block fewer others),
    # but also consider centrality: a low-degree vertex that is not central
    # is a good pick. A high-centrality vertex might be better to exclude
    # (let it block many) OR include if it has few neighbors.
    #
    # Heuristic: score = -alpha * degree + beta * centrality
    # But we need to tune alpha, beta. Let's use normalized values.
    deg_norm = degree / (degree.max() + 1e-9)
    cent_norm = centrality / (centrality.max() + 1e-9)

    # Try a few weight combinations and pick the best via a quick local evaluation?
    # But we must return a single score vector. The decoder is fixed.
    # Let's use a principled combination: prefer low degree, and among similar
    # degrees, prefer lower centrality (less "hub-like").
    #
    # Actually, for MIS, picking a vertex removes it and its neighbors.
    # A good vertex to pick is one with few neighbors (low degree) so we
    # lose fewer potential picks. Centrality is less directly helpful.
    #
    # Let's try: score = -degree * 1.0 + centrality * 0.1
    # This primarily sorts by degree, with a small tiebreaker by centrality.
    scores = -deg_norm + 0.1 * cent_norm

    # --- Local polish: try swapping a few vertices to improve the greedy result ---
    # We can't change the decoder, but we can adjust scores to change the order.
    # Let's evaluate the current greedy result.
    best_count = _greedy_decode(adjacency, scores)
    best_scores = scores.copy()

    # Try small perturbations: for each vertex, try increasing/decreasing its score
    # slightly to see if it improves the greedy result.
    # This is expensive for 200 vertices, so do a limited number of iterations.
    max_polish_iters = 20
    for _ in range(max_polish_iters):
        if time.time() > deadline:
            break
        improved = False
        # Try swapping pairs of vertices in the score order
        # This is O(n^2) per iteration, which is 40000 for n=200. Acceptable.
        for i in range(n):
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                # Swap scores of i and j
                scores[i], scores[j] = scores[j], scores[i]
                count = _greedy_decode(adjacency, scores)
                if count > best_count:
                    best_count = count
                    best_scores = scores.copy()
                    improved = True
                else:
                    # Revert
                    scores[i], scores[j] = scores[j], scores[i]
            if time.time() > deadline:
                break
        if not improved:
            break

    return best_scores
# EVOLVE-BLOCK-END
