"""Maximum Cut via fast incremental single-flip local search with multi-start and perturbation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin before 150s limit

    adj = np.array(adjacency, dtype=np.int32)
    degrees = adj.sum(axis=1).astype(np.int32)

    def compute_cut(labels):
        diff = (labels[:, None] != labels[None, :]).astype(np.int32)
        return int(np.sum(adj * diff) // 2)

    def fast_local_search(labels, max_passes=200):
        """
        Single-flip local search with incremental gain tracking.
        gain[v] = 2*cross(v) - degree(v)
        If gain[v] > 0, flipping v increases the cut by gain[v].
        After flipping v: for each neighbor u, cross(u) changes by +/-1
        (if u was opposite to v, now same -> cross(u) -= 1; else cross(u) += 1)
        gain[u] changes by +/-2 accordingly.
        """
        labels = labels.copy()
        # cross[v] = number of edges from v to vertices on the opposite side
        cross = np.zeros(n, dtype=np.int32)
        for v in range(n):
            cross[v] = int(adj[v, labels != labels[v]].sum())
        gain = 2 * cross - degrees  # gain[v] = 2*cross(v) - deg(v)

        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            for v in range(n):
                if time.time() > deadline:
                    break
                if gain[v] > 0:
                    # Flip vertex v
                    labels[v] = 1 - labels[v]
                    cross[v] = degrees[v] - cross[v]
                    gain[v] = -gain[v]
                    improved = True
                    # Update gains of neighbors
                    neighbors = np.where(adj[v] == 1)[0]
                    for u in neighbors:
                        # u's cross changes: if u was opposite to old v, now same
                        # old: labels[u] != labels[v_old] -> after flip labels[u] == labels[v_new]
                        # So cross[u] decreases by 1 if labels[u] != labels[v_old] (was opposite, now same)
                        # Actually: before flip, if labels[u] != labels[v], u was opposite to v -> cross[u] includes edge to v
                        # After flip, labels[u] == labels[v_new] -> u is same side as v -> cross[u] decreases by 1
                        if labels[u] != labels[v]:  # labels[v] is now the new value
                            # u is now same side as v (since we just flipped v)
                            # Before flip: u was opposite to v_old, so edge (u,v) was crossing
                            # After flip: u is same side as v_new, so edge is no longer crossing
                            cross[u] -= 1
                            gain[u] -= 2
                        else:
                            # u was same side as v_old, now opposite to v_new
                            # Before: edge not crossing. After: edge crossing.
                            cross[u] += 1
                            gain[u] += 2
            if not improved:
                break

        final_cut = compute_cut(labels)
        return labels, final_cut

    def spectral_init():
        degrees = adj.sum(axis=1)
        L = np.diag(degrees.astype(np.float64)) - adj.astype(np.float64)
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            idx = np.argsort(eigvals)
            fiedler = eigvecs[:, idx[1]]
            labels = (fiedler > 0).astype(np.int32)
        except:
            labels = np.random.randint(0, 2, size=n)
        return labels

    def random_init():
        return np.random.randint(0, 2, size=n)

    def sdp_init():
        try:
            k = min(8, n)
            eigvals, eigvecs = np.linalg.eigh(adj.astype(np.float64))
            idx = np.argsort(eigvals)[::-1]
            top_vecs = eigvecs[:, idx[:k]]
            random_coeff = np.random.randn(k)
            scores = top_vecs @ random_coeff
            labels = (scores > 0).astype(np.int32)
            if np.all(labels == 0) or np.all(labels == 1):
                labels = np.random.randint(0, 2, size=n)
        except Exception:
            labels = np.random.randint(0, 2, size=n)
        return labels

    def perturbation(labels, cut, num_perturbations=100):
        """Perturbation-based diversification to escape local optima."""
        best_labels = labels.copy()
        best_cut = cut

        for _ in range(num_perturbations):
            if time.time() > deadline - 5:
                break

            perturbed = best_labels.copy()
            # Flip a random subset: 5-25% of vertices
            flip_frac = random.uniform(0.05, 0.25)
            num_flips = max(1, int(n * flip_frac))
            flip_idx = random.sample(range(n), num_flips)
            for idx in flip_idx:
                perturbed[idx] = 1 - perturbed[idx]

            perturbed, cut_p = fast_local_search(perturbed, max_passes=100)
            if cut_p > best_cut:
                best_cut = cut_p
                best_labels = perturbed.copy()

        return best_labels, best_cut

    best_labels = None
    best_cut = -1

    # Phase 1: Multiple diverse initializations with local search
    inits = []

    # Spectral
    if time.time() < deadline - 30:
        lab = spectral_init()
        inits.append(lab)

    # SDP
    if time.time() < deadline - 30:
        lab = sdp_init()
        inits.append(lab)

    # Chebyshev (top eigenvector of A)
    if time.time() < deadline - 30:
        try:
            eigvals, eigvecs = np.linalg.eigh(adj.astype(np.float64))
            idx = np.argmax(eigvals)
            vec = eigvecs[:, idx]
            lab = (vec > 0).astype(np.int32)
            inits.append(lab)
        except:
            pass

    # Alternating parity
    inits.append(np.arange(n) % 2)

    # Several random inits
    for _ in range(5):
        if time.time() > deadline - 30:
            break
        inits.append(random_init())

    for lab in inits:
        if time.time() > deadline - 10:
            break
        lab, cut = fast_local_search(lab, max_passes=300)
        if cut > best_cut:
            best_cut = cut
            best_labels = lab.copy()

    # Phase 2: Perturbation search on best solution
    if best_labels is not None and time.time() < deadline - 10:
        remaining = deadline - time.time()
        num_pert = max(10, int(remaining * 3))  # Scale perturbations to available time
        best_labels, best_cut = perturbation(best_labels, best_cut, num_perturbations=num_pert)

    # Phase 3: Final polish - run local search again on best
    if best_labels is not None and time.time() < deadline - 3:
        best_labels, best_cut = fast_local_search(best_labels, max_passes=500)

    if best_labels is None:
        return np.arange(n) % 2
    return best_labels
# EVOLVE-BLOCK-END
