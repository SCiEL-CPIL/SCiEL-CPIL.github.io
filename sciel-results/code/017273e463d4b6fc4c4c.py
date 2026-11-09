"""Maximum Cut using SDP-inspired spectral initialization + aggressive local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Convert to numpy int array
    adj = np.array(adjacency, dtype=np.int32)
    degrees = adj.sum(axis=1)
    total_edges = int(degrees.sum() // 2)

    def compute_cut(labels):
        """Compute cut size using vectorized operation."""
        diff = (labels[:, None] != labels[None, :]).astype(np.int32)
        return int(np.sum(adj * diff) // 2)

    def flip_greedy(labels, max_passes=100):
        """Single-flip greedy improvement with fast delta computation."""
        labels = labels.copy()
        # Precompute for speed: for each vertex, gain from flipping
        # gain[v] = (sum of edges to opposite side) - (sum of edges to same side)
        #          = 2*(edges to opposite) - degree[v]
        # If gain > 0, flipping v increases the cut
        
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            changed = False
            # Process in random order for better mixing
            order = np.random.permutation(n)
            for v in order:
                if time.time() > deadline:
                    break
                # Compute gain from flipping v
                # Same side neighbors: adj[v, labels==labels[v]].sum()
                # Opposite side neighbors: adj[v, labels!=labels[v]].sum()
                same_side = adj[v, labels == labels[v]].sum()
                other_side = adj[v, labels != labels[v]].sum()
                if other_side > same_side:
                    labels[v] = 1 - labels[v]
                    changed = True
            if not changed:
                break
        return labels

    def pairwise_swap_search(labels, max_passes=30):
        """2-swap local search: swap labels of two vertices."""
        labels = labels.copy()
        current_cut = compute_cut(labels)
        
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            order = np.random.permutation(n)
            for idx_i in range(n):
                if time.time() > deadline:
                    break
                i = order[idx_i]
                for idx_j in range(idx_i + 1, n):
                    if time.time() > deadline:
                        break
                    j = order[idx_j]
                    if labels[i] == labels[j]:
                        continue
                    
                    # Try swap
                    labels[i], labels[j] = labels[j], labels[i]
                    new_cut = compute_cut(labels)
                    if new_cut > current_cut:
                        current_cut = new_cut
                        improved = True
                    else:
                        labels[i], labels[j] = labels[j], labels[i]
            if not improved:
                break
        return labels, current_cut

    def sdp_spectral_init():
        """
        SDP-inspired initialization:
        1. Compute top k eigenvectors of the adjacency matrix
        2. Use them to embed vertices in R^k
        3. Project onto random hyperplanes to get partitions
        Returns list of label vectors (multiple initializations)
        """
        inits = []
        try:
            k = min(8, n)
            eigvals, eigvecs = np.linalg.eigh(adj)
            # Take top k eigenvectors
            idx = np.argsort(eigvals)[::-1]
            top_vecs = eigvecs[:, idx[:k]]  # shape (n, k)
            
            # Generate multiple random hyperplane projections
            num_projections = 5
            for _ in range(num_projections):
                random_coeff = np.random.randn(k)
                scores = top_vecs @ random_coeff
                labels = (scores > 0).astype(np.int32)
                # Ensure both partitions are non-empty
                if np.all(labels == 0) or np.all(labels == 1):
                    labels = np.random.randint(0, 2, size=n)
                inits.append(labels)
        except Exception:
            pass
        
        # Also add Fiedler vector-based init
        try:
            degrees = adj.sum(axis=1)
            L = np.diag(degrees) - adj
            eigvals, eigvecs = np.linalg.eigh(L)
            idx = np.argsort(eigvals)
            fiedler = eigvecs[:, idx[1]]
            labels = (fiedler > 0).astype(np.int32)
            if not (np.all(labels == 0) or np.all(labels == 1)):
                inits.append(labels)
        except Exception:
            pass
        
        return inits

    def random_init():
        return np.random.randint(0, 2, size=n)

    def alternating_init():
        return np.arange(n) % 2

    def perturbation_search(labels, num_perturbations=20):
        """
        Apply random perturbations to escape local optima.
        """
        labels = labels.copy()
        current_cut = compute_cut(labels)
        
        for _ in range(num_perturbations):
            if time.time() > deadline:
                break
            
            # Create a perturbed version
            perturbed = labels.copy()
            flip_fraction = random.uniform(0.05, 0.25)
            num_flips = max(1, int(n * flip_fraction))
            flip_indices = random.sample(range(n), min(num_flips, n))
            for idx in flip_indices:
                perturbed[idx] = 1 - perturbed[idx]
            
            # Run local search on perturbed solution
            perturbed = flip_greedy(perturbed, max_passes=30)
            perturbed, cut = pairwise_swap_search(perturbed, max_passes=15)
            
            if cut > current_cut:
                current_cut = cut
                labels = perturbed.copy()
        
        return labels, current_cut

    best_labels = None
    best_cut = -1

    def try_solution(labels):
        nonlocal best_labels, best_cut
        if time.time() > deadline:
            return
        labels = flip_greedy(labels, max_passes=50)
        labels, cut = pairwise_swap_search(labels, max_passes=25)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # 1. SDP spectral initializations (multiple)
    if time.time() < deadline:
        sdp_inits = sdp_spectral_init()
        for init_labels in sdp_inits:
            if time.time() > deadline:
                break
            try_solution(init_labels)

    # 2. Alternating parity init
    if time.time() < deadline:
        try_solution(alternating_init())

    # 3. Random restarts with local search
    num_random_restarts = 15
    for r in range(num_random_restarts):
        if time.time() > deadline:
            break
        try_solution(random_init())

    # 4. Perturbation search on best solution
    if best_labels is not None and time.time() < deadline - 15:
        labels, cut = perturbation_search(best_labels, num_perturbations=40)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # 5. Final polish
    if best_labels is not None and time.time() < deadline - 5:
        labels, cut = pairwise_swap_search(best_labels, max_passes=50)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # 6. If we have time, do more random restarts
    if best_labels is not None and time.time() < deadline - 10:
        for r in range(10):
            if time.time() > deadline:
                break
            try_solution(random_init())

    if best_labels is None:
        return np.arange(n) % 2
    return best_labels
# EVOLVE-BLOCK-END
