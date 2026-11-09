"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from scipy.sparse.csgraph import laplacian

def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    adjacency = np.asarray(adjacency, dtype=np.int8)
    
    # 1. Spectral Initialization
    # Compute Laplacian L = D - A
    # For small n (100), dense eigen decomposition is fast and robust
    degrees = adjacency.sum(axis=1)
    L = np.diag(degrees) - adjacency
    
    # Find eigenvalues and eigenvectors
    # We want the eigenvector corresponding to the second smallest eigenvalue (index 1)
    # np.linalg.eigh returns eigenvalues in ascending order
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector is eigvecs[:, 1]
        fiedler_vector = eigvecs[:, 1]
        # Create initial partition based on sign of Fiedler vector
        # If vector is close to 0, randomize
        labels = np.sign(fiedler_vector).astype(np.int8)
        # Ensure both labels exist (if all same sign, flip half randomly)
        if len(np.unique(labels)) == 1:
            labels = np.random.randint(0, 2, size=n).astype(np.int8)
        labels[labels == -1] = 1 # Map -1 to 1, 0 stays 0 (or 1 to 1, -1 to 0)
        # Actually np.sign gives -1, 0, 1. Let's map 0 to random, -1 to 0, 1 to 1
        zeros = np.where(labels == 0)[0]
        if len(zeros) > 0:
            labels[zeros] = np.random.randint(0, 2, size=len(zeros))
        labels[labels == -1] = 0
        labels[labels == 1] = 1
        
    except Exception:
        # Fallback to random if spectral fails
        labels = np.random.randint(0, 2, size=n).astype(np.int8)

    # 2. Local Improvement (Hill Climbing)
    # We perform several rounds of greedy flips to maximize cut size.
    # This is faster than SA for small n and deterministic.
    
    def get_cut_size(labels):
        return np.sum(adjacency * (labels[:, None] != labels[None, :]))
    
    current_labels = labels.copy()
    current_score = get_cut_size(current_labels)
    
    # Multiple restarts of local search starting from spectral solution and random ones
    best_labels = current_labels.copy()
    best_score = current_score
    
    # Run local search from spectral start
    improved = True
    while improved:
        improved = False
        for v in range(n):
            # Calculate delta if we flip v
            # Delta = (edges to same side) - (edges to opp side)
            # If delta > 0, flipping improves cut
            same_count = adjacency[v, current_labels == current_labels[v]].sum()
            opp_count = adjacency[v, current_labels != current_labels[v]].sum()
            delta = same_count - opp_count
            if delta > 0:
                current_labels[v] = 1 - current_labels[v]
                current_score += delta
                improved = True
        # If no improvement in a full pass, we are at a local optimum
        
    if current_score > best_score:
        best_score = current_score
        best_labels = current_labels.copy()
        
    # Also try a few random restarts with local search to be safe
    # Since n=100 is small, this is cheap
    for _ in range(5):
        rand_labels = np.random.randint(0, 2, size=n).astype(np.int8)
        r_score = get_cut_size(rand_labels)
        
        improved = True
        while improved:
            improved = False
            for v in range(n):
                same_count = adjacency[v, rand_labels == rand_labels[v]].sum()
                opp_count = adjacency[v, rand_labels != rand_labels[v]].sum()
                delta = same_count - opp_count
                if delta > 0:
                    rand_labels[v] = 1 - rand_labels[v]
                    r_score += delta
                    improved = True
        
        if r_score > best_score:
            best_score = r_score
            best_labels = rand_labels.copy()

    return best_labels
# EVOLVE-BLOCK-END
