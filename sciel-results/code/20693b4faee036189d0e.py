"""Spectral partitioning with local search refinement for Maximum Cut."""
# EVOLVE-BLOCK-START
import numpy as np


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array if needed
    A = np.asarray(adjacency, dtype=np.float64)
    
    # Compute degree vector
    degrees = A.sum(axis=1)
    
    # Build graph Laplacian: L = D - A
    D = np.diag(degrees)
    L = D - A
    
    # Compute eigenvalues and eigenvectors
    # For n=100, this is fast enough
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        # Fallback to simple alternating pattern
        return np.arange(n) % 2
    
    # The Fiedler vector is the eigenvector corresponding to the second-smallest eigenvalue
    # eigh returns eigenvalues in ascending order, so index 1 is the second smallest
    fiedler_vector = eigenvectors[:, 1]
    
    # Create initial partition: vertices with positive Fiedler values go to 0, negative to 1
    # We can use the median or zero as threshold; median ensures balanced partition
    # For max cut, balance isn't strictly required, but using zero threshold is standard
    labels = np.where(fiedler_vector >= 0, 0, 1).astype(int)
    
    # Helper function to calculate cut size
    def cut_size(labels):
        # Edges crossing the partition
        # (labels[i] != labels[j]) * A[i][j]
        # Sum over all i < j to avoid double counting, or sum all and divide by 2
        # Using boolean mask
        mask = (labels[:, None] != labels[None, :])
        return np.sum(A * mask) / 2.0
    
    # Helper function to perform 1-flip local search
    def local_search_1flip(labels, max_passes=10):
        labels = labels.copy()
        for _ in range(max_passes):
            improved = False
            for v in range(n):
                # Count edges going to same side vs other side
                same_side = (labels == labels[v])
                other_side = (labels != labels[v])
                
                # Number of edges from v to same side
                edges_same = A[v][same_side].sum()
                # Number of edges from v to other side
                edges_other = A[v][other_side].sum()
                
                # If flipping v increases the cut, flip it
                if edges_same > edges_other:
                    labels[v] = 1 - labels[v]
                    improved = True
            
            if not improved:
                break
        return labels

    # Helper function to perform 2-swap local search
    def local_search_2swap(labels, max_passes=10):
        labels = labels.copy()
        current_cut = cut_size(labels)
        
        for _ in range(max_passes):
            improved = False
            # Iterate over all pairs of vertices with different labels
            # Swapping them changes the cut size
            for i in range(n):
                for j in range(i + 1, n):
                    if labels[i] == labels[j]:
                        continue
                    
                    # Calculate change in cut size if we swap i and j
                    # Let S_i be neighbors of i, S_j be neighbors of j
                    # If we swap i and j, 
                    # i moves from group 0 to 1 (assuming labels[i]=0, labels[j]=1)
                    # j moves from group 1 to 0
                    
                    # Let's derive the delta:
                    # Current contribution of edges involving i and j to the cut:
                    # Edges from i to others: 
                    #   If other is in same group as i, not in cut. If in other, in cut.
                    # Edges from j to others:
                    #   If other is in same group as j, not in cut. If in other, in cut.
                    
                    # It's easier to just simulate the flip and check the gain.
                    # Or use the formula:
                    # Delta = (edges from i to group_j - edges from i to group_i) 
                    #       + (edges from j to group_i - edges from j to group_j)
                    #       - 2 * A[i][j]  (because edge i-j changes status twice? No, let's be careful)
                    
                    # Let's use a simpler, safer method for n=100:
                    # Just apply the swap, calculate new cut, revert if worse.
                    # But calculating cut_size is O(n^2). Doing this for O(n^2) swaps is O(n^4) = 10^8 ops.
                    # 10^8 simple ops in Python/NumPy might be slow (seconds).
                    # Let's optimize the delta calculation.
                    
                    # Let G0 be set of nodes with label 0, G1 be set with label 1.
                    # Assume labels[i] == 0, labels[j] == 1.
                    # Neighbors of i: N(i). Neighbors of j: N(j).
                    # 
                    # Edges in cut currently:
                    # 1. Edges from i to G1 (excluding j if connected)
                    # 2. Edges from j to G0 (excluding i if connected)
                    # 3. Other edges.
                    #
                    # After swap:
                    # i is in G1, j is in G0.
                    # New edges from i to G0 (excluding j)
                    # New edges from j to G1 (excluding i)
                    
                    # Let's compute:
                    # deg_i_G0 = A[i, G0].sum() (includes i? No, A[i,i]=0 usually)
                    # deg_i_G1 = A[i, G1].sum()
                    # deg_j_G0 = A[j, G0].sum()
                    # deg_j_G1 = A[j, G1].sum()
                    
                    # Current cut contribution from i: deg_i_G1
                    # Current cut contribution from j: deg_j_G0
                    # Note: If A[i,j] is 1, it is counted in deg_i_G1 and deg_j_G0.
                    # So current cut edges involving i or j = deg_i_G1 + deg_j_G0 - (1 if A[i,j] else 0)
                    # Wait, if A[i,j]=1, it's one edge. It is counted in deg_i_G1 (since j in G1) and deg_j_G0 (since i in G0).
                    # So sum is 2. But it's 1 edge. So we subtract 1 if connected.
                    
                    # New cut contribution from i (now in G1): deg_i_G0 (neighbors in G0, excluding j? j is now in G0)
                    # Wait, j is now in G0. So if A[i,j]=1, it is in deg_i_G0.
                    # New cut contribution from j (now in G0): deg_j_G1 (neighbors in G1, excluding i? i is now in G1)
                    # If A[i,j]=1, it is in deg_j_G1.
                    
                    # New cut edges involving i or j = deg_i_G0 + deg_j_G1 - (1 if A[i,j] else 0)
                    
                    # Delta = New - Current
                    # Delta = (deg_i_G0 + deg_j_G1 - A[i,j]) - (deg_i_G1 + deg_j_G0 - A[i,j])
                    # Delta = deg_i_G0 - deg_i_G1 + deg_j_G1 - deg_j_G0
                    
                    # Let's verify.
                    # If i in G0, j in G1.
                    # deg_i_G0: neighbors of i in G0.
                    # deg_i_G1: neighbors of i in G1.
                    # deg_j_G0: neighbors of j in G0.
                    # deg_j_G1: neighbors of j in G1.
                    
                    # This formula holds.
                    
                    delta = (A[i, labels == 0].sum() - A[i, labels == 1].sum() + 
                             A[j, labels == 1].sum() - A[j, labels == 0].sum())
                    
                    # Note: A[i, labels==0] includes i? A[i,i] is 0. 
                    # A[i, labels==1] includes j? Yes, if labels[j]==1.
                    # A[j, labels==1] includes j? No, A[j,j]=0. Includes i? No, labels[i]==0.
                    # A[j, labels==0] includes i? Yes.
                    
                    # So the calculation is correct.
                    
                    if delta > 0:
                        labels[i] = 1 - labels[i]
                        labels[j] = 1 - labels[j]
                        improved = True
                        # Update current_cut if we were tracking it, but we don't need to for the logic
                        # We just need to know if improved.
            
            if not improved:
                break
        return labels

    # Run 1-flip search
    labels = local_search_1flip(labels)
    
    # Run 2-swap search
    labels = local_search_2swap(labels)
    
    # Run 1-flip search again to clean up
    labels = local_search_1flip(labels)
    
    return labels
# EVOLVE-BLOCK-END
