"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # labels are 0/1. An edge (u,v) crosses if labels[u] != labels[v].
    # cut = 0.5 * sum_{u,v} adj[u,v] * (labels[u] != labels[v])
    # Using matrix ops: 
    # Let x = 2*labels - 1 (so x is +/-1)
    # Then (labels[u] != labels[v]) is 1 when x[u]*x[v] = -1, i.e., (1 - x[u]*x[v])/2
    # So cut = 0.5 * sum_{u,v} adj[u,v] * (1 - x[u]*x[v])/2
    #        = 0.25 * (sum_{u,v} adj[u,v] - sum_{u,v} adj[u,v]*x[u]*x[v])
    #        = 0.25 * (2*E - x^T A x)  where E is number of edges (sum of upper triangle)
    # Alternatively, simpler:
    # cut = sum_v adj[v] @ (1 - labels)  ... but this counts each edge twice? No.
    # For edge (u,v) with u<v: if labels differ, it's counted in adj[u]@(1-labels) and adj[v]@(1-labels).
    # So sum_v adj[v] @ (1-labels) = 2 * cut.
    # Thus cut = 0.5 * (adj @ (1 - labels)).sum()
    # But wait, adj is symmetric. Let's verify:
    # sum_v sum_u adj[v,u] * (1-labels[u]) = sum_{u,v} adj[u,v] * (1-labels[v])
    # = sum_{u,v} adj[u,v] - sum_{u,v} adj[u,v]*labels[v]
    # For an edge (u,v) where labels[u]=0, labels[v]=1:
    #   adj[u,v]=1, adj[v,u]=1
    #   term (u,v): 1*(1-1) = 0
    #   term (v,u): 1*(1-0) = 1
    #   total = 1. But the edge crosses, so we want to count it once.
    #   Wait, the sum is over all ordered pairs. The edge contributes 1 to the sum.
    #   So sum = number of crossing edges? No, each crossing edge contributes exactly 1?
    #   Let's re-check: sum_{u,v} adj[u,v] * (1-labels[v])
    #   For a crossing edge (u,v) with labels[u]=0, labels[v]=1:
    #     (u,v): adj[u,v]*(1-labels[v]) = 1*0 = 0
    #     (v,u): adj[v,u]*(1-labels[u]) = 1*1 = 1
    #   For a crossing edge (u,v) with labels[u]=1, labels[v]=0:
    #     (u,v): adj[u,v]*(1-labels[v]) = 1*1 = 1
    #     (v,u): adj[v,u]*(1-labels[u]) = 1*0 = 0
    #   For a non-crossing edge (both 0):
    #     (u,v): 1*1 = 1, (v,u): 1*1 = 1 -> total 2? That's wrong.
    #   So this formula is incorrect.
    
    # Correct approach:
    # cut = 0.5 * sum_{u,v} adj[u,v] * (labels[u] != labels[v])
    # Let x = 2*labels - 1 (values in {-1, 1})
    # labels[u] != labels[v] iff x[u] != x[v] iff x[u]*x[v] = -1
    # (labels[u] != labels[v]) = (1 - x[u]*x[v]) / 2
    # cut = 0.5 * sum_{u,v} adj[u,v] * (1 - x[u]*x[v]) / 2
    #     = 0.25 * (sum_{u,v} adj[u,v] - sum_{u,v} adj[u,v]*x[u]*x[v])
    #     = 0.25 * (2 * num_edges - x^T A x)
    
    x = 2.0 * labels - 1.0
    xT_A_x = x @ (adj @ x)
    num_edges = 0.5 * adj.sum()
    return 0.25 * (2.0 * num_edges - xT_A_x)


def _spectral_init(adj, n, rng):
    """
    Generate initial partition using spectral embedding.
    Use the Fiedler vector (second eigenvector of the Laplacian) or
    a random projection of the top eigenvectors.
    """
    # Compute Laplacian L = D - A
    degrees = adj.sum(axis=1)
    D = np.diag(degrees)
    L = D - adj
    
    # Compute eigenvalues and eigenvectors
    # For n=100, this is fast
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        # Fallback to random
        return rng.integers(0, 2, size=n)
    
    # Sort by eigenvalue (eigh already returns sorted)
    # The smallest eigenvalue is 0 (for connected graph), next is Fiedler value
    # Use the eigenvector corresponding to the second smallest eigenvalue
    # Or use a combination of top few eigenvectors for a random hyperplane
    
    # Strategy: Use the Fiedler vector (index 1) to split
    # Or use a random linear combination of the top k eigenvectors
    
    # Let's use a random projection in the subspace spanned by the top few eigenvectors
    # to get diverse initial solutions.
    
    # Take the top min(n, 10) eigenvectors (excluding the first which is constant)
    k = min(n, 10)
    V = eigenvectors[:, 1:k+1]  # Shape (n, k)
    
    # Generate a random vector in this subspace
    w = rng.standard_normal(k)
    # Project: v = V @ w
    v = V @ w
    
    # Split by sign
    labels = (v > 0).astype(int)
    
    # Ensure balance (optional, but can help)
    # If too unbalanced, flip the minority side? Or just leave it.
    # For Max Cut, balance isn't strictly required, but helps local search.
    
    return labels


def _greedy_flip_improve(adj, labels, max_iter=100):
    """
    Greedy local search: repeatedly flip the vertex that gives the best gain
    until no improvement is possible.
    """
    n = len(labels)
    labels = labels.copy()
    
    # Precompute degrees
    degrees = adj.sum(axis=1)
    
    for _ in range(max_iter):
        improved = False
        best_gain = 0
        best_v = -1
        
        for v in range(n):
            # Gain from flipping v:
            # Current contribution of v to cut:
            #   neighbors in opposite partition
            # New contribution:
            #   neighbors in same partition (after flip, they become opposite)
            # Gain = (neighbors in same) - (neighbors in opposite)
            #      = (degree[v] - neighbors in opposite) - (neighbors in opposite)
            #      = degree[v] - 2 * (neighbors in opposite)
            
            # neighbors in opposite = adj[v] @ (1 - labels)
            # But careful: adj[v] includes self? No, no self-loops.
            # adj[v] @ (1 - labels) counts neighbors u where labels[u] != labels[v]
            
            opposite = adj[v] @ (1 - labels)
            same = degrees[v] - opposite
            
            gain = same - opposite  # = degree[v] - 2 * opposite
            
            if gain > best_gain:
                best_gain = gain
                best_v = v
        
        if best_v == -1 or best_gain <= 0:
            break
        
        labels[best_v] = 1 - labels[best_v]
        improved = True
    
    return labels


def _simulated_annealing(adj, labels, start_temp=10.0, end_temp=0.1, iterations=2000):
    """
    Simulated annealing to escape local optima.
    """
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adj, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    degrees = adj.sum(axis=1)
    
    for i in range(iterations):
        # Cool down
        temp = start_temp * (end_temp / start_temp) ** (i / iterations)
        
        # Pick random vertex to flip
        v = np.random.randint(n)
        
        # Compute gain
        opposite = adj[v] @ (1 - labels)
        same = degrees[v] - opposite
        gain = same - opposite
        
        # Accept if gain > 0 or with probability exp(gain/temp)
        if gain > 0 or (temp > 0 and np.random.random() < np.exp(gain / temp)):
            labels[v] = 1 - labels[v]
            current_cut += gain
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
    
    return best_labels, best_cut


def partition(adjacency):
    n = len(adjacency)
    adj = np.array(adjacency, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Set random seed for reproducibility within a run, but vary across calls?
    # The task says "five generated unweighted 100-node graphs", so each call is independent.
    # We can use a fixed seed for determinism, or let it be random.
    # Let's use a fixed seed for the spectral part, but random for SA.
    rng = np.random.RandomState(42)
    
    best_labels = None
    best_cut = -1
    
    # Strategy:
    # 1. Generate multiple initial solutions using spectral embedding with different random projections
    # 2. Apply greedy local search to each
    # 3. Apply simulated annealing to the best ones
    # 4. Return the best found
    
    num_spectral_inits = 50
    
    for i in range(num_spectral_inits):
        if time.time() > deadline - 5.0:
            break
        
        # Generate initial partition
        if i == 0:
            # First one: use Fiedler vector directly
            try:
                degrees = adj.sum(axis=1)
                D = np.diag(degrees)
                L = D - adj
                eigenvalues, eigenvectors = np.linalg.eigh(L)
                # Fiedler vector is the second eigenvector
                fiedler = eigenvectors[:, 1]
                labels = (fiedler > 0).astype(int)
            except:
                labels = rng.integers(0, 2, size=n)
        else:
            labels = _spectral_init(adj, n, rng)
        
        # Greedy improve
        labels = _greedy_flip_improve(adj, labels, max_iter=50)
        cut = _cut_size(adj, labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # If we have time, apply simulated annealing to the best solution
    if best_labels is not None and time.time() < deadline - 10.0:
        sa_labels, sa_cut = _simulated_annealing(adj, best_labels, start_temp=5.0, end_temp=0.01, iterations=5000)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels
    
    # Final greedy polish
    if best_labels is not None and time.time() < deadline - 1.0:
        best_labels = _greedy_flip_improve(adj, best_labels, max_iter=20)
    
    if best_labels is None:
        # Fallback
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
