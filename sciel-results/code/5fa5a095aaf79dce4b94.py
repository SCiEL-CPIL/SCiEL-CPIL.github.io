"""Phylogenetic topology recovery: NJ with vectorized distances and deadline guard."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE_MARGIN = 15.0  # seconds to leave before the 270s kill

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return 270.0 - _elapsed() - _DEADLINE_MARGIN

def _jc_distance_matrix(aln):
    """Compute JC-corrected pairwise distance matrix from alignment.
    aln: 2D numpy array of shape (n_taxa, n_sites) with values 0-3 (A,C,G,T)
    Returns: (n, n) float matrix of JC-corrected distances.
    """
    n = aln.shape[0]
    # Count mismatches: for each pair, count positions where they differ
    # Use broadcasting: aln[:, i, None] != aln[:, None, j]
    # For efficiency, compute per-site contributions
    # mismatches[i,j] = sum over sites of (aln[i,s] != aln[j,s])
    # = n_sites - sum over sites of (aln[i,s] == aln[j,s])
    
    # Vectorized: for each site, compute equality matrix
    # This is O(n^2 * L) which for n=32, L=2000 is ~2M ops per instance
    # Better: use matrix multiplication trick
    # Equality at site s: one-hot(aln[:,s]) @ one_hot(aln[:,s]).T
    
    # For small n (<=32), direct computation is fine
    mismatches = np.zeros((n, n), dtype=np.float64)
    L = aln.shape[1]
    
    # Process in chunks to avoid huge temporary arrays
    chunk_size = min(L, 200)
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        chunk = aln[:, start:end]  # (n, chunk_len)
        # For each position in chunk, compute pairwise equality
        # chunk[i, k] == chunk[j, k] for all i,j
        # Use: (chunk[:, k:k+1] == chunk[:, k:k+1].T).sum(axis=1) for each k
        # Or: one_hot approach
        # Simpler: loop over positions in chunk (chunk_len <= 200)
        for k in range(end - start):
            col = chunk[:, k]  # (n,)
            # Equality matrix: col[:, None] == col[None, :]
            eq = (col[:, None] == col[None, :]).astype(np.float64)
            mismatches += (1.0 - eq)
    
    p = mismatches / L
    np.fill_diagonal(p, 0.0)
    
    # JC correction: d = -3/4 * ln(1 - 4p/3)
    # Clip p to avoid log of negative or zero
    p_clipped = np.clip(p, 0.0, 0.7499)
    d = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    # Where p >= 0.75, set to large value
    d[p >= 0.75] = 10.0
    np.fill_diagonal(d, 0.0)
    return d

def _neighbor_joining(D, labels):
    """Perform Neighbor-Joining on distance matrix D.
    D: (n, n) distance matrix
    labels: list of n taxon labels
    Returns: Newick string of the unrooted tree.
    """
    n = len(labels)
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        # Unrooted tree with 3 taxa: any rooting gives same topology
        return f"({labels[0]},({labels[1]},{labels[2]}));"
    
    # Work with a copy
    D = D.copy()
    taxa = list(labels)
    # Tree representation: each node is either a leaf (taxon label) or a subtree
    # We'll track the Newick string for each current node
    node_newick = {i: taxa[i] for i in range(n)}
    # Active nodes
    active = list(range(n))
    # Count of active nodes
    cnt = n
    next_id = n  # for new internal nodes
    
    while cnt > 3:
        m = len(active)
        # Compute row sums: r[i] = sum of D[i][j] for j in active
        idx = np.array(active, dtype=np.int64)
        D_sub = D[np.ix_(idx, idx)]
        r = D_sub.sum(axis=1)  # (m,)
        
        # NJ criterion: Q[i,j] = (m-2)*D[i,j] - r[i] - r[j]
        # Find min Q
        Q = (m - 2) * D_sub - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        # Find the pair with minimum Q
        min_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = min_idx
        # Map back to active indices
        ai, aj = active[i], active[j]
        
        # Compute branch lengths
        dij = D_sub[i, j]
        ri, rj = r[i], r[j]
        # b_ai = 0.5 * dij + (ri - rj) / (2*(m-2))
        b_ai = 0.5 * dij + (ri - rj) / (2.0 * (m - 2))
        b_aj = dij - b_ai
        # Clamp to non-negative for stability
        b_ai = max(b_ai, 0.0)
        b_aj = max(b_aj, 0.0)
        
        # Create new node
        new_node = next_id
        next_id += 1
        
        # Build Newick for new node
        # We need to include branch lengths in the Newick for clarity,
        # but the task says branch lengths are optional. Let's include them.
        newick_str = f"({node_newick[ai]}:{b_ai:.4f},{node_newick[aj]}:{b_aj:.4f})"
        node_newick[new_node] = newick_str
        
        # Update distance matrix for new node
        # D[new][k] = 0.5 * (D[ai][k] + D[aj][k] - D[ai][aj])
        # for all k in active \ {ai, aj}
        rest = [x for x in active if x != ai and x != aj]
        if rest:
            rest_idx = np.array(rest, dtype=np.int64)
            D_new = 0.5 * (D[ai, rest_idx] + D[aj, rest_idx] - dij)
            # Update D for new node
            D[new_node, new_node] = 0.0
            D[new_node, rest_idx] = D_new
            D[rest_idx, new_node] = D_new
        
        # Remove ai, aj from active, add new_node
        active = rest + [new_node]
        cnt = len(active)
        
        # Check deadline
        if _remaining() < 5.0:
            break
    
    # Build final tree from remaining 3 nodes
    if len(active) == 3:
        a, b, c = active
        # Unrooted: (a, (b, c))
        # Get distances for branch lengths
        Dab = D[a, b] if a != b else 0.0
        Dac = D[a, c] if a != c else 0.0
        Dbc = D[b, c] if b != c else 0.0
        # Branch lengths: for unrooted 3-taxon tree
        # b_a = 0.5*(Dab + Dac - Dbc), etc.
        b_a = max(0.5 * (Dab + Dac - Dbc), 0.0)
        b_b = max(0.5 * (Dab + Dbc - Dac), 0.0)
        b_c = max(0.5 * (Dac + Dbc - Dab), 0.0)
        result = f"({node_newick[a]}:{b_a:.4f},({node_newick[b]}:{b_b:.4f},{node_newick[c]}:{b_c:.4f}));"
    elif len(active) == 2:
        a, b = active
        Dab = D[a, b]
        half = Dab / 2.0
        result = f"({node_newick[a]}:{half:.4f},{node_newick[b]}:{half:.4f});"
    elif len(active) == 1:
        result = f"({node_newick[active[0]]});"
    else:
        # Shouldn't happen, but fallback
        result = "(" + ",".join(node_newick[x] for x in active) + ");"
    
    return result

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return f"({labels[0]});" if n == 1 else "();"
    
    # Convert sequences to numpy array
    # Map A=0, C=1, G=2, T=3
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    L = len(sequences[labels[0]])
    aln = np.zeros((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        seq = sequences[lab]
        # Convert string to array
        aln[i] = [base_map[c] for c in seq]
    
    # Compute JC-corrected distance matrix
    D = _jc_distance_matrix(aln)
    
    # Check deadline
    if _remaining() < 1.0:
        # Return a star tree as fallback
        return "(" + ",".join(labels) + ");"
    
    # Run Neighbor-Joining
    newick = _neighbor_joining(D, labels)
    
    return newick
# EVOLVE-BLOCK-END
