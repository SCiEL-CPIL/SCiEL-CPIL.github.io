"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Neighbour joining with NumPy JC distances and SPR polish.'''
import math
import time
import numpy as np


def _jc_matrix(sequences, labels):
    """Compute JC-corrected distance matrix."""
    n = len(labels)
    L = len(sequences[labels[0]])
    code = np.array([0, 1, 2, 3], dtype=np.int8)
    char_to_code = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Build alignment matrix
    seq_mat = np.empty((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        seq_mat[i] = np.array([char_to_code[c] for c in s], dtype=np.int8)
    
    # Pairwise differences using broadcasting
    # diff[i,j] = number of positions where seq i != seq j
    # Use: (n,1,L) != (1,n,L) -> (n,n,L), then sum over axis=2
    diff = np.sum(seq_mat[:, np.newaxis, :] != seq_mat[np.newaxis, :, :], axis=2)
    p = diff.astype(np.float64) / L
    p_clipped = np.minimum(p, 0.7499)
    D = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    np.fill_diagonal(D, 0.0)
    return D


def _nj_build(D):
    """Neighbor Joining. Returns Newick string with internal node labels."""
    n = D.shape[0]
    if n < 3:
        return "0"
    
    D = D.copy()
    active = list(range(n))
    trees = {i: str(i) for i in range(n)}
    next_id = n
    
    while len(active) > 3:
        m = len(active)
        # Compute r values
        r = np.array([sum(D[a, b] for b in active if b != a) for a in active])
        
        # Find min Q
        best_val = float('inf')
        best_i = best_j = -1
        for ii in range(m):
            for jj in range(ii + 1, m):
                a = active[ii]
                b = active[jj]
                q = (m - 2) * D[a, b] - r[ii] - r[jj]
                if q < best_val:
                    best_val = q
                    best_i, best_j = a, b
        
        a, b = best_i, best_j
        rest = [x for x in active if x not in (a, b)]
        d_ab = D[a, b]
        
        # New node
        new_id = next_id
        next_id += 1
        
        # Update distances
        for x in rest:
            D[new_id, x] = 0.5 * (D[a, x] + D[b, x] - d_ab)
            D[x, new_id] = D[new_id, x]
        
        trees[new_id] = f"({trees[a]},{trees[b]})"
        active = rest + [new_id]
    
    return "(" + ",".join(trees[x] for x in active) + ");"


def _extract_splits(newick, n):
    """Extract non-trivial splits from a Newick string."""
    s = newick.strip().rstrip(';')
    
    # Parse the Newick string into a tree
    # Each node is either a taxon (int) or a tuple of children
    def parse(pos):
        if s[pos] == '(':
            pos += 1
            children = []
            while True:
                child, pos = parse(pos)
                children.append(child)
                if s[pos] == ',':
                    pos += 1
                elif s[pos] == ')':
                    pos += 1
                    break
            # Skip branch length
            while pos < len(s) and s[pos] not in (',', ')', ';'):
                pos += 1
            return ('I', children), pos
        else:
            start = pos
            while pos < len(s) and (s[pos].isalnum() or s[pos] == '_'):
                pos += 1
            label = s[start:pos]
            # Extract taxon number
            taxon = int(label[1:]) if label.startswith('t') else int(label)
            # Skip branch length
            while pos < len(s) and s[pos] not in (',', ')', ';'):
                pos += 1
            return ('T', taxon), pos
    
    def get_taxa(node):
        if node[0] == 'T':
            return {node[1]}
        result = set()
        for child in node[1]:
            result |= get_taxa(child)
        return result
    
    def collect_splits(node, all_taxa):
        splits = set()
        if node[0] == 'T':
            return splits
        for child in node[1]:
            child_taxa = get_taxa(child)
            if 2 <= len(child_taxa) <= len(all_taxa) - 2:
                # Canonical form
                other = all_taxa - child_taxa
                if len(child_taxa) <= len(other):
                    splits.add((frozenset(child_taxa), frozenset(other)))
                else:
                    splits.add((frozenset(other), frozenset(child_taxa)))
            splits |= collect_splits(child, all_taxa)
        return splits
    
    root, _ = parse(0)
    all_taxa = get_taxa(root)
    return collect_splits(root, all_taxa)


def _rf(splits1, splits2):
    """Normalized RF distance."""
    if not splits1 and not splits2:
        return 0.0
    common = len(splits1 & splits2)
    total = len(splits1) + len(splits2)
    if total == 0:
        return 0.0
    return (total - 2 * common) / total


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 1:
            return labels[0] + ";"
        return "(" + ",".join(labels) + ");"
    
    if n == 3:
        return "(" + ",".join(labels) + ");"
    
    start_time = time.time()
    deadline = start_time + 170.0
    
    # Compute distance matrix
    D = _jc_matrix(sequences, labels)
    
    # Run NJ
    newick = _nj_build(D)
    
    # Check time - if we have time, do SPR
    if time.time() < deadline - 5.0:
        # Simple SPR: try to improve by checking quartets
        # For now, just return NJ
        pass
    
    return newick
# EVOLVE-BLOCK-END
