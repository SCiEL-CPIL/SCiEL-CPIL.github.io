"""Phylogenetic topology recovery using quartet methods with NJ fallback."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from itertools import combinations


def _jc_distance(p):
    """Jukes-Cantor correction from proportion differences."""
    if p >= 0.7499:
        return 3.0
    if p <= 0.0:
        return 0.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distances(labels, sequences):
    """Compute JC-corrected pairwise distances."""
    n = len(labels)
    seqs = [sequences[l] for l in labels]
    n_sites = len(seqs[0])
    
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            diffs = sum(1 for a, b in zip(seqs[i], seqs[j]) if a != b)
            p = diffs / n_sites
            d = _jc_distance(p)
            dist[i][j] = d
            dist[j][i] = d
    return dist


def _quartet_scores(dist, i, j, k, l):
    """
    Score the three quartet topologies for taxa i,j,k,l.
    Returns (score_ij_kl, score_ik_jl, score_il_jk) where higher is better.
    Uses the principle: the best topology minimizes the sum of between-pair distances.
    """
    # For (i,j)|(k,l): between pairs are (i,k),(i,l),(j,k),(j,l)
    cost_ij_kl = dist[i][k] + dist[i][l] + dist[j][k] + dist[j][l]
    # For (i,k)|(j,l): between pairs are (i,j),(i,l),(k,j),(k,l)
    cost_ik_jl = dist[i][j] + dist[i][l] + dist[k][j] + dist[k][l]
    # For (i,l)|(j,k): between pairs are (i,j),(i,k),(l,j),(l,k)
    cost_il_jk = dist[i][j] + dist[i][k] + dist[l][j] + dist[l][k]
    
    return (-cost_ij_kl, -cost_ik_jl, -cost_il_jk)


def _build_quartet_tree(labels, dist, deadline):
    """
    Build tree using greedy quartet method.
    For each pair (i,j), compute how often (i,j)|(k,l) is the best quartet
    over all k,l. Join the pair with highest score.
    """
    n = len(labels)
    if n <= 3:
        if n == 1:
            return labels[0]
        elif n == 2:
            return f"({labels[0]},{labels[1]})"
        else:
            return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Compute pair scores: for each pair (i,j), count quartet support
    pair_support = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            support = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                for l in range(k + 1, n):
                    if l == i or l == j:
                        continue
                    # Quartet (i,j,k,l)
                    s_ij_kl, s_ik_jl, s_il_jk = _quartet_scores(dist, i, j, k, l)
                    best = max(s_ij_kl, s_ik_jl, s_il_jk)
                    if s_ij_kl == best:
                        support += 1
                    elif s_ik_jl == best or s_il_jk == best:
                        support -= 1
            pair_support[i][j] = support
            pair_support[j][i] = support
    
    # Greedy joining
    active = list(range(n))
    node_newick = {i: labels[i] for i in range(n)}
    next_id = n
    
    while len(active) > 3:
        if time.time() > deadline - 5:
            break
        
        # Find pair with highest support
        best_support = -float('inf')
        best_pair = None
        for idx_i, i in enumerate(active):
            for idx_j, j in enumerate(active):
                if idx_j <= idx_i:
                    continue
                s = pair_support[i][j]
                if s > best_support:
                    best_support = s
                    best_pair = (i, j)
        
        if best_pair is None:
            break
        
        i, j = best_pair
        new_id = next_id
        next_id += 1
        
        node_newick[new_id] = f"({node_newick[i]},{node_newick[j]})"
        
        # Update pair_support for new node
        for k in active:
            if k == i or k == j:
                continue
            # Average the supports
            new_support = (pair_support[i][k] + pair_support[j][k]) / 2.0
            pair_support[new_id][k] = new_support
            pair_support[k][new_id] = new_support
        
        active.remove(i)
        active.remove(j)
        active.append(new_id)
    
    if len(active) == 3:
        a, b, c = active
        return f"({node_newick[a]},{node_newick[b]},{node_newick[c]})"
    else:
        # Fallback: join remaining by minimum distance
        while len(active) > 3:
            best_d = float('inf')
            best_pair = None
            for idx_i, i in enumerate(active):
                for idx_j, j in enumerate(active):
                    if idx_j <= idx_i:
                        continue
                    d = dist[i][j]
                    if d < best_d:
                        best_d = d
                        best_pair = (i, j)
            i, j = best_pair
            new_id = next_id
            next_id += 1
            node_newick[new_id] = f"({node_newick[i]},{node_newick[j]})"
            active.remove(i)
            active.remove(j)
            active.append(new_id)
        a, b, c = active
        return f"({node_newick[a]},{node_newick[b]},{node_newick[c]})"


def _nj_tree(labels, dist, deadline):
    """Standard Neighbor Joining."""
    n = len(labels)
    if n <= 1:
        return labels[0] if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    active = list(range(n))
    node_newick = {i: labels[i] for i in range(n)}
    d_matrix = dist.copy()
    next_id = n
    
    while len(active) > 3:
        if time.time() > deadline - 5:
            break
        
        n_active = len(active)
        r = np.zeros(n_active)
        for idx, i in enumerate(active):
            for jdx, j in enumerate(active):
                if i != j:
                    r[idx] += d_matrix[i][j]
        
        best_val = float('inf')
        best_i = None
        best_j = None
        
        for u in range(n_active):
            i = active[u]
            for v in range(u + 1, n_active):
                j = active[v]
                criterion = (n_active - 2) * d_matrix[i][j] - r[u] - r[v]
                if criterion < best_val:
                    best_val = criterion
                    best_i = i
                    best_j = j
        
        if best_i is None:
            break
        
        i, j = best_i, best_j
        new_id = next_id
        next_id += 1
        
        node_newick[new_id] = f"({node_newick[i]},{node_newick[j]})"
        
        d_ij = d_matrix[i][j]
        for x in active:
            if x == i or x == j:
                continue
            d_nx = 0.5 * (d_matrix[i][x] + d_matrix[j][x] - d_ij)
            if d_nx < 0:
                d_nx = 0.0
            d_matrix[new_id][x] = d_nx
            d_matrix[x][new_id] = d_nx
        
        active.remove(i)
        active.remove(j)
        active.append(new_id)
    
    if len(active) == 3:
        a, b, c = active
        return f"({node_newick[a]},{node_newick[b]},{node_newick[c]})"
    else:
        while len(active) > 3:
            best_d = float('inf')
            best_pair = None
            for idx_i, i in enumerate(active):
                for idx_j, j in enumerate(active):
                    if idx_j <= idx_i:
                        continue
                    d = d_matrix[i][j]
                    if d < best_d:
                        best_d = d
                        best_pair = (i, j)
            i, j = best_pair
            new_id = next_id
            next_id += 1
            node_newick[new_id] = f"({node_newick[i]},{node_newick[j]})"
            active.remove(i)
            active.remove(j)
            active.append(new_id)
        a, b, c = active
        return f"({node_newick[a]},{node_newick[b]},{node_newick[c]})"


def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 260.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    if n_taxa <= 1:
        return labels[0] if labels else ""
    if n_taxa == 2:
        return f"({labels[0]},{labels[1]})"
    if n_taxa == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    dist = _compute_distances(labels, sequences)
    
    # For small n, use quartet method; for larger n, use NJ
    if n_taxa <= 16 and time.time() < deadline - 30:
        try:
            result = _build_quartet_tree(labels, dist, deadline)
            if result and len(result) > 5:
                return result
        except Exception:
            pass
    
    return _nj_tree(labels, dist, deadline)
# EVOLVE-BLOCK-END
