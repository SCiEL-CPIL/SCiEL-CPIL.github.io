<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Neighbour joining on Jukes-Cantor corrected distances (Saitou and Nei 1987).'''
import math


def _distance(a, b):
    differences = sum(1 for x, y in zip(a, b) if x != y)
    p = differences / len(a)
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    nodes = list(labels)
    trees = {name: name for name in labels}
    matrix = {(a, b): _distance(sequences[a], sequences[b]) for a in labels for b in labels if a != b}
    while len(nodes) > 3:
        n = len(nodes)
        total = {a: sum(matrix[(a, b)] for b in nodes if b != a) for a in nodes}
        best = None
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                criterion = (n - 2) * matrix[(a, b)] - total[a] - total[b]
                if best is None or criterion < best[0]:
                    best = (criterion, a, b)
        _, a, b = best
        joined = f"({trees[a]},{trees[b]})"
        new = f"n{len(nodes)}_{a}_{b}"
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            matrix[(new, x)] = matrix[(x, new)] = 0.5 * (matrix[(a, x)] + matrix[(b, x)] - matrix[(a, b)])
        nodes = rest + [new]
        trees[new] = joined
    return "(" + ",".join(trees[x] for x in nodes) + ");"
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Neighbour joining on Jukes-Cantor corrected distances (Saitou and Nei 1987).'''
import math
import time


def _jc_distance(p):
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    n = len(labels)
    seqs = [sequences[lab] for lab in labels]
    L = len(seqs[0])
    # Count differences between all pairs
    # For efficiency, use numpy if available, else fallback
    try:
        import numpy as np
        # Convert sequences to integer arrays for faster comparison
        # A=0, C=1, G=2, T=3
        int_seqs = []
        for s in seqs:
            arr = np.empty(L, dtype=np.int8)
            for i, c in enumerate(s):
                if c == 'A':
                    arr[i] = 0
                elif c == 'C':
                    arr[i] = 1
                elif c == 'G':
                    arr[i] = 2
                else:
                    arr[i] = 3
            int_seqs.append(arr)
        
        int_matrix = np.array(int_seqs, dtype=np.int8)
        # Calculate pairwise differences
        # diff[i,j] = number of positions where seq_i != seq_j
        # Broadcasting: (n, 1, L) vs (1, n, L) -> (n, n, L)
        # This might be memory heavy for large n*L, but n<=32, L<=2000 -> 32*32*2000*1 byte = 2MB. Fine.
        diff_count = np.sum(int_matrix[:, np.newaxis, :] != int_matrix[np.newaxis, :, :], axis=2)
        p_matrix = diff_count.astype(np.float64) / L
        # Apply JC correction
        # Handle p >= 0.7499
        p_matrix = np.where(p_matrix >= 0.7499, 0.7498, p_matrix)
        dist_matrix = -0.75 * np.log(1.0 - 4.0 * p_matrix / 3.0)
        np.fill_diagonal(dist_matrix, 0.0)
        return dist_matrix
    except ImportError:
        # Fallback to pure Python
        dist = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                diffs = sum(1 for x, y in zip(seqs[i], seqs[j]) if x != y)
                p = diffs / L
                d = _jc_distance(p)
                dist[i][j] = d
                dist[j][i] = d
        return dist


def _nj_newick(dist_matrix, labels):
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    current_dist = [row[:] for row in dist_matrix]
    
    while len(nodes) > 2:
        m = len(nodes)
        # Compute total distances
        total = [0.0] * m
        for i in range(m):
            for j in range(m):
                if i != j:
                    total[i] += current_dist[nodes[i]][nodes[j]]
        
        best_criterion = float('inf')
        best_i = -1
        best_j = -1
        
        for i in range(m):
            for j in range(i + 1, m):
                u = nodes[i]
                v = nodes[j]
                d_uv = current_dist[u][v]
                criterion = (m - 2) * d_uv - total[i] - total[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i = i
                    best_j = j
        
        u = nodes[best_i]
        v = nodes[best_j]
        
        # Create new node
        new_idx = n + len(nodes) - 2 # Just a unique ID, doesn't matter much as we use trees dict
        # Actually, let's just use a string key or a new integer index that we manage.
        # Since we are removing nodes, let's just pick a new index from a pool or use a counter.
        # Simpler: use the index of the new node in the list of nodes.
        # But we need to update the distance matrix.
        # Let's use a new integer ID that is not in the current nodes list.
        # Since max n is 32, and we add at most n-2 nodes, total nodes < 2*n.
        new_id = max(nodes) + 1
        
        # Calculate branch lengths
        # r_u = 0.5 * d_uv
        # r_v = 0.5 * d_uv
        # But wait, the standard NJ formula for the new node w:
        # d(w, x) = 0.5 * (d(u, x) + d(v, x) - d(u, v))
        # The branch length from u to w is 0.5 * d(u, v)
        # The branch length from v to w is 0.5 * d(u, v)
        
        d_uv = current_dist[u][v]
        branch_len = 0.5 * d_uv
        
        # Update trees
        trees[u] = f"({trees[u]},{trees[v]})" # This is wrong, we need to store the internal node structure
        # Actually, trees[i] should be the Newick string for the subtree rooted at i.
        # When we join u and v, the new node w represents the internal node.
        # The subtree for w is (subtree_u, subtree_v).
        # But we also need to keep track of the topology.
        # Let's just build the Newick string at the end or update it.
        # Since we are doing NJ, the topology is determined by the joins.
        # Let's store the join structure.
        # trees[new_id] = f"({trees[u]},{trees[v]})"
        
        # We need to update the distance matrix for the new node w
        # d(w, x) = 0.5 * (d(u, x) + d(v, x) - d(u, v))
        
        # Remove u and v from nodes
        new_nodes = [x for x in nodes if x not in (u, v)]
        new_nodes.append(new_id)
        
        # Update distance matrix
        # We need to add a row and column for new_id
        # And remove rows and columns for u and v
        # This is O(n^2) per step, which is O(n^3) total. Acceptable for n<=32.
        
        # Let's just create a new distance matrix for the new set of nodes
        # But we need to map old indices to new ones.
        # Actually, it's easier to just keep the full matrix and a list of active nodes.
        # But updating the full matrix is messy.
        # Let's just use a dictionary for distances between active nodes.
        # Or, since n is small, we can just copy the relevant parts.
        
        # Let's stick to the list of active nodes and a 2D list of distances for those nodes.
        # Re-indexing is annoying.
        
        # Alternative: Keep the original indices.
        # current_dist is n x n.
        # We just need to update the distances involving new_id.
        # d(new_id, x) for x in new_nodes (excluding new_id)
        
        # Let's just create a new 2D list for the active nodes.
        # active_nodes = new_nodes
        # new_dist = [[0.0] * len(active_nodes) for _ in range(len(active_nodes))]
        # for i, ui in enumerate(active_nodes):
        #     for j, vj in enumerate(active_nodes):
        #         if ui == new_id:
        #             # d(new_id, vj)
        #             if vj == new_id:
        #                 new_dist[i][j] = 0.0
        #             else:
        #                 # vj is in original nodes
        #                 new_dist[i][j] = 0.5 * (current_dist[u][vj] + current_dist[v][vj] - d_uv)
        #         else:
        #             if vj == new_id:
        #                 new_dist[i][j] = new_dist[j][i] # symmetry
        #             else:
        #                 new_dist[i][j] = current_dist[ui][vj]
        
        # This is getting complicated. Let's use the dictionary approach or just accept the O(n^3) with list re-indexing.
        # Given n<=32, even a naive O(n^3) is fine.
        
        # Let's restart the loop logic with a simpler data structure.
        # nodes: list of labels (strings or ints)
        # dist: dict (i, j) -> distance
        # This is what the original code did. It was slow but correct.
        # The numpy part was for the initial matrix calculation.
        # The NJ loop itself is the bottleneck if done in pure Python with dicts.
        # Let's optimize the NJ loop using lists and indices.
        
        break # Break out of this flawed implementation and rewrite properly below.
    
    # The above implementation was getting messy. Let's write a clean one.
    pass


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    # 1. Compute distance matrix using NumPy for speed
    dist_matrix = _compute_distance_matrix(sequences, labels)
    
    # 2. Neighbor Joining
    # nodes: list of indices into dist_matrix
    nodes = list(range(n))
    # trees: dict mapping node index to Newick string
    trees = {i: labels[i] for i in range(n)}
    
    # We will maintain a list of active node indices
    # and a 2D list of distances between them.
    # To avoid re-indexing the whole matrix every time, we can just use the original indices
    # and a set of active nodes. But accessing dist_matrix[i][j] is fast.
    # The problem is updating the distances for the new node.
    # We can just store the distances in a dictionary keyed by (i, j) for active nodes.
    # Or, simpler: since n is small, we can just copy the relevant submatrix every step.
    
    active_nodes = list(range(n))
    current_dist = [row[:] for row in dist_matrix]
    
    while len(active_nodes) > 2:
        m = len(active_nodes)
        
        # Compute total distances for each active node
        total = [0.0] * m
        for i in range(m):
            ui = active_nodes[i]
            for j in range(m):
                if i != j:
                    vj = active_nodes[j]
                    total[i] += current_dist[ui][vj]
        
        best_criterion = float('inf')
        best_i = -1
        best_j = -1
        
        for i in range(m):
            for j in range(i + 1, m):
                ui = active_nodes[i]
                vj = active_nodes[j]
                d_uv = current_dist[ui][vj]
                criterion = (m - 2) * d_uv - total[i] - total[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i = i
                    best_j = j
        
        u = active_nodes[best_i]
        v = active_nodes[best_j]
        d_uv = current_dist[u][v]
        
        # Create new node
        new_id = max(active_nodes) + 1
        
        # Update trees
        # The new node represents the internal node connecting u and v.
        # We don't know the branch lengths yet for the Newick string if we want to include them,
        # but the task says branch lengths are optional.
        # So we just need the topology.
        trees[new_id] = f"({trees[u]},{trees[v]})"
        
        # Update active nodes
        new_active_nodes = [x for x in active_nodes if x not in (u, v)]
        new_active_nodes.append(new_id)
        
        # Update distance matrix
        # We need to add a row/col for new_id.
        # And remove rows/cols for u and v.
        # Since we are using a 2D list indexed by original indices, we can't just "remove" easily.
        # Let's just create a new 2D list for the new set of active nodes.
        # This is O(m^2) per step.
        
        new_m = len(new_active_nodes)
        new_dist = [[0.0] * new_m for _ in range(new_m)]
        
        for i, ui in enumerate(new_active_nodes):
            for j, vj in enumerate(new_active_nodes):
                if ui == new_id:
                    if vj == new_id:
                        new_dist[i][j] = 0.0
                    else:
                        # d(new_id, vj) = 0.5 * (d(u, vj) + d(v, vj) - d(u, v))
                        # vj is in the old active nodes
                        new_dist[i][j] = 0.5 * (current_dist[u][vj] + current_dist[v][vj] - d_uv)
                else:
                    if vj == new_id:
                        new_dist[i][j] = new_dist[j][i] # symmetry
                    else:
                        new_dist[i][j] = current_dist[ui][vj]
        
        active_nodes = new_active_nodes
        current_dist = new_dist
    
    # Final 3 nodes
    # Connect them in a star topology? No, NJ leaves 3 nodes.
    # The last 3 nodes form a cherry? No, they are connected in a triangle in the distance matrix,
    # but topologically they are connected to a central node?
    # Actually, when 3 nodes remain, the topology is determined.
    # We can just connect them arbitrarily as a star or pick the closest pair.
    # Standard NJ stops when 2 nodes remain? No, usually when 3 remain, you can form the tree.
    # Let's join the closest pair of the remaining 3.
    
    if len(active_nodes) == 3:
        # Find closest pair
        u, v, w = active_nodes
        d_uv = current_dist[u][v]
        d_uw = current_dist[u][w]
        d_vw = current_dist[v][w]
        
        min_d = min(d_uv, d_uw, d_vw)
        if min_d == d_uv:
            a, b = u, v
            c = w
        elif min_d == d_uw:
            a, b = u, w
            c = v
        else:
            a, b = v, w
            c = u
        
        new_id = max(active_nodes) + 1
        trees[new_id] = f"({trees[a]},{trees[b]})"
        # Now we have new_id and c.
        # The final tree is (new_id, c)
        # But wait, the topology is ((a,b),c)
        # Let's just return the tree rooted at the last two nodes?
        # No, unrooted tree.
        # The Newick string for the final tree can be constructed by joining the last two nodes.
        # But we need to include c.
        # Actually, when 3 nodes remain, we can just pick any two to join, creating a new node,
        # and then that new node and the third node form the final tree.
        # Let's do that.
        
        # We already joined a and b into new_id.
        # Now we have new_id and c.
        # The final Newick string is (trees[new_id], trees[c])
        # But this is a rooted representation.
        # For unrooted, it's fine as long as the splits are correct.
        # The split {a,b} | {c, rest} is created.
        
        # Wait, if we join a and b, the split is {a,b} vs {c, ...}.
        # Is that correct?
        # In NJ, the last step joins the last two nodes.
        # If we have 3 nodes, we join the closest pair, say a and b.
        # The new node x represents the internal node.
        # Then we have x and c.
        # The final tree is (x, c).
        # This implies the split {a,b} | {c}.
        # But what if the true topology was (a,c) | {b}?
        # The distance matrix should guide us to the correct pair.
        
        # Let's just return the Newick string formed by the last two nodes.
        # But we need to make sure all leaves are included.
        # trees[new_id] contains a and b. trees[c] contains c.
        # So (trees[new_id], trees[c]) contains a, b, c.
        
        final_newick = f"({trees[new_id]},{trees[c]})"
        
    else:
        # Should not happen if n >= 2
        final_newick = "(" + ",".join(trees[x] for x in active_nodes) + ")"
    
    return final_newick + ";"
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
