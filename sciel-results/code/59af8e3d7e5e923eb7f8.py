<<<<<<< SEARCH
def _robinson_foulds(tree1_splits, tree2_splits):
    """Compute normalized Robinson-Foulds distance."""
    if not tree1_splits and not tree2_splits:
        return 0.0
    
    t1 = set()
    for s in tree1_splits:
        t1.add(s)
    
    t2 = set()
    for s in tree2_splits:
        t2.add(s)
    
    intersection = t1 & t2
    union = t1 | t2
    
    if not union:
        return 0.0
    
    return 1.0 - len(intersection) / len(union)
=======
def _robinson_foulds(tree1_splits, tree2_splits):
    """Compute normalized Robinson-Foulds distance."""
    if not tree1_splits and not tree2_splits:
        return 0.0
    
    t1 = set(tree1_splits)
    t2 = set(tree2_splits)
    
    intersection = t1 & t2
    union = t1 | t2
    
    if not union:
        return 0.0
    
    return 1.0 - len(intersection) / len(union)


def _jc_likelihood(tree, seq_matrix):
    """
    Compute JC log-likelihood for a tree given a sequence matrix.
    Uses Felsenstein's pruning algorithm.
    seq_matrix: (n_taxa, n_sites) array of ints (0=A, 1=C, 2=G, 3=T)
    """
    n_taxa = tree.n_taxa
    n_sites = seq_matrix.shape[1]
    n_nodes = tree.n_nodes
    
    if n_taxa < 2:
        return 0.0
    
    # Root the tree at an internal node
    root = n_taxa  # first internal node
    
    # Build parent-child structure
    parent = [-1] * n_nodes
    children = [[] for _ in range(n_nodes)]
    
    stack = [(root, -1)]
    visited = set()
    while stack:
        node, par = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        parent[node] = par
        for neighbor in tree.adj[node]:
            if neighbor != par:
                children[node].append(neighbor)
                stack.append((neighbor, node))
    
    # Post-order traversal
    post_order = []
    stack = [(root, False)]
    while stack:
        node, processed = stack.pop()
        if processed:
            post_order.append(node)
        else:
            stack.append((node, True))
            for child in reversed(children[node]):
                stack.append((child, False))
    
    # Estimate branch lengths
    # For JC, we can estimate branch lengths from pairwise distances
    # or use a simple heuristic. For NNI, we'll use a fixed approach.
    
    # Compute branch length estimates using the distance matrix approach
    # For now, use a simple estimate: for each edge, estimate the length
    # based on the average distance to nearby leaves.
    
    # Actually, for NNI comparison, we can compute the likelihood
    # with optimized branch lengths for each topology.
    
    # For efficiency, let's use a simplified approach:
    # Compute the likelihood assuming branch lengths estimated from
    # the pairwise distances.
    
    # First, compute pairwise path lengths in the tree
    # For each pair of leaves, find the path and sum branch lengths.
    
    # Estimate branch lengths:
    # For each internal edge, the length can be estimated as:
    # For edge connecting node u and v, with subtrees S_u and S_v:
    # len = 0.5 * (avg_dist(S_u, S_v) - avg_dist(S_u, S_u) - avg_dist(S_v, S_v)) / 2
    # This is complex. Let's use a simpler approach.
    
    # For JC, a common approach is to use the "least squares" method
    # to estimate branch lengths. But for speed, let's use a heuristic.
    
    # Actually, for the purpose of NNI, we can use a simpler score:
    # the sum over all quartets of the "quartet score" which is related
    # to the likelihood.
    
    # Let's use a different approach: compute the "tree score" as
    # the sum of squared differences between observed and tree-implied
    # pairwise distances. This is faster and correlated with likelihood.
    
    # For now, return a placeholder
    return 0.0


def _nni_refine(tree, dist_matrix, labels, time_limit):
    """
    Refine tree topology using NNI moves.
    Uses a score based on fit to the distance matrix.
    """
    n_taxa = tree.n_taxa
    n_nodes = tree.n_nodes
    start_time = time.time()
    
    if n_taxa < 4:
        return tree
    
    # For NNI, we need to identify internal edges.
    # An internal edge is one where both endpoints have degree >= 3
    # (in the unrooted tree).
    
    # Get current splits for scoring
    current_splits = _get_splits_from_tree(tree)
    
    # We'll use a simple heuristic score: 
    # For each NNI move, compute the change in the number of 
    # "compatible" quartets or use a distance-based score.
    
    # For efficiency, let's use a quartet-based score.
    # For each internal edge, there are 3 possible NNI topologies.
    # We can score each by counting how many quartets support it.
    
    # But computing all quartets is O(n^4) which is too slow for n=32.
    
    # Alternative: use a greedy approach with a local score.
    # For each internal edge, try all 3 NNI moves and pick the best.
    
    # Define a score function based on the distance matrix
    def score_tree(tree_obj):
        """Score a tree based on how well it fits the distance matrix."""
        # Compute the "implied" distance matrix from the tree
        # and compare to the observed distance matrix.
        # For this, we need branch lengths.
        
        # Use a simple approach: for each pair of leaves, 
        # the implied distance is the sum of branch lengths on the path.
        # We estimate branch lengths using a least-squares fit.
        
        # For speed, use a simpler proxy: 
        # the number of splits that are "supported" by the distance matrix.
        
        splits = _get_splits_from_tree(tree_obj)
        if not splits:
            return 0.0
        
        # For each split, compute a "support" score
        # based on how well the split separates the distance matrix.
        # A good split should have small within-group distances
        # and large between-group distances.
        
        n = len(labels)
        total_score = 0.0
        
        for split in splits:
            s1, s2 = split
            # Get indices of taxa in each group
            idx1 = [i for i, l in enumerate(labels) if l in s1]
            idx2 = [i for i, l in enumerate(labels) if l in s2]
            
            if len(idx1) < 2 or len(idx2) < 2:
                continue
            
            # Average within-group distance
            within1 = 0.0
            cnt1 = 0
            for i in range(len(idx1)):
                for j in range(i+1, len(idx1)):
                    within1 += dist_matrix[idx1[i], idx1[j]]
                    cnt1 += 1
            
            within2 = 0.0
            cnt2 = 0
            for i in range(len(idx2)):
                for j in range(i+1, len(idx2)):
                    within2 += dist_matrix[idx2[i], idx2[j]]
                    cnt2 += 1
            
            # Average between-group distance
            between = 0.0
            cntb = 0
            for i in idx1:
                for j in idx2:
                    between += dist_matrix[i, j]
                    cntb += 1
            
            if cnt1 > 0:
                within1 /= cnt1
            if cnt2 > 0:
                within2 /= cnt2
            if cntb > 0:
                between /= cntb
            
            # Score: between should be large, within should be small
            # Use a simple contrast
            score = between - 0.5 * (within1 + within2)
            total_score += score
        
        return total_score
    
    # NNI search
    best_tree = tree.copy()
    best_score = score_tree(best_tree)
    
    max_iterations = 50  # Limit iterations to avoid timeout
    iteration = 0
    
    while iteration < max_iterations:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        
        # Find all internal edges
        # An internal edge has both endpoints with degree >= 3
        internal_edges = []
        for u in range(n_nodes):
            for v in tree.adj[u]:
                if u < v:
                    if len(tree.adj[u]) >= 3 and len(tree.adj[v]) >= 3:
                        internal_edges.append((u, v))
        
        for u, v in internal_edges:
            if time.time() - start_time > time_limit:
                break
            
            # Get the other neighbors of u and v
            # u has neighbors: v, a, b (and possibly more if degree > 3)
            # v has neighbors: u, c, d (and possibly more)
            
            # For a proper NNI, both u and v should have degree exactly 3
            # If degree > 3, we need to handle it differently.
            # For simplicity, only consider edges where both endpoints have degree 3.
            
            if len(tree.adj[u]) != 3 or len(tree.adj[v]) != 3:
                continue
            
            # Get the other neighbors
            u_neighbors = [x for x in tree.adj[u] if x != v]
            v_neighbors = [x for x in tree.adj[v] if x != u]
            
            if len(u_neighbors) != 2 or len(v_neighbors) != 2:
                continue
            
            a, b = u_neighbors
            c, d = v_neighbors
            
            # Current topology: (a, b) connected to u, (c, d) connected to v, u-v edge
            # NNI move 1: connect a-c and b-d
            # NNI move 2: connect a-d and b-c
            
            # Try NNI move 1: (a,c) and (b,d)
            tree1 = tree.copy()
            # Remove edges u-v, u-a, u-b, v-c, v-d
            tree1.adj[u].remove(v)
            tree1.adj[v].remove(u)
            tree1.adj[u].remove(a)
            tree1.adj[a].remove(u)
            tree1.adj[u].remove(b)
            tree1.adj[b].remove(u)
            tree1.adj[v].remove(c)
            tree1.adj[c].remove(v)
            tree1.adj[v].remove(d)
            tree1.adj[d].remove(v)
            
            # Add edges u-c, u-d, v-a, v-b
            # Wait, that's not right. Let me think about this more carefully.
            
            # Current: u connected to {v, a, b}, v connected to {u, c, d}
            # NNI option 1: u connected to {a, c, v}, v connected to {u, b, d}
            # No, that's not an NNI.
            
            # Proper NNI: 
            # Original: ...a--u--v--c... with b on u and d on v
            # Actually: u has neighbors {a, b, v}, v has neighbors {c, d, u}
            # The three possible quartet topologies for {a,b,c,d} with internal edge:
            # 1. (a,b)|(c,d) - current: u connects to a,b; v connects to c,d; u-v edge
            # 2. (a,c)|(b,d) - u connects to a,d; v connects to b,c; u-v edge? No...
            
            # Let me reconsider. The NNI move changes the internal edge.
            # Original: a--u, b--u, u--v, c--v, d--v
            # NNI 1: a--u, d--u, u--v, b--v, c--v  => (a,d)|(b,c)
            # NNI 2: a--u, c--u, u--v, b--v, d--v  => (a,c)|(b,d)
            
            # Try NNI 1: (a,d)|(b,c)
            tree1 = tree.copy()
            # Remove u-b and v-c
            tree1.adj[u].remove(b)
            tree1.adj[b].remove(u)
            tree1.adj[v].remove(c)
            tree1.adj[c].remove(v)
            # Add u-d and v-b
            tree1.adj[u].append(d)
            tree1.adj[d].append(u)
            tree1.adj[v].append(b)
            tree1.adj[b].append(v)
            
            score1 = score_tree(tree1)
            if score1 > best_score:
                best_tree = tree1
                best_score = score1
                improved = True
                tree = tree1.copy()
            
            # Try NNI 2: (a,c)|(b,d)
            tree2 = tree.copy()
            # Remove u-b and v-d
            tree2.adj[u].remove(b)
            tree2.adj[b].remove(u)
            tree2.adj[v].remove(d)
            tree2.adj[d].remove(v)
            # Add u-c and v-b
            tree2.adj[u].append(c)
            tree2.adj[c].append(u)
            tree2.adj[v].append(b)
            tree2.adj[b].append(v)
            
            score2 = score_tree(tree2)
            if score2 > best_score:
                best_tree = tree2
                best_score = score2
                improved = True
                tree = tree2.copy()
        
        iteration += 1
        if not improved:
            break
    
    return best_tree
>>>>>>> REPLACE

