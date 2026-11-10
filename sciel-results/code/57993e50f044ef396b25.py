import numpy as np
import math
import time
import random
from collections import deque

def _jc_dist(a, b):
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    p = diffs / len(a)
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _nj(dist, labels):
    nodes = list(labels)
    trees = {name: name for name in labels}
    while len(nodes) > 3:
        n = len(nodes)
        r = {a: sum(dist[(a, b)] for b in nodes if b != a) for a in nodes}
        best_crit = float('inf')
        best_pair = None
        for i in range(n):
            for j in range(i+1, n):
                a, b = nodes[i], nodes[j]
                crit = (n - 2) * dist[(a, b)] - r[a] - r[b]
                if crit < best_crit:
                    best_crit = crit
                    best_pair = (a, b)
        a, b = best_pair
        joined = f"({trees[a]},{trees[b]})"
        new = f"n{len(nodes)}_{a}_{b}"
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            dist[(new, x)] = dist[(x, new)] = 0.5 * (dist[(a, x)] + dist[(b, x)] - dist[(a, b)])
        nodes = rest + [new]
        trees[new] = joined
    return "(" + ",".join(trees[x] for x in nodes) + ");"

def _parse_newick(newick):
    """Parse Newick to get adjacency list and branch lengths."""
    neighbors = {}
    branch_lengths = {}
    leaves = set()
    pos = 0
    
    def parse():
        nonlocal pos
        if newick[pos] == '(':
            pos += 1
            children = []
            while True:
                child = parse()
                children.append(child)
                if newick[pos] == ',':
                    pos += 1
                elif newick[pos] == ')':
                    pos += 1
                    break
            # Skip branch length
            bl = 0.0
            if pos < len(newick) and newick[pos] == ':':
                pos += 1
                bl_str = ''
                while pos < len(newick) and newick[pos] not in '(),;':
                    bl_str += newick[pos]
                    pos += 1
                try:
                    bl = float(bl_str)
                except:
                    bl = 0.0
            node = f"_int_{len(neighbors)}"
            neighbors[node] = set(children)
            for c in children:
                neighbors.setdefault(c, set()).add(node)
                branch_lengths[(node, c)] = bl
                branch_lengths[(c, node)] = bl
            return node
        else:
            name = ''
            while pos < len(newick) and newick[pos] not in '(),;':
                name += newick[pos]
                pos += 1
            # Skip branch length
            bl = 0.0
            if pos < len(newick) and newick[pos] == ':':
                pos += 1
                bl_str = ''
                while pos < len(newick) and newick[pos] not in '(),;':
                    bl_str += newick[pos]
                    pos += 1
                try:
                    bl = float(bl_str)
                except:
                    bl = 0.0
            neighbors.setdefault(name, set())
            leaves.add(name)
            return name
    
    newick = newick.rstrip(';')
    root = parse()
    return neighbors, branch_lengths, leaves, root

def _to_newick(neighbors, branch_lengths, leaves):
    """Convert adjacency list to Newick string."""
    internal_nodes = [n for n in neighbors if n not in leaves]
    if not internal_nodes:
        return ""
    root = internal_nodes[0]
    
    parent = {}
    children = {}
    visited = set()
    
    def build(node, par):
        parent[node] = par
        children[node] = []
        for nb in neighbors[node]:
            if nb != par and nb not in visited:
                visited.add(nb)
                children[node].append(nb)
                build(nb, node)
    
    visited.add(root)
    build(root, None)
    
    def to_string(node):
        if node in leaves:
            return node
        if len(children[node]) == 0:
            return f"({node})"
        child_strs = [to_string(c) for c in children[node]]
        return f"({','.join(child_strs)})"
    
    return to_string(root) + ";"

def _prepare_data(sequences, leaves):
    """Prepare data for vectorized likelihood calculation."""
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    L = len(next(iter(sequences.values())))
    n_leaves = len(leaves)
    data = np.zeros((L, n_leaves), dtype=np.int8)
    for i, label in enumerate(leaves):
        seq = sequences[label]
        for j, base in enumerate(seq):
            data[j, i] = base_map[base]
    return data, L, n_leaves

def _ml_likelihood_fast(neighbors, branch_lengths, leaves, data, L):
    """
    Compute ML log-likelihood for the tree using vectorized Felsenstein pruning.
    neighbors: dict of node -> set of neighbors
    branch_lengths: dict of (u, v) -> length
    leaves: set of leaf names
    data: numpy array (L, n_leaves) of base indices
    L: number of sites
    """
    # Root the tree arbitrarily at an internal node
    internal_nodes = [n for n in neighbors if n not in leaves]
    if not internal_nodes:
        return 0.0
    root = internal_nodes[0]
    
    # Build parent/children structure
    parent = {}
    children = {}
    visited = set()
    
    def build(node, par):
        parent[node] = par
        children[node] = []
        for nb in neighbors[node]:
            if nb != par and nb not in visited:
                visited.add(nb)
                children[node].append(nb)
                build(nb, node)
    
    visited.add(root)
    build(root, None)
    
    # Map leaf names to indices in data array
    leaf_list = list(leaves)
    leaf_to_idx = {leaf: i for i, leaf in enumerate(leaf_list)}
    
    # Post-order traversal
    post_order = []
    def get_post_order(node):
        for child in children[node]:
            get_post_order(child)
        post_order.append(node)
    
    get_post_order(root)
    
    # vectors[node] will be a numpy array of shape (L, 4)
    vectors = {}
    
    for node in post_order:
        if node in leaves:
            # One-hot encoding for leaves
            idx = leaf_to_idx[node]
            v = np.zeros((L, 4), dtype=np.float64)
            # data[:, idx] is the column of base indices for this leaf
            v[np.arange(L), data[:, idx]] = 1.0
            vectors[node] = v
        else:
            v = np.zeros((L, 4), dtype=np.float64)
            for child in children[node]:
                bl = branch_lengths.get((node, child), 0.0)
                if bl < 0: bl = 0.0
                # JC Transition matrix P
                # P = 1/4 * [[1+3e, 1-e, 1-e, 1-e], ...]
                e = math.exp(-4.0 * bl / 3.0)
                v_child = vectors[child]
                
                # Vectorized multiplication: v_new = v_child @ P
                # v_new[:, k] = sum_j v_child[:, j] * P[j, k]
                # P is symmetric.
                # P[j, j] = (1+3e)/4
                # P[j, k] (j!=k) = (1-e)/4
                # v_new[:, k] = (1+3e)/4 * v_child[:, k] + (1-e)/4 * sum_{j!=k} v_child[:, j]
                # sum_{j!=k} v_child[:, j] = sum(v_child) - v_child[:, k]
                
                sum_v = np.sum(v_child, axis=1)
                # v += (1/4) * ( (1+3e)*v_child + (1-e)*(sum_v - v_child) )
                # v += (1/4) * ( (1+3e - 1 + e)*v_child + (1-e)*sum_v )
                # v += (1/4) * ( (4e)*v_child + (1-e)*sum_v )
                # v += e * v_child + (1-e)/4 * sum_v
                
                # Optimization:
                # term1 = e * v_child
                # term2 = (1.0 - e) / 4.0 * sum_v[:, np.newaxis]
                # v += term1 + term2
                
                v += e * v_child + ((1.0 - e) / 4.0) * sum_v[:, np.newaxis]
            vectors[node] = v
    
    # Likelihood at root
    # Assuming uniform base frequencies (1/4 each)
    root_vec = vectors[root]
    # log( sum_k root_vec[:, k] / 4 )
    # sum_k root_vec[:, k] is the sum over states at the root
    ll = np.sum(np.log(np.sum(root_vec, axis=1) / 4.0))
    return ll

def _optimize_branch_lengths_fast(neighbors, branch_lengths, leaves, data, L, max_time=10.0):
    """Optimize branch lengths using a coarse grid search."""
    start_time = time.time()
    edges = []
    seen = set()
    for u in neighbors:
        for v in neighbors[u]:
            if (u, v) not in seen:
                edges.append((u, v))
                seen.add((u, v))
                seen.add((v, u))
    
    # Grid search for each edge
    grid = np.linspace(0.0, 1.5, 15)
    for u, v in edges:
        if time.time() - start_time > max_time:
            break
        best_ll = -float('inf')
        best_bl = 0.0
        for bl in grid:
            branch_lengths[(u, v)] = bl
            branch_lengths[(v, u)] = bl
            ll = _ml_likelihood_fast(neighbors, branch_lengths, leaves, data, L)
            if ll > best_ll:
                best_ll = ll
                best_bl = bl
        branch_lengths[(u, v)] = best_bl
        branch_lengths[(v, u)] = best_bl
    return branch_lengths

def _get_internal_edges(neighbors, leaves):
    """Get list of internal edges (u, v) where both are internal nodes."""
    edges = []
    seen = set()
    for u in neighbors:
        if u in leaves: continue
        for v in neighbors[u]:
            if v in leaves: continue
            if (u, v) not in seen and (v, u) not in seen:
                edges.append((u, v))
                seen.add((u, v))
                seen.add((v, u))
    return edges

def _nni_search(neighbors, branch_lengths, leaves, data, L, max_time=30.0):
    """
    NNI search with simulated annealing-like acceptance to escape local optima.
    """
    start_time = time.time()
    current_ll = _ml_likelihood_fast(neighbors, branch_lengths, leaves, data, L)
    best_neighbors = {k: set(v) for k, v in neighbors.items()}
    best_bl = dict(branch_lengths)
    best_ll = current_ll
    
    # Temperature for simulated annealing
    T = 10.0
    min_T = 0.1
    
    while time.time() - start_time < max_time:
        internal_edges = _get_internal_edges(neighbors, leaves)
        if not internal_edges:
            break
        
        # Pick a random internal edge
        u, v = random.choice(internal_edges)
        
        u_nbs = list(neighbors[u])
        v_nbs = list(neighbors[v])
        
        # Find the other neighbors
        a, b = [x for x in u_nbs if x != v]
        c, d = [x for x in v_nbs if x != u]
        
        # NNI Move: Swap b and c
        # Original: (a, (b, (c, d))) -- wait, NNI is on an internal edge.
        # Let's define the move properly.
        # Edge (u, v).
        # u is connected to a, b, v.
        # v is connected to c, d, u.
        # Move: remove (u,b) and (v,c), add (u,c) and (v,b).
        
        new_neighbors = {k: set(v) for k, v in neighbors.items()}
        new_bl = dict(branch_lengths)
        
        # Update neighbors
        new_neighbors[u].remove(b)
        new_neighbors[u].add(c)
        new_neighbors[b].remove(u)
        new_neighbors[b].add(v)
        new_neighbors[v].remove(c)
        new_neighbors[v].add(b)
        new_neighbors[c].remove(v)
        new_neighbors[c].add(u)
        
        # Update branch lengths for the changed edges
        # Edges changed: (u,b)->(u,c), (v,c)->(v,b)
        # We keep the length of (u,v) and optimize the others.
        # For speed, let's just keep the old lengths for now and optimize if accepted.
        # Actually, let's optimize the 3 edges involved: (u,c), (v,b), (u,v)
        
        # Optimize branch lengths for the new topology
        edges_to_opt = [(u, c), (v, b), (u, v)]
        grid = np.linspace(0.0, 1.5, 11)
        
        for e_u, e_v in edges_to_opt:
            best_ll_e = -float('inf')
            best_bl_e = 0.0
            for bl in grid:
                new_bl[(e_u, e_v)] = bl
                new_bl[(e_v, e_u)] = bl
                ll = _ml_likelihood_fast(new_neighbors, new_bl, leaves, data, L)
                if ll > best_ll_e:
                    best_ll_e = ll
                    best_bl_e = bl
            new_bl[(e_u, e_v)] = best_bl_e
            new_bl[(e_v, e_u)] = best_bl_e
        
        new_ll = _ml_likelihood_fast(new_neighbors, new_bl, leaves, data, L)
        
        # Acceptance criterion
        delta = new_ll - current_ll
        if delta > 0 or random.random() < math.exp(delta / T):
            neighbors = new_neighbors
            branch_lengths = new_bl
            current_ll = new_ll
            
            if current_ll > best_ll:
                best_ll = current_ll
                best_neighbors = {k: set(v) for k, v in neighbors.items()}
                best_bl = dict(branch_lengths)
        
        # Cool down
        T *= 0.99
        
    return best_neighbors, best_bl

def reconstruct(sequences):
    start_time = time.time()
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    
    # Compute distance matrix
    dist = {}
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            d = _jc_dist(sequences[labels[i]], sequences[labels[j]])
            dist[(labels[i], labels[j])] = d
            dist[(labels[j], labels[i])] = d
    
    # NJ
    newick = _nj(dist, labels)
    
    # Parse
    neighbors, branch_lengths, leaves, root = _parse_newick(newick)
    
    # Prepare data for fast likelihood
    data, L, n_leaves = _prepare_data(sequences, leaves)
    
    elapsed = time.time() - start_time
    if elapsed > 200:
        return newick
    
    # Optimize branch lengths
    try:
        branch_lengths = _optimize_branch_lengths_fast(neighbors, branch_lengths, leaves, data, L, max_time=20.0)
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > 240:
        return _to_newick(neighbors, branch_lengths, leaves)
    
    # NNI refinement with stochastic search
    try:
        neighbors, branch_lengths = _nni_search(neighbors, branch_lengths, leaves, data, L, max_time=240 - elapsed - 5.0)
    except Exception:
        pass
    
    return _to_newick(neighbors, branch_lengths, leaves)
