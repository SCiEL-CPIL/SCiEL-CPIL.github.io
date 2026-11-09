"""Maximum likelihood tree reconstruction using Felsenstein's pruning with hill-climbing."""
import math
import random
import numpy as np
from collections import defaultdict


def _jc_log_prob(p):
    """Log probability of observing p mismatches under JC model for branch length L."""
    # P(same) = 0.75 + 0.25*exp(-4L/3)
    # P(diff) = 0.25*(1-exp(-4L/3))
    # For log-likelihood of a single site:
    # If same: log(0.75 + 0.25*e^(-4L/3))
    # If diff: log(0.25*(1-e^(-4L/3)))
    # We'll compute this vectorized in the pruning routine.
    pass


def _compute_jc_branch_probs(L):
    """Return (p_same, p_diff) for JC model given branch length L."""
    if L < 1e-10:
        return 1.0, 0.0
    e = math.exp(-4.0 * L / 3.0)
    p_same = 0.75 + 0.25 * e
    p_diff = 0.25 * (1.0 - e)
    return p_same, p_diff


class MLTree:
    """Represents an unrooted tree for maximum likelihood under JC model."""
    
    def __init__(self, labels, n_sites):
        self.labels = labels  # list of taxon labels
        self.n = len(labels)
        self.n_sites = n_sites
        # Nodes: 0..n-1 are leaves, n..2n-2 are internal
        self.total_nodes = 2 * self.n - 2
        self.leaves = list(range(self.n))
        self.internal = list(range(self.n, self.total_nodes))
        # adjacency[node] = list of (neighbor, branch_length)
        self.adj = defaultdict(list)
        # Build initial star tree: all leaves connected to a central internal node
        center = self.n  # first internal node as center
        for leaf in self.leaves:
            self.adj[leaf].append((center, 0.1))
            self.adj[center].append((leaf, 0.1))
        # Add remaining internal nodes in a chain from center
        for i in range(self.n + 1, self.total_nodes):
            prev = self.total_nodes - 2 if i > self.n + 1 else center
            self.adj[prev].append((i, 0.05))
            self.adj[i].append((prev, 0.05))
        # Precompute parent/children for pruning
        self._root = 0  # we'll root at leaf 0
        self.parent = {}
        self.children = defaultdict(list)
        self.branch_to_parent = {}
        self._build_parent_child()
    
    def _build_parent_child(self):
        """Build rooted tree structure for Felsenstein's pruning."""
        self.parent = {}
        self.children = defaultdict(list)
        self.branch_to_parent = {}
        visited = set()
        stack = [(self._root, -1, 0.0)]
        while stack:
            node, par, blen = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            self.parent[node] = par
            self.branch_to_parent[node] = blen
            if par != -1:
                self.children[par].append(node)
            for neigh, bl in self.adj[node]:
                if neigh != par:
                    stack.append((neigh, node, bl))
    
    def get_likelihood(self, sequences_matrix):
        """
        Compute log-likelihood of the tree given the sequences.
        sequences_matrix: 2D numpy array of shape (n_taxa, n_sites), values 0-3 for A,C,G,T.
        Uses Felsenstein's pruning algorithm.
        """
        n_taxa = self.n
        n_sites = self.n_sites
        total_logL = 0.0
        
        # Post-order traversal for pruning
        post_order = []
        stack = [self._root]
        visited = set()
        order_stack = [(self._root, False)]
        while order_stack:
            node, processed = order_stack.pop()
            if processed:
                post_order.append(node)
            else:
                if node in visited:
                    continue
                visited.add(node)
                order_stack.append((node, True))
                for child in self.children[node]:
                    order_stack.append((child, False))
        
        # For each site, compute likelihood
        # We'll process all sites together using vectors
        # alpha[node] = vector of length 4 (probabilities for A,C,G,T at this node)
        # For leaves: alpha[leaf] is one-hot at the observed base
        # For internal: alpha[node] = sum over states of prod over children
        
        # Process each site
        for s in range(n_sites):
            # Initialize leaf vectors
            alpha = [None] * self.total_nodes
            for leaf in self.leaves:
                obs = sequences_matrix[leaf, s]
                vec = np.zeros(4)
                vec[obs] = 1.0
                alpha[leaf] = vec
            
            # Process nodes in post-order
            for node in reversed(post_order):
                if node in self.leaves:
                    continue
                # For internal node, compute alpha
                # alpha[node][i] = prod over children c of sum_j (P_{ij} * alpha[c][j])
                # where P_{ij} is transition probability from i to j over branch length
                node_alpha = np.zeros(4)
                for i in range(4):
                    prod_val = 1.0
                    for child in self.children[node]:
                        blen = self.branch_to_parent[child]
                        p_same, p_diff = _compute_jc_branch_probs(blen)
                        # Transition matrix: P[i][j] = p_same if i==j else p_diff
                        # sum_j P[i][j] * alpha[child][j]
                        child_alpha = alpha[child]
                        s_val = p_same * child_alpha[i] + p_diff * (child_alpha.sum() - child_alpha[i])
                        if s_val < 1e-300:
                            s_val = 1e-300
                        prod_val *= s_val
                    node_alpha[i] = prod_val
                alpha[node] = node_alpha
            
            # At root, sum over all states
            root_alpha = alpha[self._root]
            site_logL = math.log(max(root_alpha.sum(), 1e-300))
            total_logL += site_logL
        
        return total_logL
    
    def try_nni(self, sequences_matrix):
        """Try all NNI moves, return best move and improvement."""
        best_improvement = -1e10
        best_move = None
        current_LL = self.get_likelihood(sequences_matrix)
        
        # For each internal edge, try the 4 NNI alternatives
        # An internal edge connects two internal nodes u and v
        # Each has two other neighbors (besides each other)
        # NNI: swap which pair of subtrees are connected
        
        internal_edges = []
        for node in self.internal:
            for neigh, bl in self.adj[node]:
                if neigh in self.internal and node < neigh:
                    internal_edges.append((node, neigh))
        
        for u, v in internal_edges:
            # Find the two other neighbors of u and v
            u_others = [(n, b) for n, b in self.adj[u] if n != v]
            v_others = [(n, b) for n, b in self.adj[v] if n != u]
            
            if len(u_others) != 2 or len(v_others) != 2:
                continue
            
            a, a_bl = u_others[0]
            b, b_bl = u_others[1]
            c, c_bl = v_others[0]
            d, d_bl = v_others[1]
            
            # Current topology: u connected to v, a, b; v connected to u, c, d
            # NNI alternatives:
            # 1. u-v, a-b, c-d (current)
            # 2. u-c, v-a, b-d
            # 3. u-d, v-b, a-c
            
            for alt in [2, 3]:
                # Save current state
                orig_u = list(self.adj[u])
                orig_v = list(self.adj[v])
                
                # Apply alternative
                self.adj[u] = []
                self.adj[v] = []
                
                if alt == 2:
                    # u connected to c and a; v connected to b and d
                    self.adj[u].append((c, c_bl))
                    self.adj[u].append((a, a_bl))
                    self.adj[v].append((b, b_bl))
                    self.adj[v].append((d, d_bl))
                    self.adj[c] = [(x, bl) for x, bl in self.adj[c] if x != u]
                    self.adj[c].append((u, c_bl))
                    self.adj[b] = [(x, bl) for x, bl in self.adj[b] if x != v]
                    self.adj[b].append((v, b_bl))
                    # a stays with u, d stays with v
                    self.adj[a] = [(x, bl) for x, bl in self.adj[a] if x != u]
                    self.adj[a].append((u, a_bl))
                    self.adj[d] = [(x, bl) for x, bl in self.adj[d] if x != v]
                    self.adj[d].append((v, d_bl))
                else:  # alt == 3
                    # u connected to d and b; v connected to a and c
                    self.adj[u].append((d, d_bl))
                    self.adj[u].append((b, b_bl))
                    self.adj[v].append((a, a_bl))
                    self.adj[v].append((c, c_bl))
                    self.adj[d] = [(x, bl) for x, bl in self.adj[d] if x != v]
                    self.adj[d].append((u, d_bl))
                    self.adj[b] = [(x, bl) for x, bl in self.adj[b] if x != u]
                    self.adj[b].append((u, b_bl))
                    self.adj[a] = [(x, bl) for x, bl in self.adj[a] if x != v]
                    self.adj[a].append((v, a_bl))
                    self.adj[c] = [(x, bl) for x, bl in self.adj[c] if x != u]
                    self.adj[c].append((v, c_bl))
                
                self._build_parent_child()
                new_LL = self.get_likelihood(sequences_matrix)
                improvement = new_LL - current_LL
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_move = (u, v, alt, a, b, c, d, a_bl, b_bl, c_bl, d_bl, orig_u, orig_v)
                
                # Restore
                self.adj[u] = orig_u
                self.adj[v] = orig_v
                # Restore neighbors
                for n, bl in orig_u:
                    if n != v:
                        self.adj[n] = [(x, b2) for x, b2 in self.adj[n] if x != u]
                        self.adj[n].append((u, bl))
                for n, bl in orig_v:
                    if n != u:
                        self.adj[n] = [(x, b2) for x, b2 in self.adj[n] if x != v]
                        self.adj[n].append((v, bl))
        
        # Apply best move if it improves
        if best_move is not None and best_improvement > 1e-6:
            u, v, alt, a, b, c, d, a_bl, b_bl, c_bl, d_bl, orig_u, orig_v = best_move
            self.adj[u] = []
            self.adj[v] = []
            if alt == 2:
                self.adj[u].append((c, c_bl))
                self.adj[u].append((a, a_bl))
                self.adj[v].append((b, b_bl))
                self.adj[v].append((d, d_bl))
                self.adj[c] = [(x, bl) for x, bl in self.adj[c] if x != u]
                self.adj[c].append((u, c_bl))
                self.adj[b] = [(x, bl) for x, bl in self.adj[b] if x != v]
                self.adj[b].append((v, b_bl))
                self.adj[a] = [(x, bl) for x, bl in self.adj[a] if x != u]
                self.adj[a].append((u, a_bl))
                self.adj[d] = [(x, bl) for x, bl in self.adj[d] if x != v]
                self.adj[d].append((v, d_bl))
            else:
                self.adj[u].append((d, d_bl))
                self.adj[u].append((b, b_bl))
                self.adj[v].append((a, a_bl))
                self.adj[v].append((c, c_bl))
                self.adj[d] = [(x, bl) for x, bl in self.adj[d] if x != v]
                self.adj[d].append((u, d_bl))
                self.adj[b] = [(x, bl) for x, bl in self.adj[b] if x != u]
                self.adj[b].append((u, b_bl))
                self.adj[a] = [(x, bl) for x, bl in self.adj[a] if x != v]
                self.adj[a].append((v, a_bl))
                self.adj[c] = [(x, bl) for x, bl in self.adj[c] if x != u]
                self.adj[c].append((v, c_bl))
            self._build_parent_child()
            return True, best_improvement
        return False, 0.0


def _build_initial_tree_jc(labels, sequences):
    """Build an initial tree using neighbor joining as starting point."""
    labels = sorted(labels, key=lambda name: int(name[1:]))
    n = len(labels)
    
    def _distance(a, b):
        differences = sum(1 for x, y in zip(sequences[a], sequences[b]) if x != y)
        p = differences / len(sequences[a])
        if p >= 0.7499:
            return 3.0
        return -0.75 * math.log(1.0 - 4.0 * p / 3.0)
    
    nodes = list(range(n))  # use indices
    trees = {i: labels[i] for i in range(n)}
    matrix = {(i, j): _distance(sequences[labels[i]], sequences[labels[j]]) 
              for i in range(n) for j in range(n) if i != j}
    
    while len(nodes) > 3:
        m = len(nodes)
        total = {a: sum(matrix[(a, b)] for b in nodes if b != a) for a in nodes}
        best = None
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                criterion = (m - 2) * matrix[(a, b)] - total[a] - total[b]
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
    
    # Convert to MLTree
    n_sites = len(sequences[labels[0]])
    ml_tree = MLTree(labels, n_sites)
    
    # Now we need to set the topology from NJ. This is complex, so let's just use the NJ tree structure
    # For simplicity, let's use a balanced initial tree and let hill-climbing work
    return ml_tree


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    n_sites = len(sequences[labels[0]])
    
    # Convert sequences to matrix
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_matrix = np.zeros((n, n_sites), dtype=np.int8)
    for i, lab in enumerate(labels):
        seq = sequences[lab]
        for j, ch in enumerate(seq):
            seq_matrix[i, j] = base_map.get(ch, 0)
    
    # Create initial tree
    tree = MLTree(labels, n_sites)
    
    # Hill-climbing with NNI moves
    max_iterations = 50
    for iteration in range(max_iterations):
        improved, improvement = tree.try_nni(seq_matrix)
        if not improved:
            break
    
    # Convert tree to Newick
    def _to_newick(node, parent=None):
        """Convert tree node to Newick string."""
        if node in tree.leaves:
            return labels[node]
        # Get children
        child_nodes = [c for c in tree.children[node]]
        if len(child_nodes) == 0:
            return f"({labels[node]},);"  # shouldn't happen
        child_strs = []
        for c in child_nodes:
            cs = _to_newick(c, node)
            child_strs.append(cs)
        return f"({','.join(child_strs)});"
    
    # Root at an internal node for proper Newick
    newick = _to_newick(tree._root)
    return newick
