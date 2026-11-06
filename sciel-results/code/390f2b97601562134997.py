"""Phylogenetic topology recovery via NJ + Fitch parsimony NNI refinement."""
# EVOLVE-BLOCK-START
import time
import math

_DEADLINE = 250.0
_start_time = None


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _jc_distance(a, b):
    n = len(a)
    diff = sum(1 for x, y in zip(a, b) if x != y)
    p = diff / n
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _nj_build(sequences, labels):
    """Neighbour-Joining. Returns Newick string."""
    n = len(labels)
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    
    # Compute distance matrix
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            D[i][j] = d
            D[j][i] = d
    
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    next_id = n
    
    while len(nodes) > 2:
        m = len(nodes)
        r = [0.0] * m
        for i in range(m):
            for j in range(m):
                if i != j:
                    r[i] += D[nodes[i]][nodes[j]]
        
        best_criterion = float('inf')
        best_i, best_j = -1, -1
        for i in range(m):
            for j in range(i + 1, m):
                a, b = nodes[i], nodes[j]
                criterion = (m - 2) * D[a][b] - r[i] - r[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i, best_j = i, j
        
        a, b = nodes[best_i], nodes[best_j]
        new_node = next_id
        next_id += 1
        trees[new_node] = f"({trees[a]},{trees[b]})"
        
        for i in range(m):
            x = nodes[i]
            if x == a or x == b:
                continue
            new_d = 0.5 * (D[a][x] + D[b][x] - D[a][b])
            D[new_node][x] = new_d
            D[x][new_node] = new_d
        D[new_node][new_node] = 0.0
        
        nodes = [x for x in nodes if x != a and x != b]
        nodes.append(new_node)
    
    a, b = nodes[0], nodes[1]
    return f"({trees[a]},{trees[b]})"


def _fitch_score(tree, seq_int):
    """Compute Fitch parsimony score. tree is nested tuple, seq_int maps label to list of ints."""
    n_sites = len(next(iter(seq_int.values())))
    score = 0
    
    def fitch(node, pos):
        nonlocal score
        if isinstance(node, str):
            return 1 << seq_int[node][pos]
        else:
            L = fitch(node[0], pos)
            R = fitch(node[1], pos)
            inter = L & R
            if inter:
                return inter
            else:
                score += 1
                return L | R
    
    for pos in range(n_sites):
        fitch(tree, pos)
    return score


def _nni_refine(tree, seq_int, max_rounds=20):
    """NNI refinement using Fitch parsimony."""
    # Convert to mutable node list
    nodes = []
    
    def build(node):
        if isinstance(node, str):
            idx = len(nodes)
            nodes.append(('leaf', node))
            return idx
        else:
            l = build(node[0])
            r = build(node[1])
            idx = len(nodes)
            nodes.append(('internal', l, r))
            return idx
    
    root = build(tree)
    
    def to_tuple(idx):
        node = nodes[idx]
        if node[0] == 'leaf':
            return node[1]
        else:
            return (to_tuple(node[1]), to_tuple(node[2]))
    
    def score_tree():
        return _fitch_score(to_tuple(root), seq_int)
    
    current = score_tree()
    
    for _ in range(max_rounds):
        if _elapsed() > _DEADLINE - 15:
            break
        improved = False
        for i in range(len(nodes)):
            if _elapsed() > _DEADLINE - 15:
                break
            node = nodes[i]
            if node[0] != 'internal':
                continue
            # Try NNI: swap right child with left child's left
            # Original: (L, R) where L=(LL, LR)
            # NNI1: (LL, (LR, R))
            # NNI2: (LR, (LL, R))
            l, r = node[1], node[2]
            if nodes[l][0] == 'internal':
                ll, lr = nodes[l][1], nodes[l][2]
                # Save original
                orig = (node[1], node[2], nodes[l][1], nodes[l][2])
                
                # Try NNI1: (LL, (LR, R))
                nodes[i] = ('internal', ll, len(nodes))
                nodes.append(('internal', lr, r))
                new_score = score_tree()
                if new_score < current:
                    current = new_score
                    improved = True
                else:
                    # Restore
                    nodes[i] = ('internal', l, r)
                    nodes[l] = ('internal', ll, lr)
                    # Remove the extra node we added
                    if len(nodes) > orig[3] + 1:
                        nodes.pop()
                
                # Try NNI2: (LR, (LL, R))
                nodes[i] = ('internal', lr, len(nodes))
                nodes.append(('internal', ll, r))
                new_score = score_tree()
                if new_score < current:
                    current = new_score
                    improved = True
                else:
                    nodes[i] = ('internal', l, r)
                    nodes[l] = ('internal', ll, lr)
                    if len(nodes) > orig[3] + 1:
                        nodes.pop()
            elif nodes[r][0] == 'internal':
                rl, rr = nodes[r][1], nodes[r][2]
                orig = (node[1], node[2], nodes[r][1], nodes[r][2])
                
                # NNI1: ((L, RL), RR)
                nodes[i] = (len(nodes), rr)
                nodes.append(('internal', l, rl))
                new_score = score_tree()
                if new_score < current:
                    current = new_score
                    improved = True
                else:
                    nodes[i] = ('internal', l, r)
                    nodes[r] = ('internal', rl, rr)
                    if len(nodes) > orig[3] + 1:
                        nodes.pop()
                
                # NNI2: ((L, RR), RL)
                nodes[i] = (len(nodes), rl)
                nodes.append(('internal', l, rr))
                new_score = score_tree()
                if new_score < current:
                    current = new_score
                    improved = True
                else:
                    nodes[i] = ('internal', l, r)
                    nodes[r] = ('internal', rl, rr)
                    if len(nodes) > orig[3] + 1:
                        nodes.pop()
        
        if not improved:
            break
    
    return to_tuple(root)


def reconstruct(sequences):
    global _start_time
    _start_time = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    # Precompute integer sequences
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_int = {label: [char_map[c] for c in sequences[label]] for label in labels}
    
    # Build NJ tree
    newick = _nj_build(sequences, labels)
    
    # Parse Newick to nested tuple
    def parse(newick_str):
        s = newick_str.strip()
        if s.startswith('('):
            s = s[1:]
        if s.endswith(')'):
            s = s[:-1]
        if s.endswith(';'):
            s = s[:-1]
        
        def parse_node(pos):
            if s[pos] == '(':
                pos += 1
                left, pos = parse_node(pos)
                if s[pos] == ',':
                    pos += 1
                    right, pos = parse_node(pos)
                else:
                    right = None
                if s[pos] == ')':
                    pos += 1
                if right is None:
                    return left, pos
                return (left, right), pos
            else:
                start = pos
                while pos < len(s) and s[pos] not in (',', ')', '('):
                    pos += 1
                return s[start:pos], pos
        
        result, _ = parse_node(0)
        return result
    
    tree = parse(newick)
    
    # NNI refinement if time permits
    if _elapsed() < _DEADLINE - 30 and n <= 32:
        refined = _nni_refine(tree, seq_int, max_rounds=10)
        # Validate
        taxa = []
        def collect(node):
            if isinstance(node, str):
                taxa.append(node)
            else:
                collect(node[0])
                collect(node[1])
        collect(refined)
        if sorted(taxa) == labels:
            newick = ""
            def to_newick(node):
                if isinstance(node, str):
                    return node
                else:
                    return f"({to_newick(node[0])},{to_newick(node[1])})"
            newick = to_newick(refined)
    
    return newick + ";"
# EVOLVE-BLOCK-END
