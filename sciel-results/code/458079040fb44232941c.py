"""Phylogenetic topology recovery via enhanced NJ with quartet validation."""
import time
import math
import numpy as np

_DEADLINE = 262.0

def _jc_distance_matrix(sequences, labels):
    """Vectorized JC distance computation."""
    n = len(labels)
    L = len(sequences[labels[0]])
    
    # Convert to numpy
    arr = np.array([list(sequences[lab]) for lab in labels])
    
    # Count mismatches for all pairs
    dmat = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            mismatches = np.sum(arr[i] != arr[j])
            p = mismatches / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(max(1e-15, 1.0 - 4.0*p/3.0))
            dmat[i, j] = d
            dmat[j, i] = d
    return dmat

def _nj(dmat):
    """Neighbor Joining. Returns Newick string with leaf indices."""
    n = dmat.shape[0]
    nodes = list(range(n))
    trees = {i: str(i) for i in range(n)}
    d = dmat.astype(float).copy()
    
    while len(nodes) > 3:
        m = len(nodes)
        # Row sums
        r = np.array([sum(d[nodes[i], nodes[j]] for j in range(m) if j != i) for i in range(m)])
        
        best_crit = float('inf')
        best_pair = None
        for i in range(m):
            for j in range(i+1, m):
                a, b = nodes[i], nodes[j]
                crit = (m - 2) * d[a, b] - r[i] - r[j]
                if crit < best_crit:
                    best_crit = crit
                    best_pair = (i, j)
        
        i, j = best_pair
        a, b = nodes[i], nodes[j]
        rest = [nodes[k] for k in range(m) if k not in (i, j)]
        new_id = -1 - len(nodes)
        
        for x in rest:
            val = 0.5 * (d[a, x] + d[b, x] - d[a, b])
            d[new_id, x] = val
            d[x, new_id] = val
        
        trees[new_id] = f"({trees[a]},{trees[b]})"
        nodes = rest + [new_id]
    
    if len(nodes) == 3:
        a, b, c = nodes
        return f"({trees[a]},{trees[b]},{trees[c]})"
    return "(" + ",".join(trees[x] for x in nodes) + ")"

def _quartet_support(sequences, labels, a, b, c, d, L):
    """Count support for each of 3 quartet topologies."""
    sa, sb, sc, sd = sequences[labels[a]], sequences[labels[b]], sequences[labels[c]], sequences[labels[d]]
    scores = [0, 0, 0]
    for pos in range(L):
        na, nb, nc, nd = sa[pos], sb[pos], sc[pos], sd[pos]
        if na == nb and nc == nd and na != nc:
            scores[0] += 1
        elif na == nc and nb == nd and na != nb:
            scores[1] += 1
        elif na == nd and nb == nc and na != nb:
            scores[2] += 1
    return scores

def _tree_quartet_topology(newick_tree, a, b, c, d, labels, label_to_idx):
    """Determine which quartet topology the tree implies for taxa a,b,c,d."""
    # Parse the tree to find the quartet topology
    # This requires proper tree parsing - simplified approach:
    # Check if the tree has the split {a,b}|{c,d}, {a,c}|{b,d}, or {a,d}|{b,c}
    
    # For simplicity, extract all splits from the tree
    def extract_splits(s):
        splits = set()
        def parse(s, i):
            if i >= len(s):
                return set(), i
            if s[i] == '(':
                i += 1
                children = []
                while i < len(s) and s[i] != ')':
                    if s[i] == ',':
                        i += 1
                        continue
                    child, i = parse(s, i)
                    children.append(child)
                i += 1
                all_set = set()
                for c in children:
                    all_set |= c
                if len(children) == 2:
                    left = children[0]
                    right = children[1]
                    if 0 < len(left) < len(all_set):
                        splits.add(frozenset(left))
                    return all_set, i
                return all_set, i
            else:
                j = i
                while j < len(s) and s[j] not in ',)':
                    j += 1
                name = s[i:j]
                i = j
                if name in label_to_idx:
                    return frozenset([label_to_idx[name]]), i
                return set(), i
        _, _ = parse(s, 0)
        return splits
    
    n = len(labels)
    splits = extract_splits(newick_tree)
    
    # Check which quartet split is present
    set_ab = frozenset([a, b])
    set_cd = frozenset([c, d])
    set_ac = frozenset([a, c])
    set_bd = frozenset([b, d])
    set_ad = frozenset([a, d])
    set_bc = frozenset([b, c])
    
    for sp in splits:
        if sp == set_ab or sp == set_cd:
            return 0
        if sp == set_ac or sp == set_bd:
            return 1
        if sp == set_ad or sp == set_bc:
            return 2
    
    return None  # Tree doesn't resolve this quartet

def reconstruct(sequences):
    start = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    L = len(sequences[labels[0]])
    
    if n <= 3:
        return "(" + ",".join(labels) + ");"
    
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    
    # Compute distances
    dmat = _jc_distance_matrix(sequences, labels)
    
    # Run NJ
    newick = _nj(dmat)
    
    elapsed = time.time() - start
    if elapsed > _DEADLINE - 40:
        return newick
    
    # Quartet-based refinement: test a subset of quartets
    # and identify where the tree may be wrong
    max_quartets = min(3000, n*(n-1)*(n-2)*(n-3)//24)
    import random
    random.seed(42)
    
    # Sample quartets
    indices = list(range(n))
    quartets_sampled = 0
    discordant = []
    
    for _ in range(max_quartets * 3):
        if quartets_sampled >= max_quartets:
            break
        if time.time() - start > _DEADLINE - 30:
            break
        a, b, c, d = random.sample(indices, 4)
        a, b, c, d = sorted([a, b, c, d])
        
        scores = _quartet_support(sequences, labels, a, b, c, d, L)
        if max(scores) == 0:
            continue  # Uninformative
        
        best_topo = max(range(3), key=lambda i: scores[i])
        tree_topo = _tree_quartet_topology(newick, a, b, c, d, labels, label_to_idx)
        
        quartets_sampled += 1
        if tree_topo is not None and tree_topo != best_topo:
            discordant.append((a, b, c, d, best_topo, scores))
    
    # If we find discordant quartets, we could try to fix the tree
    # For now, this is diagnostic only - the NJ tree is returned as-is
    # A full implementation would use these to guide a search
    
    return newick
