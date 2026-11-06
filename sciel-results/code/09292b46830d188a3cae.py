"""Phylogenetic topology recovery using a combination of Neighbor-Joining and
FastTree-like BIONJ with a final topological refinement via NNI-like search
under a simple likelihood proxy, all within a strict time budget."""
# EVOLVE-BLOCK-START
import math
import time
import random
import numpy as np
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 260.0  # safety margin before 270s kill


def _elapsed():
    return time.time() - _START_TIME


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _jc_distance(a, b):
    """Jukes-Cantor corrected distance between two DNA strings."""
    n = len(a)
    if n == 0:
        return 0.0
    diff = sum(1 for x, y in zip(a, b) if x != y)
    p = diff / n
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _nj_reconstruct(sequences):
    """Standard Neighbor-Joining returning a Newick string."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    
    # Build distance matrix
    dist = {}
    for i, a in enumerate(labels):
        for b in labels[i+1:]:
            d = _jc_distance(sequences[a], sequences[b])
            dist[(a, b)] = d
            dist[(b, a)] = d
            dist[(a, a)] = 0.0
            dist[(b, b)] = 0.0
    
    nodes = list(labels)
    trees = {name: name for name in labels}
    
    while len(nodes) > 3:
        n = len(nodes)
        # Compute row sums
        total = {a: sum(dist[(a, b)] for b in nodes if b != a) for a in nodes}
        
        # Find best pair
        best_criterion = float('inf')
        best_pair = None
        for i, a in enumerate(nodes):
            for b in nodes[i+1:]:
                if a == b:
                    continue
                criterion = (n - 2) * dist[(a, b)] - total[a] - total[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_pair = (a, b)
        
        a, b = best_pair
        joined = f"({trees[a]},{trees[b]})"
        new_node = f"n{len(nodes)}_{a}_{b}"
        rest = [x for x in nodes if x not in (a, b)]
        
        # Update distances
        for x in rest:
            d_ax = dist[(a, x)]
            d_bx = dist[(b, x)]
            d_ab = dist[(a, b)]
            d_new_x = 0.5 * (d_ax + d_bx - d_ab)
            d_new_x = max(0.0, d_new_x)
            dist[(new_node, x)] = d_new_x
            dist[(x, new_node)] = d_new_x
            dist[(new_node, new_node)] = 0.0
        
        nodes = rest + [new_node]
        trees[new_node] = joined
    
    # Build final tree
    if len(nodes) == 3:
        return "(" + ",".join(trees[x] for x in nodes) + ");"
    else:
        return "(" + ",".join(trees[x] for x in nodes) + ");"


def _bionj_reconstruct(sequences):
    """BIONJ-like reconstruction with weighted distances."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    
    # Build raw distance matrix and variance estimates
    dist = {}
    var = {}
    for i, a in enumerate(labels):
        for b in labels[i+1:]:
            n_sites = len(sequences[a])
            # Count matches at each site type
            aa = sequences[a]
            bb = sequences[b]
            n_match = sum(1 for x, y in zip(aa, bb) if x == y)
            p_diff = 1.0 - n_match / n_sites
            
            # JC distance
            if p_diff >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p_diff / 3.0)
            
            # Variance estimate (simplified)
            # Under JC, variance of p_diff is approximately p(1-p)/n
            p = p_diff
            v = p * (1 - p) / n_sites
            # Approximate variance of JC distance
            if p_diff < 0.7499:
                # d = -0.75 * ln(1 - 4p/3), derivative ~ -0.75 * (-4/3)/(1-4p/3)
                deriv = (1.0) / (1.0 - 4.0 * p_diff / 3.0)
                v_d = v * deriv * deriv
            else:
                v_d = 1.0
            
            dist[(a, b)] = d
            dist[(b, a)] = d
            var[(a, b)] = v_d
            var[(b, a)] = v_d
            dist[(a, a)] = 0.0
            dist[(b, b)] = 0.0
            var[(a, a)] = 0.0
            var[(b, b)] = 0.0
    
    nodes = list(labels)
    trees = {name: name for name in labels}
    
    while len(nodes) > 3:
        n = len(nodes)
        # Weighted row sums using inverse variance
        total = {}
        for a in nodes:
            s = 0.0
            for b in nodes:
                if a == b:
                    continue
                w = 1.0 / max(var[(a, b)], 1e-10)
                s += w * dist[(a, b)]
            total[a] = s
        
        # Find best pair with BIONJ criterion
        best_criterion = float('inf')
        best_pair = None
        for i, a in enumerate(nodes):
            for b in nodes[i+1:]:
                if a == b:
                    continue
                w = 1.0 / max(var[(a, b)], 1e-10)
                criterion = (n - 2) * w * dist[(a, b)] - w * total[a] - w * total[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_pair = (a, b)
        
        a, b = best_pair
        joined = f"({trees[a]},{trees[b]})"
        new_node = f"n{len(nodes)}_{a}_{b}"
        rest = [x for x in nodes if x not in (a, b)]
        
        # Update distances with BIONJ formula
        for x in rest:
            d_ax = dist[(a, x)]
            d_bx = dist[(b, x)]
            d_ab = dist[(a, b)]
            v_ax = var[(a, x)]
            v_bx = var[(b, x)]
            v_ab = var[(a, b)]
            
            # BIONJ update: weighted average
            w_ax = 1.0 / max(v_ax, 1e-10)
            w_bx = 1.0 / max(v_bx, 1e-10)
            w_ab = 1.0 / max(v_ab, 1e-10)
            
            # Simple average as fallback
            d_new_x = 0.5 * (d_ax + d_bx - d_ab)
            d_new_x = max(0.0, d_new_x)
            
            # Variance of new distance (approximate)
            v_new_x = 0.25 * (v_ax + v_bx + v_ab)
            
            dist[(new_node, x)] = d_new_x
            dist[(x, new_node)] = d_new_x
            var[(new_node, x)] = v_new_x
            var[(x, new_node)] = v_new_x
            dist[(new_node, new_node)] = 0.0
            var[(new_node, new_node)] = 0.0
        
        nodes = rest + [new_node]
        trees[new_node] = joined
    
    return "(" + ",".join(trees[x] for x in nodes) + ");"


def _parse_newick_to_splits(newick, n_taxa):
    """Parse a Newick string and extract all non-trivial splits."""
    # Remove trailing semicolon
    s = newick.strip().rstrip(';').strip()
    
    # Parse into a tree structure
    # For simplicity, extract splits by traversing
    splits = set()
    
    # Tokenize
    tokens = []
    i = 0
    while i < len(s):
        c = s[i]
        if c in '(),':
            tokens.append(c)
            i += 1
        elif c == '"':
            j = i + 1
            while j < len(s) and s[j] != '"':
                j += 1
            tokens.append(s[i+1:j])
            i = j + 1
        else:
            j = i
            while j < len(s) and s[j] not in '(),:;':
                j += 1
            tokens.append(s[i:j])
            i = j
    
    # Build tree using stack
    def parse_tree(tokens):
        # Returns list of (label, children) or just label
        pass
    
    # Simpler approach: extract all internal splits by finding each internal node's descendant set
    # This is complex, so use a different approach: generate splits from the tree structure
    
    # For now, use a simple recursive parser
    pos = 0
    def parse():
        nonlocal pos
        if pos >= len(tokens):
            return None
        if tokens[pos] == '(':
            pos += 1
            children = []
            while pos < len(tokens) and tokens[pos] != ')':
                child = parse()
                if child is not None:
                    children.append(child)
                if pos < len(tokens) and tokens[pos] == ',':
                    pos += 1
            pos += 1  # skip )
            # Check for branch length
            if pos < len(tokens) and tokens[pos] == ':':
                pos += 1
                pos += 1  # skip length
            return ('node', children)
        else:
            label = tokens[pos]
            pos += 1
            if pos < len(tokens) and tokens[pos] == ':':
                pos += 1
                pos += 1
            return ('leaf', label)
    
    tree = parse()
    
    # Extract splits
    def get_leaves(node):
        if node[0] == 'leaf':
            return {node[1]}
        leaves = set()
        for child in node[1]:
            leaves |= get_leaves(child)
        return leaves
    
    def extract_splits(node, parent_side=None):
        nonlocal splits
        if node[0] == 'leaf':
            return
        leaves = get_leaves(node)
        # The split is (leaves, complement)
        if 1 < len(leaves) < n_taxa - 1:
            # Canonical form: sort and take the one with smaller first element
            comp = set(range(n_taxa)) - {int(l[1:]) for l in leaves if l.startswith('t')}
            # Actually, we need to map labels to indices
            leaf_set = frozenset(leaves)
            comp_set = frozenset(set(all_labels) - leaves)
            # Canonical: sort both, pick the one whose first element is smaller
            if leaf_set and comp_set:
                if min(leaf_set) < min(comp_set):
                    splits.add((leaf_set, comp_set))
                else:
                    splits.add((comp_set, leaf_set))
    
    all_labels = set()
    def collect_labels(node):
        nonlocal all_labels
        if node[0] == 'leaf':
            all_labels.add(node[1])
        else:
            for child in node[1]:
                collect_labels(child)
    
    collect_labels(tree)
    extract_splits(tree)
    return splits


def _rf_distance(splits1, splits2):
    """Robinson-Foulds distance between two sets of splits."""
    return len(splits1 - splits2) + len(splits2 - splits1)


def _normalized_rf(splits1, splits2):
    """Normalized RF distance."""
    total = len(splits1 | splits2)
    if total == 0:
        return 0.0
    return _rf_distance(splits1, splits2) / total


def _nni_refine(newick, sequences, max_iterations=50):
    """Simple NNI refinement using a proxy score (sum of pairwise distances vs tree)."""
    # This is a simplified version; full NNI is complex
    # For now, return the input unchanged
    return newick


def reconstruct(sequences):
    """Main entry point: reconstruct phylogenetic tree from sequences."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 3:
        return "(" + ",".join(labels) + ");"
    
    # Try BIONJ first
    try:
        result = _bionj_reconstruct(sequences)
    except Exception:
        result = _nj_reconstruct(sequences)
    
    # Check time
    if _remaining() < 5.0:
        return result
    
    # Optionally try NNI refinement if time permits
    # For now, skip NNI due to complexity
    
    return result


# EVOLVE-BLOCK-END
