"""
Phylogenetic topology recovery using Fitch parsimony split search.

Strategy:
1. Use Fitch parsimony to evaluate candidate unrooted splits (bipartitions).
2. Build the tree from the set of most parsimonious compatible splits.
3. This is a different design family: instead of distance matrices + ML hill-climbing,
   we directly search split space using parsimony scores.
"""
import time
import itertools

# Global deadline guard
_DEADLINE = 260.0
_START_TIME = None


def _fitch_parity_count(sequences, labels, split_a, split_b):
    """
    Compute Fitch parsimony score for a split (split_a, split_b).
    The score is the number of evolutionary changes needed on the edge
    that separates the two groups, given the observed data.
    
    For a split, we compute the minimum number of changes required
    if we root the tree at the internal edge corresponding to this split.
    
    Simplified approach: compute Fitch score for the whole tree implied
    by this split being a single edge, using the two subtrees.
    """
    # For a split, the parsimony score contribution of the internal edge
    # is determined by whether the ancestral sets of the two sides overlap.
    # We use a simplified Fitch: compute the set of possible ancestral states
    # at the node for each side, then check if they overlap.
    
    # Actually, for split-based parsimony, we compute the total parsimony
    # of the tree where this split is present, compared to without it.
    # A simpler approach: for each split, compute the parsimony cost of
    # the two induced subtrees independently.
    
    # Fitch algorithm on a group of taxa:
    # Returns (min_cost, ancestral_set)
    
    def fitch_group(taxa):
        """Fitch parsimony on a set of taxa. Returns (cost, ancestral_state_set)."""
        if len(taxa) == 1:
            return 0, {sequences[taxa][0]}  # placeholder, we'll do full alignment
        
        # For full alignment Fitch, we need to process all sites.
        # But for split scoring, we can use a simplified per-site approach.
        pass
    
    # Simplified: for a split, the parsimony score is computed by
    # running Fitch on each side and checking the edge cost.
    # Let's use a different approach: compute the total parsimony of
    # the tree with this split as an edge.
    
    # Actually, the cleanest approach for split-based parsimony:
    # For each candidate split, compute the parsimony score of the
    # induced subtree on each side, plus the cost of the connecting edge.
    
    # For a split A|B, the parsimony score is:
    # score(A) + score(B) + edge_cost(A, B)
    # where edge_cost is 0 if the ancestral sets overlap, 1 otherwise.
    
    # To compute score(A) and score(B), we need to know the tree topology
    # within each group. This is circular.
    
    # Alternative: use the "parsimony score of a split" as defined by
    # Farris: the number of sites where the two groups have different
    # majority states.
    
    n_sites = len(sequences[labels[0]])
    score = 0
    for site in range(n_sites):
        # Get the set of bases in group A and group B at this site
        bases_a = set(sequences[lab][site] for lab in split_a)
        bases_b = set(sequences[lab][site] for lab in split_b)
        # If the sets are disjoint, this site supports the split (cost 1)
        # If they overlap, cost 0
        if bases_a.isdisjoint(bases_b):
            score += 1
    return score


def _fitch_split_cost_full(sequences, labels, split_a, split_b):
    """
    Compute Fitch parsimony score for a bipartition split.
    Uses the standard Fitch algorithm on the two induced subtrees.
    
    For a split A|B, we compute:
    - The Fitch parsimony score of the subtree on A
    - The Fitch parsimony score of the subtree on B  
    - The cost of the edge connecting them (0 if ancestral sets overlap, 1 if not)
    
    But we don't know the internal topology, so we use a lower bound:
    the minimum possible parsimony score for each group.
    """
    n_sites = len(sequences[labels[0]])
    total_cost = 0
    
    for site in range(n_sites):
        # Fitch on group A at this site
        set_a = set(sequences[lab][site] for lab in split_a)
        # Fitch on group B at this site
        set_b = set(sequences[lab][site] for lab in split_b)
        
        # The cost of the edge between the two groups:
        # If set_a and set_b overlap, cost = 0
        # Otherwise, cost = 1
        if set_a.isdisjoint(set_b):
            total_cost += 1
    
    return total_cost


def _generate_candidate_splits(labels):
    """
    Generate candidate unrooted splits (bipartitions) for the given taxa.
    For n taxa, there are many possible splits. We generate a manageable subset.
    
    Strategy: generate all splits of size 1 to n/2 (by symmetry).
    For small n (<=16), we can generate all. For larger n, we sample.
    """
    n = len(labels)
    splits = []
    
    if n <= 12:
        # Generate all splits of size 1 to n//2
        for k in range(1, n // 2 + 1):
            for combo in itertools.combinations(range(n), k):
                split_a = [labels[i] for i in combo]
                split_b = [labels[i] for i in range(n) if i not in combo]
                # Normalize: ensure the first element of split_a < first element of split_b
                # to avoid duplicates
                if split_a[0] < split_b[0]:
                    splits.append((tuple(split_a), tuple(split_b)))
    else:
        # For larger n, generate a subset
        # Always include all 1|n-1 splits
        for i in range(n):
            split_a = (labels[i],)
            split_b = tuple(labels[j] for j in range(n) if j != i)
            splits.append((split_a, split_b))
        
        # Include some random 2|n-2 splits
        import random
        rng = random.Random(42)
        num_2splits = min(500, n * (n - 1) // 2)
        for _ in range(num_2splits):
            i, j = rng.sample(range(n), 2)
            split_a = tuple(sorted([labels[i], labels[j]]))
            split_b = tuple(labels[k] for k in range(n) if k not in (i, j))
            if split_b:
                splits.append((split_a, tuple(sorted(split_b))))
        
        # Include some random 3|n-3 splits
        num_3splits = min(300, n * (n - 1) * (n - 2) // 6)
        for _ in range(num_3splits):
            combo = rng.sample(range(n), 3)
            split_a = tuple(sorted([labels[i] for i in combo]))
            split_b = tuple(labels[k] for k in range(n) if k not in combo)
            if split_b:
                splits.append((split_a, tuple(sorted(split_b))))
    
    return splits


def _splits_compatible(split1, split2):
    """
    Check if two splits are compatible (can coexist in a tree).
    Two splits A|B and C|D are compatible if at most one of the
    four intersections A∩C, A∩D, B∩C, B∩D is non-empty.
    """
    a1, b1 = split1
    a2, b2 = split2
    
    inter1 = set(a1) & set(a2)
    inter2 = set(a1) & set(b2)
    inter3 = set(b1) & set(a2)
    inter4 = set(b1) & set(b2)
    
    non_empty = sum(1 for x in [inter1, inter2, inter3, inter4] if x)
    return non_empty <= 1


def _build_tree_from_splits(labels, splits):
    """
    Build a tree from a set of compatible splits.
    Uses a greedy approach: start with the most parsimonious splits
    and build the tree incrementally.
    
    For a set of compatible splits, the tree can be reconstructed
    by finding the minimal representation.
    
    Simplified approach: use the splits to define the tree structure
    via a recursive decomposition.
    """
    n = len(labels)
    if n <= 2:
        return tuple(labels)
    if n == 3:
        return (tuple(labels[:2]), labels[2])
    
    # Find a split that partitions the taxa into two non-empty groups
    # and recursively build subtrees
    # Use the splits to guide the decomposition
    
    # Sort splits by parsimony score (ascending)
    # For now, use a simple heuristic: find a balanced split
    
    # Try to find a split from our set that divides the taxa
    for split_a, split_b in splits:
        sa = set(split_a)
        sb = set(split_b)
        if sa and sb and sa | sb == set(labels):
            # This split is valid
            # Recursively build subtrees
            # For simplicity, just return a star-like tree guided by this split
            sub_a = _build_tree_from_splits(list(split_a), [])
            sub_b = _build_tree_from_splits(list(split_b), [])
            return (sub_a, sub_b)
    
    # Fallback: star tree
    if n == 2:
        return tuple(labels)
    return (tuple(labels[:n // 2]), tuple(labels[n // 2:]))


def _build_tree_greedy(labels, scored_splits):
    """
    Build tree greedily from scored splits.
    Start with the full set of taxa, repeatedly split using the best
    compatible split until we reach single taxa.
    """
    n = len(labels)
    if n <= 2:
        return tuple(labels)
    if n == 3:
        return (tuple(labels[:2]), labels[2])
    
    # Find the best split (lowest parsimony score) that divides current taxa
    current_taxa = set(labels)
    
    best_split = None
    best_score = float('inf')
    
    for split_a, split_b, score in scored_splits:
        sa = set(split_a)
        sb = set(split_b)
        # Check if this split is relevant to current_taxa
        if sa | sb != current_taxa:
            continue
        if not sa or not sb:
            continue
        if score < best_score:
            best_score = score
            best_split = (sa, sb)
    
    if best_split is None:
        # No good split found, return a balanced split
        mid = n // 2
        return (tuple(labels[:mid]), tuple(labels[mid:]))
    
    sa, sb = best_split
    sub_a = _build_tree_greedy(list(sa), scored_splits)
    sub_b = _build_tree_greedy(list(sb), scored_splits)
    return (sub_a, sub_b)


def _tree_to_newick(tree):
    """Convert nested tuple tree to Newick string."""
    if isinstance(tree, str):
        return tree
    if isinstance(tree, tuple) and len(tree) == 1:
        return _tree_to_newick(tree[0])
    parts = []
    for child in tree:
        parts.append(_tree_to_newick(child))
    return "(" + ",".join(parts) + ")"


def reconstruct(sequences):
    """
    Reconstruct phylogenetic tree from DNA sequences using parsimony split search.
    Returns Newick string.
    """
    global _START_TIME
    _START_TIME = time.time()

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 2:
        return "(" + ",".join(labels) + ");"
    if n == 3:
        return "(" + ",".join(labels) + ");"

    # Step 1: Generate candidate splits
    candidate_splits = _generate_candidate_splits(labels)
    
    # Step 2: Score each split using Fitch parsimony
    scored_splits = []
    for split_a, split_b in candidate_splits:
        if time.time() - _START_TIME > _DEADLINE - 10:
            break
        score = _fitch_split_cost_full(sequences, labels, split_a, split_b)
        scored_splits.append((split_a, split_b, score))
    
    # Sort by score (ascending - lower is better)
    scored_splits.sort(key=lambda x: x[2])
    
    # Step 3: Build tree from the best splits
    # Use greedy approach: pick the best split that divides the full set
    tree = _build_tree_greedy(labels, scored_splits)
    
    # Step 4: Convert to Newick
    newick = _tree_to_newick(tree)
    return newick + ";"


# For testing
if __name__ == "__main__":
    seqs = {
        't0': 'ACGTACGTACGTACGT',
        't1': 'ACGTACGTACGTACGT',
        't2': 'ACGTACGTACGTACGA',
        't3': 'ACGTACGTACGTACGT',
    }
    result = reconstruct(seqs)
    print(result)
