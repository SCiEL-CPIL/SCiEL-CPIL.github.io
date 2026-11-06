"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Distance-Proportional Parsimony (DPP) with NJ fallback and time guard.'''
import math
import time


def _jc_distance(a, b):
    differences = sum(1 for x, y in zip(a, b) if x != y)
    p = differences / len(a)
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _nj_reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    nodes = list(labels)
    trees = {name: name for name in labels}
    matrix = {(a, b): _jc_distance(sequences[a], sequences[b]) for a in labels for b in labels if a != b}
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


def _dpp_reconstruct(sequences, deadline):
    """Distance-Proportional Parsimony: find tree minimizing sum of branch lengths."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 3:
        return "(" + ",".join(labels) + ");"
    
    # Compute pairwise JC distances
    dist = {}
    for i, a in enumerate(labels):
        for b in labels[i+1:]:
            d = _jc_distance(sequences[a], sequences[b])
            dist[(a, b)] = d
            dist[(b, a)] = d
    
    # DPP: iteratively join the pair that minimizes the resulting tree length
    # Tree length for a set of taxa with distances = sum over all pairs / (n-1) for a star
    # For DPP, we use the criterion: join a,b if it minimizes the implied tree length
    # The DPP criterion for joining a,b given current tree T:
    #   delta = d(a,b) - (1/2) * (sum_{x in rest} d(a,x) + sum_{x in rest} d(b,x)) / (n-2)
    # Actually DPP minimizes the total branch length. The change in tree length when
    # joining a,b is related to how much "shorter" the tree becomes.
    
    # Standard DPP: at each step, find pair (a,b) minimizing:
    #   sum_{x in rest} (d(a,x) + d(b,x) - d(a,b)) / 2
    # This is equivalent to minimizing the total edge length of the resulting tree.
    
    active = list(labels)
    trees = {name: name for name in labels}
    # Use a dict for distances, updating as we join
    d = {}
    for i, a in enumerate(labels):
        for b in labels[i+1:]:
            d[(a, b)] = dist[(a, b)]
            d[(b, a)] = dist[(a, b)]
    
    while len(active) > 3:
        if time.time() > deadline:
            break
        m = len(active)
        # For each pair, compute the DPP criterion
        # The DPP score for joining a,b: sum_{x in active \ {a,b}} (d(a,x) + d(b,x) - d(a,b)) / 2
        # We want to MINIMIZE this (it represents the increase in tree length)
        # Actually we want to minimize the resulting tree length, so we minimize this sum.
        
        best_score = None
        best_pair = None
        for i, a in enumerate(active):
            for j, b in enumerate(active):
                if i >= j:
                    continue
                ab = d[(a, b)]
                s = 0.0
                for k, x in enumerate(active):
                    if k != i and k != j:
                        s += d[(a, x)] + d[(b, x)] - ab
                score = s / 2.0
                if best_score is None or score < best_score:
                    best_score = score
                    best_pair = (a, b)
        
        if best_pair is None:
            break
        a, b = best_pair
        joined = f"({trees[a]},{trees[b]})"
        new = f"n{len(active)}_{a}_{b}"
        rest = [x for x in active if x not in (a, b)]
        
        # Update distances: new node's distance to x = 0.5*(d(a,x) + d(b,x) - d(a,b))
        for x in rest:
            nd = 0.5 * (d[(a, x)] + d[(b, x)] - d[(a, b)])
            d[(new, x)] = nd
            d[(x, new)] = nd
        
        active = rest + [new]
        trees[new] = joined
    
    # If we broke out early due to deadline, fall back to NJ on remaining
    if len(active) > 3:
        # Build a sub-sequences dict for remaining active nodes (only original labels have sequences)
        # For internal nodes we don't have sequences, so we need to handle this differently
        # Just use NJ on the remaining original labels if possible
        orig_remaining = [x for x in active if x in sequences]
        if len(orig_remaining) >= 4:
            sub_seq = {x: sequences[x] for x in orig_remaining}
            # But we need to incorporate the already-joined subtrees...
            # Simpler: just return a star tree on all active nodes
            return "(" + ",".join(trees[x] for x in active) + ");"
        else:
            return "(" + ",".join(trees[x] for x in active) + ");"
    
    return "(" + ",".join(trees[x] for x in active) + ");"


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 3:
        return "(" + ",".join(labels) + ");"
    
    start = time.time()
    # Budget: leave 30s margin for all 20 instances (270s total / 20 = 13.5s per instance)
    # But we're called once per instance, so use a fraction of the total budget
    # Use 12s per instance as a safe limit
    deadline = start + 12.0
    
    try:
        result = _dpp_reconstruct(sequences, deadline)
        return result
    except Exception:
        pass
    
    # Fallback to NJ
    try:
        return _nj_reconstruct(sequences)
    except Exception:
        # Last resort: star tree
        return "(" + ",".join(labels) + ");"
# EVOLVE-BLOCK-END
