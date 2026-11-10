"""Reconstruct phylogenetic topology using Quartet Fitting (quartet-based consensus)."""
import numpy as np
import time
import math
import itertools

_START_TIME = None
_DEADLINE = 260.0  # safety margin before 270s kill

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _elapsed()

def _quartet_scores(seq_a, seq_b, seq_c, seq_d, L):
    """Count compatible sites for each of the 3 quartet topologies.
    Topologies: 0: (a,b)|(c,d), 1: (a,c)|(b,d), 2: (a,d)|(b,c)
    Returns (score0, score1, score2) as floats.
    """
    # For JC model, a site supports topology (a,b)|(c,d) if
    # a,b have same char and c,d have same char (and the pairs differ)
    # More precisely: under JC, the quartet likelihood for topology (a,b)|(c,d)
    # is maximized when a,b agree and c,d agree.
    # A simpler additive score: for each site, count how many of the 3
    # "pair-agreement" patterns match.
    
    # For each site, the 4 characters determine which topology is supported.
    # Under JC with unknown branch lengths, the best-fit topology for a single site
    # is the one where the two leaves with the same character are paired together.
    # If all 4 chars same or all different, all topologies tie.
    # If 3 same, 1 different: the topology pairing the two that share the majority char wins.
    # If 2+2 split: the topology matching the split wins.
    
    # Vectorized approach:
    # For topology 0 (a,b)|(c,d): score += 1 if a==b and c==d (and not all same)
    # Actually, a cleaner approach: use the "quartet distance" or "compatibility" count.
    
    # Let's use a well-known approach: for each site, determine which topology
    # is best supported, and accumulate.
    
    # Convert to integer codes 0-3
    # a, b, c, d are numpy arrays of ints
    
    # For topology (a,b)|(c,d):
    # This topology is supported when a==b and c==d, OR when a==c and b==d is NOT the case...
    # Actually, let me think more carefully.
    
    # Under JC, for a single site with characters (ca, cb, cc, cd):
    # The likelihood under topology (a,b)|(c,d) with branch lengths (la, lb, lc, ld, le)
    # where e is internal: P = sum over internal states...
    # 
    # A simpler, well-established approach for quartet fitting:
    # Count for each topology the number of sites where the "pair" characters match.
    # Topology (a,b)|(c,d): count sites where (a==b) XOR (c==d) is false, i.e., 
    #   both pairs agree or both pairs disagree... 
    # 
    # Actually, the standard approach: for each site, the topology that groups
    # the two identical characters together is the best one. Let me just count:
    
    # For each site position:
    #   if all 4 same: all topologies get 0 (or 1/3 each)
    #   if 3 same 1 diff: the topology pairing the two majority chars gets 1
    #   if 2-2 split: the topology matching the split gets 1
    #   if all 4 different: all get 0 (or 1/3 each)
    
    # This is equivalent to: for topology (a,b)|(c,d), score += 1 if a==b and c==d
    #   (with the constraint that not all are the same, but that's handled naturally)
    
    # Let me use: score_topo = number of sites where the two leaves in each pair match
    #   and the two pairs have different characters.
    
    ab_match = (seq_a == seq_b)
    cd_match = (seq_c == seq_d)
    ac_match = (seq_a == seq_c)
    bd_match = (seq_b == seq_d)
    ad_match = (seq_a == seq_d)
    bc_match = (seq_b == seq_c)
    
    # Topology 0: (a,b)|(c,d) - supported when a==b and c==d and the pairs differ
    score0 = np.sum(ab_match & cd_match & (seq_a != seq_c))
    # Topology 1: (a,c)|(b,d) - supported when a==c and b==d and the pairs differ
    score1 = np.sum(ac_match & bd_match & (seq_a != seq_b))
    # Topology 2: (a,d)|(b,c) - supported when a==d and b==c and the pairs differ
    score2 = np.sum(ad_match & bc_match & (seq_a != seq_b))
    
    return float(score0), float(score1), float(score2)

def _quartet_fitting(sequences, labels):
    """Run Quartet Fitting: count quartet support and build tree via greedy/consensus."""
    n = len(labels)
    L = len(sequences[labels[0]])
    
    if n <= 3:
        if n == 1:
            return labels[0] + ";"
        elif n == 2:
            return "(" + labels[0] + "," + labels[1] + ");"
        else:
            return "(" + labels[0] + "," + labels[1] + "," + labels[2] + ");"
    
    # Convert sequences to integer arrays for fast comparison
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_ints = {}
    for label in labels:
        seq_ints[label] = np.array([char_map[c] for c in sequences[label]], dtype=np.int8)
    
    # Count quartet scores
    # For each 4-taxon subset, determine the best topology
    # quartet_scores[quartet_tuple] = (score0, score1, score2)
    # where the quartet is (a,b,c,d) with a<b<c<d (by index)
    
    # We'll store: for each quartet (i,j,k,l) with i<j<k<l:
    #   best_topo: 0=(i,j)|(k,l), 1=(i,k)|(j,l), 2=(i,l)|(j,k)
    #   margin: difference between best and second-best
    
    # For efficiency, we'll accumulate a "splits" weight matrix
    # Each quartet vote adds to the corresponding split.
    
    # A split is represented by the frozenset of one side of the bipartition
    # For n taxa, we only need to track splits where both sides have size >= 2
    
    # Initialize split weights
    # Use a dict: frozenset -> weight
    split_weights = {}
    
    idx_to_label = {i: labels[i] for i in range(n)}
    
    # Process all quartets
    quartets = list(itertools.combinations(range(n), 4))
    total_quartets = len(quartets)
    
    # If too many quartets, sample
    if total_quartets > 50000 and _time_left() < 120:
        # Sample quartets
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(total_quartets, size=50000, replace=False)
        quartets = [quartets[i] for i in sorted(sample_idx)]
    
    for qi, (i, j, k, l) in enumerate(quartets):
        if qi % 10000 == 0 and _time_left() < 30:
            break
            
        a = seq_ints[labels[i]]
        b = seq_ints[labels[j]]
        c = seq_ints[labels[k]]
        d = seq_ints[labels[l]]
        
        s0, s1, s2 = _quartet_scores(a, b, c, d, L)
        
        scores = [s0, s1, s2]
        best = max(scores)
        second_best = max([s for s in scores if s != best]) if scores.count(best) == 1 else max(scores)
        
        if best == 0:
            # All tied, skip or give uniform weight
            continue
        
        margin = best - second_best
        if margin <= 0:
            continue
        
        # Determine which split the best topology implies
        # Topology 0: (i,j)|(k,l) -> split {i,j} or {k,l}
        # Topology 1: (i,k)|(j,l) -> split {i,k} or {j,l}
        # Topology 2: (i,l)|(j,k) -> split {i,l} or {j,k}
        
        if s0 == best:
            split = frozenset([i, j])
        elif s1 == best:
            split = frozenset([i, k])
        else:
            split = frozenset([i, l])
        
        # Normalize split to canonical form (smaller side first, or by min element)
        # We'll use the frozenset with the smallest minimum element
        # Actually, for a bipartition {A}|{B}, we store the side with smaller min element
        # or if equal, the lexicographically smaller one
        other = frozenset(set(range(n)) - split)
        # Canonical: use the side whose smallest element is smaller
        if min(split) < min(other):
            canonical = split
        else:
            canonical = other
        
        split_weights[canonical] = split_weights.get(canonical, 0.0) + margin
    
    # Now build a tree from the weighted splits
    # Use a greedy approach: start with all taxa, repeatedly find the best split
    # that partitions the current set, and build a binary tree.
    
    # Alternative: use the splits to build a tree via a "greedy consensus" approach
    # Start with the full set of taxa. Find the highest-weight split that 
    # properly partitions the current set. Recurse.
    
    def build_tree(taxon_set):
        if len(taxon_set) <= 1:
            return labels[list(taxon_set)[0]] if taxon_set else ""
        if len(taxon_set) == 2:
            return "(" + labels[list(taxon_set)[0]] + "," + labels[list(taxon_set)[1]] + ");"
        if len(taxon_set) == 3:
            lst = sorted(taxon_set)
            return "(" + labels[lst[0]] + "," + labels[lst[1]] + "," + labels[lst[2]] + ");"
        
        # Find the best split that partitions taxon_set into two non-empty parts
        best_split = None
        best_weight = 0.0
        
        for split, weight in split_weights.items():
            if weight <= best_weight:
                continue
            
            left = split & taxon_set
            right = taxon_set - split
            
            if len(left) < 2 or len(right) < 2:
                continue
            
            # Check that this is a valid split of the current set
            # (both sides must be non-empty subsets of taxon_set)
            if len(left) >= 2 and len(right) >= 2:
                best_split = (left, right)
                best_weight = weight
        
        if best_split is None:
            # No good split found, return a star tree
            lst = sorted(taxon_set)
            return "(" + ",".join(labels[i] for i in lst) + ");"
        
        left_tree = build_tree(best_split[0])
        right_tree = build_tree(best_split[1])
        return "(" + left_tree + "," + right_tree + ");"
    
    all_taxa = frozenset(range(n))
    newick = build_tree(all_taxa)
    
    return newick

def reconstruct(sequences):
    global _START_TIME
    _START_TIME = time.time()
    
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    n = len(labels)
    
    if n <= 3:
        if n == 1:
            return labels[0] + ";"
        elif n == 2:
            return "(" + labels[0] + "," + labels[1] + ");"
        else:
            return "(" + labels[0] + "," + labels[1] + "," + labels[2] + ");"
    
    newick = _quartet_fitting(sequences, labels)
    
    return newick
