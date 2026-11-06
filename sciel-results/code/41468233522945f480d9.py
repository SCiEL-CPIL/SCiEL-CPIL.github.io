import time
import random
import numpy as np
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)  # 560
N_WORDS = (N_TRIPLES + 63) // 64  # 9

# Precompute triple index lookup
_triple_idx = {}
for _i, _t in enumerate(ALL_TRIPLES):
    _triple_idx[_t] = _i

def _tidx(a, b, c):
    """Get index of triple {a,b,c} (all distinct)."""
    if a > b: a, b = b, a
    if b > c: b, c = c, b
    if a > b: a, b = b, a
    return _triple_idx[(a, b, c)]

def _has_presence(presence, idx):
    """Check if triple idx is in the presence bitmask."""
    return (presence[idx >> 6] >> (idx & 63)) & 1

def _set_presence(presence, idx, val):
    """Set or clear triple idx in presence."""
    word = idx >> 6
    bit = 1 << (idx & 63)
    if val:
        presence[word] |= bit
    else:
        presence[word] &= ~bit

def _find_witnesses(a, b, c, presence, max_witnesses=10):
    """Find up to max_witnesses K(2,2,2)s that block triple (a,b,c)."""
    witnesses = []
    excluded = {a, b, c}
    
    for a_p in range(V):
        if a_p in excluded:
            continue
        idx_apbc = _tidx(a_p, b, c)
        if not _has_presence(presence, idx_apbc):
            continue
        for b_p in range(V):
            if b_p in excluded or b_p == a_p:
                continue
            idx_abpc = _tidx(a, b_p, c)
            if not _has_presence(presence, idx_abpc):
                continue
            idx_apbpc = _tidx(a_p, b_p, c)
            if not _has_presence(presence, idx_apbpc):
                continue
            for c_p in range(V):
                if c_p in excluded or c_p == a_p or c_p == b_p:
                    continue
                idx_abcp = _tidx(a, b, c_p)
                if not _has_presence(presence, idx_abcp):
                    continue
                idx_apbcp = _tidx(a_p, b, c_p)
                if not _has_presence(presence, idx_apbcp):
                    continue
                idx_abpcp = _tidx(a, b_p, c_p)
                if not _has_presence(presence, idx_abpcp):
                    continue
                idx_apbpcp = _tidx(a_p, b_p, c_p)
                if not _has_presence(presence, idx_apbpcp):
                    continue
                
                # Found a witness
                witness = [
                    idx_apbc,
                    idx_abpc,
                    idx_apbpc,
                    idx_abcp,
                    idx_apbcp,
                    idx_abpcp,
                    idx_apbpcp
                ]
                witnesses.append(witness)
                if len(witnesses) >= max_witnesses:
                    return witnesses
    return witnesses

def _greedy(order, deadline):
    """Greedy: add triples in order, skipping conflicts."""
    presence = [0] * N_WORDS
    count = 0
    for idx in order:
        if time.time() >= deadline:
            break
        a, b, c = ALL_TRIPLES[idx]
        witnesses = _find_witnesses(a, b, c, presence, 1)
        if witnesses:
            continue
        presence[idx >> 6] |= (1 << (idx & 63))
        count += 1
    return count, presence

def _spectral_clustering(deadline):
    """
    Build the conflict graph (560x560) where edges connect triples that cannot
    coexist (i.e., their union contains a K(2,2,2)). Then compute the Laplacian
    and use eigenvectors to identify clusters of low-conflict triples.
    Returns a list of cluster orders (lists of triple indices) sorted by
    expected density of independent sets.
    """
    # Build conflict matrix: conflict[i][j] = 1 if triple i and j cannot both be in a valid system
    # Two triples conflict if adding one creates a K(2,2,2) with the other already present?
    # Actually, conflict means: there exists a K(2,2,2) that contains both triples.
    # Precompute: for each pair of triples, check if they are in a common K(2,2,2).
    
    # For speed, we'll compute a sparse representation.
    # conflict[i] = set of j such that i and j are in some K(2,2,2)
    
    # A K(2,2,2) involves 6 points: a,a',b,b',c,c' and 8 triples.
    # Two triples are in conflict if they appear together in at least one K(2,2,2).
    
    # Efficient approach: for each triple i, find all triples j that share a K(2,2,2) with i.
    # This is expensive to precompute fully, but we can sample or use structure.
    
    # Alternative: use the fact that if triple (a,b,c) is in the system, then any triple
    # that is one of the 7 "neighbors" in a potential K(2,2,2) centered at (a,b,c) with
    # some (a',b',c') creates a conflict if all 8 are present.
    
    # For spectral clustering, we need the adjacency of the conflict graph.
    # Let's build it by iterating over all K(2,2,2)s.
    
    # Number of K(2,2,2)s: choose 6 points from 16, then partition into 3 pairs.
    # C(16,6) * 15 = 8008 * 15 = 120120. Each K(2,2,2) has 8 triples.
    
    # For each K(2,2,2), all 8 triples are pairwise conflicting.
    # So for each K(2,2,2) with triples t1..t8, for each pair (ti, tj), add edge.
    
    # This is 120120 * 28 = 3.36M edge additions. Feasible.
    
    # Build conflict sets
    conflict = [set() for _ in range(N_TRIPLES)]
    
    # Iterate over all ways to choose 6 points and partition into 3 pairs
    from itertools import permutations
    
    # Actually, let's be smarter. A K(2,2,2) is determined by 6 distinct points
    # a,a',b,b',c,c' and the 8 triples:
    # (a,b,c), (a,b,c'), (a,b',c), (a,b',c'), (a',b,c), (a',b,c'), (a',b',c), (a',b',c')
    # So we need to choose 6 points and assign them to roles (a,a',b,b',c,c').
    # But the assignment matters for which triples are formed.
    
    # Let's iterate over all 6-subsets and all partitions into 3 ordered pairs.
    # For a 6-set {p0,p1,p2,p3,p4,p5}, a partition into 3 pairs is a perfect matching.
    # There are 15 perfect matchings on 6 points.
    # For each matching, we also need to order the pairs (which is a,b,c) and order within pairs.
    # But the K(2,2,2) is the same regardless of ordering.
    
    # Let's just iterate over all 6-subsets and for each, generate all 15 matchings,
    # and for each matching, generate the 8 triples.
    
    six_subsets = list(combinations(range(V), 6))
    
    # For each 6-subset, generate all perfect matchings
    # A perfect matching on 6 points: pair up p0 with one of p1..p5 (5 choices),
    # then pair up the next unpaired with one of the remaining 3 (3 choices),
    # then the last two are paired (1 choice). Total 15.
    
    def _matchings(points):
        """Generate all perfect matchings of 6 points as list of 3 pairs."""
        pts = list(points)
        results = []
        def _rec(remaining, current):
            if not remaining:
                results.append(tuple(sorted(current)))
                return
            first = remaining[0]
            for i in range(1, len(remaining)):
                second = remaining[i]
                new_remaining = remaining[1:i] + remaining[i+1:]
                _rec(new_remaining, current + [(first, second)])
        _rec(pts, [])
        return results
    
    for six in six_subsets:
        for matching in _matchings(six):
            # matching is a tuple of 3 pairs: ((a,a'),(b,b'),(c,c')) or some ordering
            # We need to assign which pair is (a,a'), which is (b,b'), which is (c,c')
            # And within each pair, which is the "prime" and which is not.
            # But the 8 triples are symmetric in this sense.
            # Let's just take the matching as 3 pairs and generate all 8 triples.
            
            pairs = list(matching)  # 3 pairs
            # For each permutation of the 3 pairs, and for each pair, 2 orderings
            # But this overcounts. Let's just fix an ordering.
            
            # Actually, for the conflict graph, we just need the set of 8 triples.
            # The 8 triples for a given K(2,2,2) with pairs (x1,x2), (y1,y2), (z1,z2) are:
            # (x1,y1,z1), (x1,y1,z2), (x1,y2,z1), (x1,y2,z2),
            # (x2,y1,z1), (x2,y1,z2), (x2,y2,z1), (x2,y2,z2)
            
            (x1, x2), (y1, y2), (z1, z2) = pairs
            
            triples_in_k = [
                (x1, y1, z1), (x1, y1, z2), (x1, y2, z1), (x1, y2, z2),
                (x2, y1, z1), (x2, y1, z2), (x2, y2, z1), (x2, y2, z2)
            ]
            
            # Get indices
            indices_in_k = []
            for t in triples_in_k:
                idx = _tidx(t[0], t[1], t[2])
                indices_in_k.append(idx)
            
            # All pairs in this K(2,2,2) are conflicting
            for i in range(8):
                for j in range(i+1, 8):
                    conflict[indices_in_k[i]].add(indices_in_k[j])
                    conflict[indices_in_k[j]].add(indices_in_k[i])
    
    # Now build the Laplacian
    # L = D - A
    # For spectral clustering, we want to find clusters of low internal conflict.
    # Actually, we want independent sets, so we want to find dense regions in the
    # complement of the conflict graph.
    
    # Let's use the conflict graph directly. We want to find sets of triples with
    # few conflicts between them.
    
    # Build adjacency matrix
    A = np.zeros((N_TRIPLES, N_TRIPLES), dtype=np.float64)
    for i in range(N_TRIPLES):
        for j in conflict[i]:
            A[i, j] = 1
            A[j, i] = 1
    
    # Degree
    d = A.sum(axis=1)
    
    # Laplacian
    L = np.diag(d) - A
    
    # Compute eigenvectors
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Use the first few eigenvectors (smallest eigenvalues) for clustering
        # Typically use 2-5 eigenvectors
        k_clusters = 5
        X = eigvecs[:, 1:k_clusters+1]  # Skip the first (constant) eigenvector
        
        # Normalize rows
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        X = X / norms
        
        # K-means clustering
        from scipy.cluster.vq import kmeans2
        centroids, labels = kmeans2(X, k_clusters, minit='++')
        
        # Create clusters
        clusters = [[] for _ in range(k_clusters)]
        for i in range(N_TRIPLES):
            clusters[labels[i]].append(i)
        
        # Sort clusters by size (larger clusters first)
        clusters.sort(key=len, reverse=True)
        
        return clusters
        
    except Exception:
        # Fallback: just return all triples as one cluster
        return [list(range(N_TRIPLES))]

def _local_search(presence, count, deadline):
    """Local search with Add and Swap moves."""
    present_set = set()
    for i in range(N_TRIPLES):
        if (presence[i >> 6] >> (i & 63)) & 1:
            present_set.add(i)
    
    absent_list = [i for i in range(N_TRIPLES) if i not in present_set]
    
    best_count = count
    best_presence = presence[:]
    best_present_set = set(present_set)
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.99
    
    while T > T_min and time.time() < deadline:
        move_type = random.random()
        
        if move_type < 0.3:
            # Add move
            if not absent_list:
                T *= cooling_rate
                continue
            new_idx = random.choice(absent_list)
            a, b, c = ALL_TRIPLES[new_idx]
            witnesses = _find_witnesses(a, b, c, presence, 1)
            if not witnesses:
                presence[new_idx >> 6] |= (1 << (new_idx & 63))
                present_set.add(new_idx)
                absent_list.remove(new_idx)
                count += 1
                if count > best_count:
                    best_count = count
                    best_presence = presence[:]
                    best_present_set = set(present_set)
        elif move_type < 0.5:
            # Remove move
            if not present_set:
                T *= cooling_rate
                continue
            old_idx = random.choice(list(present_set))
            presence[old_idx >> 6] &= ~(1 << (old_idx & 63))
            present_set.remove(old_idx)
            absent_list.append(old_idx)
            count -= 1
        else:
            # Swap move
            if not present_set or not absent_list:
                T *= cooling_rate
                continue
            new_idx = random.choice(absent_list)
            a, b, c = ALL_TRIPLES[new_idx]
            witnesses = _find_witnesses(a, b, c, presence, 15)
            if not witnesses:
                presence[new_idx >> 6] |= (1 << (new_idx & 63))
                present_set.add(new_idx)
                absent_list.remove(new_idx)
                count += 1
                if count > best_count:
                    best_count = count
                    best_presence = presence[:]
                    best_present_set = set(present_set)
            else:
                candidates = set()
                for w in witnesses:
                    for t in w:
                        candidates.add(t)
                candidates = [t for t in candidates if t in present_set]
                if candidates:
                    random.shuffle(candidates)
                    for t_rem in candidates[:5]:
                        presence[t_rem >> 6] &= ~(1 << (t_rem & 63))
                        present_set.remove(t_rem)
                        remaining_witnesses = _find_witnesses(a, b, c, presence, 1)
                        if not remaining_witnesses:
                            presence[new_idx >> 6] |= (1 << (new_idx & 63))
                            present_set.add(new_idx)
                            absent_list.remove(new_idx)
                            if count > best_count:
                                best_count = count
                                best_presence = presence[:]
                                best_present_set = set(present_set)
                            break
                        else:
                            presence[t_rem >> 6] |= (1 << (t_rem & 63))
                            present_set.add(t_rem)
        
        T *= cooling_rate
    
    for i in range(N_TRIPLES):
        bit = 1 << (i & 63)
        word = i >> 6
        if i in best_present_set:
            presence[word] |= bit
        else:
            presence[word] &= ~bit
    
    return best_count, presence

def construct():
    start = time.time()
    deadline = start + 115  # Safety margin
    
    # Phase 1: Spectral clustering to identify low-conflict regions
    clusters = _spectral_clustering(deadline)
    
    # Phase 2: Greedy construction within each cluster, then merge
    # Start with empty presence
    presence = [0] * N_WORDS
    count = 0
    
    # Process clusters in order of size (largest first)
    for cluster in clusters:
        # Within each cluster, do greedy addition
        # Sort by some heuristic (e.g., degree in conflict graph)
        # For now, just random order within cluster
        random.shuffle(cluster)
        
        for idx in cluster:
            if time.time() >= deadline:
                break
            a, b, c = ALL_TRIPLES[idx]
            if _has_presence(presence, idx):
                continue
            witnesses = _find_witnesses(a, b, c, presence, 1)
            if witnesses:
                continue
            presence[idx >> 6] |= (1 << (idx & 63))
            count += 1
    
    # Phase 3: Local search to improve
    count, presence = _local_search(presence, count, deadline)
    
    # Extract result
    result = []
    for i in range(N_TRIPLES):
        if (presence[i >> 6] >> (i & 63)) & 1:
            result.append(list(ALL_TRIPLES[i]))
    
    return result

random.seed(42)
