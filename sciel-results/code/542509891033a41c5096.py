import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 180
    
    # Precompute all forbidden triples: 1/a = 1/b + 1/c => a = bc/(b+c)
    # with a < b < c (distinct, and a is the smallest since 1/a > 1/b + 1/c)
    triples = []
    for b in range(2, N + 1):
        for c in range(b + 1, N + 1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a < b:
                    triples.append((a, b, c))
    
    # Build adjacency: for each element, which triples it's in
    triples_by_elem = {i: [] for i in range(1, N+1)}
    for idx, (a, b, c) in enumerate(triples):
        triples_by_elem[a].append(idx)
        triples_by_elem[b].append(idx)
        triples_by_elem[c].append(idx)
    
    def can_add(s_set, candidate, triples, triples_by_elem):
        """Check if adding candidate to s_set creates a forbidden triple."""
        for tidx in triples_by_elem[candidate]:
            (a, b, c) = triples[tidx]
            others = [a, b, c]
            others.remove(candidate)
            if all(o in s_set for o in others):
                return False
        return True
    
    # Spectral clustering on conflict graph to identify dense communities
    # Build conflict graph: two elements conflict if they appear in a common triple
    conflict_graph = np.zeros((N+1, N+1), dtype=np.int32)
    for (a, b, c) in triples:
        conflict_graph[a, b] += 1
        conflict_graph[a, c] += 1
        conflict_graph[b, c] += 1
    
    # Use degree as a proxy for "hubbiness" - small numbers have higher degree
    degrees = np.sum(conflict_graph, axis=1)
    
    # Identify dense communities using spectral embedding
    # Focus on elements 1-99 (the "small" region)
    small_elements = list(range(1, 100))
    n_small = len(small_elements)
    
    # Build subgraph for small elements
    conflict_sub = np.zeros((n_small, n_small), dtype=np.float64)
    for i, ei in enumerate(small_elements):
        for j, ej in enumerate(small_elements):
            if i != j:
                conflict_sub[i, j] = conflict_graph[ei, ej]
    
    # Spectral embedding: compute Laplacian eigenvalues
    D = np.diag(np.sum(conflict_sub, axis=1))
    L = D - conflict_sub
    
    # Get eigenvectors for smallest eigenvalues (excluding trivial ones)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Use first few non-trivial eigenvectors for clustering
        n_clusters = min(5, n_small // 10 + 1)
        # Skip the first eigenvector (trivial)
        cluster_vectors = eigenvectors[:, 1:n_clusters+1]
        
        # Simple k-means-like clustering
        from scipy.cluster.vq import kmeans2
        if n_clusters < n_small:
            labels, _ = kmeans2(cluster_vectors, n_clusters, minit='points')
        else:
            labels = np.arange(n_small)
    except Exception:
        # Fallback: cluster by degree
        labels = np.zeros(n_small, dtype=int)
        sorted_by_degree = np.argsort(degrees[small_elements])[::-1]
        for i, idx in enumerate(sorted_by_degree):
            labels[idx] = i // (n_small // max(1, n_clusters))
    
    # For each cluster, identify the "hub" elements (high degree within cluster)
    # Strategy: in dense clusters, remove hubs to break many triples at once
    # In sparse outer regions, keep most elements
    
    # Build initial set: start with [100, 180] (no internal triples)
    # Then for small elements, use cluster-aware greedy
    
    best_set = set(range(100, N+1))
    best_size = len(best_set)
    
    # Multiple strategies with cluster awareness
    for attempt in range(500):
        if time.time() > deadline - 15.0:
            break
        
        if attempt == 0:
            # Original backward greedy
            order = list(range(99, 0, -1))
        elif attempt == 1:
            # Forward greedy
            order = list(range(1, 100))
        elif attempt == 2:
            # Degree-sorted (most constrained first)
            order = sorted(range(1, 100), key=lambda x: len(triples_by_elem[x]), reverse=True)
        elif attempt == 3:
            # Degree-sorted (least constrained first)
            order = sorted(range(1, 100), key=lambda x: len(triples_by_elem[x]))
        elif attempt == 4:
            # Cluster-aware: process clusters in order of density, within cluster by degree
            # Dense clusters first (remove hubs), then sparse
            cluster_density = {}
            for label in set(labels):
                members = [small_elements[i] for i in range(n_small) if labels[i] == label]
                # Count conflicts within cluster
                density = 0
                for i in range(len(members)):
                    for j in range(i+1, len(members)):
                        density += conflict_graph[members[i], members[j]]
                cluster_density[label] = (density, members)
            
            # Sort clusters by density (dense first)
            sorted_clusters = sorted(cluster_density.items(), key=lambda x: -x[1][0])
            order = []
            for label, (density, members) in sorted_clusters:
                # Within cluster, sort by degree (hubs first)
                members_sorted = sorted(members, key=lambda x: degrees[x], reverse=True)
                order.extend(members_sorted)
        elif attempt == 5:
            # Anti-cluster: sparse clusters first
            cluster_density = {}
            for label in set(labels):
                members = [small_elements[i] for i in range(n_small) if labels[i] == label]
                density = 0
                for i in range(len(members)):
                    for j in range(i+1, len(members)):
                        density += conflict_graph[members[i], members[j]]
                cluster_density[label] = (density, members)
            
            sorted_clusters = sorted(cluster_density.items(), key=lambda x: x[1][0])
            order = []
            for label, (density, members) in sorted_clusters:
                members_sorted = sorted(members, key=lambda x: degrees[x])
                order.extend(members_sorted)
        else:
            # Random with cluster bias
            order = list(range(99, 0, -1))
            random.shuffle(order)
            # With some probability, bias toward low-degree elements
            if random.random() < 0.3:
                order.sort(key=lambda x: degrees[x])
        
        cur = set(range(100, N+1))
        
        for cand in order:
            if time.time() > deadline - 12.0:
                break
            if can_add(cur, cand, triples, triples_by_elem):
                cur.add(cand)
        
        if len(cur) > best_size:
            best_set = set(cur)
            best_size = len(cur)
    
    # Local search: try to swap elements to improve
    improved = True
    while improved and time.time() < deadline - 15.0:
        improved = False
        cur = set(best_set)
        for cand in range(1, 100):
            if time.time() > deadline - 12.0:
                break
            if cand in cur:
                continue
            # Find all triples that would be completed by adding cand
            blocking_triples = []
            for tidx in triples_by_elem[cand]:
                (a, b, c) = triples[tidx]
                others = [x for x in (a, b, c) if x != cand]
                if others[0] in cur and others[1] in cur:
                    blocking_triples.append((a, b, c))
            
            if not blocking_triples:
                cur.add(cand)
                improved = True
                continue
            
            # Try removing one element from the union of blocking triples
            removable = set()
            for (a, b, c) in blocking_triples:
                for x in (a, b, c):
                    if x != cand:
                        removable.add(x)
            
            for rem in removable:
                if rem not in cur:
                    continue
                # Quick check: does removing rem fix all blocking triples?
                all_fixed = True
                for (a, b, c) in blocking_triples:
                    others = [x for x in (a, b, c) if x != cand]
                    if rem not in others:
                        all_fixed = False
                        break
                if not all_fixed:
                    continue
                
                test_set = cur - {rem}
                if can_add(test_set, cand, triples, triples_by_elem):
                    best_set = test_set | {cand}
                    cur = set(best_set)
                    improved = True
                    break
    
    return list(best_set)

