"""Rewritten program using spectral initialization + multi-restart local search."""
import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    d = 7
    n = 2 ** d
    m = d * 2 ** (d - 1)  # 448
    
    # Generate all edges
    edges = []
    edge_to_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_to_idx[(x, y)] = idx
    
    # Generate all faces (2-dimensional squares)
    # Each face is defined by a pair of dimensions (i, j) and assignments to other bits
    # A face has 4 edges
    face_set = set()
    for x in range(n):
        for i in range(d):
            for j in range(i + 1, d):
                y = x ^ (1 << i)
                z = x ^ (1 << j)
                w = x ^ (1 << i) ^ (1 << j)
                e1 = (min(x, y), max(x, y))
                e2 = (min(y, w), max(y, w))
                e3 = (min(w, z), max(w, z))
                e4 = (min(z, x), max(z, x))
                idxs = tuple(sorted([edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4]]))
                face_set.add(frozenset(idxs))
    
    faces = list(face_set)
    num_faces = len(faces)
    face_edges = [list(f) for f in faces]
    
    # Build edge_faces: for each edge, which faces contain it
    edge_faces = [[] for _ in range(m)]
    for fi, f in enumerate(face_edges):
        for ei in f:
            edge_faces[ei].append(fi)
    
    # For each face, precompute the 4 edge indices
    # face_edges[fi] = [e1, e2, e3, e4]
    
    # Helper: count selected edges in a face
    def face_count(selected, fi):
        c = 0
        for ei in face_edges[fi]:
            if selected[ei]:
                c += 1
        return c
    
    # Helper: check if adding edge ei would create a violation
    def would_violate(selected, ei):
        for fi in edge_faces[ei]:
            if face_count(selected, fi) == 3:
                return True
        return False
    
    # Spectral initialization
    # Build a conflict graph where two edges conflict if they share a face
    # Actually, we want to maximize edges such that no face has all 4 edges.
    # This is equivalent to: for each face, at least one edge must be absent.
    # We can model this as a maximum independent set problem in a hypergraph,
    # but let's try a different approach: use the fact that the constraint is local.
    
    # Alternative: use a weighted approach where we assign weights to edges
    # and try to maximize the sum subject to constraints.
    
    # Let's try a simple spectral method:
    # Build a matrix A where A[i][j] = 1 if edges i and j share a face, 0 otherwise.
    # The Laplacian L = D - A. The Fiedler vector (second eigenvector) gives a partition.
    # We can use this to initialize a solution.
    
    # Build adjacency matrix
    A = np.zeros((m, m), dtype=np.int8)
    for fi, f in enumerate(face_edges):
        for a in range(4):
            for b in range(a + 1, 4):
                A[f[a]][f[b]] = 1
                A[f[b]][f[a]] = 1
    
    # Compute degree
    deg = A.sum(axis=1)
    
    # Laplacian
    L = np.diag(deg.astype(np.float64)) - A.astype(np.float64)
    
    # Compute eigenvectors
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Fiedler vector is the second smallest eigenvalue's eigenvector
        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        fiedler = eigenvectors[:, idx[1]]
        
        # Threshold at median to get a partition
        threshold = np.median(fiedler)
        init_selected = (fiedler > threshold).astype(np.int8)
    except:
        init_selected = np.ones(m, dtype=np.int8)
    
    # Repair the initialization: remove edges that violate constraints
    # We need to ensure no face has all 4 edges selected.
    # Greedy repair: for each face with 4 selected edges, remove one.
    for fi in range(num_faces):
        if face_count(init_selected, fi) == 4:
            # Remove a random edge from this face
            candidates = [ei for ei in face_edges[fi] if init_selected[ei] == 1]
            if candidates:
                ei = random.choice(candidates)
                init_selected[ei] = 0
    
    # Now run local search / simulated annealing
    best_selected = init_selected.copy()
    best_count = int(init_selected.sum())
    
    # Multi-restart local search
    num_restarts = 20
    steps_per_restart = 50000
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # For the first restart, use spectral init. For others, random or perturbation.
        if restart == 0:
            selected = best_selected.copy()
        elif restart < 5:
            # Perturbation of best
            selected = best_selected.copy()
            # Randomly flip some edges
            for _ in range(random.randint(10, 50)):
                ei = random.randint(0, m - 1)
                selected[ei] = 1 - selected[ei]
            # Repair
            for fi in range(num_faces):
                if face_count(selected, fi) == 4:
                    candidates = [ei for ei in face_edges[fi] if selected[ei] == 1]
                    if candidates:
                        ei = random.choice(candidates)
                        selected[ei] = 0
        else:
            # Random restart
            selected = np.random.randint(0, 2, m, dtype=np.int8)
            # Repair
            for fi in range(num_faces):
                if face_count(selected, fi) == 4:
                    candidates = [ei for ei in face_edges[fi] if selected[ei] == 1]
                    if candidates:
                        ei = random.choice(candidates)
                        selected[ei] = 0
        
        count = int(selected.sum())
        
        # Local search: hill climbing with occasional restarts
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            # Pick a random edge
            ei = random.randint(0, m - 1)
            
            if selected[ei] == 1:
                # Try to remove
                selected[ei] = 0
                delta = -1
            else:
                # Try to add
                if would_violate(selected, ei):
                    # Reject
                    continue
                selected[ei] = 1
                delta = 1
            
            if delta > 0:
                count += 1
                if count > best_count:
                    best_count = count
                    best_selected = selected.copy()
            else:
                # Accept with probability exp(delta / T)
                if random.random() < math.exp(delta / T):
                    count += delta
                else:
                    # Revert
                    selected[ei] = 1 - selected[ei]
            
            T *= cooling
        
        if time.time() > deadline:
            break
    
    # Final verification and return
    out = []
    for ei in range(m):
        if best_selected[ei] == 1:
            u, v = edges[ei]
            out.append([u, v])
    
    return out
