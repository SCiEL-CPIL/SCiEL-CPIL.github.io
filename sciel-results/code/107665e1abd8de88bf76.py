"""Packing trees T1..T20 into K20."""
import time
import random
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def _edge_set(tree_edges):
    """Return a frozenset of frozensets representing the edges of a tree."""
    return frozenset(frozenset(e) for e in tree_edges)

def _generate_embedding(tree, n=20, rng=None):
    """
    Generate a random embedding of a tree into K_n.
    Returns a list of length k where position i is the host vertex for tree vertex i.
    """
    if rng is None:
        rng = random
    k = len(tree) + 1 if tree else 1
    # For tree with k vertices, we need k distinct host vertices
    if k == 1:
        return [rng.randrange(n)]
    # Pick k distinct vertices
    host_vertices = rng.sample(range(n), k)
    # Assign tree vertices to host vertices via a random permutation
    perm = list(range(k))
    rng.shuffle(perm)
    # embedding[i] = host_vertices[perm[i]]
    embedding = [host_vertices[perm[i]] for i in range(k)]
    return embedding

def _embedding_edges(embedding, tree_edges):
    """
    Given an embedding (list mapping tree vertex -> host vertex) and tree edges,
    return the set of host edges (as frozensets) used by this embedding.
    """
    edges = set()
    for (u, v) in tree_edges:
        hu, hv = embedding[u], embedding[v]
        edges.add(frozenset((hu, hv)))
    return edges

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin
    
    trees = instance_trees(20, 611)
    # trees[k-1] is the edge list for T_k (k vertices)
    # T1: 0 edges, T2: 1 edge, ..., T20: 19 edges
    
    # Total edges available in K20
    total_edges = 20 * 19 // 2  # 190
    
    # We'll embed trees from largest to smallest to reduce search space
    # Order: T20, T19, ..., T1
    order = list(range(19, -1, -1))  # indices 19 down to 0, i.e., T20 to T1
    
    # For each tree, precompute its edge count
    tree_edge_counts = [len(t) for t in trees]
    
    # Best solution found so far
    best_solution = [None] * 20
    best_score = 0
    
    # We'll use a randomized backtracking search
    # For each tree, we'll try multiple random embeddings
    
    def search(pos, used_edges, current_solution, current_score, rng):
        nonlocal best_solution, best_score, deadline, start_time
        
        if time.time() > deadline:
            return
        
        if pos == len(order):
            # All trees processed
            if current_score > best_score:
                best_score = current_score
                best_solution = list(current_solution)
            return
        
        tree_idx = order[pos]
        tree = trees[tree_idx]
        k = tree_idx + 1  # number of vertices
        
        if k == 1:
            # T1 has no edges, always embeddable
            current_solution[tree_idx] = [0]  # arbitrary
            search(pos + 1, used_edges, current_solution, current_score, rng)
            current_solution[tree_idx] = None
            return
        
        # Try random embeddings for this tree
        max_attempts = 200 if k <= 10 else 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return
            
            embedding = _generate_embedding(tree, 20, rng)
            emb_edges = _embedding_edges(embedding, tree)
            
            # Check if any edge conflicts with used_edges
            conflict = False
            for e in emb_edges:
                if e in used_edges:
                    conflict = True
                    break
            
            if not conflict:
                # This embedding is valid
                new_used = used_edges | emb_edges
                current_solution[tree_idx] = embedding
                search(pos + 1, new_used, current_solution, current_score + len(tree), rng)
                current_solution[tree_idx] = None
        
        # If no embedding worked, this branch fails (backtrack)
    
    # Run multiple random restarts
    rng = random.Random(42)
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        current_solution = [None] * 20
        current_score = 0
        search(0, set(), current_solution, current_score, rng)
        
        if best_score == total_edges:
            break
    
    # Ensure the result is properly formatted
    result = [None] * 20
    for i in range(20):
        if best_solution[i] is not None:
            # Convert to list of Python ints
            result[i] = [int(x) for x in best_solution[i]]
    
    return result
