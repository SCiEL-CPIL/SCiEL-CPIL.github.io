"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph
from scipy.optimize import linprog

D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
from scipy.optimize import linprog
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Generate all integer vectors in Z^11 with norm squared = 4
    # These are vectors with exactly two non-zero entries, each +/-1
    # OR vectors with one non-zero entry of +/-2 (but norm squared = 4 means 2^2=4)
    # Wait: norm squared = 4. So sum of squares = 4.
    # Options:
    # 1. Two entries are +/-1, rest 0 (1^2 + 1^2 = 2) -> NO, that's norm squared 2
    # 2. One entry is +/-2, rest 0 (2^2 = 4) -> YES
    # 3. Four entries are +/-1, rest 0 (1+1+1+1 = 4) -> YES
    # 4. One entry +/-2 and... no, 4 + anything > 4
    # 5. Two entries: one +/-1 and one +/-? 1 + x^2 = 4 => x^2 = 3, not integer
    # So: type A: one coordinate is +/-2, rest 0. Count: 11 * 2 = 22
    #     type B: four coordinates are +/-1, rest 0. Count: C(11,4) * 2^4 = 330 * 16 = 5280
    # Total: 5302 vectors
    
    # Actually, let me reconsider. The task says "minimum pairwise distance is at least the maximum vector norm"
    # and "Normalizing each vector to norm 2 certifies a kissing configuration"
    # So we want vectors of equal norm, and pairwise distance >= norm.
    # If all vectors have norm r, then pairwise distance >= r.
    # Distance between u and v: ||u-v|| >= r
    # ||u-v||^2 = ||u||^2 + ||v||^2 - 2u.v = 2r^2 - 2u.v >= r^2
    # So 2r^2 - 2u.v >= r^2 => r^2 >= 2u.v => u.v <= r^2/2
    # For integer vectors with norm squared = 4 (so r=2), we need u.v <= 2.
    # But wait, the current seed uses norm squared 2 vectors (two +/-1s).
    # Let me re-read: "minimum pairwise distance is at least the maximum vector norm"
    # If max norm is R, then min distance >= R.
    # For normalized to norm 2: each vector has norm 2. Distance >= 2.
    # ||u-v||^2 >= 4 => 4 + 4 - 2u.v >= 4 => 4 >= 2u.v => u.v <= 2.
    
    # So we want integer vectors of norm 2 (norm squared 4) with pairwise dot product <= 2.
    # But actually, any set of vectors where min pairwise distance >= max norm works.
    # Let's just use norm squared 4 vectors for simplicity, and require u.v <= 2 for all pairs.
    
    # Generate all vectors with norm squared = 4
    vectors = []
    idx = 0
    
    # Type A: one coordinate is +/-2
    for i in range(D):
        for sign in (1, -1):
            v = [0] * D
            v[i] = 2 * sign
            vectors.append(v)
    
    # Type B: four coordinates are +/-1
    for coords in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = [0] * D
            for c, s in zip(coords, signs):
                v[c] = s
            vectors.append(v)
    
    n = len(vectors)
    vectors = np.array(vectors, dtype=np.int64)
    
    # Compute dot product matrix
    # Dot[i,j] = vectors[i] . vectors[j]
    # We need Dot[i,j] <= 2 for all i != j in our selected set
    # And Dot[i,i] = 4 for all i
    
    # For branch and bound, we need a conflict graph where edge (i,j) exists if Dot[i,j] > 2
    # We want maximum independent set in this conflict graph
    
    # Compute conflict matrix
    # This is an n x n matrix, n ~ 5302
    # Dot product matrix computation: vectors @ vectors.T
    # This is 5302 x 11 @ 11 x 5302 = manageable
    
    dot = vectors @ vectors.T
    # Conflict: dot[i,j] > 2 for i != j
    # Note: dot[i,i] = 4, so we mask the diagonal
    conflict = (dot > 2).astype(np.int8)
    np.fill_diagonal(conflict, 0)
    
    # Now we need maximum independent set in the graph defined by conflict matrix
    # This is equivalent to maximum clique in the complement graph
    # Or we can use branch and bound directly on independent set
    
    # For branch and bound with LP relaxation:
    # The LP relaxation of maximum independent set:
    # Maximize sum(x_i)
    # Subject to: x_i + x_j <= 1 for each edge (i,j)
    #             0 <= x_i <= 1
    #
    # This is a large LP with ~n variables and ~E constraints
    # n ~ 5302, E could be large
    
    # Let's estimate E: for each pair, probability dot > 2
    # For random vectors of this type, dot products are small
    # Let's just compute it and see
    
    # Get edge list
    edges = np.argwhere(conflict == 1)
    # Remove duplicates (i,j) and (j,i)
    edges = edges[edges[:, 0] < edges[:, 1]]
    m = len(edges)
    
    # For branch and bound, we'll use a recursive approach with LP bounds
    # But LP on 5302 variables and potentially many constraints is expensive
    # Let's use a greedy heuristic first, then try to improve
    
    # Greedy: sort by degree (ascending), pick vertex, remove its neighbors
    degrees = conflict.sum(axis=1)
    order = np.argsort(degrees)
    
    selected = []
    remaining = set(range(n))
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    
    for v in order:
        if v in remaining:
            selected.append(v)
            remaining.discard(v)
            for u in adj[v]:
                remaining.discard(u)
    
    greedy_size = len(selected)
    
    # Now try local search / improvement
    # Start with greedy solution, try to add more or swap
    
    # Actually, let's try a different approach: use the fact that this is a specific structured problem
    # The kissing number in dimension 11 is known to be 506 (from literature)
    # But we need to actually construct it
    
    # Let's try a smarter greedy: use weighted greedy where weight is 1/degree
    # Or try multiple random orderings
    
    best_selected = set(selected)
    best_size = greedy_size
    
    # Try several random perturbations of the greedy
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        
        # Random ordering biased toward low degree
        weights = 1.0 / (degrees + 1)
        # Sample without replacement
        order_rand = np.random.choice(n, size=n, replace=False, p=weights / weights.sum())
        
        selected_trial = []
        remaining = set(range(n))
        
        for v in order_rand:
            if v in remaining:
                selected_trial.append(v)
                remaining.discard(v)
                for u in adj[v]:
                    remaining.discard(u)
        
        if len(selected_trial) > best_size:
            best_size = len(selected_trial)
            best_selected = set(selected_trial)
    
    # Local search: try to improve by swapping
    selected_list = list(best_selected)
    selected_set = set(selected_list)
    
    # Try adding a non-selected vertex if it doesn't conflict
    # And removing a selected vertex that conflicts with many non-selected
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Try to add vertices
        for v in range(n):
            if v in selected_set:
                continue
            # Check if v conflicts with any selected vertex
            if not any(u in selected_set for u in adj[v]):
                selected_set.add(v)
                improved = True
        
        # Try swaps: remove one selected, add two non-selected
        # This is expensive, skip for now
    
    best_size = len(selected_set)
    best_selected = list(selected_set)
    
    if time.time() > deadline - 5:
        # Return best found
        if len(best_selected) == 0:
            # Fallback: return just the type A vectors (22 of them)
            return vectors[:22]
        return vectors[best_selected]
    
    # Return the best independent set found
    return vectors[best_selected]
# EVOLVE-BLOCK-END
