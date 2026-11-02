"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11
import time

# EVOLVE-BLOCK-START
import numpy as np
import itertools
from scipy.optimize import linprog

def _all_weight_vectors():
    """Generate all nonzero integer vectors in R^11 with entries in {-1,0,1}."""
    vecs = []
    # We'll iterate over all possible patterns of nonzero entries.
    # For each subset of positions, assign +1 or -1 to each.
    # But that's 2^11 * 2^k which is too many. Let's be smarter.
    # Actually, the kissing number problem in dimension 11 is known to be 506 (or nearby).
    # The vectors that form kissing configurations typically have small norms.
    # Let's generate vectors with L2 norm squared up to some limit.
    
    # Start with vectors of norm squared 2 (two non-zero entries of ±1)
    # and norm squared 1 (one non-zero entry of ±1)
    # and maybe norm squared 3, 4, etc.
    
    # For norm squared 1: 11*2 = 22 vectors
    # For norm squared 2: C(11,2)*4 = 55*4 = 220 vectors
    # For norm squared 3: C(11,3)*8 = 165*8 = 1320 vectors
    # For norm squared 4: could be 4 ones, or one 2 and... but entries are only -1,0,1
    # So norm squared 4: C(11,4)*16 = 330*16 = 5280 vectors
    
    # Let's generate all vectors with entries in {-1,0,1} and norm squared <= 4
    # Total: 22 + 220 + 1320 + 5280 = 6842 vectors
    
    for k in range(1, 5):  # 1 to 4 non-zero entries
        for positions in itertools.combinations(range(D), k):
            for signs in itertools.product([1, -1], repeat=k):
                v = [0] * D
                for pos, sign in zip(positions, signs):
                    v[pos] = sign
                vecs.append(v)
    return np.array(vecs, dtype=np.int64)


def _conflict_matrix(V):
    """
    Build conflict matrix: conflict[i,j] = True if vectors i and j conflict.
    Two vectors conflict if their distance < max_norm.
    max_norm = max of all vector norms.
    """
    n = len(V)
    norms_sq = np.sum(V ** 2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Distance squared between i and j: ||Vi - Vj||^2 = ||Vi||^2 + ||Vj||^2 - 2*Vi.Vj
    # Conflict if ||Vi - Vj||^2 < max_norm_sq
    # i.e., ||Vi||^2 + ||Vj||^2 - 2*Vi.Vj < max_norm_sq
    
    # Compute dot product matrix
    dot = V @ V.T
    
    # dist_sq[i,j] = norms_sq[i] + norms_sq[j] - 2*dot[i,j]
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dot
    
    # Conflict if dist_sq < max_norm_sq (strictly less than)
    # Note: dist_sq[i,i] = 0, so diagonal is always "conflict" but we exclude i==j
    conflict = dist_sq < max_norm_sq
    np.fill_diagonal(conflict, False)
    
    return conflict


def _solve_max_independent_set(conflict, time_limit=100):
    """
    Solve maximum independent set using LP relaxation with branch and bound.
    Actually, let's use a simpler approach: greedy + local search with simulated annealing.
    """
    n = len(conflict)
    adj = conflict  # adj[i,j] = True means i and j cannot both be in the set
    
    # Greedy initialization
    selected = set()
    remaining = set(range(n))
    while remaining:
        # Pick vertex with minimum degree in remaining subgraph
        best = None
        best_deg = float('inf')
        for v in remaining:
            deg = sum(1 for u in remaining if u != v and adj[v, u])
            if deg < best_deg:
                best_deg = deg
                best = v
        selected.add(best)
        remaining.discard(best)
        # Remove neighbors
        to_remove = [u for u in remaining if adj[best, u]]
        for u in to_remove:
            remaining.discard(u)
    
    best_set = set(selected)
    best_size = len(best_set)
    
    # Local search: try to add vertices, remove conflicting ones
    # Simulated annealing style
    import random
    random.seed(42)
    
    current_set = set(best_set)
    current_size = len(current_set)
    
    # Precompute adjacency lists
    adj_list = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                adj_list[i].add(j)
    
    def can_add(v, current):
        return all(u not in current for u in adj_list[v])
    
    # Try to grow the set
    for _ in range(5000):
        # Pick a random vertex not in current set
        candidates = [v for v in range(n) if v not in current_set]
        if not candidates:
            break
        v = random.choice(candidates)
        if can_add(v, current_set):
            current_set.add(v)
            if len(current_set) > best_size:
                best_set = set(current_set)
                best_size = len(current_set)
        else:
            # Try to swap: remove one conflicting vertex, add v
            conflicts = [u for u in current_set if adj[v, u]]
            if conflicts:
                # Try removing a random conflict and adding v
                u = random.choice(conflicts)
                # Check if after removing u, v can be added and we don't lose more than we gain
                new_set = current_set - {u} | {v}
                # Verify no conflicts in new_set
                valid = True
                for a in new_set:
                    for b in new_set:
                        if a < b and adj[a, b]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    current_set = new_set
                    if len(current_set) > best_size:
                        best_set = set(current_set)
                        best_size = len(current_set)
    
    return best_set


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s
    
    # Generate all candidate vectors
    V = _all_weight_vectors()
    n = len(V)
    
    # Build conflict matrix
    conflict = _conflict_matrix(V)
    
    # Solve maximum independent set
    selected = _solve_max_independent_set(conflict, time_limit=100)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 130:
        # Return what we have
        result = V[list(selected)]
        if len(result) == 0:
            # Fallback to simple construction
            vectors = []
            for i, j in itertools.combinations(range(D), 2):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = [0] * D
                        v[i], v[j] = si, sj
                        vectors.append(v)
            result = np.array(vectors, dtype=np.int64)
        return result
    
    result = V[list(selected)]
    if len(result) == 0:
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        result = np.array(vectors, dtype=np.int64)
    
    return result
# EVOLVE-BLOCK-END
