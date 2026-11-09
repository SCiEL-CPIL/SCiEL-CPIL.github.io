"""Construct a minimum maximal no-isosceles set on an N x N grid."""
N = 16

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import sys

def _build_conflict_graph(n):
    """
    Build a conflict graph where an edge (i, j) exists if there exists some k
    such that points i, j, k form an isosceles triangle.
    Actually, for the independent set formulation:
    We want a set S such that no three points in S form an isosceles triangle.
    This is NOT a simple pairwise constraint. It's a 3-uniform hypergraph independent set.
    
    However, we can reformulate: 
    A set S is "no-isosceles" iff for every pair (a, b) in S, there is no c in S 
    such that {a, b, c} is isosceles.
    
    This is hard to make into a binary MIS directly. Let me think differently.
    
    Actually, let's use a different approach: backtracking with pruning.
    We want a maximal set S (no point can be added) that is no-isosceles and as small as possible.
    
    Strategy: 
    1. Precompute for each pair (i, j) the set of points k that form isosceles with i, j.
    2. Use backtracking: try to build a small set S that is maximal.
    
    For maximality: for every point p not in S, there exist a, b in S such that {p, a, b} is isosceles.
    
    Let's use a different strategy: 
    - Start with all points, remove points while maintaining no-isosceles and maximality.
    - Or: search for small sets directly.
    
    Given the time limit (300s), let's use a randomized approach with good pruning:
    1. Precompute isosceles triples or at least isosceles pair conflicts.
    2. Use simulated annealing / local search with careful state tracking.
    
    Actually, let me precompute a 3D boolean array or use a set of forbidden triples.
    """
    n2 = n * n
    idx = lambda x, y: x * n + y
    
    # dist2[i][j] = squared distance between point i and point j
    points = [(x, y) for x in range(n) for y in range(n)]
    dist2 = np.zeros((n2, n2), dtype=np.int32)
    for i, (x1, y1) in enumerate(points):
        for j, (x2, y2) in enumerate(points):
            dist2[i, j] = (x1 - x2) ** 2 + (y1 - y2) ** 2
    
    # For each pair (i, j), precompute which k make {i, j, k} isosceles
    # isosceles_conflict[i][j] = set of k such that dist2(i,j)==dist2(i,k) or 
    #   dist2(i,j)==dist2(j,k) or dist2(i,k)==dist2(j,k)
    
    # For efficiency, let's store for each pair (i,j) a bitset of conflicting k's
    # But n2=256, so we can use a list of sets or numpy arrays.
    
    # Let's precompute: for each triple (i, j, k) with i<j<k, whether it's isosceles
    # isos[i][j][k] = True if isosceles
    # This is 256^3 / 6 ≈ 2.8M entries, feasible.
    
    # Actually, for the local search approach, we need to check:
    # 1. Is S no-isosceles? (no triple in S is isosceles)
    # 2. Is S maximal? (every point outside S is "blocked" by some pair in S)
    
    # Let's precompute for each point p and each pair (a, b): is {p, a, b} isosceles?
    # blocked[p][a][b] = True if p is blocked by pair (a, b)
    # This is 256^3 = 16.7M booleans, feasible with numpy.
    
    # isos[p, a, b] = True if {p, a, b} form isosceles triangle
    isos = np.zeros((n2, n2, n2), dtype=bool)
    for i in range(n2):
        for j in range(i + 1, n2):
            for k in range(j + 1, n2):
                d1 = dist2[i, j]
                d2 = dist2[i, k]
                d3 = dist2[j, k]
                if d1 == d2 or d1 == d3 or d2 == d3:
                    isos[i, j, k] = True
                    isos[i, k, j] = True
                    isos[j, i, k] = True
                    isos[j, k, i] = True
                    isos[k, i, j] = True
                    isos[k, j, i] = True
    
    return points, dist2, isos

def construct():
    n = N
    n2 = n * n
    points, dist2, isos = _build_conflict_graph(n)
    
    import random
    rng = random.Random(42)
    
    def is_no_isosceles(S):
        """Check if set S (list of indices) has no isosceles triple."""
        S_set = set(S)
        for i, a in enumerate(S):
            for b in S[i+1:]:
                # Check if there exists c in S such that isos[a][b][c]
                for c in S:
                    if c != a and c != b and isos[a, b, c]:
                        return False
        return True
    
    def is_maximal(S, n2, isos):
        """Check if S is maximal: every point not in S is blocked by some pair in S."""
        S_set = set(S)
        S_list = list(S_set)
        for p in range(n2):
            if p in S_set:
                continue
            # Check if p is blocked: exists a, b in S such that isos[p][a][b]
            blocked = False
            for i, a in enumerate(S_list):
                for b in S_list[i+1:]:
                    if isos[p, a, b]:
                        blocked = True
                        break
                if blocked:
                    break
            if not blocked:
                return False
        return True
    
    def count_isosceles_triples(S):
        """Count number of isosceles triples in S."""
        count = 0
        S_set = set(S)
        for i, a in enumerate(S):
            for b in S[i+1:]:
                for c in S:
                    if c != a and c != b and isos[a, b, c]:
                        count += 1
        return count // 6  # Each triple counted 6 times
    
    # Strategy: Start with a large valid set, then try to shrink it.
    # Greedy construction in random order
    def greedy_construct(order):
        S = []
        for p in order:
            # Check if adding p keeps S no-isosceles
            valid = True
            for i, a in enumerate(S):
                for b in S[i+1:]:
                    if isos[p, a, b]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S.append(p)
        return S
    
    # Try multiple random orderings
    best = None
    best_size = n2 + 1
    
    for seed in range(200):
        rng2 = random.Random(seed)
        order = list(range(n2))
        rng2.shuffle(order)
        S = greedy_construct(order)
        
        if len(S) < best_size:
            best = S
            best_size = len(S)
        
        # Local search: try to remove points
        S = list(S)
        for _ in range(50):  # iterations
            progress = False
            for i in range(len(S)):
                candidate = S[:i] + S[i+1:]
                if not is_no_isosceles(candidate):
                    continue
                if is_maximal(candidate, n2, isos):
                    S = candidate
                    best_size = len(S)
                    if len(S) < best_size:
                        best = list(S)
                    progress = True
                    break
            if not progress:
                break
        
        if len(S) < best_size:
            best = list(S)
            best_size = len(S)
    
    # Final pass: try to remove more
    if best is None:
        best = list(range(n2))  # fallback
    else:
        for _ in range(1000):
            progress = False
            for i in range(len(best)):
                candidate = best[:i] + best[i+1:]
                if not is_no_isosceles(candidate) or not is_maximal(candidate, n2, isos):
                    continue
                best = candidate
                progress = True
                break
            if not progress:
                break
    
    # Convert back to (x, y) pairs
    result = [points[i] for i in best]
    return result

# EVOLVE-BLOCK-END
