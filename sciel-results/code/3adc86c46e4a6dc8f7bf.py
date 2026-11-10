# EVOLVE-BLOCK-START
import numpy as np
import random
import math


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 (F_7[alpha], alpha^2 = 3).

    The Nikodym property requires that for every point x in F_49^2, there is a
    direction v = (1, m) or v = (0, 1) such that all other points on the line
    through x in direction v belong to N.

    We aim to maximize the complement S = F_49^2 \ N (the excluded set).
    This is formulated as a Maximum Independent Set problem on a conflict graph,
    solved via a greedy algorithm with random restarts and blocked-array optimization.

    Returns:
        np.ndarray of shape (k, 2, 2) with dtype int64, listing points of N.
    """
    p = 7
    k = 3  # alpha^2 = 3 in F_7

    # Total number of points
    n_total = p * p * p * p  # 2401

    # Precompute all points as flat indices 0..2400
    # Index = ((xa * p + xb) * p + ya) * p + yb
    def to_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    def to_coords(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        return xa, xb, ya, yb

    # Field operations
    def add_field(a1, b1, a2, b2):
        return ((a1 + a2) % p, (b1 + b2) % p)

    def mul_field(a1, b1, a2, b2):
        # (a1 + b1*alpha)(a2 + b2*alpha) = a1*a2 + b1*b2*k + (a1*b2 + b1*a2)*alpha
        ra = (a1 * a2 + b1 * b2 * k) % p
        rb = (a1 * b2 + b1 * a2) % p
        return ra, rb

    # Directions: (0, 1) and (1, m) for m in F_49
    directions = []
    directions.append(((0, 0), (0, 1)))  # vertical
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))

    n_dirs = len(directions)  # 50

    # Precompute neighbors for each point
    # neighbors[i] is a list of indices j such that i and j are on a common line
    # We need to build the conflict graph: edge (i, j) if i and j are collinear.
    # Note: If i and j are on the same line, they conflict.
    
    neighbors = [[] for _ in range(n_total)]
    
    # Efficient way to build neighbors:
    # For each line, all pairs of points on that line are neighbors.
    # Instead of iterating all points and all directions (which creates duplicates),
    # we can iterate over all unique lines.
    # However, since N is small (2401), we can just iterate point -> direction -> others,
    # and add to a set, then convert to list.
    
    # To save time, let's just build the adjacency list directly.
    # For each point i, and each direction d, find all other points j on the line.
    # Add j to neighbors[i] and i to neighbors[j].
    
    # We use a set for each point to avoid duplicate neighbors
    neighbor_sets = [set() for _ in range(n_total)]
    
    for i in range(n_total):
        xa, xb, ya, yb = to_coords(i)
        for d_idx in range(n_dirs):
            (dx_a, dx_b), (dy_a, dy_b) = directions[d_idx]
            # Iterate t in F_49 \ {0}
            for t_a in range(p):
                for t_b in range(p):
                    if t_a == 0 and t_b == 0:
                        continue
                    tx_a, tx_b = mul_field(t_a, t_b, dx_a, dx_b)
                    ty_a, ty_b = mul_field(t_a, t_b, dy_a, dy_b)
                    nxa = (xa + tx_a) % p
                    nx_b = (xb + tx_b) % p
                    nya = (ya + ty_a) % p
                    ny_b = (yb + ty_b) % p
                    j = to_idx(nxa, nx_b, nya, ny_b)
                    neighbor_sets[i].add(j)
                    
    # Convert sets to lists for faster iteration
    for i in range(n_total):
        neighbors[i] = list(neighbor_sets[i])

    # Greedy MIS with random restarts
    # Strategy:
    # 1. Maintain a 'blocked' boolean array.
    # 2. Maintain a list of 'available' points (indices not blocked).
    # 3. While available is not empty:
    #    a. Pick a point from available (random or heuristic).
    #    b. Add to S.
    #    c. Mark it and its neighbors as blocked.
    #    d. Remove blocked points from available.
    
    # To make it faster, we can use a "lazy deletion" or just rebuild the available list?
    # Rebuilding is O(N). N=2401. Number of steps in MIS is at most N.
    # Total O(N^2) per run. 2401^2 ~ 5.7M ops.
    # We can do many restarts within 600s.
    
    best_S = set()
    n_starts = 200 # Increase restarts
    rng = random.Random(42)
    
    # Precompute indices for faster access?
    # Let's use a simple greedy:
    # At each step, pick the point with the smallest degree in the remaining graph?
    # That's expensive to update.
    # Let's stick to random pick from available, but shuffle the order initially?
    # Or just pick randomly from available each time.
    
    # Optimization: Instead of maintaining a dynamic list of available points which requires
    # filtering, we can just iterate through a shuffled list and skip blocked ones?
    # No, that's O(N) per step, total O(N^2). Same complexity.
    # But "pick random from available" requires knowing what is available.
    # Let's use a list `available` and remove blocked items.
    # Removing from a list is O(N).
    
    # Alternative: Use a boolean array `is_available`.
    # Pick a random index from 0..N-1, check if is_available[idx]. If yes, select it.
    # If no, try again.
    # Expected number of tries is N / |available|.
    # This is efficient.
    
    for start in range(n_starts):
        S = set()
        is_available = [True] * n_total
        available_count = n_total
        
        while available_count > 0:
            # Find a random available point
            # This loop might take a few iterations if available_count is small
            while True:
                idx = rng.randint(0, n_total - 1)
                if is_available[idx]:
                    break
            
            # Select idx
            S.add(idx)
            is_available[idx] = False
            available_count -= 1
            
            # Block neighbors
            for nb in neighbors[idx]:
                if is_available[nb]:
                    is_available[nb] = False
                    available_count -= 1
        
        if len(S) > len(best_S):
            best_S = S.copy()

    # Construct N = complement of S
    n_points = n_total - len(best_S)
    points = []
    for i in range(n_total):
        if i not in best_S:
            xa, xb, ya, yb = to_coords(i)
            points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
