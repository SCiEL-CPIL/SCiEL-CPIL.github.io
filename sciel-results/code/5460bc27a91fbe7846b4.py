import time
import random
import numpy as np

N = 64
TOTAL = N * N

# Precompute squared distances
xs = np.repeat(np.arange(N), N)
ys = np.tile(np.arange(N), N)
dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
dist2 = (dx * dx + dy * dy).astype(np.int32)

# Build conflict graph
# Two points conflict if they are part of ANY isosceles triangle with a third point.
# We use a bitset representation for the conflict graph to speed up MIS heuristics.
# conflict[i] is a numpy array of bools or a bitset. Given memory, a list of sets or sparse matrix might be better.
# However, for MIS, we need to quickly find neighbors of a node.
# Let's build a list of lists of neighbors.
neighbors = [[] for _ in range(TOTAL)]

# Optimization: We can compute conflicts more efficiently.
# Point p and q conflict if there exists r such that (p,r) and (q,r) have same dist, 
# OR (p,q) and (p,r) have same dist, OR (p,q) and (q,r) have same dist.
# This is O(N^3) to compute naively. N=64, N^3 = 262k, which is fine.
# But we can do it smarter.
# For each point r, group other points by distance to r.
# If r is the apex, any two points in the same group conflict with each other (via r).
# This adds edges between points in the same group.
# What if r is a base vertex? 
# If p is apex, then q and r are base. dist(p,q) == dist(p,r).
# So for each p, group other points by dist to p. Any two in same group conflict.
# This covers all cases where the "equal" sides share a vertex.
# Is that all? 
# Isosceles triangle has two equal sides. The vertex where they meet is the apex.
# So yes, for any isosceles triangle, there is an apex. 
# So if we connect any two points that share a common apex with equal distance, we cover all conflicts.
# Wait, do we need to connect p and q if they form a base of an isosceles triangle with apex r?
# Yes, because if p and q are in the set, and we add r, it forms an isosceles triangle.
# So p and q cannot both be in the set IF r is also in the set.
# But the conflict is not just between p and q. It's a hyperedge (p, q, r).
# My MIS formulation above was slightly wrong. "Nodes are grid points, edges connect points that cannot coexist".
# If p and q conflict, it means they cannot be in the SAME independent set?
# No, p and q CAN be in the same set, as long as NO r exists such that (p,q,r) is isosceles.
# So the conflict graph is NOT simply "p and q cannot be together".
# The constraint is: No three points form an isosceles triangle.
# This is a 3-uniform hypergraph independent set problem.
# MIS on a graph is for 2-uniform constraints.
# So I must reformulate.
#
# Revised Strategy:
# The problem is: Find the largest subset S of points such that for any distinct p, q, r in S, they do not form an isosceles triangle.
# This is a 3-SAT-like constraint or a 3-uniform hypergraph independent set.
# Greedy addition (current) works because it checks constraints incrementally.
# The "MIS on conflict graph" idea in the prompt was a bit of a red herring if interpreted strictly as pairwise conflicts, because the conflict is ternary.
# HOWEVER, we can still use a different search method: Local Search / Hill Climbing on the set of points.
#
# New Strategy: 
# 1. Start with a random valid set (using greedy).
# 2. Try to improve it by:
#    a. Adding a random point if it doesn't violate constraints.
#    b. Swapping a point in the set with a point outside, if the swap is valid and maintains size (or increases if we remove one and add two? No, swap is 1-for-1).
#    c. Actually, a better local search for independent sets in hypergraphs:
#       - Try to add a point. If it fails, identify the "conflicting" triples.
#       - Try to remove one point from the conflicting triples and add the new point.
#
# Let's implement a "Greedy + Local Search" approach.
# 1. Greedy construction to get a baseline.
# 2. Local search: 
#    - Maintain the set S.
#    - Precompute for each point p outside S, the list of points in S that conflict with p (i.e., form an isosceles triangle with some other point in S).
#      Actually, we need to know if adding p creates ANY isosceles triangle with TWO points in S.
#      Let's precompute for each pair (u, v) in S, the set of points w that form an isosceles triangle with u and v.
#      This is too expensive to maintain dynamically.
#
# Alternative Local Search:
# 1. Start with empty set.
# 2. Randomly add points (greedy).
# 3. Once stuck, try "kick" moves: Remove k random points from S, then greedily re-add points (including the removed ones and new ones) in a random order.
# This is a form of Randomized Local Search (RLS) or Simulated Annealing.
#
# Let's try:
# 1. Greedy to get a seed set S.
# 2. Repeat until time limit:
#    - Pick a random subset of S to remove (e.g., 5% of points).
#    - Create a list of all points (or a random subset of all points).
#    - Greedily add points from this list to the reduced set S', ensuring validity.
#    - If |S'| > |S|, accept S'.
#
# This is essentially "Random Restart with Perturbation".
# The current program does "Random Restart with Greedy".
# The difference: Current starts from scratch. This one starts from a good solution and perturbs it.
# This allows the search to explore the neighborhood of good solutions rather than just finding random local optima.

def construct():
    start_time = time.time()
    deadline = start_time + 295
    
    best_set = []
    best_size = 0
    
    # Helper: Check if point p can be added to set S
    # Returns True if valid, False otherwise.
    # We need to check if there exist u, v in S such that (p, u, v) is isosceles.
    # This is O(|S|^2) per check.
    # To speed up, we can precompute for each p, the set of pairs (u, v) that form isosceles with p.
    # But S changes.
    
    # Let's optimize the validity check.
    # For a given p and S, we need to ensure no u, v in S form an isosceles triangle with p.
    # This means:
    # 1. No u, v in S such that dist(p, u) == dist(p, v). (p is apex)
    # 2. No u, v in S such that dist(u, v) == dist(p, u). (u is apex)
    # 3. No u, v in S such that dist(u, v) == dist(p, v). (v is apex)
    
    # We can maintain for the current set S:
    # - For each point p, the distances to all points in S.
    # This is hard to maintain incrementally for all p.
    
    # Let's stick to the O(|S|^2) check but make it fast with NumPy.
    # S is a list of indices.
    # dist_matrix = dist2[p, S] gives distances from p to all in S.
    # Check 1: len(set(dist_matrix)) == len(dist_matrix)
    # Check 2 & 3: For each u in S, we need to check if there is v in S such that dist(u, v) == dist(p, u).
    # This is equivalent to: For each u in S, let d_pu = dist(p, u). We need to check if there is v in S (v != u) such that dist(u, v) == d_pu.
    # We can precompute for each u, the set of distances to all other points in the grid.
    # dist_to_all[u] is a set of squared distances from u to all other points.
    # Then for a given p and S:
    # For each u in S:
    #   d_pu = dist2[p, u]
    #   We need to know if there is v in S such that dist2[u, v] == d_pu.
    #   This is a query: "Is there a v in S such that dist2[u, v] == d_pu?"
    #   We can precompute for each u and each distance d, the list of points v such that dist2[u, v] == d.
    #   Let neighbors_by_dist[u][d] = list of v.
    #   Then we check if any v in neighbors_by_dist[u][d_pu] is in S.
    #   Checking "is v in S" is O(1) if S is a set.
    
    # Precomputation:
    # neighbors_by_dist[u] is a dict: distance -> list of points
    # This takes O(N^3) memory? No, O(N^2 * avg_degree).
    # N=64, N^2=4096. Avg degree ~ 4096. Total edges ~ 16M.
    # Storing this might be heavy but doable.
    # Alternatively, just use the O(|S|^2) check with NumPy vectorization.
    # |S| is likely around 10-20. So |S|^2 is 100-400.
    # 5000 restarts * 400 checks * 100 ops = 200M ops. Might be slow in Python.
    
    # Let's try the precomputation approach for speed.
    # neighbors_by_dist[u] = {d: [v1, v2, ...]}
    neighbors_by_dist = [dict() for _ in range(TOTAL)]
    for u in range(TOTAL):
        for v in range(TOTAL):
            if u == v: continue
            d = dist2[u, v]
            if d not in neighbors_by_dist[u]:
                neighbors_by_dist[u][d] = []
            neighbors_by_dist[u][d].append(v)
            
    # Convert lists to sets for O(1) lookup? No, we need to check if ANY v in the list is in S.
    # So we keep lists.
    
    def is_valid_addition(p, S_set, S_list):
        # S_set is a set of indices in S for O(1) lookup.
        # S_list is the list of indices in S.
        if not S_list:
            return True
            
        # Check 1: Distances from p to all in S must be distinct.
        dists_to_S = dist2[p, S_list]
        if len(set(dists_to_S)) != len(dists_to_S):
            return False
            
        # Check 2 & 3: For each u in S, check if there is v in S such that dist(u, v) == dist(p, u).
        for i, u in enumerate(S_list):
            d_pu = dists_to_S[i]
            # Get all v such that dist(u, v) == d_pu
            candidates = neighbors_by_dist[u].get(d_pu, [])
            for v in candidates:
                if v in S_set:
                    # v is in S. 
                    # We must ensure v != u (guaranteed by dist > 0) and v != p (p is not in S yet).
                    # If v is in S, then (p, u, v) forms an isosceles triangle with apex u.
                    return False
                    
        return True

    def greedy_construct(order):
        S = []
        S_set = set()
        for p in order:
            if is_valid_addition(p, S_set, S):
                S.append(p)
                S_set.add(p)
        return S

    # Main Loop: Randomized Local Search
    for attempt in range(5000):
        if time.time() > deadline:
            break
            
        if attempt == 0:
            # First attempt: Pure greedy with random order
            perm = list(range(TOTAL))
            random.shuffle(perm)
            current_set = greedy_construct(perm)
        else:
            # Perturb the best set found so far
            if not best_set:
                perm = list(range(TOTAL))
                random.shuffle(perm)
                current_set = greedy_construct(perm)
            else:
                # Remove a small random subset of best_set
                k = max(1, len(best_set) // 10)
                to_remove = random.sample(best_set, k)
                to_remove_set = set(to_remove)
                current_set = [p for p in best_set if p not in to_remove_set]
                current_set_set = set(current_set)
                
                # Now try to add points. 
                # We prioritize the removed points and random new points.
                # Create a list of candidates: removed points + random subset of all points.
                candidates = list(to_remove)
                # Add some random new points to explore
                new_points = random.sample(list(set(range(TOTAL)) - current_set_set), min(50, TOTAL - len(current_set)))
                candidates.extend(new_points)
                random.shuffle(candidates)
                
                # Greedily add from candidates
                for p in candidates:
                    if p in current_set_set:
                        continue
                    if is_valid_addition(p, current_set_set, current_set):
                        current_set.append(p)
                        current_set_set.add(p)
                        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
    return [(idx // N, idx % N) for idx in best_set]

# Run construct to ensure no syntax errors
if __name__ == "__main__":
    res = construct()
    print(len(res))
