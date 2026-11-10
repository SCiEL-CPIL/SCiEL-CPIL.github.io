"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import itertools

def is_perfect_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def dist_sq(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def are_collinear(p1, p2, p3):
    # (p2-p1) x (p3-p1) == 0
    dx1 = p2[0] - p1[0]
    dy1 = p2[1] - p1[1]
    dx2 = p3[0] - p1[0]
    dy2 = p3[1] - p1[1]
    return dx1 * dy2 - dy1 * dx2 == 0

def are_concyclic(p1, p2, p3, p4):
    # Determinant of circle equation
    # | x^2+y^2  x  y  1 |
    # | x^2+y^2  x  y  1 | = 0
    # | x^2+y^2  x  y  1 |
    # | x^2+y^2  x  y  1 |
    
    def det4(m):
        # m is 4x4 list of lists
        # Use fraction-free or just integer arithmetic since inputs are integers
        # Cofactor expansion along first row
        res = 0
        for j in range(4):
            sign = 1 if j % 2 == 0 else -1
            # Minor 3x3
            m1 = [row[:j] + row[j+1:] for row in m[1:]]
            # det3 for m1
            d = (m1[0][0]*(m1[1][1]*m1[2][2] - m1[1][2]*m1[2][1]) -
                 m1[0][1]*(m1[1][0]*m1[2][2] - m1[1][2]*m1[2][0]) +
                 m1[0][2]*(m1[1][0]*m1[2][1] - m1[1][1]*m1[2][0]))
            res += sign * m[0][j] * d
        return res

    r1 = [p1[0]*p1[0] + p1[1]*p1[1], p1[0], p1[1], 1]
    r2 = [p2[0]*p2[0] + p2[1]*p2[1], p2[0], p2[1], 1]
    r3 = [p3[0]*p3[0] + p3[1]*p3[1], p3[0], p3[1], 1]
    r4 = [p4[0]*p4[0] + p4[1]*p4[1], p4[0], p4[1], 1]
    
    return det4([r1, r2, r3, r4]) == 0

def construct():
    # Strategy: Find a clique of size 6 in the graph of integer lattice points
    # where edges connect points with integer distance.
    # Then filter for no 3 collinear and no 4 concyclic.
    
    # 1. Generate candidates
    # A bound of 100-150 is usually sufficient for small cliques.
    # Let's try a slightly larger bound to increase density but keep it manageable.
    bound = 80
    candidates = []
    for x in range(-bound, bound + 1):
        for y in range(-bound, bound + 1):
            candidates.append((x, y))
            
    n = len(candidates)
    
    # 2. Precompute adjacency (integer distance)
    # adj[i] is a set of indices j such that dist(candidates[i], candidates[j]) is integer.
    adj = [set() for _ in range(n)]
    
    # To speed up, we can precompute squared distances or just check on the fly.
    # Given n ~ 160^2 = 25600, pairs ~ 300M. This is too slow for pure Python nested loops if done naively.
    # We need a faster way or smaller bound.
    # Let's reduce bound to 50 first. 100x100 = 10000 points. Pairs ~ 50M. Still slow.
    # Let's use a smaller bound or a smarter generation.
    # Actually, for a clique of 6, we don't need a huge graph.
    # Let's try bound = 40. 80x80 = 6400 points. Pairs ~ 20M. Doable in ~10-20s? Maybe tight.
    # Let's optimize the loop.
    
    # Alternative: Generate only points that are part of known integer distance structures?
    # Or just use a smaller bound and rely on the high density of integer distances near the origin.
    
    # Let's try bound = 30. 60x60 = 3600 points. Pairs ~ 6.5M. Very fast.
    bound = 30
    candidates = []
    for x in range(-bound, bound + 1):
        for y in range(-bound, bound + 1):
            candidates.append((x, y))
    n = len(candidates)
    
    adj = [set() for _ in range(n)]
    
    for i in range(n):
        xi, yi = candidates[i]
        for j in range(i + 1, n):
            xj, yj = candidates[j]
            d2 = (xi - xj)**2 + (yi - yj)**2
            if is_perfect_square(d2):
                adj[i].add(j)
                adj[j].add(i)
                
    # 3. Search for a clique of size 6
    # We can use a backtracking search or a greedy approach with restarts.
    # Given the small n, we can try to find a 6-clique.
    # Tomita's algorithm or simple backtracking with ordering by degree.
    
    # Sort nodes by degree to improve backtracking pruning
    # But we also need to check geometric constraints.
    # Let's use a randomized greedy approach with backtracking for the last few steps.
    
    best_set = None
    
    # To ensure we find a valid set, we might need to try multiple random orderings or start nodes.
    # Since the graph is dense near the origin, let's start with high-degree nodes.
    nodes_sorted = sorted(range(n), key=lambda i: len(adj[i]), reverse=True)
    
    # We will perform a depth-first search for a clique of size 6.
    # To avoid getting stuck in a local minimum that violates geometric constraints,
    # we can generate multiple candidate cliques and verify them.
    
    found_cliques = []
    
    def search_cliques(current_clique, candidates_list, target_size=6):
        if len(current_clique) == target_size:
            # Verify geometric constraints
            # Check collinearity
            valid = True
            for i in range(len(current_clique)):
                for j in range(i + 1, len(current_clique)):
                    for k in range(j + 1, len(current_clique)):
                        if are_collinear(candidates[current_clique[i]], 
                                         candidates[current_clique[j]], 
                                         candidates[current_clique[k]]):
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            
            if valid:
                # Check concyclicity
                for i in range(len(current_clique)):
                    for j in range(i + 1, len(current_clique)):
                        for k in range(j + 1, len(current_clique)):
                            for l in range(k + 1, len(current_clique)):
                                if are_concyclic(candidates[current_clique[i]], 
                                                 candidates[current_clique[j]], 
                                                 candidates[current_clique[k]], 
                                                 candidates[current_clique[l]]):
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                    if not valid: break
                if valid:
                    found_cliques.append(current_clique[:])
                    return # Stop searching this branch, we found a valid one
            
            return
        
        if not candidates_list:
            return
            
        # Pick a candidate
        # Heuristic: pick the one with highest degree in the remaining subgraph?
        # For simplicity and speed, just iterate.
        # To find *a* solution quickly, randomization helps avoid worst-case backtracking.
        
        # We can limit the search depth or number of solutions found.
        
        for idx, node in enumerate(candidates_list):
            # Check if node is connected to all in current_clique
            if all(node in adj[c] for c in current_clique):
                # Recurse
                new_candidates = [c for c in candidates_list[idx+1:] if c in adj[node]]
                search_cliques(current_clique + [node], new_candidates, target_size)
                if found_cliques:
                    return # Early exit if we found a valid clique
                
    # Start search from a few promising nodes
    # We can just start with the highest degree node
    if nodes_sorted:
        start_node = nodes_sorted[0]
        # Initial candidates for the second point must be neighbors of start_node
        init_cands = [n_idx for n_idx in nodes_sorted if n_idx in adj[start_node]]
        # Filter init_cands to only include nodes "after" start_node in some ordering?
        # In clique finding, we usually restrict to nodes with index > current to avoid duplicates.
        # But here we just need *a* set.
        
        # Let's try a simpler randomized greedy approach which is often faster for small cliques
        # if we restart many times.
        
        best_set = None
        for _ in range(200):
            # Randomly shuffle the sorted nodes? Or just pick random starting points?
            # Let's pick a random high-degree node to start
            if not nodes_sorted: continue
            start_idx = random.choice(nodes_sorted[:50])
            
            clique = [start_idx]
            # Candidates for next node: neighbors of all in clique
            # Intersection of adj[clique[0]], adj[clique[1]], ...
            candidates_next = set(adj[clique[0]])
            for c in clique[1:]:
                candidates_next.intersection_update(adj[c])
            
            # Sort candidates_next by degree (in the whole graph) to prefer "central" nodes
            candidates_next_sorted = sorted(candidates_next, key=lambda x: len(adj[x]), reverse=True)
            
            # Greedily add nodes
            for node in candidates_next_sorted:
                if len(clique) >= 6:
                    break
                # Add node
                clique.append(node)
                # Update candidates_next to intersection with adj[node]
                candidates_next = candidates_next.intersection(adj[node])
                candidates_next.discard(node)
                # Re-sort? Might be expensive. Let's just filter the existing list.
                # Actually, we can just build the new list from the old one
                # But the list `candidates_next_sorted` is no longer valid.
                # Let's just re-compute the intersection and sort.
                # For speed, we can just iterate through the *original* sorted list and check membership?
                # No, that's O(N).
                # Let's just re-sort the small set `candidates_next`.
                candidates_next_sorted = sorted(candidates_next, key=lambda x: len(adj[x]), reverse=True)
            
            if len(clique) == 6:
                # Verify geometric constraints
                valid = True
                pts = [candidates[i] for i in clique]
                for i in range(6):
                    for j in range(i+1, 6):
                        for k in range(j+1, 6):
                            if are_collinear(pts[i], pts[j], pts[k]):
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                
                if valid:
                    for i in range(6):
                        for j in range(i+1, 6):
                            for k in range(j+1, 6):
                                for l in range(k+1, 6):
                                    if are_concyclic(pts[i], pts[j], pts[k], pts[l]):
                                        valid = False
                                        break
                                if not valid: break
                            if not valid: break
                        if not valid: break
                    if valid:
                        best_set = pts
                        break
                        
        if best_set:
            return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_set]}

    # Fallback if 6 not found
    # Try to find 5
    best_set_5 = None
    for _ in range(200):
        if not nodes_sorted: continue
        start_idx = random.choice(nodes_sorted[:50])
        clique = [start_idx]
        candidates_next = set(adj[clique[0]])
        candidates_next_sorted = sorted(candidates_next, key=lambda x: len(adj[x]), reverse=True)
        
        for node in candidates_next_sorted:
            if len(clique) >= 5:
                break
            clique.append(node)
            candidates_next = candidates_next.intersection(adj[node])
            candidates_next.discard(node)
            candidates_next_sorted = sorted(candidates_next, key=lambda x: len(adj[x]), reverse=True)
            
        if len(clique) == 5:
            pts = [candidates[i] for i in clique]
            valid = True
            for i in range(5):
                for j in range(i+1, 5):
                    for k in range(j+1, 5):
                        if are_collinear(pts[i], pts[j], pts[k]):
                            valid = False
                            break
                        if not valid: break
                    if not valid: break
                if not valid: break
            if valid:
                best_set_5 = pts
                break

    if best_set_5:
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_set_5]}
        
    # Final fallback: known 5-point set
    points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}
# EVOLVE-BLOCK-END
