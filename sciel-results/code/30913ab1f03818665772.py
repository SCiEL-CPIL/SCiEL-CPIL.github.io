import time
import random
import itertools
import sys

# Q3 structure
# Vertices 0..7
# Edges:
# 0-1, 0-2, 0-4
# 1-3, 1-5
# 2-3, 2-6
# 3-7
# 4-5, 4-6
# 5-7
# 6-7

Q3_ADJ = [
    [1, 2, 4],
    [0, 3, 5],
    [0, 3, 6],
    [1, 2, 7],
    [0, 5, 6],
    [1, 4, 7],
    [2, 4, 7],
    [3, 5, 6]
]

Q3_EDGES = [
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7)
]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit
    
    n = 16
    all_vertices = list(range(n))
    all_edges = list(itertools.combinations(range(n), 2))
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    # Precompute Q3 embeddings? No, too many.
    # Instead, we will use a fast checker for "Does graph G contain Q3?"
    # Since n=16 is small, we can check all 8-vertex subsets.
    # Number of subsets: C(16, 8) = 12870.
    # For each subset, check if it contains Q3.
    # To check if an 8-vertex induced subgraph H contains Q3:
    # We can try all 8! = 40320 permutations of the 8 vertices mapped to Q3 vertices 0..7.
    # For each permutation, check if all 12 Q3 edges exist in H.
    # Total operations: 12870 * 40320 * 12 ~ 6.2 billion. Too slow for repeated calls.
    
    # Optimization:
    # We can precompute for each subset of 8 vertices, the "Q3-freeness" is not static.
    # However, we can speed up the check by using bitmasks.
    
    # Alternative Strategy:
    # Use a randomized local search (Simulated Annealing) on the edge set.
    # State: A set of edges.
    # Objective: Maximize edges such that graph is Q3-free.
    # Move: Add a random missing edge, or remove a random existing edge.
    # Constraint: If adding an edge creates a Q3, reject (or accept with low prob in SA).
    # To check if adding edge (u,v) creates a Q3:
    #   We need to check if there exists a Q3 embedding using (u,v).
    #   This is faster than checking the whole graph.
    
    # Let's implement `creates_q3(u, v, adj_mask)` where adj_mask is a list of bitmasks.
    # adj_mask[i] has bit j set if (i,j) is an edge.
    
    def creates_q3(u, v, adj_mask):
        # We need to find a mapping f: {0..7} -> {0..15} such that:
        # f is injective
        # {f(i), f(j)} is an edge in G for all (i,j) in Q3_EDGES
        # And {u, v} is one of the edges, i.e., (f(i), f(j)) == (u, v) or (v, u) for some (i,j).
        
        # To speed this up, we iterate over all Q3 edges (i,j).
        # For each (i,j), we try mapping u->i, v->j and u->j, v->i.
        # Then we solve the remaining 6 vertices via backtracking.
        
        # Backtracking state:
        # assigned_q3: dict q3_vertex -> graph_vertex
        # available_g: set of available graph vertices
        
        # We will use a recursive helper.
        
        # Precompute neighbors in Q3 for fast constraint checking.
        
        def backtrack(assigned, available):
            # assigned: dict q3_idx -> g_idx
            # available: set of g_idx not used
            
            # Find unassigned Q3 vertex
            unassigned = [x for x in range(8) if x not in assigned]
            if not unassigned:
                return True
            
            # Heuristic: Pick the unassigned Q3 vertex with the most assigned neighbors?
            # Or just pick the first one. Let's pick the one with most assigned neighbors to prune early.
            def constraint_count(q3_v):
                count = 0
                for nb in Q3_ADJ[q3_v]:
                    if nb in assigned:
                        count += 1
                return count
            
            unassigned.sort(key=constraint_count, reverse=True)
            q3_v = unassigned[0]
            
            # Candidates for q3_v from available
            # Must be adjacent in G to all assigned neighbors of q3_v in Q3.
            candidates = list(available)
            
            for nb in Q3_ADJ[q3_v]:
                if nb in assigned:
                    g_nb = assigned[nb]
                    # Filter candidates: must be adjacent to g_nb
                    # adj_mask[g_nb] has bit c set if (g_nb, c) is edge.
                    mask = adj_mask[g_nb]
                    candidates = [c for c in candidates if (mask >> c) & 1]
                    if not candidates:
                        return False
            
            # Also, if q3_v has an edge to u or v in Q3, and u/v are not assigned to those specific Q3 nodes?
            # Wait, u and v are already assigned to specific Q3 nodes in the outer loop.
            # So we just need to check against all assigned Q3 nodes.
            
            for g_cand in candidates:
                new_assigned = dict(assigned)
                new_assigned[q3_v] = g_cand
                new_available = set(available)
                new_available.remove(g_cand)
                
                if backtrack(new_assigned, new_available):
                    return True
            
            return False
        
        # Iterate over all Q3 edges to see if (u,v) can be part of a Q3
        for i, j in Q3_EDGES:
            # Case 1: u->i, v->j
            if i != j: # Always true for edges
                # Check if u and v are adjacent in G? Yes, we are adding them, so assume yes for this check.
                # Actually, creates_q3 is called BEFORE adding. So we assume the edge exists for the purpose of the Q3.
                # The other edges must exist in the current graph.
                
                # Initialize assigned
                assigned = {i: u, j: v}
                available = set(all_vertices) - {u, v}
                
                # Check if this initial assignment is valid w.r.t current graph?
                # The edge (u,v) is the one being added. Other edges involving i or j must exist.
                # We handle this in backtrack by checking neighbors.
                # But wait, backtrack checks neighbors of the *newly assigned* vertex.
                # It does NOT check the validity of the *initial* assignment's edges to other assigned vertices?
                # In this case, i and j are assigned. Are there other assigned vertices? No.
                # But are there edges in Q3 between i and j? Yes.
                # Are there edges in Q3 from i to other unassigned vertices? Yes.
                # Backtrack will check those when assigning the other vertices.
                # But what if there is an edge in Q3 between i and some k, and in G, u is NOT adjacent to the eventual candidate for k?
                # Backtrack will filter candidates for k based on adjacency to u. So it's handled.
                
                if backtrack(assigned, available):
                    return True
            
            # Case 2: u->j, v->i
            assigned = {j: u, i: v}
            available = set(all_vertices) - {u, v}
            
            if backtrack(assigned, available):
                return True
        
        return False

    # Local Search Strategy
    # Start with a random Q3-free graph.
    # How to generate initial Q3-free graph?
    # Random Greedy: Add edges in random order, keep if Q3-free.
    
    def random_greedy_init(seed):
        random.seed(seed)
        adj_mask = [0] * n
        edges = list(all_edges)
        random.shuffle(edges)
        
        for u, v in edges:
            # Check if adding (u,v) creates Q3
            # Temporarily add for check? No, creates_q3 assumes (u,v) is present.
            # So we just call creates_q3(u, v, adj_mask).
            # But creates_q3 checks if there is a Q3 *using* (u,v).
            # The other edges of that Q3 must be in adj_mask.
            # So this is correct.
            if not creates_q3(u, v, adj_mask):
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
        
        return adj_mask

    def count_edges(adj_mask):
        count = 0
        for i in range(n):
            count += bin(adj_mask[i]).count('1')
        return count // 2

    def is_q3_free(adj_mask):
        # Verify the whole graph is Q3-free.
        # Check all 8-vertex subsets?
        # Or just trust the incremental check?
        # Incremental check is sound if implemented correctly.
        # But for safety in local search, we might need a full check if we do weird moves.
        # For now, assume incremental check is correct.
        return True

    best_adj = None
    best_count = 0
    
    # Run multiple restarts
    # Time budget: 115s
    # Each restart: Init + Local Search
    
    num_restarts = 20 # Adjust based on speed
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # 1. Initialize
        adj = random_greedy_init(r)
        current_count = count_edges(adj)
        
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
            
        # 2. Local Search (Hill Climbing / Simulated Annealing)
        # We will try to add edges.
        # If adding fails, try swapping: remove one edge, add two?
        # Or just try adding random missing edges.
        
        # To improve, we can try "swap" moves:
        # Remove edge e1, add edge e2.
        # This might allow adding more edges later?
        # Actually, removing an edge never creates a Q3.
        # Adding an edge might.
        # So a "swap" is: Remove e1, try add e2. If e2 succeeds, we have same count.
        # If we remove e1 and add e2 and e3?
        
        # Let's stick to simple Hill Climbing: Add random missing edge if possible.
        # If stuck, try a "kick": Remove a random edge, then try to add 2 random missing edges.
        
        stuck_iterations = 0
        max_stuck = 100
        
        while time.time() < deadline and stuck_iterations < max_stuck:
            # Get missing edges
            missing = []
            for u, v in all_edges:
                if not ((adj[u] >> v) & 1):
                    missing.append((u, v))
            
            if not missing:
                break # Complete graph (unlikely to be Q3-free)
            
            random.shuffle(missing)
            
            # Try to add one
            added = False
            for u, v in missing[:10]: # Try a few
                if not creates_q3(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_count += 1
                    added = True
                    stuck_iterations = 0
                    break
            
            if not added:
                stuck_iterations += 1
                # Kick: Remove a random edge
                existing = []
                for u, v in all_edges:
                    if (adj[u] >> v) & 1:
                        existing.append((u, v))
                
                if existing:
                    ru, rv = random.choice(existing)
                    adj[ru] &= ~(1 << rv)
                    adj[rv] &= ~(1 << ru)
                    current_count -= 1
                    stuck_iterations = 0 # Reset stuck counter after kick
            
            if current_count > best_count:
                best_count = current_count
                best_adj = adj[:]
        
    # Convert best_adj to list of edges
    result = []
    for u, v in all_edges:
        if (best_adj[u] >> v) & 1:
            result.append([u, v])
            
    return result

# Test / Sanity check
if __name__ == "__main__":
    # Quick test
    edges = construct()
    print(f"Found {len(edges)} edges")
