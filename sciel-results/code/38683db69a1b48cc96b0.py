"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools
import random
import sys
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 115.0  # safety margin under 150s kill

    n = 20
    num_colors = 20

    # 1. Generate all edges for K_n
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)

    # 2. Generate all 4-cycles
    # A 4-cycle is defined by 4 distinct vertices {a, b, c, d} and a cyclic ordering.
    # In K_n, any 4 distinct vertices form 3 distinct 4-cycles (subsets of edges).
    # Cycle 1: a-b-c-d-a (edges: ab, bc, cd, da)
    # Cycle 2: a-b-d-c-a (edges: ab, bd, dc, ca)
    # Cycle 3: a-c-b-d-a (edges: ac, cb, bd, da)
    
    cycles = []
    vertices = list(range(n))
    
    # Precompute edge indices for all pairs to speed up cycle generation
    # adj_idx[u][v] = edge index
    adj_idx = [[-1]*n for _ in range(n)]
    for i, (u, v) in enumerate(all_edges):
        adj_idx[u][v] = i
        adj_idx[v][u] = i

    for a, b, c, d in itertools.combinations(vertices, 4):
        # Cycle 1: a-b, b-c, c-d, d-a
        e1 = adj_idx[a][b]
        e2 = adj_idx[b][c]
        e3 = adj_idx[c][d]
        e4 = adj_idx[d][a]
        cycles.append((e1, e2, e3, e4))
        
        # Cycle 2: a-b, b-d, d-c, c-a
        e1 = adj_idx[a][b]
        e2 = adj_idx[b][d]
        e3 = adj_idx[d][c]
        e4 = adj_idx[c][a]
        cycles.append((e1, e2, e3, e4))
        
        # Cycle 3: a-c, c-b, b-d, d-a
        e1 = adj_idx[a][c]
        e2 = adj_idx[c][b]
        e3 = adj_idx[b][d]
        e4 = adj_idx[d][a]
        cycles.append((e1, e2, e3, e4))

    # 3. Solve CSP
    # Variables: edge indices 0..num_edges-1
    # Values: colors 0..num_colors-1
    
    # We will try to solve for the full graph first.
    # If it fails, we remove edges.
    
    # Optimization: Pre-compute constraints per edge
    # For each edge, which other edges is it in a 4-cycle with?
    # Actually, the constraint is "all 4 distinct". This is an AllDifferent constraint on the set of 4 edges.
    
    # Simple Backtracking with Forward Checking
    # State: colors[edge_idx] = color or -1
    
    def solve_csp(edges_to_consider, time_limit):
        """
        edges_to_consider: list of edge indices to include in the graph.
        Returns a list of [u, v, c] for valid edges, or None if infeasible.
        """
        if not edges_to_consider:
            return []
            
        # Filter cycles that are fully contained in edges_to_consider
        edge_set = set(edges_to_consider)
        active_cycles = []
        for cyc in cycles:
            if all(e in edge_set for e in cyc):
                active_cycles.append(cyc)
                
        # If no cycles, any coloring works.
        if not active_cycles:
            # Assign arbitrary colors
            result = []
            for idx in edges_to_consider:
                u, v = all_edges[idx]
                result.append([u, v, 0])
            return result

        # Heuristic: Order variables (edges) by number of constraints (cycles they are in)
        # Count how many active cycles each edge is part of
        edge_constraint_count = defaultdict(int)
        for cyc in active_cycles:
            for e in cyc:
                edge_constraint_count[e] += 1
                
        # Sort edges to consider by constraint count (descending)
        # We need to solve for the colors of these edges.
        # Let's map the active edges to a smaller index space for the solver?
        # No, keep original indices for simplicity, but process in order.
        
        # We need to assign colors to ALL edges in edges_to_consider.
        # Edges not in any active cycle can be assigned arbitrarily later.
        # Let's identify 'critical' edges (in >= 1 cycle) and 'free' edges.
        
        critical_edges = [e for e in edges_to_consider if edge_constraint_count[e] > 0]
        free_edges = [e for e in edges_to_consider if edge_constraint_count[e] == 0]
        
        # Sort critical edges by constraint count (Most Constrained Variable)
        critical_edges.sort(key=lambda e: -edge_constraint_count[e])
        
        # Backtracking state
        colors = [-1] * num_edges
        # For each edge, maintain a set of available colors? 
        # With 20 colors and 190 edges, domains are large.
        # Constraint: In a 4-cycle (e1,e2,e3,e4), c1!=c2, c1!=c3, c1!=c4, c2!=c3, c2!=c4, c3!=c4.
        # This is equivalent to: c1,c2,c3,c4 are all distinct.
        
        # We can use a standard backtracking with forward checking.
        # To speed up, we can check conflicts on assignment.
        
        def is_consistent(edge_idx, color_val):
            # Check if assigning color_val to edge_idx violates any constraint with already assigned neighbors in cycles
            # We need to check all cycles containing edge_idx.
            # If another edge in that cycle is assigned a color, it must be different.
            # Since the constraint is AllDifferent on the 4, if any other edge in the cycle has color_val, it's a conflict.
            
            for cyc in active_cycles:
                if edge_idx in cyc:
                    for other_e in cyc:
                        if other_e != edge_idx:
                            if colors[other_e] == color_val:
                                return False
            return True
            
        def backtrack(idx_in_order):
            if time.time() > time_limit:
                return False
                
            if idx_in_order == len(critical_edges):
                # All critical edges assigned.
                # Assign free edges arbitrarily (e.g., 0)
                # But we must ensure free edges don't conflict?
                # By definition, free edges are in NO active cycle.
                # So they don't participate in any 4-cycle constraint.
                # So any color is valid for them.
                for e in free_edges:
                    colors[e] = 0
                return True
                
            edge_idx = critical_edges[idx_in_order]
            
            # Try colors 0..19
            # Heuristic: Try colors that are least used? Or just 0..19
            for c in range(num_colors):
                if is_consistent(edge_idx, c):
                    colors[edge_idx] = c
                    if backtrack(idx_in_order + 1):
                        return True
                    colors[edge_idx] = -1
                    
            return False

        # Start backtracking
        if backtrack(0):
            # Build result
            result = []
            for idx in edges_to_consider:
                u, v = all_edges[idx]
                result.append([u, v, colors[idx]])
            return result
        else:
            return None

    # Strategy:
    # 1. Try K_20 (all 190 edges).
    # 2. If fails, remove 1 random edge, try again.
    # 3. Repeat until success or time limit.
    
    # To be smarter about removal:
    # If K_20 fails, it's likely due to a dense cluster of cycles.
    # We can try removing edges from the "core" of the graph.
    # For simplicity, we will start with all edges and remove randomly.
    # However, solving K_20 might take a long time if it's infeasible.
    # We should set a shorter time limit for the "hard" full graph attempt.
    
    current_edges = list(range(num_edges))
    best_result = None
    best_edge_count = 0
    
    # Attempt 1: Full Graph
    # Give it 10 seconds. If it fails, we assume it's infeasible or too hard, and start pruning.
    # Actually, if it's feasible, it might find a solution quickly with good heuristics.
    # If it's infeasible, it will exhaust the search space.
    # A smarter approach: Incremental pruning.
    
    # Let's try a few random subsets of high density.
    # Start with 190, then 185, 180...
    
    # But random removal might break the graph into components, making it easier.
    # Let's try to keep it connected.
    
    # We'll use a simple loop:
    # Try current_edges.
    # If success, return.
    # If fail, remove 1 random edge from current_edges.
    
    # To avoid getting stuck in a bad random sequence, we can restart with a new random seed/order occasionally.
    
    # Let's optimize the solver slightly:
    # The `is_consistent` check iterates over all active cycles containing the edge.
    # Pre-compute `cycles_by_edge` for the current active set?
    # Since active set changes when we remove edges, we have to rebuild it.
    # Rebuilding `active_cycles` takes O(num_cycles) which is ~15k. This is fast.
    
    # Let's run the loop.
    # We will allow up to 2000 removals (which means trying graphs of size 190, 189, ...).
    # But 2000 * (solver time) might be too long.
    # Let's set a global deadline.
    
    # To make it more robust, we can try multiple random removal sequences.
    # Sequence 1: Random removal.
    # Sequence 2: Remove edges incident to highest degree vertices.
    
    # Let's stick to random removal for now, but with a time check inside the solver.
    
    # Optimization: If the number of edges is small enough, the CSP is easy.
    # ex(20, C4) is ~89. So graphs with < 90 edges are trivially solvable (C4-free).
    # We want to go much higher.
    
    # Let's try to solve.
    
    # Note: The solver as written uses recursion. Python recursion limit might be an issue for 190 depth?
    # 190 is fine for default limit (1000).
    
    # Let's run it.
    
    # We will use a list of edges to consider.
    # We start with all edges.
    
    # To speed up failure detection, we can check for "obvious" infeasibility?
    # Hard to do.
    
    # Let's just run the loop.
    
    # Random seed for reproducibility of the search path
    random.seed(42)
    
    # We will try to remove edges in a specific order to maintain density.
    # Shuffle the list of edges to remove.
    removal_order = list(range(num_edges))
    random.shuffle(removal_order)
    
    # We start with all edges.
    active_edge_set = set(current_edges)
    
    # We will iterate through removal_order.
    # At each step, we have a set of edges.
    # We try to solve.
    # If solved, return.
    # If not, remove the next edge from removal_order and try again.
    
    # To avoid solving the same instance multiple times if the first one was just "hard but feasible",
    # we should give the solver a reasonable time limit.
    # If it finds a solution, great. If it times out, we prune.
    
    # However, if it times out, it might have been close.
    # For simplicity, we assume timeout = infeasible for that density level.
    
    # Time limit per solve attempt: 5 seconds?
    # If we have 115s total, we can do ~20 attempts.
    # So we will try densities: 190, 185, 180, ...
    
    step_size = 5
    
    # We will maintain a list of edges.
    # Initially all.
    # We remove `step_size` edges at a time.
    
    # Let's create a list of "candidate graphs" by removing chunks of edges.
    # Graph 0: All 190 edges.
    # Graph 1: Remove 5 random edges.
    # Graph 2: Remove 10 random edges.
    # ...
    
    # This is better than removing 1 by 1 because it's faster to fail on a "too dense" graph
    # and we skip to a potentially feasible one.
    
    # But what if 190 is feasible? We find it immediately.
    # What if 185 is feasible but 190 is not? We find it in step 1.
    
    # Let's implement this chunked removal.
    
    # We need to be careful: removing edges changes the active cycles.
    
    # Let's pre-generate the chunks.
    # We need to remove edges such that the graph remains "dense".
    # Random removal is fine.
    
    # Let's try up to 190 edges, then 185, 180, ... down to 100.
    # If we reach 100 and haven't found a solution, we can fall back to a C4-free construction
    # (which guarantees a solution with ~89 edges) to ensure we don't return 0.
    
    # But the C4-free construction is the baseline. We want to beat it.
    # So if we fail at 100, we return the best C4-free graph we can construct quickly.
    
    # Let's refine the solver to be as fast as possible.
    # The `is_consistent` check is O(degree_in_cycles).
    # In K_20, each edge is in many cycles.
    # Number of 4-cycles containing a specific edge (u,v):
    # We need to choose 2 other vertices x, y from the remaining 18.
    # The cycles are (u,v,x,y) and (u,v,y,x).
    # Actually, for a fixed edge (u,v), a 4-cycle is determined by 2 other vertices x,y.
    # The cycle is u-v-x-y-u or u-v-y-x-u.
    # So there are \binom{18}{2} = 153 such pairs.
    # Each pair defines 2 cycles?
    # Let's check:
    # Vertices u,v,x,y.
    # Cycles involving (u,v):
    # 1. u-v-x-y-u (edges uv, vx, xy, yu)
    # 2. u-v-y-x-u (edges uv, vy, yx, xu)
    # Yes, 2 cycles per pair of other vertices.
    # So each edge is in 2 * 153 = 306 cycles.
    # This is a lot. 306 checks per color assignment.
    # 20 colors * 306 checks = 6120 operations per variable.
    # 190 variables * 6120 = 1.1M operations per level of search tree?
    # Backtracking depth 190.
    # This will be slow if it goes deep.
    # But if it fails early (conflict), it's fast.
    # If it's feasible, it might find a solution quickly with good heuristics.
    
    # Let's try it.
    
    # We will use a list of edges to keep.
    # We start with all edges.
    
    # To speed up the "infeasible" detection, we can use a SAT-like clause learning?
    # Too complex.
    
    # Let's just run the chunked removal.
    
    # We will keep a list of edges to remove.
    # We shuffle all edges.
    # We take the first 5, remove them, try solve.
    # If fail, take next 5, remove them, try solve.
    
    # This ensures we are always trying a graph that is a subset of the previous one.
    
    # Let's do it.
    
    # We need to update `active_cycles` efficiently?
    # Recalculating `active_cycles` takes 15k checks. 15k is fast.
    
    # Let's go.
    
    # Fallback C4-free construction (from original code, simplified)
    def construct_c4_free():
        adj = [0] * n
        all_edges_local = [(u, v) for u in range(n) for v in range(u + 1, n)]
        np.random.seed(42)
        edge_indices = np.random.permutation(len(all_edges_local))
        result = []
        for idx in edge_indices:
            u, v = all_edges_local[idx]
            is_c4 = False
            for w in range(n):
                if w == u or w == v:
                    continue
                if (adj[v] >> w) & 1:
                    if (adj[w] & adj[u]) != 0:
                        is_c4 = True
                        break
            if not is_c4:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                result.append([u, v, 0])
        return result

    # Main Loop
    # We will try to solve for graphs of size 190, 185, 180, ...
    
    # Prepare removal list
    removal_list = list(range(num_edges))
    random.shuffle(removal_list)
    
    # We will remove in chunks of 5
    chunk_size = 5
    removed_count = 0
    
    # We want to try until time runs out.
    # Each solve attempt gets a time limit.
    # Let's say 5 seconds per attempt.
    
    # But if the first attempt (190 edges) is feasible, it might take longer than 5s?
    # If it's feasible, the search tree might be small.
    # If it's infeasible, it will explode.
    # So 5s is a good cutoff for "is it likely feasible?".
    
    # Let's run.
    
    # We keep track of the best result found.
    # Since we are removing edges, the first success is the densest.
    
    # We iterate.
    
    # To be safe, we will also run the C4-free constructor in parallel? No, sequential.
    # We will run C4-free at the end if we fail.
    
    # Let's try the CSP solver.
    
    # We need to pass the list of active edges to the solver.
    
    # We will maintain a set of active edges.
    active_edges = set(range(num_edges))
    
    # We will remove chunks.
    
    # We will try up to 20 chunks (100 edges removed -> 90 edges left).
    # 90 is close to C4-free limit.
    
    # Actually, let's try up to 100 chunks (500 edges removed -> impossible, max 190).
    # So max 38 chunks of 5.
    
    for i in range(40):
        if time.time() > DEADLINE - 10:
            break
            
        # Determine current active edges
        # We remove the next chunk from removal_list
        if removed_count < len(removal_list):
            # Remove next chunk
            end_idx = min(removed_count + chunk_size, len(removal_list))
            for e in removal_list[removed_count:end_idx]:
                active_edges.remove(e)
            removed_count = end_idx
            
        # Try to solve
        # Time limit for this specific solve: 5 seconds
        solve_time_limit = time.time() + 5.0
        
        # Pass list of active edges
        active_edge_list = list(active_edges)
        
        # Solve
        result = solve_csp(active_edge_list, solve_time_limit)
        
        if result is not None:
            # Success!
            # Verify result quickly?
            # The solver should guarantee it.
            # But let's check edge count.
            if len(result) > best_edge_count:
                best_result = result
                best_edge_count = len(result)
            
            # Since we are removing edges, the first success is the best.
            # We can return immediately.
            return result
            
    # If we reached here, we failed to find a solution for densities >= 90.
    # Fall back to C4-free construction.
    # This guarantees a valid result with ~89 edges.
    
    # Check time
    if time.time() > DEADLINE - 5:
        # Return a very sparse graph to be safe
        return [[0, i, 0] for i in range(1, n)]
        
    fallback_result = construct_c4_free()
    return fallback_result

# EVOLVE-BLOCK-END
