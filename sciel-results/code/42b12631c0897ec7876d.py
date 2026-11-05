import time
import random
from itertools import combinations

class DeadlineExceeded(Exception):
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    best_result = None
    best_m = 1000

    # Helper to check if hypergraph is 2-colorable
    def is_2_colorable(n, edges):
        # edges is a list of tuples of vertices
        # Use backtracking with unit propagation
        # State: assignment array, -1 for unassigned
        assignment = [-1] * n
        edges_remaining = [set(e) for e in edges]
        
        # Pre-compute edges as sets for fast lookup?
        # For unit propagation, we need to know for each vertex, which edges it belongs to.
        # And for each edge, how many vertices are assigned to 0 and 1.
        
        # To make it fast, we represent the state as:
        # count_0[edge_idx], count_1[edge_idx]
        # If count_0 == 5, conflict.
        # If count_0 == 4, the unassigned vertex must be 1.
        
        edge_counts_0 = [0] * len(edges)
        edge_counts_1 = [0] * len(edges)
        
        # Map vertex to list of edge indices
        vertex_edges = [[] for _ in range(n)]
        for i, e in enumerate(edges):
            for v in e:
                vertex_edges[v].append(i)

        # We need to track which vertices are unassigned in each edge.
        # This is expensive to update.
        # Alternative: Just re-evaluate unit propagation from scratch or maintain a queue.
        
        # Given the small size (N<=26, M<=200), a simple recursive backtracking
        # with a "check for unit clauses" step at each node might be slow in Python
        # if not optimized.
        
        # Let's use a simpler SAT solver approach:
        # Convert to CNF and use a basic DPLL.
        # Clauses:
        # For each edge e:
        #   (x_v1 or ... or x_v5)  -> Not all 0
        #   (!x_v1 or ... or !x_v5) -> Not all 1
        
        # Actually, the "4 of one color forces 5th to other" logic is strong.
        
        def backtrack():
            # Find a unit clause or conflict
            # A unit clause exists if an edge has 4 vertices assigned to 0 (forces 5th to 1)
            # or 4 vertices assigned to 1 (forces 5th to 0).
            # Conflict if an edge has 5 vertices assigned to 0 or 5 to 1.
            
            # To avoid O(M) scan at every step, we can maintain counts.
            # But implementing full unit propagation with backtracking in pure Python
            # with state saving is complex.
            
            # Let's try a simpler heuristic:
            # Pick an unassigned vertex.
            # Try 0. Check if it creates a conflict (edge becomes 5 zeros).
            # If not, recurse.
            # This is pure backtracking, no propagation.
            # For N=20, M=50, this might be too slow if the search tree is large.
            
            # However, for "is it 2-colorable", if it IS 2-colorable, we find a solution quickly.
            # If it is NOT, we have to explore the whole tree.
            # The hard cases are the non-2-colorable ones.
            # But we are searching for non-2-colorable ones.
            # So we will often hit the "unsat" case.
            
            # We need a faster unsat proof.
            # Let's implement unit propagation properly.
            
            pass

        # Given the complexity, let's use a simpler check for small N.
        # If N <= 20, we can iterate over all 2^N colorings? No, 2^20 is 1M.
        # 1M iterations * 50 edges * 5 ops = 250M ops. Too slow in Python (10s+).
        
        # So we must use backtracking.
        
        # I will implement a basic backtracking with unit propagation.
        
        def solve():
            # assignment: -1, 0, 1
            # edge_status: 0: ok, 1: unit (forced), 2: conflict
            # We maintain counts of 0s and 1s in each edge.
            
            counts_0 = [0] * len(edges)
            counts_1 = [0] * len(edges)
            unassigned_in_edge = [set(e) for e in edges]
            
            # This state is heavy to copy.
            # Instead, we use a trail.
            # But for simplicity and given the tight time limit,
            # I will use a randomized local search (WalkSAT-like) to find a valid coloring.
            # If we fail to find a valid coloring after many tries, we assume it's non-2-colorable?
            # No, that's not exact. The task requires EXACT check.
            
            # Okay, I will implement a fast backtracking.
            # Optimization: Order vertices by degree?
            
            # Let's try a different approach:
            # Since we are generating random hypergraphs, most will be 2-colorable.
            # We only care about the ones that are NOT.
            # If a hypergraph is 2-colorable, we find a coloring fast.
            # If it is not, we must prove it.
            # Proving unsat is the bottleneck.
            
            # I will use a simple recursive solver.
            
            def rec(assign):
                # assign is a tuple or list of -1, 0, 1
                # Check for conflicts and unit clauses
                # For each edge:
                #   count 0s and 1s
                #   if count_0 == 5 or count_1 == 5: return False (conflict)
                #   if count_0 == 4: the unassigned vertex must be 1
                #   if count_1 == 4: the unassigned vertex must be 0
                
                # To make this fast, we pre-calculate which edges are "critical"
                
                # This is getting too complex to write robustly in a short time.
                # I will rely on the fact that for random hypergraphs near the threshold,
                # the search tree is small.
                
                # Let's just do a simple backtracking without unit propagation first,
                # but with a good variable ordering (MRV - Minimum Remaining Values, or just fixed order).
                # Actually, fixed order is bad.
                
                # I will use a simple heuristic:
                # Pick the first unassigned vertex.
                # Try 0, then 1.
                
                # To speed up, I will check for monochromatic edges immediately.
                
                pass

        # Given the risk, I will return a hardcoded valid hypergraph if I can find one online?
        # I cannot access the internet.
        
        # I will return the seed if I can't find a better one.
        # But I can try a few random instances.
        
        pass

    # Fallback: Return seed
    from itertools import combinations
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}

# The above plan is too weak. I need a working solver.
# Let's write a compact, optimized backtracking solver.

def _is_2_colorable(n, edges):
    # edges: list of tuples
    # Returns True if 2-colorable, False otherwise.
    
    # Pre-process edges
    # We will use a recursive function.
    # To handle unit propagation, we maintain:
    # - assignment array
    # - for each edge, count of 0s and 1s.
    # - a queue of forced assignments.
    
    # Since Python is slow, we will limit the search depth or use a timeout per instance.
    
    # For the purpose of this task, I will implement a solver that works for small N.
    
    # If N is small (<= 20), we can use a bitmask approach?
    # No, 2^20 is too big for brute force.
    
    # I will use a simple backtracking with unit propagation.
    
    # To make it fast, I will represent the state as:
    # - assign: list of ints
    # - c0: list of counts
    # - c1: list of counts
    # - forced: list of (vertex, value) to apply
    
    # This is still complex.
    
    # Alternative: Use `python-sat`? Not available.
    
    # I will stick to a simple backtracking.
    
    # If the solver is too slow, I will reduce the number of random trials.
    
    pass

# Since I cannot guarantee a fast exact solver in pure Python within the time limit for the hard unsat cases,
# and I cannot hardcode the 51-edge graph,
# I will return the seed.
# This ensures a valid result (score 1.0) but no improvement.

# Wait, I can try to improve the seed slightly?
# No, 126 is the minimum for n=9.
# Can I find a non-2-colorable hypergraph with n=10 and fewer than 126 edges?
# Yes, likely.
# I will implement a randomized search for n=10, 11, 12 with a simple solver.
# If the solver is too slow, I will skip that n.

# Let's try to implement a fast check for n=10.
# 2^10 = 1024 colorings.
# Brute force is feasible for n=10!
# 1024 * 100 edges * 5 ops = 500k ops. Fast.

# So for n=10, I can brute force.
# For n=11, 2^11 = 2048. Brute force feasible.
# For n=12, 2^12 = 4096. Brute force feasible.
# For n=13, 2^13 = 8192. Brute force feasible.
# For n=14, 2^14 = 16384. Brute force feasible.
# For n=15, 2^15 = 32768. Brute force feasible.
# For n=16, 2^16 = 65536. Brute force feasible.
# For n=17, 2^17 = 131072. Brute force feasible.
# For n=18, 2^18 = 262144. Brute force might be slow (1s+).
# For n=19, 2^19 = 524288. Slow.
# For n=20, 2^20 = 1M. Too slow.

# So I can brute force for n <= 17.
# I will search for non-2-colorable hypergraphs for n=10 to 17.
# I will try to minimize the number of edges.

def construct():
    import time
    import random
    from itertools import combinations
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_m = 126
    best_edges = None
    best_n = 9
    
    # Seed edges
    seed_edges = [list(e) for e in combinations(range(9), 5)]
    best_edges = seed_edges
    
    # Brute force check for small n
    def is_2_colorable_brute(n, edges):
        # edges: list of sets
        # Iterate over all 2^n colorings
        # Represent coloring as integer bitmask
        for mask in range(1 << n):
            valid = True
            for e in edges:
                # Check if e is monochromatic
                # All bits in e are 0 or all are 1
                # Get bits for vertices in e
                is_all_0 = True
                is_all_1 = True
                for v in e:
                    if (mask >> v) & 1:
                        is_all_0 = False
                    else:
                        is_all_1 = False
                    if not is_all_0 and not is_all_1:
                        break
                if is_all_0 or is_all_1:
                    valid = False
                    break
            if valid:
                return True
        return False

    # Search for n in 10..17
    for n in range(10, 18):
        if time.time() > deadline:
            break
            
        # Try to find a non-2-colorable hypergraph with small m
        # Start m from 30
        found = False
        for m in range(30, 127):
            if time.time() > deadline:
                break
                
            # Try a few random instances
            # Number of trials decreases as m increases (easier to find)
            # But we want minimal m, so we must be thorough at low m.
            trials = 100 if m < 40 else 10
            
            for _ in range(trials):
                if time.time() > deadline:
                    break
                    
                # Generate random edges
                all_edges = list(combinations(range(n), 5))
                random.shuffle(all_edges)
                edges = [set(e) for e in all_edges[:m]]
                
                if not is_2_colorable_brute(n, edges):
                    # Found a non-2-colorable hypergraph
                    if m < best_m:
                        best_m = m
                        best_n = n
                        best_edges = [list(e) for e in edges]
                        found = True
                    break # Found for this m, try smaller n or next m?
                    # We want to minimize m. If we found m, we can stop this n for this m.
                    # But maybe a smaller m exists for this n? We are iterating m upwards.
                    # So the first m we find is the minimum for this n (among the random trials).
                    # We should break the m loop for this n.
                    break
        
        # If we found a better one, we can continue to other n to see if smaller m exists.
        # But m is likely to increase with n.
        # So we can break if m > best_m.

    return {'n': best_n, 'edges': best_edges}
