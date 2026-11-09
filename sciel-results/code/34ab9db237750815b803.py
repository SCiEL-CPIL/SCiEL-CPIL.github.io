import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    all_edges = []
    edge_index = {}
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = len(all_edges)
            all_edges.append((i, j))
    m = len(all_edges) # 120 edges

    # Precompute all Q3 subgraphs as frozensets of edge indices
    # A Q3 is an 8-vertex subgraph with 12 edges.
    # We iterate over all 8-vertex subsets.
    q3s = []
    
    # Helper to check if a set of 8 vertices and a specific edge pattern is a Q3
    # Actually, easier: 
    # For each 8-set, generate all 1680 candidate Q3 edge sets? 
    # No, just check if the induced subgraph on 8 vertices contains a Q3.
    # But we need the specific edge sets to build the conflict structure.
    
    # Efficient generation of Q3 edge sets in K16:
    # 1. Choose 8 vertices.
    # 2. The number of Q3 subgraphs in K8 is 70 (bipartitions) * 24 (matchings) / 2? 
    #    Actually, let's just enumerate.
    
    # To speed up, we can just store the list of Q3 edge sets.
    # There are 12870 8-sets.
    # For each 8-set, we can find all Q3s.
    
    # A Q3 is bipartite. Parts A, B of size 4.
    # Edges are A x B minus a perfect matching.
    
    # We will generate the list of Q3 edge-sets (as tuples of edge indices)
    
    for v8 in itertools.combinations(range(n), 8):
        # Get indices of edges within v8
        # There are 28 edges in K8.
        # We need to find subsets of 12 edges that form Q3.
        
        # Iterate over all 35 bipartitions of v8 into A and B
        vlist = list(v8)
        for i in range(8):
            for j in range(i+1, 8):
                for k in range(j+1, 8):
                    for l in range(k+1, 8):
                        A = [vlist[i], vlist[j], vlist[k], vlist[l]]
                        B = [x for x in vlist if x not in A]
                        
                        # For each of the 24 perfect matchings in K4,4
                        # The edges of Q3 are (A x B) \ Matching
                        # We need to map these to edge indices in K16
                        
                        # To avoid checking all 24 matchings explicitly in a slow loop,
                        # we can precompute the matchings for a canonical 4x4 grid.
                        
                        # Let's just check all 24 matchings
                        # Permutations of B for matching with A
                        for perm in itertools.permutations(B):
                            # Matching is {(A[0], perm[0]), (A[1], perm[1]), ...}
                            # The Q3 edges are all (a, b) in A x B except these 4.
                            
                            # Build the set of edge indices
                            # We can do this quickly
                            # But 12870 * 35 * 24 is ~10 million iterations. Might be too slow in Python if not optimized.
                            
                            # Optimization: 
                            # Precompute the 24 matchings as sets of pairs (0,1,2,3) indices.
                            # Then for a given A, B, map indices to vertex ids.
                            
                            pass # We'll do a faster check below

    # Faster Precomputation:
    # We don't need to precompute ALL Q3s if we use a fast "is Q3 present" check during search.
    # But for Local Search (Simulated Annealing), knowing the conflicts is helpful.
    # However, 12870 * 35 * 24 = 10.7M iterations. In Python, this might take 10-20 seconds.
    # We can prune: 
    # A Q3 is a specific graph. 
    # Let's try to generate them once and store them.
    
    q3_list = []
    
    # Precompute the 24 matchings on {0,1,2,3}
    matchings = []
    for p in itertools.permutations([0, 1, 2, 3]):
        matchings.append(frozenset([(i, p[i]) for i in range(4)]))
        
    # We also need to map (a, b) to edge index
    # This is done via edge_index dict
    
    for v8 in itertools.combinations(range(n), 8):
        vlist = list(v8)
        # 35 ways to split into A, B
        for i in range(8):
            for j in range(i+1, 8):
                for k in range(j+1, 8):
                    for l in range(k+1, 8):
                        A = [vlist[i], vlist[j], vlist[k], vlist[l]]
                        B = [x for x in vlist if x not in A]
                        
                        # Map local indices to global vertices
                        # A[0], A[1], A[2], A[3]
                        # B[0], B[1], B[2], B[3]
                        
                        # For each matching, the Q3 edges are:
                        # {(A[r], B[c]) for r in 0..3, c in 0..3} \ {(A[r], B[perm[r]])}
                        
                        # We can construct the edge set for the "full" K4,4 part
                        # and subtract the matching.
                        
                        # To make it faster, we can represent the Q3 by its 12 edges.
                        
                        for match in matchings:
                            # This loop is heavy. 
                            # Let's check if we can skip.
                            # Actually, 10M iterations of simple set ops might be okay if we use bitmasks or lists.
                            
                            # Let's try a different approach for the search that doesn't require precomputing all Q3s.
                            # Use a fast "check if Q3 exists" function.
                            # And use Greedy + Local Search.
                            
                            pass

    # RE-STRATEGY: 
    # Precomputation of all Q3s is risky for time.
    # Instead, use a fast `has_q3` check and a smart local search.
    
    # Fast `has_q3`:
    # 1. Iterate 8-sets.
    # 2. For each 8-set, check if it contains Q3.
    #    This can be done by checking if the induced subgraph has a Q3.
    #    We can precompute for a 8-bit mask of edges in the 8-set?
    #    There are 28 edges in K8. 2^28 states is too big.
    #    
    #    However, we can just use the bipartition check.
    #    For a fixed 8-set and fixed graph, checking if it contains Q3:
    #    Iterate 35 bipartitions.
    #    For each bipartition, check if the edges between A and B form a Q3.
    #    A Q3 in A x B is K4,4 minus a perfect matching.
    #    So, count missing edges in A x B.
    #    If missing edges count is 4 and they form a perfect matching, it's a Q3.
    #    This check is O(1) per bipartition (just 16 edge checks).
    #    Total per 8-set: 35 * 16 = 560 ops.
    #    Total for all 8-sets: 12870 * 560 ~ 7M ops.
    #    This is fast enough in Python (likely < 1 sec) if implemented with bitmasks.
    
    # Let's implement this fast check.
    
    adj = [0] * n
    
    def rebuild_adj():
        for i in range(n):
            adj[i] = 0
        for (u, v) in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    # We will maintain `edges` as a set of tuples.
    # And `adj` as list of ints.
    
    # Fast Q3 check
    def has_q3_fast():
        # Precompute masks for the 35 bipartitions?
        # No, just iterate.
        
        # Optimization: Only check 8-sets that have enough edges.
        # Q3 has 12 edges.
        
        for v8 in itertools.combinations(range(n), 8):
            # Count edges in induced subgraph
            # We can do this quickly
            ec = 0
            # We can use a bitmask for the 8 vertices?
            # Or just nested loops.
            for i in range(8):
                for j in range(i+1, 8):
                    u, v = v8[i], v8[j]
                    if (adj[u] >> v) & 1:
                        ec += 1
            if ec < 12:
                continue
                
            # Check bipartitions
            vlist = list(v8)
            for i in range(8):
                for j in range(i+1, 8):
                    for k in range(j+1, 8):
                        for l in range(k+1, 8):
                            A = [vlist[i], vlist[j], vlist[k], vlist[l]]
                            B = [x for x in vlist if x not in A]
                            
                            # Check if A x B is Q3
                            # Missing edges must be 4 and form a matching
                            missing = []
                            valid = True
                            rows_used = [False]*4
                            cols_used = [False]*4
                            
                            for ri, a in enumerate(A):
                                for ci, b in enumerate(B):
                                    if not ((adj[a] >> b) & 1):
                                        if rows_used[ri] or cols_used[ci]:
                                            valid = False
                                            break
                                        rows_used[ri] = True
                                        cols_used[ci] = True
                                        missing.append((a, b))
                                if not valid:
                                    break
                            
                            if valid and len(missing) == 4:
                                return True, missing # Return a witness? Or just True
        return False, None

    # Initialize
    random.seed(42)
    # Start with a dense graph
    edges = set()
    for e in itertools.combinations(range(n), 2):
        if random.random() < 0.9:
            edges.add(e)
            
    rebuild_adj()
    
    # Phase 1: Remove edges to make Q3-free
    # Use a greedy approach: find a Q3, remove an edge from it.
    # To be efficient, we can cache the first Q3 found?
    
    # We will use a local search (Simulated Annealing style) to maximize edges.
    
    # State: `edges`
    # We want to maximize len(edges).
    
    # Initial cleanup
    for _ in range(1000):
        found, _ = has_q3_fast()
        if not found:
            break
        if time.time() > deadline - 10:
            break
        # Find a Q3 again to get an edge to remove
        # This is slow. 
        # Better: In has_q3_fast, return the edge to remove.
        
    # Let's modify has_q3_fast to return a witness edge
    def find_q3_witness():
        for v8 in itertools.combinations(range(n), 8):
            ec = 0
            for i in range(8):
                for j in range(i+1, 8):
                    u, v = v8[i], v8[j]
                    if (adj[u] >> v) & 1:
                        ec += 1
            if ec < 12:
                continue
            vlist = list(v8)
            for i in range(8):
                for j in range(i+1, 8):
                    for k in range(j+1, 8):
                        for l in range(k+1, 8):
                            A = [vlist[i], vlist[j], vlist[k], vlist[l]]
                            B = [x for x in vlist if x not in A]
                            rows_used = [False]*4
                            cols_used = [False]*4
                            valid = True
                            missing_edges = []
                            for ri, a in enumerate(A):
                                for ci, b in enumerate(B):
                                    if not ((adj[a] >> b) & 1):
                                        if rows_used[ri] or cols_used[ci]:
                                            valid = False
                                            break
                                        rows_used[ri] = True
                                        cols_used[ci] = True
                                        missing_edges.append((a, b))
                                if not valid:
                                    break
                            if valid and len(missing_edges) == 4:
                                # Found Q3. The edges of Q3 are (A x B) \ missing_edges
                                # Pick a random edge from the Q3 to remove
                                # Actually, pick an edge that is in many Q3s?
                                # For now, pick random.
                                # List of edges in this Q3:
                                q3_edges = []
                                for a in A:
                                    for b in B:
                                        if (a, b) not in missing_edges and (b, a) not in missing_edges:
                                            # Normalize edge
                                            e = (min(a,b), max(a,b))
                                            if e in edges:
                                                q3_edges.append(e)
                                if q3_edges:
                                    return random.choice(q3_edges)
        return None

    # Phase 1: Greedy Removal
    while time.time() < deadline - 20:
        e_to_remove = find_q3_witness()
        if e_to_remove is None:
            break
        edges.remove(e_to_remove)
        u, v = e_to_remove
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    # Phase 2: Local Search / Simulated Annealing
    # Try to add edges.
    # If adding an edge creates a Q3, we can try to swap it with another edge.
    
    best_edges = set(edges)
    best_count = len(edges)
    
    # We will run a loop trying to add random non-edges.
    non_edges = [e for e in itertools.combinations(range(n), 2) if e not in edges]
    
    # To make it faster, we can use a "kick" strategy.
    # Pick a non-edge e. Add it.
    # If Q3 is formed, find that Q3.
    # Remove a random edge f from that Q3 (f != e).
    # This is a valid move if the new graph is Q3-free? 
    # No, removing f might not fix other Q3s? 
    # Actually, if we add e and it creates *only* Q3s that contain e, and we remove f from one of them, 
    # we might still have other Q3s containing e.
    # So we need to ensure the graph is Q3-free after the move.
    
    # Safer: 
    # 1. Add e.
    # 2. Check if Q3-free.
    # 3. If not, find a Q3. Remove an edge from it.
    # 4. Check if Q3-free again.
    # 5. If yes, accept if it improves or with probability.
    # 6. If no, revert.
    
    # This is effectively a random walk on the space of Q3-free graphs.
    
    temp = 1.0
    cooling = 0.999
    
    current_edges = set(edges)
    current_count = len(edges)
    
    # We need a faster check for "is this specific graph Q3-free"
    # We can reuse find_q3_witness.
    
    for _ in range(5000): # Iterations
        if time.time() > deadline - 5:
            break
            
        if not non_edges:
            break
            
        e = random.choice(non_edges)
        # Add e
        u, v = e
        current_edges.add(e)
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # Check validity
        witness = find_q3_witness()
        
        if witness is None:
            # Valid and improved
            current_count += 1
            non_edges.remove(e)
            if current_count > best_count:
                best_edges = set(current_edges)
                best_count = current_count
        else:
            # Invalid. Try to fix by removing witness?
            # Or just revert.
            # Let's try to remove witness and see if it becomes valid.
            # This is a "swap" move.
            w_u, w_v = witness
            if witness != e:
                # Remove witness
                current_edges.remove(witness)
                adj[w_u] &= ~(1 << w_v)
                adj[w_v] &= ~(1 << w_u)
                
                # Check if valid now
                witness2 = find_q3_witness()
                if witness2 is None:
                    # Valid swap
                    # Count remains same (added 1, removed 1)
                    # But we might have broken a previous Q3? 
                    # No, we started from a valid state, added e (invalid), removed w (valid).
                    # So current_edges is valid.
                    # Update non_edges
                    non_edges.remove(e)
                    non_edges.append(witness)
                    # Accept if random < temp? Or always? 
                    # Since count is same, we can accept to explore.
                    if random.random() < temp:
                        pass # Keep the swap
                    else:
                        # Revert
                        current_edges.remove(e)
                        current_edges.add(witness)
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        adj[w_u] |= (1 << w_v)
                        adj[w_v] |= (1 << w_u)
                        non_edges.remove(witness)
                        non_edges.append(e)
                else:
                    # Still invalid. Revert everything
                    current_edges.remove(e)
                    current_edges.add(witness)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    adj[w_u] |= (1 << w_v)
                    adj[w_v] |= (1 << w_u)
            else:
                # Witness is e itself? Impossible if we picked e from non_edges and it's in the Q3.
                # Just revert
                current_edges.remove(e)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
        temp *= cooling

    # Final cleanup: try to add any remaining non-edges
    for e in list(non_edges):
        if time.time() > deadline - 1:
            break
        u, v = e
        current_edges.add(e)
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if find_q3_witness() is None:
            best_edges = set(current_edges)
            best_count = len(current_edges)
            non_edges.remove(e)
        else:
            current_edges.remove(e)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)

    return [list(e) for e in best_edges]

if __name__ == "__main__":
    edges = construct()
    print(len(edges))
