"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

N = 40

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin before 90s kill

    # STRATEGY: Double-ring construction
    # - Two 20-cycles: even vertices {0,2,...,38} and odd vertices {1,3,...,39}
    # - Rungs connecting the two rings to create cycles of all lengths 3..40
    # - Each rung (e_i, o_j) can create cycles using arcs on both rings
    #
    # Ring E: 0-2-4-...-38-0 (20 edges)
    # Ring O: 1-3-5-...-39-1 (20 edges)
    # Total ring edges: 40
    #
    # We need cycles of lengths 3..40.
    # Length 40: Hamiltonian cycle using all vertices.
    # Length 39: 39-vertex cycle.
    # ...
    # Length 3: triangle.
    #
    # A rung (e, o) where e is on ring E and o is on ring O:
    # If e and o are "adjacent" in the interleaved order (e.g., 0 and 1),
    # then e-o plus the path on E from e back to e (trivial) doesn't work.
    # Instead, e-o plus a path on E from e to some e' and a path on O from o to some o'
    # and a rung e'-o' creates a cycle.
    #
    # Simpler approach: Use a single Hamiltonian cycle as backbone + minimal chords,
    # but choose the Hamiltonian cycle and chords jointly to minimize total edges.
    #
    # Actually, let's try a different structural approach:
    # Use a "theta-like" structure: two vertices connected by multiple internally
    # disjoint paths of different lengths.
    #
    # Theta graph: vertices a, b, and three internally disjoint paths P1, P2, P3
    # from a to b of lengths l1, l2, l3 (number of edges).
    # Cycles: P1+P2 (length l1+l2), P1+P3 (l1+l3), P2+P3 (l2+l3).
    # Total edges: l1+l2+l3. Total vertices: 2 + (l1-1)+(l2-1)+(l3-1) = l1+l2+l3-1.
    #
    # For N=40, we need 40 vertices. So l1+l2+l3-1 = 40, i.e., l1+l2+l3 = 41.
    # Cycles: l1+l2, l1+l3, l2+l3.
    # We need all cycle lengths 3..40. A theta only gives 3 cycles. Not enough.
    #
    # Generalization: "multi-theta" or "cactus of cycles".
    #
    # Let's go back to the chorded Hamiltonian cycle but optimize the chord placement.
    # Current: 40 + 19 = 59 edges.
    #
    # Can we reduce below 59?
    # Key insight: A chord (u,v) on a Hamiltonian cycle of length N creates exactly
    # 2 new cycle lengths: d+1 and N-d+1 where d is the distance along the cycle.
    # But if we have MULTIPLE chords, they can interact to create ADDITIONAL cycles
    # that don't use just one chord. For example, two chords (a,b) and (c,d) can
    # create a cycle that uses both chords plus path segments.
    #
    # However, for the PURPOSE of this task, we just need to CERTIFY one cycle per
    # length. We don't need the graph to have ONLY those cycles. So we can use
    # interactions to get "free" cycle lengths.
    #
    # New idea: Use fewer chords but exploit chord-chord interactions.
    #
    # Alternative structural approach: "Ladder" graph.
    # Two paths of length 20 (vertices 0..19 and 20..39) connected by rungs.
    # Path 1: 0-1-2-...-19 (19 edges)
    # Path 2: 20-21-...-39 (19 edges)
    # Rungs: (i, i+20) for i in 0..19 (20 edges)
    # Total: 19+19+20 = 58 edges.
    #
    # Cycles in a ladder: The "rungs" create 20 cycles of length 4 (each rung pair
    # with path segments). Not pancyclic.
    #
    # Let's try yet another approach: Start from the 59-edge solution and try to
    # REMOVE edges while maintaining pancyclicity, using a greedy deletion + repair.
    #
    # Actually, let me reconsider the double-ring more carefully.
    #
    # Ring E: 0-2-4-...-38-0 (positions 0,1,...,19 on the ring)
    # Ring O: 1-3-5-...-39-1 (positions 0,1,...,19 on the ring)
    #
    # If I add rung (0,1): creates no cycle by itself (just an edge).
    # If I add rungs (0,1) and (2,3): 
    #   Cycle: 0-1-3-2-0? Need edges (1,3) and (2,0). 
    #   (1,3) is on ring O. (2,0) is on ring E.
    #   So 0-1-3-2-0 is a 4-cycle. Length 4.
    #
    # If I add rungs (0,1) and (0,3):
    #   Cycle: 0-1-3-0. Need (0,1), (1,3), (3,0). 
    #   (0,1) rung, (1,3) ring O, (3,0) rung. Length 3!
    #
    # So with rungs (0,1) and (0,3), we get a triangle 0-1-3-0.
    #
    # General: rung (e_i, o_j) where e_i = 2*i, o_j = 2*j+1.
    # The "interleaved" neighbors of e_i are o_{i} = 2*i+1 and o_{i-1} = 2*i-1.
    #
    # Let me think about this differently. 
    #
    # BEST APPROACH: Keep the proven 59-edge construction as a fallback, but try
    # to find a sparser construction using a randomized local search that starts
    # from the 59-edge graph and attempts edge deletions, checking pancyclicity
    # after each deletion.
    #
    # Pancyclicity check: For each length 3..40, verify there exists a simple cycle
    # of that length. This is expensive (NP-hard in general), but for N=40 with
    # ~59 edges, we can use the fact that our cycles are structured.
    #
    # Actually, the simplest structural improvement:
    # The 59-edge solution uses chords (0,2), (0,3), ..., (0,20).
    # Chord (0,20) creates cycles of length 21 and 21 (degenerate - same cycle twice).
    # So chord (0,20) is "wasteful" - it only gives one new length (21) instead of two.
    #
    # If I remove chord (0,20), I lose length 21. But length 21 is also created by
    # chord (0,19) which gives lengths 20 and 22. Wait no:
    # Chord (0,19): cycles of length 20 (0-1-...-19-0) and 22 (0-19-20-...-39-0).
    # Chord (0,20): cycles of length 21 (0-1-...-20-0) and 21 (0-20-21-...-39-0).
    # Both are length 21! So chord (0,20) gives length 21 twice.
    #
    # Can I get length 21 from another chord? 
    # Length 21 = k+1 => k=20 (chord 0-20) or N-k+1=21 => k=20 (same).
    # So only chord (0,20) gives length 21 in this construction.
    #
    # What if I use a different chord for length 21?
    # In the Hamiltonian cycle 0-1-...-39-0, a chord (u,v) creates cycles of length
    # (v-u mod N)+1 and (u-v mod N)+1.
    # For length 21: need (v-u mod 40) = 20 or (u-v mod 40) = 20.
    # So chords (0,20), (1,21), (2,22), ..., (19,39) all create a length-21 cycle.
    #
    # If I use chord (1,21) instead of (0,20), does it interfere with other cycles?
    # Chord (1,21) creates cycles: 1-2-...-21-1 (length 21) and 1-21-22-...-39-0-1 (length 21).
    # Both length 21. Same issue.
    #
    # The fundamental issue: on a 40-cycle, any chord at distance 20 creates two
    # length-21 cycles. There's no way to avoid this with a single chord on a 40-cycle.
    #
    # UNLESS: We use a non-symmetric chord set where some lengths are covered by
    # chord-chord interactions rather than single chords.
    #
    # Let me try: remove chord (0,20) and see if length 21 can be formed by
    # combining other chords.
    # With chords (0,2)..(0,19), can we form a 21-cycle?
    # A 21-cycle would need to use at least two chords (since no single chord gives 21
    # without (0,20)).
    # Example: 0-1-...-19-0-2-3-...-? No, that repeats 0.
    # Example: 0-1-...-18-19-0-2-3-...-20-? Need to get back to 0 with 21 vertices.
    # This is getting complicated. Let's just try the computational approach.
    #
    # FINAL STRATEGY: 
    # 1. Start with the 59-edge construction.
    # 2. Try to delete each chord one at a time and check if all cycle lengths 3..40
    #    are still achievable (using BFS/DFS for cycle detection of specific lengths).
    # 3. If a chord can be deleted, delete it and continue.
    # 4. Return the sparsest graph found.
    #
    # For cycle detection of length L in a graph with ~59 edges and 40 vertices,
    # we can use color-coding or simple backtracking (feasible for N=40, E~59).
    
    def build_base_graph():
        """Build the 59-edge chorded Hamiltonian cycle."""
        edges = set()
        # Hamiltonian cycle
        for i in range(N):
            edges.add((min(i, (i+1)%N), max(i, (i+1)%N)))
        # Chords (0, k) for k = 2..20
        for k in range(2, 21):
            edges.add((0, k))
        return edges
    
    def has_cycle_of_length(graph_edges, L):
        """Check if there's a simple cycle of exactly L vertices in the graph."""
        if L < 3 or L > N:
            return False, None
        if L == N:
            # Check for Hamiltonian cycle - just verify the base cycle works
            # In our construction, the Hamiltonian cycle is always present
            # unless we've deleted a ring edge. For safety, do a quick check.
            pass
        
        # Build adjacency
        adj = {i: set() for i in range(N)}
        for u, v in graph_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # For small L (3,4,5), use direct enumeration
        if L == 3:
            for u in range(N):
                for v in adj[u]:
                    if v > u:
                        for w in adj[v]:
                            if w > v and w in adj[u]:
                                return True, [u, v, w]
            return False, None
        
        if L == 4:
            for u in range(N):
                for v in adj[u]:
                    if v > u:
                        for w in adj[v]:
                            if w > v:
                                for x in adj[w]:
                                    if x > w and x in adj[u]:
                                        return True, [u, v, w, x]
            return False, None
        
        # For larger L, use backtracking with pruning
        # This is expensive, so only do it if needed
        found = [False]
        cycle = [None]
        
        def backtrack(path, visited, target_len):
            if found[0]:
                return
            if len(path) == target_len:
                # Check if last vertex connects to first
                if path[0] in adj[path[-1]]:
                    found[0] = True
                    cycle[0] = list(path)
                return
            # Pruning: not enough remaining vertices
            remaining = target_len - len(path)
            if remaining > N - len(visited):
                return
            for nxt in adj[path[-1]]:
                if nxt not in visited and nxt != path[0]:
                    path.append(nxt)
                    visited.add(nxt)
                    backtrack(path, visited, target_len)
                    path.pop()
                    visited.remove(nxt)
                    if found[0]:
                        return
        
        # Try from each starting vertex
        for start in range(N):
            if found[0]:
                break
            path = [start]
            visited = {start}
            backtrack(path, visited, L)
        
        if found[0]:
            return True, cycle[0]
        return False, None
    
    def verify_pancyclic(graph_edges):
        """Verify all cycle lengths 3..40 exist. Returns cycles dict or None."""
        cycles = {}
        for L in range(3, N+1):
            exists, cyc = has_cycle_of_length(graph_edges, L)
            if not exists:
                return None
            cycles[L] = cyc
        return cycles
    
    def construct():
        start_time = time.time()
        deadline = 80.0
        
        # Start with 59-edge construction
        edges = build_base_graph()
        
        # Try to delete chords (not ring edges) to reduce edge count
        # Chords are (0, k) for k in 2..20
        chord_list = [(0, k) for k in range(2, 21)]
        
        # Try deleting each chord
        for chord in chord_list:
            if time.time() - start_time > deadline - 10:
                break
            if chord not in edges:
                continue
            edges_copy = set(edges)
            edges_copy.discard(chord)
            
            # Check if still pancyclic
            cycles = verify_pancyclic(edges_copy)
            if cycles is not None:
                edges = edges_copy
                # Successfully deleted a chord
        
        # Build final result
        edge_list = sorted(set(edges))
        
        # Get cycles for each length
        final_cycles = verify_pancyclic(edges)
        if final_cycles is None:
            # Fallback: use the base construction
            edges = build_base_graph()
            edge_list = sorted(set(edges))
            final_cycles = verify_pancyclic(edges)
            if final_cycles is None:
                # Shouldn't happen, but provide the known-good construction
                cycles = []
                for L in range(3, N+1):
                    if L == N:
                        cycles.append(list(range(N)))
                    else:
                        k1 = L - 1
                        if 2 <= k1 <= 20:
                            cycles.append(list(range(0, k1+1)))
                        else:
                            k2 = N - L + 1
                            if 2 <= k2 <= 20:
                                cycles.append([0] + list(range(k2, N)))
                            else:
                                raise ValueError(f"Cannot find cycle of length {L}")
                return {'edges': edge_list, 'cycles': cycles}
        
        cycles_out = []
        for L in range(3, N+1):
            cycles_out.append(final_cycles[L])
        
        return {'edges': edge_list, 'cycles': cycles_out}
    
    # Call the inner construct
    return construct()
# EVOLVE-BLOCK-END
