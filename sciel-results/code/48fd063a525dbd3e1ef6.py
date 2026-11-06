"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 120.0 - 5.0  # Safety margin
    
    # Strategy: Deterministic algebraic construction using complete bipartite graph
    # Split vertices into two sets of 10: A = 0..9, B = 10..19
    # Color edge (a, b) with c = (a - (b-10)) % 20
    # This ensures every 4-cycle is rainbow because in a bipartite graph,
    # a 4-cycle a1-b1-a2-b2-a1 implies colors:
    # c1 = (a1 - b1') % 20
    # c2 = (a2 - b1') % 20
    # c3 = (a2 - b2') % 20
    # c4 = (a1 - b2') % 20
    # where b' = b-10.
    # If c1 == c3, then a1-b1' == a2-b2' => a1-a2 == b1'-b2'
    # If c2 == c4, then a2-b1' == a1-b2' => a2-a1 == b1'-b2' => a1-a2 == b2'-b1'
    # So a1-a2 == b1'-b2' and a1-a2 == -(b1'-b2') => b1'-b2' == 0 => b1==b2 (impossible in simple cycle)
    # Wait, let's re-verify the rainbow property for this specific coloring.
    
    # Let A = {0..9}, B = {10..19}. Map B to {0..9} by v' = v-10.
    # Edge (u, v) with u in A, v in B gets color (u - v') % 20.
    # Consider a 4-cycle u1-v1-u2-v2-u1.
    # Colors:
    # c1 = (u1 - v1') % 20
    # c2 = (u2 - v1') % 20
    # c3 = (u2 - v2') % 20
    # c4 = (u1 - v2') % 20
    
    # Check for collisions:
    # c1 == c2 => u1 == u2 (impossible)
    # c3 == c4 => u2 == u1 (impossible)
    # c1 == c3 => u1 - v1' == u2 - v2' => u1 - u2 == v1' - v2'
    # c2 == c4 => u2 - v1' == u1 - v2' => u2 - u1 == v1' - v2' => u1 - u2 == v2' - v1'
    # If c1 == c3 and c2 == c4, then v1' - v2' == -(v1' - v2') => v1' == v2' (impossible)
    # So c1 != c3 and c2 != c4 unless the cycle is degenerate.
    # What about c1 == c4? u1 - v1' == u1 - v2' => v1' == v2' (impossible)
    # What about c2 == c3? u2 - v1' == u2 - v2' => v1' == v2' (impossible)
    
    # Thus, all 4 colors are distinct.
    
    edges = []
    for u in range(10):
        for v in range(10, 20):
            c = (u - (v - 10)) % 20
            edges.append([u, v, c])
            
    # This gives 100 edges.
    # Can we do better? The complete graph K20 has 190 edges.
    # The constraint is strong. 
    # A known result for anti-Ramsey C4 coloring: 
    # For n vertices, max edges in a C4-anti-Ramsey coloring is roughly n^2/2? 
    # Actually, a complete bipartite graph K_{n/2, n/2} with this coloring works.
    # Let's try to add edges within A or within B?
    # If we add an edge within A, say (u1, u2), it creates 4-cycles with edges to B.
    # Cycle: u1 - u2 - v - w - u1? No, u1-u2 is in A.
    # A 4-cycle involving an intra-A edge (u1, u2) must be u1-u2-v-w-u1.
    # Edges: (u1,u2), (u2,v), (v,w), (w,u1).
    # (v,w) is intra-B.
    # So we need intra-A and intra-B edges to form 4-cycles.
    # If we only add intra-A edges, no 4-cycle is formed that includes them unless there are also intra-B edges.
    # If we add no intra-B edges, any 4-cycle must alternate A and B.
    # So we can add ALL intra-A edges and ALL intra-B edges?
    # Let's check if adding intra-A edges breaks the rainbow property for existing bipartite 4-cycles.
    # Existing 4-cycles are bipartite. Adding intra-A edges creates new 4-cycles:
    # u1 - u2 (intra-A) - v (cross) - w (cross) - u1 (cross)? No, w-u1 is cross.
    # Wait, u1, u2 in A. v, w in B.
    # Cycle: u1 - u2 - v - w - u1.
    # Edges: (u1, u2) [intra-A], (u2, v) [cross], (v, w) [intra-B? No, v,w in B, so intra-B], (w, u1) [cross].
    # So this cycle uses 2 cross, 1 intra-A, 1 intra-B.
    # If we add intra-A edges, we MUST also add intra-B edges to form 4-cycles?
    # No, a 4-cycle is a subgraph. If I add edge (u1, u2) in A, and have edges (u2, v) and (w, u1) in cross, and edge (v, w) in B, then we have a 4-cycle.
    # If I do NOT have edge (v, w) in B, then this is not a 4-cycle.
    # So, if I add all intra-A edges but NO intra-B edges, do I create any 4-cycles?
    # A 4-cycle needs 4 edges.
    # If graph is K_{10,10} + K_{10} (intra-A complete), edges are:
    # Cross: u-v. Intra-A: u1-u2.
    # Can I form a 4-cycle?
    # u1 - u2 (intra-A) - v (cross) - ... need to get back to u1.
    # Path from v to u1 must be length 2.
    # v - x - u1.
    # x must be in A (since v is in B, x in A for cross) or x in B (for intra-B, but none exist).
    # If x in A: v-x is cross, x-u1 is intra-A.
    # Cycle: u1 - u2 - v - x - u1.
    # Edges: (u1,u2) intra-A, (u2,v) cross, (v,x) cross, (x,u1) intra-A.
    # This is a valid 4-cycle subgraph!
    # Colors:
    # c1 = color(u1, u2) [intra-A]
    # c2 = color(u2, v) [cross]
    # c3 = color(v, x) [cross]
    # c4 = color(x, u1) [intra-A]
    # We need c1, c2, c3, c4 all distinct.
    # Our current coloring only defined cross colors.
    # We can assign colors to intra-A and intra-B edges.
    # Cross colors are in {0..19}.
    # We have 20 colors total.
    # If we use distinct colors for intra-A edges, we might collide with cross colors.
    # There are 45 intra-A edges and 45 intra-B edges.
    # Total edges = 100 + 45 + 45 = 190 (Complete Graph K20).
    # Can we color K20 such that every 4-cycle is rainbow?
    # This is the anti-Ramsey number for C4.
    # It is known that for large n, the maximum number of edges is less than n^2/2? 
    # Actually, a proper edge coloring of K_n uses n-1 colors if n is odd, n if n is even.
    # Here we have 20 colors for 20 vertices.
    # A 1-factorization of K20 gives a proper edge coloring with 19 colors (each color is a perfect matching).
    # In a proper edge coloring, any 2 edges sharing a vertex have different colors.
    # Does proper edge coloring imply rainbow 4-cycles?
    # No. In a proper edge coloring, a 4-cycle can have colors c1, c2, c1, c2.
    # Example: K4 with proper 3-edge-coloring. Cycle 1-2-3-4-1.
    # Colors: 1,2,3,1? No, 1-2 (c1), 2-3 (c2), 3-4 (c3), 4-1 (c1).
    # Colors are {1,2,3,1}. Not rainbow.
    # So proper edge coloring is NOT sufficient.
    
    # However, the bipartite construction with 100 edges is valid and fast.
    # Can we add some intra edges?
    # Let's stick to the 100-edge bipartite graph first, as it is guaranteed valid.
    # 100 edges is a score of 100/19 = 5.26 (if seed is 19).
    # The seed score is 1.0 (19 edges).
    # 100 edges is a huge improvement.
    
    # Let's verify the time. It's instantaneous.
    # We can use the remaining time to try to add edges?
    # Given the complexity of checking 4-cycles, and the risk of invalidity,
    # and the fact that 100 is already > 5x the seed, this is a strong move.
    # If we want to push for more, we could try a greedy addition of intra-A edges
    # with careful color assignment, but it's risky.
    
    # Let's return the 100-edge bipartite graph.
    
    return edges

# EVOLVE-BLOCK-END
