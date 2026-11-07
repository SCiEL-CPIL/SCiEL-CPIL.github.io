"""Rewritten program: construct a maximal edge-critical diameter-2 graph on 16 vertices."""
import time

def construct():
    """Return edges of K_{8,8} (64 edges), which is edge-critical for diameter 2."""
    n = 16
    # Partition into two sets of 8: A = {0..7}, B = {8..15}
    # K_{8,8} has all edges between A and B, no edges within A or within B.
    # Diameter is exactly 2: any two vertices in the same part have distance 2
    # (via any vertex in the other part), and vertices in different parts have distance 1.
    # Edge-criticality: removing any edge (u in A, v in B) disconnects u and v directly,
    # and since u and v are in different parts, their only common neighbors are... wait.
    # Actually in K_{8,8}, u in A and v in B are adjacent. If we remove edge (u,v),
    # can u and v still be connected in 2 steps? u is connected to all of B except v,
    # and v is connected to all of A except u. So u -> b (any other in B) -> ... but b is in B,
    # and v is in B. b and v are not adjacent (both in B). So u -> b doesn't help reach v in 2 steps.
    # u -> a (any other in A) -> v? a and v are adjacent (a in A, v in B). So u-a-v is a path of length 2!
    # Wait, that means removing (u,v) still leaves distance 2 between u and v. So K_{8,8} is NOT edge-critical!
    
    # Let me reconsider. The task says "Deleting EACH individual edge must increase its diameter above two or disconnect it."
    # For K_{8,8}: remove edge (u,v) where u in A, v in B.
    # u and v are still at distance 2 via any other a in A (u-a-v) or any other b in B (u-b-v... wait u-b is edge, b-v is not edge since both in B).
    # u-a-v: u-a is edge (both... u in A, a in A - NO, no edges within A in K_{8,8}!).
    # So u is only connected to B. a in A is only connected to B.
    # Path u -> b -> a -> v would be length 3.
    # u -> b (b in B, b != v): u-b is edge. b to v: both in B, no edge.
    # So there's no path of length 2 from u to v after removing (u,v).
    # Distance becomes... u can reach all of B (except v) in 1 step, then from any b in B\{v}, b can reach all of A in 1 step, then from any a in A, a can reach all of B in 1 step. So u to v is at distance 3: u-b-a-v where b in B\{v}, a in A.
    # So diameter becomes 3! K_{8,8} IS edge-critical for diameter 2.
    
    # Great, so K_{8,8} works with 64 edges.
    
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    
    return edges
