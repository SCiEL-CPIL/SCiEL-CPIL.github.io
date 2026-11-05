"""Rewritten program: construct dense 4-cycle-rainbow edge-coloured graph on 20 vertices."""
import time
import random
from itertools import combinations

N = 20
N_COLORS = 20
DEADLINE_MARGIN = 10.0  # seconds to spare before the 150s kill

def _elapsed(start):
    return time.time() - start

def _validate(edges):
    """Check that every 4-cycle has four distinct colours.
    edges: list of [u, v, c]
    Returns True if valid.
    """
    # Build adjacency with colours
    adj = {}
    for u, v, c in edges:
        adj.setdefault(u, {})[v] = c
        adj.setdefault(v, {})[u] = c

    # Enumerate all 4-cycles: (a, b, c, d) with a < b < c < d and edges a-b, b-c, c-d, d-a
    # A 4-cycle is a set of 4 vertices {a,b,c,d} and a cyclic ordering.
    # For 4 vertices, there are 3 possible 4-cycles (3 perfect matchings of the complement... 
    # actually for 4 vertices, a 4-cycle uses 4 of the 6 possible edges in a cycle).
    # A 4-cycle on vertices {a,b,c,d} exists for each of the 3 distinct cycle orderings:
    # (a,b,c,d), (a,b,d,c), (a,c,b,d) — but these are the same cycle traversed differently.
    # Actually, for 4 specific vertices, the number of distinct 4-cycles (as subgraphs) is 
    # the number of ways to arrange them in a cycle up to rotation and reversal = (4-1)!/2 = 3.
    # The 3 cycles are:
    #   a-b-c-d-a
    #   a-b-d-c-a
    #   a-c-b-d-a
    # Each uses 4 edges. We need all 4 edges to be present in the graph.
    
    verts = list(range(N))
    for a, b, c, d in combinations(verts, 4):
        # Three possible 4-cycles on these 4 vertices
        cycles = [
            [(a,b), (b,c), (c,d), (d,a)],
            [(a,b), (b,d), (d,c), (c,a)],
            [(a,c), (c,b), (b,d), (d,a)],
        ]
        for cyc in cycles:
            # Check all 4 edges exist
            cols = []
            valid_cycle = True
            for (u, v) in cyc:
                if v not in adj.get(u, {}):
                    valid_cycle = False
                    break
                cols.append(adj[u][v])
            if valid_cycle:
                if len(set(cols)) != 4:
                    return False
    return True

def _num_edges(edges):
    return len(edges)

def _try_greedy_add(edges, adj, colors_used_on_edge, remaining_pairs, rng, deadline_start):
    """Try to add edges greedily, checking 4-cycle constraints incrementally."""
    # This is a helper; the main logic is in construct.
    pass

def construct():
    start = time.time()
    rng = random.Random(42)
    
    # Strategy: start with a known valid dense structure and try to add more edges.
    # A good base: the complete graph K_n is too dense (too many 4-cycles).
    # Instead, use a structured approach.
    
    # Key insight: if we assign colours such that for any 4-cycle, the 4 edges get distinct colours,
    # one sufficient condition is: for any two vertices u,v, the colour of edge (u,v) is determined
    # by a function f(u,v) such that for any 4-cycle a-b-c-d-a, the four values f(a,b), f(b,c), f(c,d), f(d,a) are distinct.
    
    # A known construction: use a finite field / cyclic group.
    # Let vertices be elements of Z_20. Let edge (u,v) have colour (u+v) mod 20 or (u-v) mod 20.
    # Check: for 4-cycle a-b-c-d-a, colours are (a+b)%20, (b+c)%20, (c+d)%20, (d+a)%20.
    # These need to be distinct. Not guaranteed.
    
    # Alternative: colour edge (u,v) with (u * v) mod p for some prime p > 20? But we only have 20 colours.
    
    # Let's try: vertices 0..19, colour of edge (u,v) = (u + v) % 20.
    # 4-cycle a-b-c-d-a: colours (a+b)%20, (b+c)%20, (c+d)%20, (d+a)%20.
    # Counter-example: a=0,b=1,c=2,d=3: colours 1,3,5,3. Not distinct! So this fails.
    
    # Try colour (u - v) mod 20 (symmetric: use min/max or abs).
    # Actually for undirected, let's use (u + v) % 20 but that failed.
    
    # Let's think differently. A bipartite graph with parts of size 10 each.
    # In a bipartite graph, 4-cycles are a-b-c-d where a,c in L and b,d in R.
    # If we colour edge (l, r) with colour depending on l and r such that for any two pairs (l1,r1),(l2,r2) forming a 4-cycle, the colours are distinct...
    # A 4-cycle in bipartite: l1-r1-l2-r2-l1. Edges: (l1,r1),(r1,l2),(l2,r2),(r2,l1).
    # Colours: c(l1,r1), c(l2,r1), c(l2,r2), c(l1,r2).
    # If c(l,r) = (l * r) mod 20, then for l1=0: all colours 0. Fails.
    
    # Use c(l,r) = (l + r) mod 20 with l in 0..9, r in 0..9 (renumber).
    # 4-cycle: l1,r1,l2,r2. Colours: l1+r1, l2+r1, l2+r2, l1+r2 (mod 20).
    # Need these 4 distinct. l1+r1, l2+r1, l2+r2, l1+r2.
    # Let l1=0, l2=1, r1=0, r2=1: colours 0, 1, 2, 1. Not distinct!
    
    # The problem is hard. Let's use a randomized search with validation.
    
    # Approach: 
    # 1. Start with a random sparse valid graph.
    # 2. Try to add edges one by one, checking validity.
    # 3. Use local search / simulated annealing to maximize edges.
    
    # Given time constraints, let's do a greedy + local search.
    
    # First, let's try a structured construction that's known to work for small n.
    # For n=20, maybe use a union of matchings with careful colouring.
    
    # Actually, let's just do a smart random search.
    
    # Initialize with empty graph
    edges = []
    adj = {}  # adj[u] = {v: colour}
    
    # Precompute all pairs
    all_pairs = [(u, v) for u in range(N) for v in range(u+1, N)]
    rng.shuffle(all_pairs)
    
    def add_edge(u, v, c):
        adj.setdefault(u, {})[v] = c
        adj.setdefault(v, {})[u] = c
        edges.append([u, v, c])
    
    def remove_edge(u, v):
        if u in adj and v in adj[u]:
            del adj[u][v]
        if v in adj and u in adj[v]:
            del adj[v][u]
        edges[:] = [e for e in edges if not (e[0]==u and e[1]==v) and not (e[0]==v and e[1]==u)]
    
    def check_validity_incremental(u, v, c, adj):
        """Check if adding edge (u,v) with colour c violates any 4-cycle constraint.
        A new 4-cycle must include the new edge (u,v).
        A 4-cycle containing edge (u,v): u-v-x-y-u where x != u,v and y != u,v,x.
        The other three edges are (v,x), (x,y), (y,u).
        For each such x,y, check if all three edges exist and colours are distinct from c and each other.
        """
        # Find all x such that (v,x) exists
        vs = list(adj.get(v, {}).keys())
        for x in vs:
            if x == u:
                continue
            # Find all y such that (x,y) exists and (y,u) exists
            xs = list(adj.get(x, {}).keys())
            for y in xs:
                if y == u or y == v or y == x:
                    continue
                # Check if (y,u) exists
                if u not in adj.get(y, {}):
                    continue
                # We have a 4-cycle: u-v-x-y-u
                c_vx = adj[v][x]
                c_xy = adj[x][y]
                c_yu = adj[y][u]
                cols = [c, c_vx, c_xy, c_yu]
                if len(set(cols)) != 4:
                    return False
        return True
    
    # Greedy: try to add each pair with a random colour
    for u, v in all_pairs:
        if _elapsed(start) > 150 - DEADLINE_MARGIN:
            break
        # Try random colours
        trial_colors = list(range(N_COLORS))
        rng.shuffle(trial_colors)
        added = False
        for c in trial_colors:
            if check_validity_incremental(u, v, c, adj):
                add_edge(u, v, c)
                added = True
                break
        # If no colour works, skip this pair for now
    
    # Local search: try to re-colour edges or swap colours to enable more additions
    # Also try to add skipped pairs
    
    # Let's do a simple improvement pass: for each missing pair, try all colours
    missing = [(u, v) for u in range(N) for v in range(u+1, N) if v not in adj.get(u, {})]
    
    improved = True
    while improved and _elapsed(start) < 150 - DEADLINE_MARGIN:
        improved = False
        for u, v in missing:
            if _elapsed(start) > 150 - DEADLINE_MARGIN:
                break
            trial_colors = list(range(N_COLORS))
            rng.shuffle(trial_colors)
            for c in trial_colors:
                if check_validity_incremental(u, v, c, adj):
                    add_edge(u, v, c)
                    missing.remove((u, v))
                    improved = True
                    break
    
    # Try to remove edges that are "blocking" others and re-add with different colours
    # This is more complex; skip for now due to time.
    
    # Final validation
    if not _validate(edges):
        # Should not happen, but if it does, fall back to star
        return [[0, v, 0] for v in range(1, 20)]
    
    return edges

# Test
if __name__ == "__main__":
    result = construct()
    print(f"Number of edges: {len(result)}")
    print(f"Valid: {_validate(result)}")
