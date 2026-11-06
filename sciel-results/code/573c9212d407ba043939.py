import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit with margin
    
    n = 20
    num_colors = 20
    
    # Try to build a dense graph with rainbow 4-cycles
    # Strategy: Use a structured colouring based on vertex indices
    
    # Approach: Assign colours based on (u mod k, v mod k) pattern
    # or use a 1-factorisation approach
    
    # Let's try: colour edge (u,v) = (u + v) % 20 for u < v
    # Check if this gives rainbow 4-cycles
    
    def check_4cycle_rainbow(color_func, edges_set):
        """Check if all 4-cycles are rainbow."""
        # Build adjacency
        adj = {i: set() for i in range(n)}
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find all 4-cycles: a-b-c-d-a where a<b<c<d not required, 
        # but to avoid duplicates, require a < c and b < d with a,b,c,d distinct
        # A 4-cycle is defined by 4 vertices {a,b,c,d} and a pairing into opposite pairs
        # Actually, a 4-cycle is a sequence a-b-c-d-a with all distinct.
        # To enumerate without duplicates: fix a as smallest, then consider orderings
        
        # For each set of 4 vertices, there are 3 ways to form a 4-cycle
        # (3 perfect matchings of the 4 vertices into 2 opposite pairs)
        
        for a, b, c, d in combinations(range(n), 4):
            # Three possible 4-cycles on these 4 vertices:
            # 1) a-b-c-d-a: edges ab, bc, cd, da
            # 2) a-b-d-c-a: edges ab, bd, dc, ca
            # 3) a-c-b-d-a: edges ac, cb, bd, da
            
            cycles = [
                [(a,b), (b,c), (c,d), (d,a)],
                [(a,b), (b,d), (d,c), (c,a)],
                [(a,c), (c,b), (b,d), (d,a)]
            ]
            
            for cycle in cycles:
                # Check if all 4 edges exist
                if all(tuple(sorted(e)) in edges_set for e in cycle):
                    # Check rainbow
                    colors = [color_func(min(e), max(e)) for e in cycle]
                    if len(set(colors)) != 4:
                        return False
        return True
    
    # Try colouring: c(u,v) = (u + v) % 20
    def color_sum(u, v):
        return (u + v) % 20
    
    # Try on complete graph first
    all_edges = set()
    for u in range(n):
        for v in range(u+1, n):
            all_edges.add((u, v))
    
    # Check if complete graph works with sum colouring
    if check_4cycle_rainbow(color_sum, all_edges):
        return [[u, v, color_sum(u, v)] for u, v in sorted(all_edges)]
    
    # If not, try to find maximum subset
    # Use greedy: start with all edges, remove edges that violate
    
    def get_violations(color_func, edges_set):
        """Return set of edges that are in violating 4-cycles."""
        adj = {i: set() for i in range(n)}
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        
        bad_edges = set()
        
        for a, b, c, d in combinations(range(n), 4):
            cycles = [
                [(a,b), (b,c), (c,d), (d,a)],
                [(a,b), (b,d), (d,c), (c,a)],
                [(a,c), (c,b), (b,d), (d,a)]
            ]
            
            for cycle in cycles:
                if all(tuple(sorted(e)) in edges_set for e in cycle):
                    colors = [color_func(min(e), max(e)) for e in cycle]
                    if len(set(colors)) != 4:
                        for e in cycle:
                            bad_edges.add(tuple(sorted(e)))
        
        return bad_edges
    
    # Greedy removal
    current_edges = set(all_edges)
    while True:
        if time.time() > deadline - 5:
            break
        bad = get_violations(color_sum, current_edges)
        if not bad:
            break
        # Remove one bad edge (the one in most violations, or just first)
        current_edges.discard(bad.pop())
    
    if time.time() > deadline - 5:
        pass
    
    # Verify
    if check_4cycle_rainbow(color_sum, current_edges):
        return [[u, v, color_sum(u, v)] for u, v in sorted(current_edges)]
    
    # Fallback: try other colourings or return star
    # Try colouring c(u,v) = (u * 2 + v) % 20
    def color_alt(u, v):
        return (u * 2 + v) % 20
    
    current_edges = set(all_edges)
    while True:
        if time.time() > deadline - 5:
            break
        bad = get_violations(color_alt, current_edges)
        if not bad:
            break
        current_edges.discard(bad.pop())
    
    if check_4cycle_rainbow(color_alt, current_edges):
        return [[u, v, color_alt(u, v)] for u, v in sorted(current_edges)]
    
    # Fallback to star
    return [[0, v, 0] for v in range(1, 20)]
