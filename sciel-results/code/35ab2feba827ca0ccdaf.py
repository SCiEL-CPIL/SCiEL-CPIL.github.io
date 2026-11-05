"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # 1. Precompute the conflict graph
    # Nodes are 0..179 representing integers 1..180
    # An edge exists between i and j if adding both creates a reciprocal triple with some k in [1,180]
    # Actually, we need a graph where independent sets are valid subsets.
    # A subset S is valid if there are NO three distinct a,b,c in S such that 1/a = 1/b + 1/c.
    # This is a 3-uniform hypergraph constraint, not a graph constraint.
    # However, we can convert this to a graph problem?
    # No, MIS on a graph forbids edges (pairs). Here we forbid triangles of a specific type.
    # Wait, the current program checks triples.
    # If we use a graph where edge (u,v) exists if u,v are part of ANY reciprocal triple with a third element in [1,180],
    # then an independent set in this graph would be a set with NO pair (u,v) that is part of any triple.
    # This is a STRONGER condition than required. It might yield a smaller set.
    # But it allows us to use standard MIS algorithms.
    # Let's check if this approximation is good.
    # If u,v are in the set, and they form a triple with w, then w might not be in the set.
    # The constraint is: no (a,b,c) in S.
    # If we build a graph G where (u,v) is an edge if there EXISTS w in [1,180] such that {u,v,w} is a reciprocal triple,
    # then any Independent Set of G is a valid subset for the task.
    # Is it optimal? Probably not, but it's a lower bound.
    # Can we do better?
    # We can use a branch and bound directly on the hypergraph, but that's harder.
    # Let's try the graph approximation first. It's a standard technique for "forbidden substructures" if we over-approximate the conflict.
    
    # Precompute conflicts
    n = 180
    # conflict[i][j] = True if i and j (1-indexed) cannot both be in the set
    # because there exists k such that 1/i = 1/j + 1/k OR 1/j = 1/i + 1/k OR 1/k = 1/i + 1/j
    # Note: 1/a = 1/b + 1/c implies a < b and a < c.
    # So for a pair (u, v) with u < v:
    # Case 1: u is 'a'. Then 1/u = 1/v + 1/k => k = uv/(u+v). If k is integer and k != u, k != v, then (u,v,k) is a triple.
    # Case 2: v is 'a'. Impossible since a must be smallest.
    # Case 3: k is 'a'. Then 1/k = 1/u + 1/v => k = uv/(u+v). Same as Case 1.
    # Wait, is it possible that u and v are 'b' and 'c'?
    # If u and v are b and c, then a = uv/(u+v). If a is in [1,180] and a != u, a != v, then {a,u,v} is a triple.
    # So, u and v conflict if there exists an integer a in [1,180] such that a = uv/(u+v) and a != u, a != v.
    # This is the same condition: uv % (u+v) == 0 and a = uv//(u+v) is in [1,180].
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            prod = i * j
            ssum = i + j
            if prod % ssum == 0:
                a = prod // ssum
                if 1 <= a <= n and a != i and a != j:
                    # i and j conflict
                    adj[i - 1].append(j - 1)
                    adj[j - 1].append(i - 1)
    
    # 2. Solve Maximum Independent Set on this graph
    # Use Tomita's algorithm (branch and bound with coloring)
    
    # Precompute colors for the graph? 
    # Tomita's algorithm colors the remaining candidates at each step.
    # We will implement a recursive search.
    
    # To speed up, we can order vertices by degree or use a heuristic.
    # Let's use a standard B&B.
    
    best_size = 0
    best_set = []
    
    # Convert to 0-indexed
    # We want to find a max independent set.
    # This is equivalent to finding a max clique in the complement graph.
    # Let's build the complement graph for Tomita's algorithm, which is designed for Max Clique.
    # Or we can adapt Tomita for Max Independent Set directly.
    # Max Independent Set in G = Max Clique in G_complement.
    
    # Let's build G_complement adjacency.
    # n=180 is small enough for O(n^2) complement construction.
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                # i and j are NOT in conflict, so they CAN be together.
                # In the complement graph, they are connected.
                comp_adj[i].add(j)
                comp_adj[j].add(i)
                
    # Convert sets to lists for faster access? Or keep sets for O(1) check.
    # Tomita's algorithm usually uses lists of candidates.
    
    # Let's implement Max Clique on comp_adj.
    
    # Global variables for search
    current_clique = []
    best_clique = []
    
    def color_sort(candidates):
        """Greedy coloring of the induced subgraph on candidates.
        Returns list of (color, vertex) sorted by color descending."""
        # candidates is a list of vertex indices
        # We color the subgraph induced by candidates.
        # Greedy: process vertices in some order, assign smallest available color.
        # To make it effective for B&B, we want vertices with high color to be processed first?
        # Actually, Tomita sorts by color and uses the color as an upper bound.
        
        # We need to color the graph induced by 'candidates' in comp_adj.
        # Let's do a simple greedy coloring.
        # Order: maybe by degree in the induced subgraph? Or just arbitrary?
        # Standard Tomita: color in the order they appear in the candidate list, 
        # but the list itself is often sorted by degree or previous color.
        
        # Let's just do a simple greedy pass.
        # Map vertex -> color
        color_of = {}
        # Sort candidates by degree in the induced subgraph (descending) to get better coloring?
        # Or just use the order provided.
        # For now, let's just iterate through candidates.
        # To be safe and fast, we can just assign colors 1..k.
        
        # Actually, a better greedy coloring for B&B is to color the complement of the candidate graph?
        # No, we are finding a clique in comp_adj. So we color comp_adj[candidates].
        
        # Let's use a simple greedy coloring algorithm.
        # We want to find the chromatic number (or an upper bound on it) for the induced subgraph.
        # Wait, the upper bound for the clique size is the number of colors used?
        # Yes, if the graph is k-colorable, the max clique size is <= k.
        
        # Greedy coloring:
        # vertices = list of candidates
        # We assign colors 1, 2, ...
        # We want to minimize the number of colors? No, we want a valid coloring.
        # The number of colors used is an upper bound on the clique size.
        # We want this upper bound to be tight.
        # So we should try to use FEW colors? No, any valid coloring gives an upper bound.
        # If we use few colors, the upper bound is small, which prunes more.
        # So we want a good (low color count) coloring.
        
        # Simple greedy:
        # For each vertex in some order, assign the smallest color not used by its neighbors in the induced subgraph.
        
        # Order: Let's use the order they were added to the candidate list, which is often reverse of processing order.
        # Actually, in Tomita, the candidate list is maintained as a stack.
        
        # Let's just do a simple greedy coloring on the list.
        # To improve, we can sort by degree in the induced subgraph (descending).
        
        # Calculate degrees in induced subgraph
        cand_set = set(candidates)
        degrees = {}
        for v in candidates:
            deg = 0
            for u in comp_adj[v]:
                if u in cand_set:
                    deg += 1
            degrees[v] = deg
            
        # Sort by degree descending
        sorted_cands = sorted(candidates, key=lambda x: degrees[x], reverse=True)
        
        color_of = {}
        next_color = 1
        for v in sorted_cands:
            # Find colors used by neighbors
            used_colors = set()
            for u in comp_adj[v]:
                if u in cand_set and u in color_of:
                    used_colors.add(color_of[u])
            # Find smallest available color
            c = 1
            while c in used_colors:
                c += 1
            color_of[v] = c
            if c > next_color:
                next_color = c
        
        # Now, we have color_of for each vertex.
        # We want to return the list of vertices sorted by color descending?
        # In Tomita, we process vertices in order of decreasing color.
        # And we use the color as the bound.
        
        # Return list of (color, vertex)
        colored = [(color_of[v], v) for v in candidates]
        # Sort by color descending
        colored.sort(key=lambda x: x[0], reverse=True)
        return colored

    def search(candidates, clique_size):
        nonlocal best_size, best_clique, current_clique
        
        if not candidates:
            if clique_size > best_size:
                best_size = clique_size
                best_clique = current_clique[:]
            return
        
        # Get colored candidates
        colored = color_sort(candidates)
        
        # Process in order of decreasing color
        for color, v in colored:
            # Upper bound: current clique size + color
            if clique_size + color <= best_size:
                break # No need to explore this vertex or any with lower color
            
            # Add v to clique
            current_clique.append(v)
            
            # New candidates: neighbors of v in the induced subgraph of remaining candidates
            # The remaining candidates are those in 'colored' that come after v?
            # No, the candidates for the next level are the intersection of:
            # 1. Neighbors of v in comp_adj
            # 2. Candidates that are "after" v in the current ordering?
            # Actually, in Tomita, we maintain a list P of candidates.
            # When we pick v from P, the new P is {u in P | u is neighbor of v AND u is after v in P?}
            # No, the order matters for the coloring bound.
            # Let's stick to the standard implementation:
            # P is a list of vertices.
            # We color P.
            # We iterate through P in order of color (descending).
            # When we select v, the new candidate list is the set of vertices in P that are neighbors of v.
            # But we must maintain the relative order?
            # Actually, we can just take the intersection of P and neighbors(v).
            # And re-color that new set.
            
            # To avoid modifying P while iterating, we create a new list.
            # But we need to ensure we don't pick vertices "before" v in some sense?
            # No, for Max Clique, the order in P is just for the coloring bound.
            # The new P is simply P' = { u in P | u != v and (u,v) in E }.
            
            new_candidates = [u for u, _ in colored if u != v and u in comp_adj[v]]
            # Wait, the 'colored' list contains all candidates.
            # We need to filter it.
            # Also, we should probably maintain the order of 'colored' for the next step?
            # The next step will re-color anyway.
            
            search(new_candidates, clique_size + 1)
            
            current_clique.pop()
            
            # Remove v from the candidate list for subsequent iterations in this loop?
            # In the standard algorithm, we process v, then remove v from P for the next iteration of the loop.
            # Since we are iterating over 'colored', we effectively "remove" v by not considering it again.
            # But the 'new_candidates' for the NEXT vertex in the loop should not include v.
            # And it shouldn't include vertices that were removed?
            # Actually, the standard algorithm updates P in place or creates a new list.
            # Let's refine the loop.
            
            # The loop above is slightly inefficient because it recalculates new_candidates from the full 'colored' list.
            # It should be from the remaining candidates.
            
            # Let's restructure the search function to be more standard.
            
    # Let's rewrite the search part more carefully.
    
    def max_clique_search(P, clique):
        """
        P: list of candidate vertices
        clique: current list of vertices in the clique
        """
        nonlocal best_size, best_clique
        
        if not P:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique[:]
            return
            
        # Color the induced subgraph on P
        # We want an upper bound on the clique size in P.
        # Greedy coloring gives an upper bound.
        
        # We'll use a simple greedy coloring that returns the max color used.
        # And we sort P by color descending.
        
        # To speed up, we can use a precomputed order or just sort by degree.
        # Let's sort P by degree in the induced subgraph (descending) before coloring.
        # This often yields fewer colors.
        
        P_set = set(P)
        # Calculate degrees
        degs = {}
        for v in P:
            d = 0
            for u in comp_adj[v]:
                if u in P_set:
                    d += 1
            degs[v] = d
            
        # Sort by degree desc
        P_sorted = sorted(P, key=lambda x: degs[x], reverse=True)
        
        # Greedy coloring
        color_of = {}
        max_color = 0
        for v in P_sorted:
            used = set()
            for u in comp_adj[v]:
                if u in P_set and u in color_of:
                    used.add(color_of[u])
            c = 1
            while c in used:
                c += 1
            color_of[v] = c
            if c > max_color:
                max_color = c
                
        # Now, we have a coloring.
        # We process vertices in order of decreasing color.
        # We need to maintain the list of "remaining" candidates as we go.
        
        # Create a list of (color, vertex)
        items = [(color_of[v], v) for v in P_sorted]
        # Sort by color descending
        items.sort(key=lambda x: x[0], reverse=True)
        
        # We will iterate through these items.
        # But we must be careful: when we pick a vertex v, the new P is the intersection of the CURRENT P and neighbors(v).
        # And we should only consider vertices that are "after" v in the original P?
        # No, the standard algorithm picks v, recurses on N(v) ∩ P', where P' is P \ {v} (or P minus processed vertices).
        
        # Let's use a list `remaining` that we shrink.
        remaining = list(P_sorted) # Start with all candidates
        # Actually, we need to process them in the order of 'items'.
        # Let's just iterate through 'items'.
        
        for color, v in items:
            # Bound check
            if len(clique) + color <= best_size:
                break
                
            # v is in remaining?
            if v not in remaining:
                continue
                
            # Remove v from remaining
            remaining.remove(v)
            
            # New candidates: neighbors of v that are in remaining
            new_P = [u for u in remaining if u in comp_adj[v]]
            
            # Recurse
            max_clique_search(new_P, clique + [v])
            
            # Check time
            if time.time() > deadline:
                return

    # Initialize
    # Start with all vertices as candidates
    initial_P = list(range(n))
    
    # Run search
    max_clique_search(initial_P, [])
    
    # Convert best_clique (0-indexed) to 1-indexed set
    if best_clique:
        result = [x + 1 for x in best_clique]
    else:
        # Fallback: if something went wrong, return a known valid set (e.g., upper half)
        result = list(range(91, 181))
        
    # Verify the result (just in case)
    # The graph approximation ensures validity, but let's double check the logic.
    # If u,v are in the set, they are not connected in the conflict graph.
    # This means there is NO w such that {u,v,w} is a triple.
    # So no triple exists.
    # So the result is valid.
    
    return result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    print(len(res))
