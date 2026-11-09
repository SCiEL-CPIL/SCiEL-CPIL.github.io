"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    results = [None, None, None]
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for idx, spec in enumerate(specs):
        if time.time() > deadline:
            # Return whatever we have, but this shouldn't happen for small instances
            break
        n, k, p, seed = spec
        cliques = solve_instance(n, k, p, seed, deadline)
        results[idx] = cliques
    
    return results

def solve_instance(n, k, p, seed, deadline):
    from random import Random
    
    # 1. Generate the graph
    rng = Random(seed)
    C = set(range(k))  # Clique vertices
    I = set(range(k, n))  # Independent set vertices
    
    # Cross edges: (c, i) where c in C, i in I
    cross_edges = []
    for c in C:
        for i in I:
            if rng.random() < p:
                cross_edges.append((c, i))
    
    # 2. Build shadow sets for each I-vertex
    # shadow[i] = set of C-vertices connected to i
    shadow = {i: set() for i in I}
    for c, i in cross_edges:
        shadow[i].add(c)
    
    # 3. Group I-vertices by their shadow set
    # Vertices with identical shadow sets are "equivalent" in terms of which C-vertices they connect to.
    groups = defaultdict(list)
    for i in I:
        s = tuple(sorted(shadow[i]))
        groups[s].append(i)
    
    # 4. Construct cliques
    cliques = []
    
    # For each group of equivalent I-vertices, we need to cover the edges from C to these I-vertices.
    # The edges are (c, i) for c in shadow[i], i in group.
    # Since all i in the group have the same shadow S, the edges form a complete bipartite graph between S and the group (if we consider only cross edges).
    # Wait, the clique must contain vertices that are all mutually adjacent.
    # In a split graph, a clique can contain:
    # - Any subset of C (since C is a clique)
    # - Any subset of I, BUT only if they are all connected to the same C-vertices?
    # No. If a clique contains i1, i2 in I, it cannot contain any other vertices except those in C that are connected to BOTH i1 and i2.
    # Since i1 and i2 are in the same group, they have the same shadow S.
    # So a clique containing i1, i2 can also contain any subset of S.
    # However, we must cover ALL edges.
    # The edges to cover for this group are {(c, i) | c in S, i in group}.
    # Note: Edges within C are already covered by the main clique? No, we haven't added the main clique yet.
    # Edges within I are none (independent set).
    
    # Strategy:
    # For a group with shadow S and members G (list of I-vertices):
    # We need to cover edges (c, i) for all c in S, i in G.
    # A clique can contain:
    # - A subset of C, say C_sub
    # - A subset of G, say G_sub
    # Condition: Every vertex in G_sub must be connected to every vertex in C_sub.
    # Since all i in G are connected to exactly S, C_sub must be a subset of S.
    # If C_sub is empty, the clique is just G_sub (which is an independent set, so not a clique unless |G_sub|<=1, but cliques must have size >= 2).
    # Wait, a clique of size 1 is not allowed.
    # If C_sub is not empty, then C_sub ∪ G_sub is a clique iff G_sub ⊆ G and C_sub ⊆ S.
    # Actually, any c in C_sub is connected to any i in G_sub (since C_sub ⊆ S).
    # And C_sub is a clique within C.
    # So C_sub ∪ G_sub is a valid clique.
    
    # We want to minimize the number of cliques.
    # This is equivalent to covering the edges of the bipartite graph between S and G with bicliques (complete bipartite subgraphs).
    # Each biclique corresponds to a clique C_sub ∪ G_sub where C_sub ⊆ S, G_sub ⊆ G.
    # Note: The edges within C_sub are also covered by this clique. But we need to ensure ALL edges in the graph are covered exactly once.
    # The edges within C (the main clique) must be covered.
    # The edges between C and I are the cross edges.
    # The edges within I are none.
    
    # Let's handle the "main" clique first?
    # If we put all C in one clique, it covers all edges within C.
    # Then we only need to cover cross edges.
    # But a cross edge (c, i) is covered by a clique containing c and i.
    # If we use a clique {c} ∪ {i1, i2, ...}, it covers edges (c, i1), (c, i2), etc.
    # It also covers edges within {c} (none) and within {i1, ...} (none).
    # So we can decompose the problem into:
    # 1. Cover edges within C.
    # 2. Cover cross edges.
    
    # However, a clique can cover edges from both categories simultaneously.
    # E.g., clique {c1, c2, i1} covers (c1, c2), (c1, i1), (c2, i1).
    # This might be more efficient than separate cliques.
    
    # Let's try a greedy approach on the cross edges, but allow including C-vertices.
    # Actually, since C is a clique, any subset of C is a clique.
    # The most efficient way to cover cross edges is to group I-vertices that share common neighbors in C.
    
    # Let's refine the grouping strategy.
    # For each group G with shadow S:
    # We need to cover edges between S and G.
    # We can form cliques of the form {c} ∪ G' where c in S, G' ⊆ G.
    # Or {c1, c2} ∪ G' where c1, c2 in S, G' ⊆ G.
    # The latter covers more edges per clique.
    
    # Let's try to cover the bipartite graph between S and G.
    # This is a biclique cover problem.
    # For small S and G, we can try to find a good cover.
    
    # Heuristic:
    # 1. If S is empty, no edges to cover.
    # 2. If G is empty, no edges to cover.
    # 3. Sort C-vertices in S by degree (number of neighbors in G). All have degree |G| since it's complete bipartite?
    #    No, the shadow is S, so every c in S is connected to every i in G.
    #    So the bipartite graph is K_{|S|, |G|}.
    #    We need to cover K_{|S|, |G|} with bicliques.
    #    A biclique is defined by C_sub ⊆ S and G_sub ⊆ G.
    #    The number of bicliques needed to cover K_{m, n} is min(m, n) if we use star bicliques?
    #    Actually, we can cover it with min(m, n) bicliques if we allow overlapping?
    #    No, edge-disjoint cover.
    #    Wait, the cliques must be edge-disjoint.
    #    So we need a partition of the edges of K_{|S|, |G|} into complete bipartite subgraphs.
    #    This is equivalent to partitioning the edges of a complete bipartite graph into bicliques.
    #    The minimum number of bicliques to partition K_{m, n} is max(m, n)?
    #    No. If m=1, n=1, 1 biclique.
    #    If m=2, n=2, we can use 2 bicliques: {c1}x{g1,g2} and {c2}x{g1,g2}? No, that overlaps on no edges?
    #    Edges: (c1,g1), (c1,g2), (c2,g1), (c2,g2).
    #    Biclique 1: {c1} x {g1, g2} covers (c1,g1), (c1,g2).
    #    Biclique 2: {c2} x {g1, g2} covers (c2,g1), (c2,g2).
    #    Total 2. max(2,2)=2.
    #    If m=2, n=3.
    #    Biclique 1: {c1} x {g1, g2, g3}
    #    Biclique 2: {c2} x {g1, g2, g3}
    #    Total 2. max(2,3)=3? No, 2 is better.
    #    So it's min(m, n)?
    #    Let's check m=3, n=2.
    #    Biclique 1: {c1, c2} x {g1, g2} -> covers 4 edges.
    #    Remaining: (c3,g1), (c3,g2).
    #    Biclique 2: {c3} x {g1, g2} -> covers 2 edges.
    #    Total 2. min(3,2)=2.
    #    It seems min(m, n) bicliques are sufficient.
    #    Can we do better?
    #    If m=2, n=2, can we do 1? No, K_{2,2} is not a biclique.
    #    So min(m, n) is the answer for partitioning K_{m,n} into bicliques?
    #    Actually, yes. You can always partition into min(m,n) bicliques by taking stars from the smaller side.
    #    Is it optimal?
    #    The rank of the adjacency matrix of K_{m,n} over GF(2) is 2 (if m,n > 0).
    #    But we are not doing XOR cover, we are doing partition.
    #    It is known that the minimum number of bicliques to partition K_{m,n} is min(m,n).
    
    # So for each group G with shadow S, we need min(|S|, |G|) cliques.
    # Each clique will be of the form C_sub ∪ G_sub.
    # If |S| <= |G|, we use |S| cliques, each containing one c from S and all of G?
    # No, that would overlap on edges?
    # If we use {c} ∪ G for each c in S, the edges (c, i) are covered by clique {c} ∪ G.
    # Are these cliques edge-disjoint?
    # Clique 1: {c1} ∪ G. Edges: (c1, i) for all i in G.
    # Clique 2: {c2} ∪ G. Edges: (c2, i) for all i in G.
    # They share no edges because c1 != c2.
    # So yes, this works.
    # But wait, the clique {c1} ∪ G also contains edges within G? No, G is independent.
    # And edges within {c1}? No.
    # So this is a valid partition.
    
    # However, we also have the edges within C (the main clique).
    # If we use cliques like {c1} ∪ G, we are using single C-vertices.
    # We still need to cover edges between C-vertices.
    # We can add a final clique containing all of C.
    # But then the edges (c1, c2) are covered by the main clique.
    # Do the cliques {c1} ∪ G cover any edges within C? No.
    # So we can just add the main clique C at the end.
    
    # BUT, is it better to merge some C-vertices into the G-cliques?
    # E.g., instead of {c1} ∪ G and {c2} ∪ G and {c1, c2} (main),
    # can we do {c1, c2} ∪ G?
    # This covers edges (c1, c2), (c1, i), (c2, i).
    # This saves one clique (the main one for c1, c2) but uses a larger clique.
    # However, we only need to cover edges within C once.
    # If we put all C in one clique, we cover all (c, c') edges.
    # Then we just need to cover cross edges.
    # The cross edges can be covered by min(|S|, |G|) cliques per group.
    # Total cliques = 1 (main C) + sum(min(|S_g|, |G_g|) for each group g).
    
    # Can we do better?
    # What if we don't use a single main clique?
    # We could partition C into subsets, and assign each subset to cover some cross edges.
    # But C is a clique, so any subset is a clique.
    # The edges within C must be covered.
    # If we split C into C1, C2, we need cliques to cover edges within C1, within C2, and between C1, C2.
    # The edges between C1 and C2 can be covered by a clique C1 ∪ C2?
    # If we use clique C1 ∪ C2, it covers all edges within C1 ∪ C2.
    # So we might as well keep C together?
    # Unless merging C with G reduces the number of G-cliques?
    # The number of G-cliques depends on |S| and |G|.
    # S is the shadow of G.
    # If we merge C-vertices into G-cliques, does it change S? No.
    # Does it change the number of cliques needed?
    # If we use a clique C_sub ∪ G_sub, it covers edges (c, i) for c in C_sub, i in G_sub.
    # It also covers edges within C_sub.
    # If we cover edges within C using these merged cliques, we might save the "main" clique.
    # But we need to cover ALL edges within C.
    # This becomes a complex optimization.
    
    # Let's stick with the simple strategy first:
    # 1. One clique for all C.
    # 2. For each group G with shadow S, use min(|S|, |G|) cliques to cover cross edges.
    
    # Let's implement this.
    
    # Step 1: Main clique
    if len(C) >= 2:
        cliques.append(list(C))
    elif len(C) == 1:
        # If C has 1 vertex, it's not a clique of size >= 2.
        # But it might be part of a cross clique.
        # We'll handle it in the cross cliques.
        pass
    
    # Step 2: Cross cliques
    for S_tuple, G in groups.items():
        S = set(S_tuple)
        if not S or not G:
            continue
        
        m = len(S)
        n = len(G)
        
        if m == 0 or n == 0:
            continue
            
        # We need min(m, n) cliques.
        # If m <= n, use m cliques, each with one c and all G.
        # If n < m, use n cliques, each with all S and one i.
        
        if m <= n:
            for c in S:
                clique = [c] + list(G)
                if len(clique) >= 2:
                    cliques.append(clique)
        else:
            for i in G:
                clique = list(S) + [i]
                if len(clique) >= 2:
                    cliques.append(clique)
    
    # Check if any C-vertex was isolated (no cross edges) and not in main clique?
    # If C has size 1 and no cross edges, it's isolated.
    # But the task says "every edge must lie in exactly one listed clique".
    # Isolated vertices don't need to be in any clique?
    # "Every listed set must be a clique with at least two vertices".
    # It doesn't say every vertex must be in a clique.
    # So isolated vertices are fine.
    
    # However, if C has size >= 2, we added the main clique.
    # If C has size 1, and it has cross edges, it's in a cross clique.
    # If C has size 1 and no cross edges, it's isolated.
    
    # What if C has size 0?
    # Then all vertices are in I.
    # If there are no cross edges, no cliques.
    # If there are cross edges... wait, if C is empty, there are no cross edges.
    # So no cliques.
    
    # Let's double check the "exactly one" constraint.
    # Edges within C: Covered by main clique.
    # Cross edges: Covered by cross cliques.
    # Edges within I: None.
    # Are any edges covered twice?
    # Main clique covers (c1, c2).
    # Cross clique {c1} ∪ G covers (c1, i).
    # No overlap.
    
    # Are any edges missed?
    # Edges within C: All covered by main clique.
    # Cross edges: For each group G with shadow S, we cover all (c, i) for c in S, i in G.
    # Since every cross edge (c, i) belongs to exactly one group (the group of i), and c is in S (shadow of i), it is covered.
    
    # So this is a valid clique partition.
    
    # Can we improve?
    # The number of cliques is 1 (if |C|>=2) + sum(min(|S_g|, |G_g|)).
    
    # Let's try to optimize the cross cliques.
    # Instead of using stars, can we use larger bicliques?
    # For K_{m, n}, we established min(m, n) is optimal for partition.
    # So we can't do better per group.
    
    # But can we merge groups?
    # Groups are defined by shadow sets.
    # If two groups have different shadows, their I-vertices are not equivalent.
    # Can we form a clique containing i1 from G1 and i2 from G2?
    # Only if they share common neighbors in C.
    # If S1 != S2, they might still share some neighbors.
    # But if we put i1 and i2 in a clique, the clique can only contain C-vertices in S1 ∩ S2.
    # This might reduce the number of cliques if we can pack more I-vertices together.
    
    # However, the edges (c, i1) for c in S1 \ S2 must be covered by other cliques.
    # This gets complicated.
    
    # Given the time constraint and the fact that the simple strategy is already O(n^2) or less,
    # and the theoretical bound is 3n^2/16, let's see how many cliques we get.
    
    # For n=90, k=30, p=0.7.
    # |I| = 60.
    # Each i has ~42 neighbors in C.
    # Shadow sets are likely all distinct or few groups.
    # If all shadows are distinct, each group has size 1.
    # Then min(|S|, 1) = 1 for each group.
    # So 60 cliques.
    # Plus 1 main clique.
    # Total 61.
    # 3n^2/16 = 3*8100/16 = 1518.
    # So 61 is much better.
    
    # What if shadows are identical?
    # Then group size is large.
    # min(|S|, |G|) could be smaller.
    
    # This strategy seems very good.
    
    # Let's verify the "at least two vertices" constraint.
    # Main clique: |C| >= 2.
    # Cross cliques:
    # If m <= n: clique is [c] + G. Size 1 + |G|. Since n >= 1, size >= 2.
    # If n < m: clique is S + [i]. Size |S| + 1. Since m >= 1, size >= 2.
    
    # So all cliques have size >= 2.
    
    # One edge case: If C has size 1.
    # Main clique not added.
    # Cross cliques:
    # If m=1, n=1. Clique [c, i]. Size 2.
    # If m=1, n>1. Clique [c] + G. Size 1+n.
    # If m>1, n=1. Clique S + [i]. Size m+1.
    
    # What if C has size 0?
    # No cliques.
    
    # What if there are no cross edges?
    # No groups with non-empty S.
    # Only main clique if |C|>=2.
    
    # This seems robust.
    
    # Let's add a check to ensure all edges are covered.
    # (Optional, for debugging)
    
    return cliques

# EVOLVE-BLOCK-END
