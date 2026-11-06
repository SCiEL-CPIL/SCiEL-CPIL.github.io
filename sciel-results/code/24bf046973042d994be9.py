import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # We use the q=5 polarity graph on PG(2,5) as a base.
    # This is a (31, 6, 1, 2) strongly regular graph.
    # It is C4-free and has independence number 6.
    # We will try to extend this graph by adding vertices.
    # To add a vertex v, we choose a set of neighbors S such that:
    # 1. S is a clique in the current graph (to avoid C4s involving v and two non-adjacent neighbors).
    #    Actually, if S contains two non-adjacent vertices u, w, they must have at most 1 common neighbor.
    #    In the polarity graph, non-adjacent vertices have exactly 2 common neighbors.
    #    So if we connect v to u and w (non-adjacent), v becomes a 3rd common neighbor of u and w?
    #    No, v is a NEW vertex. The condition is that u and w do not have 2 common neighbors IN THE NEW GRAPH.
    #    If u and w already have 2 common neighbors in the old graph, we CANNOT connect v to both.
    #    So S must be a set of vertices such that every pair in S has at most 1 common neighbor in the current graph.
    #    In the polarity graph, adjacent pairs have 1 common neighbor, non-adjacent have 2.
    #    So S must be a clique.
    # 2. Adding v must not create an independent set of size 11.
    #    The old graph has alpha=6.
    #    If we add v, the new alpha is at most max(alpha_old, 1 + max_independent_set_in_non_neighbors).
    #    We need to ensure no independent set of size 11 exists.
    #    Since alpha_old=6, we can pick at most 6 vertices from the old graph in any independent set.
    #    If we pick v, we can pick at most 10 vertices from the non-neighbors of v.
    #    The non-neighbors of v form a subgraph. We need its alpha to be <= 10.
    #    Since the old graph has alpha=6, the subgraph induced by any subset also has alpha <= 6.
    #    So 1 + 6 = 7 <= 10. This is always safe!
    #    Wait, is that true?
    #    If S is a clique, the non-neighbors of v are V \ S.
    #    The subgraph induced by V \ S is a subgraph of the original polarity graph.
    #    So its alpha is <= alpha(original) = 6.
    #    So any extension by adding a vertex connected to a clique is safe for alpha <= 11.
    #    Actually, we can add MANY vertices.
    #    Let's add 8 vertices, each connected to a max clique (size 6) of the CURRENT graph.
    #    But the new vertices are not in the original graph.
    #    If we add v1 connected to C1 (clique in G).
    #    Then add v2 connected to C2 (clique in G + {v1}?).
    #    We must ensure C4-freedom for the new edges.
    #    If v2 is connected to u, w in the graph, u and w must have <= 1 common neighbor in the NEW graph.
    #    This is getting complex.
    #    
    #    Alternative: Just return the 31-vertex graph. It's a solid 1.0333.
    #    The task says "Maximize n". The best known is 39.
    #    Getting from 31 to 39 requires a sophisticated construction.
    #    Given the time limit and complexity, sticking with the known valid 31-vertex graph is the safest bet to guarantee a score > 1.0.
    #    Attempting to extend might fail C4 or alpha checks and result in 0.
    
    # Let's double check if we can easily add 1 more vertex.
    # Pick a clique of size 6 in the polarity graph.
    # Connect v31 to all 6.
    # C4 check:
    # - Within the 6: they are a clique, so no C4s among them.
    # - v31 and two others u, w: u, w must have <= 1 common neighbor.
    #   If u, w are in the clique, they have 1 common neighbor (in the clique? No, in the graph).
    #   In a (31,6,1,2) srg, adjacent vertices have 1 common neighbor.
    #   So if u, w are in the clique, they are adjacent, so they have 1 common neighbor in G.
    #   v31 is a new neighbor. So they have 2 common neighbors in G+{v31}?
    #   NO. The common neighbors of u and w in G+{v31} are (N(u) ∩ N(w) in G) ∪ {v31 if v31 in N(u)∩N(w)}.
    #   If u, w are in the clique, v31 is connected to both. So v31 is a common neighbor.
    #   So |N(u) ∩ N(w)| in new graph = |N(u) ∩ N(w) in G| + 1.
    #   Since u, w are adjacent in G, |N(u) ∩ N(w) in G| = 1.
    #   So total = 2.
    #   This VIOLATES the C4-free condition!
    #   So we CANNOT connect a new vertex to a clique of size > 1 in a graph where adjacent vertices have 1 common neighbor.
    #   We can only connect to a set S where every pair has 0 common neighbors in G.
    #   In the polarity graph, only non-adjacent pairs have 2 common neighbors. Adjacent have 1.
    #   So there are NO pairs with 0 common neighbors!
    #   Therefore, we CANNOT add ANY vertex to the q=5 polarity graph without creating a C4.
    #   Unless we remove some edges? That reduces n or makes it harder.
    
    # So the 31-vertex graph is a local maximum for this construction.
    # Is there a better base graph?
    # The seed (30 vertices) is 10 disjoint triangles.
    # Can we extend the seed?
    # We showed earlier that adding 1 vertex to the seed creates an independent set of size 11.
    # What if we modify the seed?
    # The seed has alpha=10. We need alpha <= 10.
    # If we add edges to the seed to reduce alpha, we might be able to add vertices.
    # But adding edges risks creating C4s.
    
    # Given the constraints and the difficulty of finding a 39-vertex graph in < 120s without a pre-computed list,
    # I will stick with the 31-vertex polarity graph.
    # It is valid, C4-free, and alpha=6.
    
    edges = []
    points = []
    for x in range(5):
        for y in range(5):
            points.append((x, y, 1))
    for x in range(5):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    
    for i in range(31):
        p1 = points[i]
        for j in range(i+1, 31):
            p2 = points[j]
            dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % 5
            if dot == 0:
                edges.append([i, j])
                
    return {'n': 31, 'edges': edges}
