"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 120
    # We need k=5, g=9. The generalized hexagon of order (4,4) is a 5-regular bipartite
    # graph with girth 12 and 210 vertices.  Its incidence graph is the generalized
    # hexagon H(4,4).
    #
    # However, a more compact construction: the generalized hexagon of order (q,q)
    # has (q+1)(q^2+q+1) points and the same number of lines, and 2(q+1)(q^2+q+1) total
    # vertices in the incidence graph.  For q=4: (5)(21)=105 points, 105 lines, 210 vertices.
    #
    # But the task asks for k=5, g=9.  The girth-12 hexagon satisfies girth >= 9.
    #
    # We need the explicit incidence structure of H(4,4).  The generalized hexagon H(q,q)
    # for q prime power has a well-known explicit construction.  Let me use the standard one.
    #
    # H(q,q) construction:  Points are triples (x,y,z) in GF(q)^3 plus points at infinity.
    # Actually, the standard generalized hexagon H(q,q) (also called the "hexagon of order (q,q)")
    # can be constructed as follows.
    #
    # For GF(q), define points as:
    #   P = {(x, y, z) : x,y,z in GF(q)}  (q^3 points)
    #   plus "points at infinity"
    #
    # Actually, the cleanest known explicit construction of H(q,q) uses the following:
    #   Points:  (x, y, z) with x,y,z in GF(q), and points of the form (x, y, infinity), (x, infinity, z), (infinity, y, z), (infinity, infinity, z), (infinity, y, infinity), (infinity, infinity, infinity)
    #   This is getting complicated.
    #
    # Let me use a different well-known construction.  The generalized hexagon of order (4,4)
    # is isomorphic to the point-line incidence graph of the generalized hexagon H(4,4).
    #
    # Actually, for q=4 (a prime power), H(4,4) is a well-studied object.  Let me use the
    # explicit construction from the literature.
    #
    # The generalized hexagon H(q,q) (q prime power) has the following explicit description:
    #   Points:  GF(q)^3  (q^3 points)
    #   Lines:  "planes"  x * a + y * b + z = 0  for (a,b) in GF(q)^2, plus "vertical" lines
    #   Wait, that's not right either.
    #
    # Let me use a completely different approach.  I'll construct a 5-regular bipartite graph
    # with girth >= 9 using a known explicit construction, and verify the girth.
    #
    # Actually, the seed program already constructs a valid 5-regular graph with girth >= 9
    # using the Lazebnik-Ustimenko-Woldar construction.  The issue is the number of vertices.
    #
    # Let me think about what's actually achievable.  The task says the smallest known 5-regular
    # graph of girth 9 has 1116 vertices.  The seed program's output is likely close to that.
    #
    # Wait, let me re-read: "The smallest known such graph has 1116 vertices" and "the best lower
    # bound is 428."  So the seed program likely produces a graph with ~1116 vertices or so.
    #
    # The hexagon of order (4,4) has 210 vertices and girth 12, which is much better!
    # But wait - is H(4,4) really 5-regular?  Each point is on q+1 = 5 lines, and each line
    # contains q+1 = 5 points.  Yes!  So the incidence graph of H(4,4) is 5-regular bipartite
    # with 210 vertices and girth 12.
    #
    # This is a huge improvement over 1116.  I need the explicit edge list.
    #
    # Let me construct H(4,4) explicitly.  For GF(4), I'll use the representation with
    # elements {0, 1, alpha, alpha+1} where alpha^2 + alpha + 1 = 0.
    #
    # The generalized hexagon H(q,q) construction:
    #   Points:  (x, y, z) in GF(q)^3  (q^3 points)
    #   Lines:  For each (a, b) in GF(q)^2, the line L_{a,b} = {(x, y, z) : x*a + y*b + z = 0}
    #           plus "lines at infinity"
    #
    # Wait, that gives q^2 lines of q points each = q^3 incidences, but we need q^3 points
    # each on q+1 lines = q^3(q+1) incidences.  So I need q+1 lines per point.
    #
    # The correct construction of H(q,q):
    #   Points:  GF(q)^3
    #   Lines:  The set of all "affine planes" through the origin in a 4-dimensional space,
    #           projected... this is getting too complicated without the reference.
    #
    # Let me use a different, simpler known construction.
    #
    # Actually, for the specific case q=4, I can use the following known fact:
    # H(4,4) is the incidence graph of the generalized hexagon of order (4,4).
    # It has 105 points and 105 lines.
    #
    # Since I don't have the explicit edge list memorized, let me use a computational approach:
    # construct the graph from its definition and verify.
    #
    # The generalized hexagon H(q,q) for prime power q can be constructed as follows:
    #   GF(q) with elements 0, 1, ..., q-1 (for prime q)
    #   Points:  (x, y, z) for x, y, z in GF(q)
    #   For each point p = (x, y, z), the q+1 lines through p are:
    #     - "horizontal":  {p' = (x', y, z') : x' - x = a*(z' - z) for some a in GF(q)}  ... 
    #
    # This is the "standard" construction but I'm not 100% sure of the exact formula.
    #
    # Let me take a completely different approach.  I'll use the known fact that H(4,4) exists
    # and has 210 vertices, and I'll construct it using a verified method.
    #
    # Actually, the simplest correct approach:  use the seed construction (which is known valid)
    # and just return it.  The score is 1.0, meaning it matches the seed.  To improve, I need
    # a better construction.
    #
    # Let me try yet another approach:  the "Cousin" of H(4,4) or a direct construction.
    #
    # For GF(4) = {0, 1, w, w^2} where w^2 = w + 1:
    #   The generalized hexagon H(4,4) has a well-known explicit construction.
    #
    # Points:  105 points.  Lines:  105 lines.  Each point on 5 lines, each line has 5 points.
    #
    # I'll use the following explicit construction from the literature on generalized hexagons:
    #
    # H(q,q) construction for q prime power:
    #   Let F = GF(q).
    #   Points:  (x, y, z) in F^3,  (x, y, inf),  (x, inf, z),  (inf, y, z),
    #            (inf, inf, z),  (inf, y, inf),  (inf, inf, inf)
    #   This gives q^3 + q^2*3 + q*2 + 1 = q^3 + 3q^2 + 2q + 1 points.
    #   For q=4:  64 + 48 + 8 + 1 = 121.  That's not 105.
    #
    # Hmm.  Let me count differently.
    #   The generalized hexagon H(q,q) has (q+1)(q^2+q+1) points.
    #   For q=4:  5 * 21 = 105.  Yes.
    #
    # The points can be described as:
    #   (x, y, z) with x, y, z in F  ->  q^3 = 64 points
    #   (x, y, inf) with x, y in F  ->  q^2 = 16 points
    #   (x, inf, z) with x, z in F  ->  q^2 = 16 points  [but some overlap with above?]
    #   (inf, y, z) with y, z in F  ->  q^2 = 16 points
    #   (inf, inf, z) with z in F  ->  q = 4 points
    #   (inf, y, inf) with y in F  ->  q = 4 points
    #   (inf, inf, inf)  ->  1 point
    #   Total:  64 + 16 + 16 + 16 + 4 + 4 + 1 = 121.  Still not 105.
    #
    # So this coordinate system doesn't work directly.  The actual point set is smaller.
    #
    # I think the correct description is:
    #   Points:  (x, y, z) in F^3  (q^3 points)
    #   plus  points at infinity forming a "line at infinity" with q+1 points,
    #   plus  additional points.
    #   q^3 + (q+1) + ... = 105 for q=4.
    #   64 + 41 = 105.  So 41 points at infinity?  That doesn't decompose nicely.
    #
    # I'm spending too much time trying to recall the exact construction.  Let me use a
    # different strategy:  I'll keep the seed construction as a fallback, and try to
    # construct H(4,4) using a verified computational method.
    #
    # Actually, the best strategy given time constraints:  the seed program is valid and
    # produces a correct answer.  The score is 1.0.  To improve, I need fewer vertices.
    #
    # Let me use the following approach:  construct a 5-regular bipartite graph with
    # girth >= 9 using a random construction + girth verification, and keep the best
    # (smallest n) valid graph found.  This is a search approach.
    #
    # But the task says the smallest known is 1116.  A random search won't find something
    # smaller than the known best in 120 seconds.
    #
    # Wait - the hexagon H(4,4) has 210 vertices!  That's way below 1116.  So either:
    #   1. H(4,4) doesn't actually have girth 12 (it does - it's a generalized hexagon)
    #   2. The task's k and g are not k=5, g=9 for this specific evaluation
    #   3. There's something I'm missing
    #
    # The task says: "the evaluator passes this task's degree k and girth g".  So k and g
    # are parameters passed in.  The hexagon H(4,4) works specifically for k=5, g=9 (or any
    # g <= 12).
    #
    # I MUST find the explicit edge list for H(4,4).  Let me use the standard construction.
    #
    # The generalized hexagon H(q,q) (q prime power) - standard construction:
    #   F = GF(q)
    #   Points:  F^3  (q^3 points)
    #   Lines:  For each (a, b) in F^2, the line  x = a*y + b*z  ... no.
    #
    # OK here's the actual construction.  The generalized hexagon H(q,q) is the incidence
    # structure with:
    #   Points:  (x, y, z) in GF(q)^3
    #   Lines:  "tuples" (a, b, c, d) with the incidence relation:
    #           (x,y,z) is on line (a,b,c,d) iff  x = a, y = b, z = c  ... no, that's trivial.
    #
    # I give up trying to recall it from memory.  Let me use a computational approach:
    # generate a candidate graph and verify.
    #
    # Strategy:  Use the seed construction (known valid) as the base.  It produces a valid
    # 5-regular graph with girth >= 9.  The score is 1.0.  Without a better explicit
    # construction, I can't improve.
    #
    # BUT WAIT:  let me re-read the seed code more carefully.
    #
    # The seed uses Lazebnik-Ustimenko-Woldar CD(d, q) with q = k = 5.
    # d = max(3, g-5 if (g-5)%2 else g-4) = for g=9: g-5=4, 4%2==0, so d = g-4 = 5.
    # So d=5, q=5.  The graph has (q+1)(q^d) vertices per side?  Let me compute.
    #
    # Actually the seed returns the connected component of the zero point.  For d=5, q=5,
    # the points are tuples (None, 0, y2, y3, y4, y5) where y_i in GF(5).  So 5^4 = 625
    # points per side, total 1250 vertices.  Close to 1116 but not exact.
    #
    # Hmm, actually the first component element is None, and the rest are in GF(5).  So
    # (d) elements after the None: y2, y3, y4, y5 (4 elements) -> 5^4 = 625.  Total 1250.
    #
    # Anyway, 1250 > 210.  So if I can construct H(4,4), I'd get a much better score.
    #
    # Let me try ONE more time to recall the H(q,q) construction.
    #
    # From the theory of generalized polygons:
    #   H(q,q) for q prime power:
    #   Points:  (x, y, z) in F^3  where F = GF(q)
    #   Lines:  A line is a set of 5 points.  The lines through (x0, y0, z0) are:
    #     L1:  {(x0, y0, z) : z in F}  (varying z)
    #     L2:  {(x, y0, z0) : x in F}  (varying x)
    #     L3:  {(x, y, z) : x - x0 = a*(y - y0), z - z0 = b*(y - y0) for fixed a,b}  ... 
    #
    # No, that gives too many or too few.
    #
    # The actual construction (from Tits):
    #   H(q,q) points:  F^3
    #   Lines through (x,y,z):
    #     1. {(x, y, w) : w in F}
    #     2. {(u, y, z) : u in F}  
    #     3. {(x + a*t, y + b*t, z + c*t) : t in F} for (a,b,c) in some set
    #
    # This is getting nowhere without the reference.
    #
    # FINAL DECISION:  I'll implement the seed construction (which is valid) and return it.
    # This guarantees a valid answer with score 1.0.  Without a verified explicit H(4,4)
    # construction, I risk producing an invalid graph (score 0) which is worse.
    
    # Use the seed construction
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
# EVOLVE-BLOCK-END
