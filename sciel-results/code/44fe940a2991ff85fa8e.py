"""
Constructs a 3-regular triangle-free graph on 12 vertices (the dodecahedral graph's
bipartite half, or equivalently the bipartite double cover of the Petersen graph).
This graph has girth 6, so all simple cycles have length at least 6.

Actually, let's reconsider. The dodecahedral graph is 3-regular on 20 vertices with girth 5.
We need 12 vertices.

Let me think about what 3-regular triangle-free graphs exist on 12 vertices:
- Two K_{3,3} components: cycle lengths {4, 6} -> 2 distinct lengths
- The 3-regular bipartite graph that is the bipartite double cover of Petersen:
  This has 20 vertices, too many.
- What about a single 12-vertex 3-regular triangle-free graph?

The key question: can we have only 1 distinct cycle length?
If all cycles have the same length L, then the graph is a "cactus-like" structure or a specific
regular structure. For a 3-regular graph where every cycle has the same length, this is very restrictive.

Actually, for a connected 3-regular graph, if all cycles have length L, then:
- If L=4: This would be a graph where every cycle is a 4-cycle. This is possible (e.g., K_{3,3} has cycles of length 4 and 6, so no). Actually, K_{3,3} has both 4-cycles and 6-cycles.
- Can a 3-regular graph have ALL cycles of the same length? 

For a 3-regular graph, the girth is the shortest cycle. If all cycles have length L, then the graph is a "L-girth" graph where no other cycle lengths exist.

Known result: A connected 3-regular graph with all cycles of the same length must be either:
- A single cycle (not 3-regular)
- Or a very specific structure

Actually, let's think about this differently. The minimum number of distinct cycle lengths for a 3-regular triangle-free graph on 12 vertices:

Option 1: Two disjoint K_{3,3}: cycle lengths {4, 6} -> 2
Option 2: A single 3-regular triangle-free graph on 12 vertices with girth >= 4.

What 3-regular triangle-free graphs exist on 12 vertices?
- The complement of a perfect matching in K_{6,6}? No, that's 12 vertices but not 3-regular.
- The 3-regular bipartite graphs on 12 vertices: These are 6+6 bipartite 3-regular graphs.
  - K_{3,3} + K_{3,3}: 2 components
  - A connected 3-regular bipartite graph on 12 vertices.

For a connected 3-regular bipartite graph on 12 vertices (6+6 split), what cycle lengths are possible?
- 4-cycles exist if there's a "square" in the bipartite structure.
- 6-cycles, 8-cycles, etc.

Actually, let me try to construct a specific graph. The "truncated cube" is 3-regular on 12 vertices with girth 4. It has cycles of length 4, 6, and 8.

What about the "Petersen graph's bipartite double cover"? That's 20 vertices.

Let me try a different approach. What if we use a 3-regular graph with girth 6?
The smallest 3-regular graph with girth 6 is the Heawood graph (14 vertices). So no 12-vertex 3-regular graph can have girth 6.

So the best we can do is girth 4 or 5.

For girth 5: The Petersen graph is 3-regular on 10 vertices with girth 5. We need 12 vertices.
Can we extend the Petersen graph to 12 vertices while maintaining 3-regularity and girth 5?

Actually, let's just go with two K_{3,3} components which gives 2 distinct cycle lengths. This is likely optimal.

But wait - can we do better? Can we have a graph where all cycles are the same length?

For a 3-regular graph, if every cycle has length exactly 4, then the graph must be a disjoint union of K_{2,2} components (which are 2-regular, not 3-regular) or... actually this is impossible for 3-regular graphs because at each vertex, the three edges create multiple cycle structures.

The answer is: 2 distinct cycle lengths is the minimum for a 3-regular triangle-free graph.

Let me construct two K_{3,3} graphs.
"""


def construct():
    """
    Constructs two disjoint K_{3,3} components.
    Each K_{3,3} has 6 vertices, degree 3, girth 4.
    Cycle lengths in K_{3,3} are 4 and 6.
    Total distinct cycle lengths: {4, 6} -> 2.
    """
    edges = []
    
    # First K_{3,3} on vertices 0,1,2 and 3,4,5
    part1 = [0, 1, 2]
    part2 = [3, 4, 5]
    for u in part1:
        for v in part2:
            edges.append([int(u), int(v)])
            
    # Second K_{3,3} on vertices 6,7,8 and 9,10,11
    part1 = [6, 7, 8]
    part2 = [9, 10, 11]
    for u in part1:
        for v in part2:
            edges.append([int(u), int(v)])
            
    # Deduplicate and normalize
    seen = set()
    final = []
    for e in edges:
        u, v = int(e[0]), int(e[1])
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            final.append([u, v])
    
    final.sort()
    return final
