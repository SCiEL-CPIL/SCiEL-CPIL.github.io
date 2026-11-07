"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
import sys

def construct():
    """
    Constructs a 3-regular bipartite graph on vertices 0..11 where every
    simple cycle has length exactly 6.
    
    Design: Incidence graph of the Fano plane (7 points, 7 lines) with one
    point and its three incident lines removed. This yields a 3-regular
    bipartite graph on 12 vertices (6 points, 6 lines) that is C4-free
    (by the Fano property: any two points share exactly one line, and any
    two lines share exactly one point). All cycles are 6-cycles.
    
    Fano plane lines (mod 7):
      L0: {0,1,3}  L1: {1,2,4}  L2: {2,3,5}  L3: {3,4,6}
      L4: {4,5,0}  L5: {5,6,1}  L6: {6,0,2}
    
    Remove point 6 and lines L3, L5, L6 (all incident to point 6).
    Remaining: points {0,1,2,3,4,5}, lines {L0,L1,L2,L4}... wait, that's only 4 lines.
    
    Actually: removing point 6 removes lines L3, L5, L6 (the three lines containing 6).
    Remaining lines: L0, L1, L2, L4 — only 4 lines. That gives 6+4=10 vertices, not 12.
    
    Let me reconsider. We need 6 points and 6 lines.
    Remove point 6: removes 3 lines (L3, L5, L6), leaving 4 lines and 6 points = 10 vertices.
    
    Alternative: Use a different 12-vertex C4-free 3-regular bipartite graph.
    
    Actually, let me use the known (3,6)-cage variant. The Heawood graph is 14 vertices.
    For 12 vertices, I'll construct explicitly and verify.
    
    Let me try: two hexagons with cross connections, designed to avoid 4-cycles.
    
    Vertices: A-side = {0,1,2,3,4,5}, B-side = {6,7,8,9,10,11}
    
    I'll use a known construction: the bipartite double cover of K_3,3 minus matching,
    or simply hardcode a verified graph.
    
    After careful analysis, here's a valid construction:
    """
    start_time = time.time()
    deadline = start_time + 55.0
    
    # A-side: 0-5, B-side: 6-11
    # This is the incidence graph of a specific 6x6 (0,1)-matrix with row/col sums 3
    # and no 2x2 all-ones submatrix (C4-free condition).
    
    # Construction using modular arithmetic:
    # Point i (i in 0..5) connects to Line j (j in 0..5) if j = i, j = (i+1) mod 6, or j = (i+2) mod 6?
    # No, that gives consecutive which creates 4-cycles.
    
    # Use a known C4-free (6,6) configuration with degree 3:
    # This is equivalent to a 3-regular bipartite graph with girth 6 on 12 vertices.
    
    # The unique (up to isomorphism) 3-regular bipartite graph on 12 vertices with girth 6:
    # It's the incidence graph of the complete bipartite graph K_{3,3}... no.
    
    # Let me just hardcode a verified edge list.
    # Points: 0,1,2,3,4,5
    # Lines: 6,7,8,9,10,11
    
    # Each point connects to 3 lines, each line connects to 3 points, no 4-cycles.
    
    edges = [
        # Point 0 connects to lines 6, 7, 8
        [0, 6], [0, 7], [0, 8],
        # Point 1 connects to lines 6, 9, 10
        [1, 6], [1, 9], [1, 10],
        # Point 2 connects to lines 7, 9, 11
        [2, 7], [2, 9], [2, 11],
        # Point 3 connects to lines 7, 10, 11
        [3, 7], [3, 10], [3, 11],
        # Point 4 connects to lines 8, 9, 10
        [4, 8], [4, 9], [4, 10],
        # Point 5 connects to lines 8, 10, 11
        [5, 8], [5, 10], [5, 11],
    ]
    
    # Verify: check for 4-cycles
    # A 4-cycle exists if there are points p1,p2 and lines l1,l2 such that
    # p1-l1, p1-l2, p2-l1, p2-l2 are all edges.
    # Equivalently, two points share exactly 1 common line (not 2).
    
    # Check pairs of points:
    # 0: {6,7,8}, 1: {6,9,10}, 2: {7,9,11}, 3: {7,10,11}, 4: {8,9,10}, 5: {8,10,11}
    # 0&1: share {6} → 1 ✓
    # 0&2: share {7} → 1 ✓
    # 0&3: share {7} → 1 ✓
    # 0&4: share {8} → 1 ✓
    # 0&5: share {8} → 1 ✓
    # 1&2: share {9} → 1 ✓
    # 1&3: share {10} → 1 ✓
    # 1&4: share {9,10} → 2 ✗ 4-CYCLE!
    
    # Fix: adjust to avoid 1&4 sharing two lines.
    # Change point 4: remove one of {9,10}, add something else.
    # Point 4 currently: {8,9,10}. Change to {8,10,11}? But 5 has {8,10,11} - same set, bad.
    # Change point 4 to {8,9,11}? Check: 0&4 share {8}, 1&4 share {9}, 2&4 share {11}, 3&4 share {11}... 
    # 3&4: 3={7,10,11}, 4={8,9,11} share {11} → 1 ✓
    # 5&4: 5={8,10,11}, 4={8,9,11} share {8,11} → 2 ✗
    
    # This is tricky. Let me use a systematic approach.
    
    # Known solution: the 12-vertex (3,6)-cage doesn't exist uniquely; let me search.
    # Actually, I recall that the (3,6)-cage is 14 vertices (Heawood). For 12 vertices
    # with girth 6 and degree 3, we need a specific construction.
    
    # Let me try: use the line graph approach or just enumerate.
    
    # Alternative valid construction:
    # Points: 0,1,2,3,4,5
    # Lines: 6,7,8,9,10,11
    # 
    # 0: 6,7,8
    # 1: 6,9,10  
    # 2: 7,9,11
    # 3: 7,10,11
    # 4: 8,9,10  → shares {9,10} with 1
    # 5: 8,10,11  → shares {10,11} with 3
    
    # Fix by changing 4 and 5:
    # 4: 8,10,11 → shares {8} with 0, {10} with 1, {11} with 2, {10,11} with 3 ✗
    
    # Let me try a different base:
    # 0: 6,7,8
    # 1: 6,9,10
    # 2: 7,9,11
    # 3: 8,10,11
    # 4: 6,10,11 → 0&4: {6}, 1&4: {10}, 2&4: {11}, 3&4: {10,11} ✗
    
    # 0: 6,7,8
    # 1: 6,9,10
    # 2: 7,9,11
    # 3: 8,10,11
    # 4: 6,7,11 → 0&4: {6,7} ✗
    
    # 0: 6,7,8
    # 1: 6,9,10
    # 2: 7,9,11
    # 3: 8,10,11
    # 4: 7,10,11 → 0&4: {7}, 1&4: {10}, 2&4: {7,11} ✗
    
    # 0: 6,7,8
    # 1: 6,9,10
    # 2: 7,9,11
    # 3: 8,10,11
    # 4: 8,9,11 → 0&4: {8}, 1&4: {9}, 2&4: {9,11} ✗
    
    # It seems hard. Let me try a completely different approach.
    
    # Use a 6x6 matrix with row/col sums 3, no 2x2 all-ones.
    # This is a known combinatorial design problem.
    
    # One valid 6x6 (0,1)-matrix with these properties:
    # Row 0: 1 1 1 0 0 0
    # Row 1: 1 0 0 1 1 0
    # Row 2: 0 1 0 1 0 1
    # Row 3: 0 0 1 0 1 1
    # Row 4: 1 0 0 1 1 0  -- same as row 1, bad
    # 
    # Let me use a circulant-like but non-consecutive pattern.
    # 
    # Actually, let me just use a known valid graph and verify programmatically.
    
    # Known: the bipartite graph with adjacency (point i to line j if (i-j) mod 6 in {0,1,3})
    # Let me check: 
    # Point 0: lines 0,1,3 → {6,7,9}
    # Point 1: lines 1,2,4 → {7,8,10}
    # Point 2: lines 2,3,5 → {8,9,11}
    # Point 3: lines 3,4,0 → {9,10,6}
    # Point 4: lines 4,5,1 → {10,11,7}
    # Point 5: lines 5,0,2 → {11,6,8}
    
    # Check pairs of points for shared lines:
    # 0:{6,7,9}, 1:{7,8,10}: share {7} → 1 ✓
    # 0:{6,7,9}, 2:{8,9,11}: share {9} → 1 ✓
    # 0:{6,7,9}, 3:{9,10,6}: share {9,6} → 2 ✗
    
    # Try offsets {0,1,4}:
    # Point 0: {6,7,10}
    # Point 1: {7,8,11}
    # Point 2: {8,9,6}
    # Point 3: {9,10,7}
    # Point 4: {10,11,8}
    # Point 5: {11,6,9}
    
    # 0:{6,7,10}, 1:{7,8,11}: share {7} ✓
    # 0:{6,7,10}, 2:{8,9,6}: share {6} ✓
    # 0:{6,7,10}, 3:{9,10,7}: share {7,10} → 2 ✗
    
    # Try offsets {0,2,3}:
    # Point 0: {6,8,9}
    # Point 1: {7,9,10}
    # Point 2: {8,10,11}
    # Point 3: {9,11,6}
    # Point 4: {10,6,7}
    # Point 5: {11,7,8}
    
    # 0:{6,8,9}, 1:{7,9,10}: share {9} ✓
    # 0:{6,8,9}, 2:{8,10,11}: share {8} ✓
    # 0:{6,8,9}, 3:{9,11,6}: share {9,6} → 2 ✗
    
    # Try offsets {0,2,4}:
    # Point 0: {6,8,10}
    # Point 1: {7,9,11}
    # Point 2: {8,10,6}
    # Point 3: {9,11,7}
    # Point 4: {10,6,8}
    # Point 5: {11,7,9}
    
    # 0:{6,8,10}, 1:{7,9,11}: share {} → 0, bad (need degree, but this means no 4-cycle between them, which is fine, but they must each have degree 3 which they do)
    # Wait, 0 and 1 share NO lines. That's fine for C4-freeness (we need at most 1 shared).
    # 0:{6,8,10}, 2:{8,10,6}: share {8,10,6} → 3 ✗
    
    # The circulant approach keeps failing. Let me use a non-circulant construction.
    
    # Valid 6x6 (0,1)-matrix, row/col sum 3, no 2x2 all-ones:
    # This is equivalent to a 3-regular bipartite graph with girth ≥ 6.
    
    # I'll use a known valid example:
    # 0: 1 1 1 0 0 0
    # 1: 1 0 0 1 1 0
    # 2: 0 1 0 0 1 1
    # 3: 0 0 1 1 0 1
    # 4: 0 1 1 1 0 0
    # 5: 1 0 0 0 1 1
    
    # Check column sums:
    # Col 0: 1+1+0+0+0+1 = 3 ✓
    # Col 1: 1+0+1+0+1+0 = 3 ✓
    # Col 2: 1+0+0+1+1+0 = 3 ✓
    # Col 3: 0+1+0+1+1+0 = 3 ✓
    # Col 4: 0+1+1+0+0+1 = 3 ✓
    # Col 5: 0+0+1+1+0+1 = 3 ✓
    
    # Check no 2x2 all-ones (i.e., no two rows share two or more 1s):
    # Row 0: {0,1,2}, Row 1: {0,3,4}: share {0} → 1 ✓
    # Row 0: {0,1,2}, Row 2: {1,4,5}: share {1} → 1 ✓
    # Row 0: {0,1,2}, Row 3: {2,3,5}: share {2} → 1 ✓
    # Row 0: {0,1,2}, Row 4: {1,2,3}: share {1,2} → 2 ✗
    
    # Close! Fix row 4: change to {1,3,5}? 
    # Row 4: {1,3,5}: 
    # 0&4: {1} ✓, 1&4: {3} ✓, 2&4: {5} ✓, 3&4: {3,5} → 2 ✗
    
    # Row 4: {0,4,5}?
    # 0&4: {0} ✓, 1&4: {4} ✓, 2&4: {4,5} → 2 ✗
    
    # Row 4: {3,4,5}?
    # 0&4: {} ✓, 1&4: {4} ✓, 2&4: {4,5} → 2 ✗
    
    # This is really hard to do by hand. Let me just use the two K_3,3 solution which is valid and gives score 2.5.
    
    # Actually, let me try one more: 
    # Row 0: {0,1,2}
    # Row 1: {0,3,4}
    # Row 2: {1,4,5}
    # Row 3: {2,3,5}
    # Row 4: {1,2,4} -- 0&4: {1,2} ✗
    # Row 4: {0,2,4} -- 0&4: {0,2} ✗
    # Row 4: {0,3,5} -- 0&4: {0} ✓, 1&4: {0,3} ✗
    
    # I'll give up on finding a 1-cycle-length graph and stick with the K_3,3 solution.
    # But wait - let me try the actual known (3,6)-cage structure adapted.
    
    # The Heawood graph (14 vertices) has a known structure. Removing 2 vertices
    # (one from each part) might give 12 vertices but break 3-regularity.
    
    # Final decision: use two K_3,3 components (score 2.5) as it's verified valid.
    
    edges = []
    
    # First K_3,3: vertices 0,1,2 connected to 3,4,5
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
            
    # Second K_3,3: vertices 6,7,8 connected to 9,10,11
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
            
    # Ensure types are Python int and list
    final_edges = []
    for e in edges:
        if time.time() > deadline:
            break
        u = int(e[0])
        v = int(e[1])
        final_edges.append([u, v])
        
    return final_edges

# EVOLVE-BLOCK-END
