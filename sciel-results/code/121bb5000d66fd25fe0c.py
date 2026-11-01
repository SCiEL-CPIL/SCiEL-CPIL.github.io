<<<<<<< SEARCH
def _direct_bfs_construct(k, g, deadline):
    """
    Build a k-regular graph with girth >= g by growing from a root vertex
    using non-backtracking BFS. At each layer, new vertices are created to
    avoid short cycles. The boundary (vertices at depth >= g//2 - 1) is then
    matched to achieve regularity without creating short cycles.
    
    For g=16, k=3: layers 0..7 from root. At layer 7, we have 2*3^6 = 1458
    boundary vertices (in the tree). We need to match them in pairs (each
    vertex needs 1 more edge) to get 3-regularity. The key constraint is
    that matched vertices must be "far apart" in the tree (distance >= g
    in the tree, or at least not creating a cycle < g).
    
    Actually for g=16, a cycle of length 16 through the root would go
    root->...->v (depth 8) and back. So we need depth >= 8 for the
    "far side". Let's use depth 8: 2*3^7 = 4374 boundary vertices.
    That gives n = sum of tree up to depth 7 + boundary matching.
    
    Total vertices in full binary-3 tree to depth d: 1 + 3 + 3*3 + ... = 1 + 3*(3^d - 1)/2
    For d=7: 1 + 3*(2187-1)/2 = 1 + 3*1093 = 3280. Plus we need to match
    boundary vertices among themselves.
    
    Actually, let me think more carefully. In a 3-regular graph with girth 16:
    - From any vertex, the non-backtracking tree has 3 vertices at depth 1,
      3*2=6 at depth 2, 3*2^2=12 at depth 3, ..., 3*2^(d-1) at depth d.
    - Total up to depth 7: 3*(2^8-1) = 3*255 = 765, plus root = 766.
    - At depth 7, we have 3*2^6 = 192 vertices, each needing 1 more edge.
    - We match these 192 vertices in 96 pairs. Each pair adds an edge.
    - Total vertices: 766. Total edges: 766*3/2 = 1149.
    - Tree edges: 765. Matching edges: 96. Total: 861. That's not 1149.
    
    Wait, I'm confusing myself. Let me reconsider.
    
    In a 3-regular graph, from vertex v:
    - 3 neighbors at distance 1
    - 3*2 = 6 at distance 2 (each neighbor has 2 new neighbors, excluding v)
    - 3*2^2 = 12 at distance 3
    - ...
    - 3*2^(d-1) at distance d
    
    For girth 16, the tree is unique up to distance 7 (since 2*7+1 = 15 < 16).
    At distance 8, vertices might start to be shared (creating cycles of length 16).
    
    So the minimal structure: 1 + 3 + 6 + 12 + 24 + 48 + 96 + 192 = 383 vertices
    in the "tree part" up to depth 7. At depth 7, each vertex has used 2 of its
    3 edges (one to parent, one to... wait, no.
    
    Actually: root has 3 edges to depth-1 vertices. Each depth-1 vertex has 3 edges:
    1 to root, 2 to depth-2. Each depth-2 vertex: 1 to parent, 2 to depth-3.
    ... Each depth-7 vertex: 1 to parent, 2 to... it needs 2 more neighbors.
    
    So at depth 7, we have 3*2^6 = 192 vertices, each needing 2 more edges.
    Total "dangling" half-edges: 192 * 2 = 384. We need to pair them up: 192 edges.
    
    Total vertices: 1 + 3 + 6 + 12 + 24 + 48 + 96 + 192 = 383.
    Total edges: 383 * 3 / 2 = 574.5... not integer! 383 is odd, so 383*3/2 is not integer.
    
    Hmm, that's a problem. The tree up to depth 7 has 383 vertices, but 383 is odd,
    so we can't have a 3-regular graph on 383 vertices (sum of degrees = 383*3 = 1149,
    which is odd, impossible).
    
    So we need at least 384 vertices. This means we need to add at least 1 more vertex
    or restructure. The Moore bound says n >= 510 for non-bipartite girth 16, 3-regular.
    Wait, let me recompute the Moore bound.
    
    For k-regular, girth g (odd): n >= k * sum_{i=0}^{(g-1)/2} (k-1)^i
    For k=3, g=16 (even): n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i = 2 * sum_{i=0}^{7} 2^i = 2*255 = 510.
    
    So n >= 510. The tree up to depth 7 has 383 vertices from one root, but we need
    the graph to be 3-regular globally. The Moore bound of 510 accounts for this.
    
    Let me try a different approach: build a tree of depth 8 from a root (to get
    enough vertices), then match boundary vertices.
    
    Tree to depth 8: 1 + 3 + 6 + 12 + 24 + 48 + 96 + 192 + 384 = 766 vertices.
    766 is even, so 3-regular is possible (766*3/2 = 1149 edges).
    At depth 8: 384 vertices, each needing 1 more edge (used 2: parent + ... wait).
    
    Actually: depth-0 (root): 3 edges used (to depth 1). Done.
    Depth-1: 3 vertices, each has 1 edge to root, needs 2 more.
    Depth-2: 6 vertices, each has 1 edge to parent, needs 2 more.
    ...
    Depth-7: 192 vertices, each has 1 edge to parent, needs 2 more.
    Depth-8: 384 vertices, each has 1 edge to parent, needs 2 more.
    
    Wait, that's not right either. In the tree:
    - Root connects to 3 depth-1 vertices.
    - Each depth-1 vertex connects to root + 2 depth-2 vertices = 3 edges. Done!
    - Each depth-2 vertex connects to 1 depth-1 parent + 2 depth-3 = 3 edges. Done!
    - ...
    - Each depth-7 vertex connects to 1 depth-6 parent + 2 depth-8 = 3 edges. Done!
    - Each depth-8 vertex connects to 1 depth-7 parent, needs 2 more.
    
    So only depth-8 vertices need more edges: 384 vertices * 2 = 768 half-edges = 384 edges.
    Total edges: (tree edges) + 384 = (766-1) + 384 = 765 + 384 = 1149. Check: 766*3/2 = 1149. ✓
    
    So we need to match 384 vertices (at depth 8) in pairs, adding 384 edges.
    The constraint: no two matched vertices should be connected by a path of length
    < 15 in the tree (otherwise the cycle would be < 16).
    
    Two depth-8 vertices are connected by a tree path of length = 2*8 - 2*LCA_depth
    where LCA_depth is the depth of their lowest common ancestor.
    For the cycle to be >= 16: tree_path_length + 1 (the matching edge) >= 16
    => tree_path_length >= 15
    => 16 - 2*LCA_depth >= 15
    => 2*LCA_depth <= 1
    => LCA_depth <= 0
    
    So the LCA of any two matched vertices must be the root (depth 0) or they must
    be in different "branches" from the root. Since the root has 3 children, vertices
    in different root-branches have LCA = root (depth 0), giving path length 16.
    Vertices in the same root-branch have LCA at depth >= 1, giving path length <= 14,
    which would create a cycle of length <= 15. NOT ALLOWED.
    
    So we can only match vertices from DIFFERENT root-branches!
    Branch 0 has 384/3... wait, 384 = 3 * 128. So each branch has 128 depth-8 vertices.
    We need to match vertices across branches: 128 from branch 0, 128 from branch 1, 128 from branch 2.
    Total cross-branch pairs: we need 384/2 = 192 pairs, but each pair must be cross-branch.
    With 3 branches of 128 each, the maximum matching is: pair branch 0 with branch 1 (128 pairs)
    and branch 0 with branch 2... but branch 0 only has 128 vertices.
    
    Actually: we have 384 vertices to pair up into 192 edges, each edge connecting
    two vertices from different branches. This is a 3-partite matching problem.
    Branch sizes: 128, 128, 128. Total: 384. We need 192 edges, each using one vertex
    from each of two different branches.
    
    Max edges: floor(384/2) = 192. Can we achieve this? Yes: pair 128 from B0 with 128 from B1,
    and... wait that uses all of B0 and B1, leaving 128 from B2 unmatched. That gives only 128 edges.
    
    Better: pair 64 from B0 with 64 from B1, 64 from B0 with 64 from B2, 64 from B1 with 64 from B2.
    That's 64+64+64 = 192 edges. ✓
    
    So the construction works! n = 766 vertices.
    
    But wait - we also need to check that the matching edges don't create cycles
    shorter than 16 that don't go through the root. Two vertices in different branches
    have tree distance 16, so the cycle through the matching edge has length 17 (16 tree
    edges + 1 matching edge). That's fine.
    
    But could two matching edges create a shorter cycle? E.g., if we match v1-v2 (B0-B1)
    and v3-v4 (B0-B1), could v1-v2-v4-v3-v1 form a short cycle? The tree distance between
    v2 and v4 (both in B1) is at most 14 (if they share a parent at depth 7). So
    v1-v2 (matching) + tree(v2,v4) + v4-v3 (matching) + tree(v3,v1) could be short.
    tree(v2,v4) >= 0 (if v2=v4, but they're distinct), actually min tree distance between
    two depth-8 vertices in the same branch is 2 (if they share a parent at depth 7).
    tree(v3,v1) >= 2 similarly.
    So cycle: 1 + 2 + 1 + 2 = 6. That's way too short!
    
    So we need to be more careful. The matching must ensure that no two matching edges
    create a short cycle. This is the hard part.
    
    Let me reconsider. The constraint is: for any two matching edges (a,b) and (c,d),
    the cycle a-b + tree(b,c) + c-d + tree(d,a) must have length >= 16.
    Since tree(b,c) >= 2 (both in same branch, min distance 2) and tree(d,a) >= 2,
    we get cycle >= 1+2+1+2 = 6. So we need tree(b,c) + tree(d,a) >= 14.
    
    This means b and c must be "far apart" in the tree. If b is in B1 and c is in B1,
    their tree distance is 16 - 2*LCA_depth. For distance >= 14, we need LCA_depth <= 1.
    So b and c must be in different sub-branches of B1 (different depth-1 children of B1's root child).
    
    B1's root child (call it b1) has 2 children at depth 2, each with 2 children at depth 3, etc.
    At depth 8, B1 has 2^7 = 128 vertices. They're organized as 2 subtrees of 64 each
    (under b1's two children).
    
    So within B1, we can split into 2 groups of 64. Similarly for B0 and B2.
    
    Matching: B0_group0 (64) <-> B1_group0 (64): 64 edges
              B0_group1 (64) <-> B2_group0 (64): 64 edges  
              B1_group1 (64) <-> B2_group1 (64): 64 edges
    
    Check: for two edges in the first group, b,c both in B1_group0. Their LCA is at depth 2
    (b1's child), so tree distance = 16-4 = 12. d,a both in B0_group0, LCA at depth 2,
    distance = 12. Cycle: 1+12+1+12 = 26 >= 16. ✓
    
    For edges in different groups: e.g., (a in B0_g0, b in B1_g0) and (c in B0_g1, d in B2_g0).
    tree(b,c): b in B1_g0, c in B0_g1. LCA is root, distance = 16. ✓
    tree(d,a): d in B2_g0, a in B0_g0. LCA is root, distance = 16. ✓
    Cycle: 1+16+1+16 = 34. ✓
    
    For (a in B0_g0, b in B1_g0) and (c in B1_g1, d in B2_g0):
    tree(b,c): b in B1_g0, c in B1_g1. LCA is b1 (depth 1), distance = 16-2 = 14.
    tree(d,a): d in B2_g0, a in B0_g0. LCA is root, distance = 16.
    Cycle: 1+14+1+16 = 32. ✓
    
    For (a in B0_g0, b in B1_g0) and (c in B0_g1, d in B2_g1):
    tree(b,c): b in B1_g0, c in B0_g1. LCA is root, distance = 16.
    tree(d,a): d in B2_g1, a in B0_g0. LCA is root, distance = 16.
    Cycle: 1+16+1+16 = 34. ✓
    
    Great! So the construction works with n = 766.
    
    But wait, I need to double-check: are there any other short cycles I'm missing?
    A cycle could use multiple matching edges. E.g., 3 matching edges:
    (a1,b1), (a2,b2), (a3,b3) forming a cycle a1-b1-tree-b2-a2-tree-b3-a3-tree-a1.
    Length: 3 (matching edges) + tree(b1,b2) + tree(a2,b3) + tree(a3,a1).
    Min tree distances: if all in "compatible" groups, min is 12 or 14 or 16.
    So min cycle with 3 matching edges: 3 + 12 + 12 + 12 = 39. Fine.
    
    What about a cycle using 1 matching edge and going around the tree?
    The tree is a tree, so any cycle must use at least 1 matching edge.
    With 1 matching edge (a,b): cycle is a-b + tree(b,a). tree(b,a) = 16 (different branches).
    Cycle length = 17. Fine.
    
    I think this construction is valid! n = 766.
    
    But the Moore bound is 510, and best known is 936. 766 < 936, so this would be an improvement!
    But wait, is 766 actually achievable? Let me verify the tree structure more carefully.
    
    Actually, I realize there might be an issue. The "tree" I'm constructing is not quite
    a tree in the graph - it's the non-backtracking tree from the root. In the actual graph,
    two different paths from the root could reach the same vertex (creating a cycle). But I'm
    constructing the graph FROM the tree, so each vertex is unique. The issue is whether
    the resulting graph is simple (no multi-edges) and whether all constraints are satisfied.
    
    Let me code this up properly.
    """
    if k != 3:
        return None
    
    # Build the non-backtracking tree from root to depth 8
    # Each node: (parent, branch_index) where branch_index is 0,1,2 for which child
    # Root: depth 0
    # Depth 1: 3 nodes (children of root)
    # Depth d: each node has 2 children (excluding parent)
    
    # Represent tree nodes by their path from root
    # Node at depth d is identified by a tuple of length d: (c1, c2, ..., cd)
    # where ci is 0 or 1 (which child of the previous node)
    # The root's children are indexed 0, 1, 2 (the 3 branches)
    
    # Actually, let me use a simpler representation.
    # Level 0: [0] (root)
    # Level 1: [1, 2, 3] (3 children of root)
    # Level 2: each level-1 node has 2 children
    # Level d: each level-(d-1) node has 2 children
    
    # Assign IDs:
    # Root: 0
    # Level 1: 1, 2, 3
    # Level 2: 4..9 (2 per level-1 node)
    # Level d: 3*2^(d-1) nodes, IDs from sum_{i=1}^{d-1} 3*2^(i-1) to sum_{i=1}^{d} 3*2^(i-1)
    
    # Total nodes up to level 8: 1 + 3 + 6 + 12 + 24 + 48 + 96 + 192 + 384 = 766
    
    n = 766
    edges = []
    
    # Build tree edges
    # Root (0) connects to 1, 2, 3
    edges.append((0, 1))
    edges.append((0, 2))
    edges.append((0, 3))
    
    # Level 1 nodes: 1, 2, 3
    # Level 2: node 1 gets children 4,5; node 2 gets 6,7; node 3 gets 8,9
    # Level 3: node 4 gets 10,11; node 5 gets 12,13; etc.
    
    # General: level d nodes have IDs from start_d to end_d
    # start_1 = 1, end_1 = 3
    # start_d = 1 + 3*(2^(d-1) - 1) for d >= 1... let me just compute
    
    # Level 1: IDs 1-3 (3 nodes)
    # Level 2: IDs 4-9 (6 nodes)
    # Level 3: IDs 10-21 (12 nodes)
    # Level d: start = 1 + 3*(2^(d-1)-1), count = 3*2^(d-1)
    
    level_start = [0] * 10  # 1-indexed
    level_start[1] = 1
    for d in range(2, 9):
        level_start[d] = level_start[d-1] + 3 * (2 ** (d-2))
    # level_start[1]=1, level_start[2]=4, level_start[3]=10, level_start[4]=22,
    # level_start[5]=46, level_start[6]=94, level_start[7]=190, level_start[8]=382
    
    # Parent of level-d node:
    # Level 1: parent is root (0)
    # Level d (d>=2): node at position p in level d (0-indexed) has parent at position p//2 in level d-1
    
    # Add tree edges
    for d in range(1, 9):
        count_d = 3 * (2 ** (d-1))
        start_d = level_start[d]
        if d == 1:
            # Children of root
            for i in range(3):
                child_id = start_d + i
                edges.append((0, child_id))
        else:
            count_prev = 3 * (2 ** (d-2))
            start_prev = level_start[d-1]
            for i in range(count_d):
                parent_idx = i // 2
                parent_id = start_prev + parent_idx
                child_id = start_d + i
                edges.append((parent_id, child_id))
    
    # Now add matching edges at level 8
    # Level 8: 384 nodes, IDs 382 to 765
    # Group by branch (which child of root) and sub-branch
    # Branch 0: nodes descended from node 1 (level 1)
    # Branch 1: nodes descended from node 2
    # Branch 2: nodes descended from node 3
    
    # Within each branch, split into 2 groups (by which child of the level-1 node)
    # Branch 0, group 0: descended from node 1's child 0 (node 4)
    # Branch 0, group 1: descended from node 1's child 1 (node 5)
    # etc.
    
    # Level 8 nodes in branch b, group g:
    # They descend from level-2 node at position (b*2 + g) in level 2
    # Level 2 node ID: 4 + b*2 + g
    # From level 2 to level 8: 6 more levels, each with 2 children
    # So 2^6 = 64 nodes per (branch, group)
    
    # Group IDs:
    # Branch 0, Group 0: descend from node 4 (level 2)
    # Branch 0, Group 1: descend from node 5
    # Branch 1, Group 0: descend from node 6
    # Branch 1, Group 1: descend from node 7
    # Branch 2, Group 0: descend from node 8
    # Branch 2, Group 1: descend from node 9
    
    # For a level-2 node with ID x, its level-8 descendants:
    # Level 3: 2 children, Level 4: 4, ..., Level 8: 64
    # The level-8 descendants of level-2 node x are contiguous in the level-8 ID range.
    
    # Level 8 start: 382
    # Level 2 node 4 (first node in level 2) has level-8 descendants:
    # Level 3: 10, 11 (2 nodes)
    # Level 4: 22-25 (4 nodes)
    # Level 5: 46-53 (8 nodes)
    # Level 6: 94-101 (8... wait, 16 nodes: 94-109)
    # Level 7: 190-221 (32 nodes)
    # Level 8: 382-445 (64 nodes)
    
    # In general, level-2 node at position p (0-indexed) in level 2 has level-8 descendants
    # starting at 382 + p*64, count 64.
    
    # Matching:
    # B0G0 (382-445) <-> B1G0 (510-573): 64 edges
    # B0G1 (446-509) <-> B2G0 (574-637): 64 edges
    # B1G1 (574-637) <-> B2G1 (638-701): wait, that's wrong.
    
    # Let me recompute:
    # Level 2 positions: 0,1,2,3,4,5 (nodes 4,5,6,7,8,9)
    # Position 0 (node 4): B0G0, level-8 IDs 382-445
    # Position 1 (node 5): B0G1, level-8 IDs 446-509
    # Position 2 (node 6): B1G0, level-8 IDs 510-573
    # Position 3 (node 7): B1G1, level-8 IDs 574-637
    # Position 4 (node 8): B2G0, level-8 IDs 638-701
    # Position 5 (node 9): B2G1, level-8 IDs 702-765
    
    # Matching:
    # B0G0 <-> B1G0: pair 382+i with 510+i for i=0..63
    # B0G1 <-> B2G0: pair 446+i with 638+i for i=0..63
    # B1G1 <-> B2G1: pair 574+i with 702+i for i=0..63
    
    for i in range(64):
        edges.append((382 + i, 510 + i))
        edges.append((446 + i, 638 + i))
        edges.append((574 + i, 702 + i))
    
    # Verify: total edges should be n*k/2 = 766*3/2 = 1149
    # Tree edges: 765 (each non-root node has exactly one parent edge)
    # Matching edges: 192
    # Total: 765 + 192 = 957. That's not 1149!
    
    # I made an error. Let me recount.
    # Root: 3 edges (to level 1)
    # Level 1: 3 nodes, each has 1 edge to root + 2 edges to level 2 = 3 edges each. Total from level 1: 3*3 = 9, but 3 are shared with root. New edges: 6.
    # Level 2: 6 nodes, each has 1 edge to parent + 2 to level 3. New edges: 12.
    # Level 3: 12 nodes, new edges: 24.
    # Level 4: 24 nodes, new edges: 48.
    # Level 5: 48 nodes, new edges: 96.
    # Level 6: 96 nodes, new edges: 192.
    # Level 7: 192 nodes, new edges: 384.
    # Level 8: 384 nodes, new edges: 0 (they're leaves, need 2 more edges each).
    
    # Total tree edges: 3 + 6 + 12 + 24 + 48 + 96 + 192 + 384 = 765. ✓
    # Level 8 vertices need 2 more edges each: 384*2 = 768 half-edges = 384 edges.
    # But I only added 192 matching edges!
    
    # I see the problem. Each level-8 vertex needs 2 more edges, not 1.
    # So I need 384 matching edges, not 192.
    # But 384 vertices can only form 192 pairs (each edge uses 2 vertices).
    # 192 edges * 2 = 384 vertex-endpoints. But we need 768 vertex-endpoints.
    
    # This means the tree-to-depth-8 approach doesn't work directly.
    # Each leaf needs 2 edges, but pairing only gives 1.
    
    # Solution: extend to depth 9.
    # Level 9: 768 nodes. Each has 1 edge to parent, needs 2 more.
    # Total vertices: 766 + 768 = 1534.
    # That's too many.
    
    # Alternative: use depth 7 for the tree, and match level-7 vertices.
    # Level 7: 192 vertices, each needs 2 more edges. 192*2 = 384 half-edges = 192 edges.
    # Total vertices: 383. But 383 is odd, can't be 3-regular.
    
    # Alternative: depth 7 tree + add some extra vertices.
    # 383 vertices in tree, level 7 has 192 vertices needing 2 edges each.
    # Add 1 extra vertex (total 384, even). The extra vertex needs 3 edges.
    # Total half-edges to satisfy: 192*2 + 3 = 387. Odd! Can't pair.
    
    # Add 2 extra vertices (total 385, odd). No good.
    # Add 3 extra vertices (total 386, even). Half-edges: 192*2 + 3*3 = 384+9 = 393. Odd!
    # Add 4 extra (total 387, odd). No.
    # Add 5 extra (total 388, even). Half-edges: 384 + 15 = 399. Odd!
    # Add 6 extra (total 389, odd). No.
    # Add 7 extra (total 390, even). Half-edges: 384 + 21 = 405. Odd!
    
    # Hmm, 192*2 + 3*m = 384 + 3m. For this to be even, 3m must be even, so m must be even.
    # m=0: 384 (even), total vertices 383 (odd). No.
    # m=2: 390 (even), total 385 (odd). No.
    # m=4: 396 (even), total 387 (odd). No.
    # m=6: 402 (even), total 389 (odd). No.
    
    # 383 + m must be even, so m must be odd. But 384+3m must be even, so m must be even.
    # Contradiction! So depth-7 tree + extra vertices doesn't work.
    
    # Back to depth 8: 766 vertices (even). Level 8: 384 vertices, each needs 2 edges.
    # 384*2 = 768 half-edges = 384 edges needed.
    # But 384 vertices can only be paired into 192 edges.
    # We need 384 edges but can only make 192 from pairing.
    
    # The issue: in the tree, level-8 vertices only have 1 edge (to parent). They need 2 more.
    # But when we pair them, each pair gives 1 edge to each vertex, so each vertex gets 1 more edge.
    # They still need 1 more edge each!
    
    # So the tree approach to depth 8 gives vertices at depth 8 with degree 1 (parent only).
    # They need degree 3, so 2 more edges each. Pairing gives 1. We need another 1 per vertex.
    # That means we need to add MORE edges, but all vertices are already in the graph.
    
    # I think I'm confusing myself. Let me restart the counting.
    
    # In the TREE:
    # Root (depth 0): 3 neighbors (depth 1). Degree 3. Done.
    # Depth 1: 3 vertices. Each has 1 neighbor (root) + 2 neighbors (depth 2). Degree 3. Done.
    # Depth 2: 6 vertices. Each has 1 parent + 2 children (depth 3). Degree 3. Done.
    # ...
    # Depth 7: 192 vertices. Each has 1 parent + 2 children (depth 8). Degree 3. Done.
    # Depth 8: 384 vertices. Each has 1 parent (depth 7). Degree 1. Needs 2 more.
    
    # So the tree has 766 vertices, 765 edges. Vertices at depth 0-7 have degree 3.
    # Vertices at depth 8 have degree 1. We need to add edges to bring depth-8 vertices to degree 3.
    # Each depth-8 vertex needs 2 more edges. Total: 384*2 = 768 half-edges = 384 edges.
    # But these 384 edges must connect depth-8 vertices to OTHER vertices.
    # The only available vertices are the other 383 depth-8 vertices (since depth 0-7 are full).
    # So we need to add 384 edges among the 384 depth-8 vertices.
    # Each depth-8 vertex needs degree 2 in this subgraph.
    # A 2-regular graph on 384 vertices is a union of cycles.
    # We need each cycle to have length >= 16 (to maintain girth 16).
    # 384 vertices in cycles of length >= 16: e.g., 24 cycles of length 16, or 2 cycles of length 192, etc.
    
    # But wait - the "distance" between two depth-8 vertices in the full graph is not just
    # their tree distance. If we add an edge between two depth-8 vertices, the cycle formed
    # is: v1 - v2 (new edge) + tree_path(v2, v1). The tree path goes up to their LCA and back down.
    # For vertices in different branches (LCA = root), tree path length = 16. Cycle = 17. OK.
    # For vertices in the same branch but different sub-branches (LCA at depth 1), tree path = 14. Cycle = 15. NOT OK!
    
    # So we can only connect depth-8 vertices from DIFFERENT root-branches.
    # Branch sizes: 128 each. We need a 2-regular graph on 384 vertices where each vertex
    # is connected to 2 vertices from other branches.
    
    # This is a 3-partite 2-regular graph. We can make it a single cycle of length 384
    # that alternates between branches, or multiple cycles.
    
    # Simplest: 3 cycles, each of length 128, going B0->B1->B2->B0->... 
    # But a cycle of length 128 in the "matching subgraph" means 128 edges.
    # 3 cycles * 128 edges = 384 edges. ✓
    # Each cycle: v0(B0) - v1(B1) - v2(B2) - v3(B0) - ... - v127(B2) - v0(B0)
    # Check girth: consecutive vertices are in different branches, so tree distance = 16.
    # Cycle in full graph: 128 matching edges + 128 tree paths of length 16 = 128 + 128*16 = 2176.
    # That's way more than 16. But we need to check for SHORTER cycles too.
    
    # A shorter cycle might use 2 matching edges: v1(B0)-v2(B1)-v3(B2)-v4(B0) where
    # v1-v2 and v3-v4 are matching edges, and v2-v3, v4-v1 are tree paths.
    # v2(B1) to v3(B2): different branches, tree distance 16.
    # v4(B0) to v1(B0): same branch! Tree distance could be as low as 2.
    # Cycle: 1 + 16 + 1 + 2 = 20. OK, >= 16.
    
    # What about v1(B0)-v2(B1)-v3(B0) where v1-v2 is matching, v2-v3 is tree, v3-v1 is matching?
    # v2(B1) to v3(B0): different branches, tree distance 16.
    # v3(B0) to v1(B0): same branch, min tree distance 2.
    # Cycle: 1 + 16 + 1 = 18. OK.
    
    # What about 3 matching edges: v1(B0)-v2(B1)-v3(B2)-v4(B0) with v1-v2, v2-v3, v3-v4 matching?
    # Wait, that's 3 matching edges and 0 tree edges between them? No, in the full graph,
    # v2 and v3 are connected by the matching edge directly. So the cycle is v1-v2-v3-v4-v1
    # with edges: matching, matching, matching, tree(v4,v1).
    # tree(v4,v1): both B0, min distance 2.
    # Cycle: 3 + 2 = 5. WAY TOO SHORT!
    
    # So we can't have 3 consecutive matching edges in a cycle. The matching subgraph
    # must not contain any path of length 2 where the endpoints are in the same branch.
    
    # In my proposed cycle B0->B1->B2->B0->..., consecutive vertices alternate branches.
    # A path of length 2: B0->B1->B2 (endpoints B0 and B2, different). OK.
    # B1->B2->B0 (endpoints B1 and B0, different). OK.
    # B2->B0->B1 (endpoints B2 and B1, different). OK.
    # So no path of length 2 has same-branch endpoints. ✓
    
    # What about a path of length 1 (single matching edge)? Endpoints are always
    # in different branches by construction. ✓
    
    # What about a cycle using 4 matching edges? v1(B0)-v2(B1)-v3(B2)-v4(B0)-v5(B1)-v1(B0)?
    # Wait, that's not a valid cycle in the matching subgraph unless v5-v1 is an edge.
    # In the 128-cycle, v5 is not adjacent to v1 (they're 4 apart in the cycle).
    # So this isn't a cycle in the matching subgraph.
    
    # I think the 3-cycle-of-128 construction works. Let me verify more carefully.
    # The matching subgraph is 3 disjoint cycles of length 128: C0 (B0 vertices), C1 (B1), C2 (B2)?
    # No, that's not right. Each cycle goes through all 3 branches.
    
    # Let me redefine: Cycle: x0(B0), y0(B1), z0(B2), x1(B0), y1(B1), z1(B2), ..., x42(B0), y42(B1), z42(B2), back to x0.
    # That's 128 vertices per cycle, 42.67 groups... 128/3 is not integer.
    
    # Let me use 3 cycles of length 128: 
    # Cycle 1: 64 B0 + 64 B1 vertices, alternating: x0(B0), y0(B1), x1(B0), y1(B1), ..., x63(B0), y63(B1), back to x0.
    # That's 128 vertices, 128 edges. Each vertex has degree 2 in the matching subgraph. ✓
    # But we have 128 B0, 128 B1, 128 B2 vertices. Cycle 1 uses 64 B0 + 64 B1.
    # Cycle 2: 64 B0 + 64 B2. Uses remaining 64 B0 + 64 B2.
    # Cycle 3: 64 B1 + 64 B2. Uses remaining 64 B1 + 64 B2.
    # Total: 128 B0 + 128 B1 + 128 B2. ✓
    
    # Check Cycle 1: x0(B0)-y0(B1)-x1(B0)-y1(B1)-...-x63(B0)-y63(B1)-x0(B0)
    # Consecutive vertices are in different branches. ✓
    # Path of length 2: x0-y0-x1: endpoints x0(B0) and x1(B0), same branch!
    # Tree distance x0 to x1: both in B0, min 2.
    # Cycle in full graph: x0-y0 (matching) + y0-x1 (tree, length 16) + x1-x0 (tree, length >= 2).
    # Wait, x1-x0 is NOT a matching edge. It's a tree path.
    # So the cycle is: x0 - y0 (matching edge) - x1 (tree path y0 to x1, length 16) - x0 (tree path x1 to x0, length >= 2).
    # Total: 1 + 16 + 2 = 19. OK.
    
    # But wait, is there a shorter cycle? What about x0-y0-x1 where x0-y0 is matching and y0-x1 is tree?
    # That's a path, not a cycle. To close it, we need x1 back to x0, which is tree path.
    # So the cycle is 1 + 16 + tree(x1,x0) >= 1 + 16 + 2 = 19. ✓
    
    # What about a cycle using 2 matching edges from Cycle 1 and 1 from Cycle 2?
    # That would require a vertex to be in two cycles, which isn't the case.
    
    # I think this works! n = 766, girth >= 16.
    
    # But wait, I need to double-check that no two vertices in the same "pair" of the
    # matching are actually the same vertex (no self-loops) and that edges are unique.
    
    # Let me code this up.
    
    # Actually, let me just verify the girth computationally for this specific construction.
    # If it passes, great. If not, I'll fall back to other methods.
    
    # Let me build the graph and verify.
    if time.time() > deadline - 5:
        return None
    
    # Build tree
    n = 766
    edges = []
    
    # Level starts (1-indexed levels)
    # Level 1: start=1, count=3
    # Level 2: start=4, count=6
    # Level 3: start=10, count=12
    # Level 4: start=22, count=24
    # Level 5: start=46, count=48
    # Level 6: start=94, count=96
    # Level 7: start=190, count=192
    # Level 8: start=382, count=384
    
    level_info = [
        (1, 3),    # level 1: start=1, count=3
        (4, 6),    # level 2
        (10, 12),  # level 3
        (22, 24),  # level 4
        (46, 48),  # level 5
        (94, 96),  # level 6
        (190, 192),# level 7
        (382, 384) # level 8
    ]
    
    # Tree edges
    # Root (0) to level 1
    for i in range(3):
        edges.append((0, 1 + i))
    
    # Level d to level d+1
    for d in range(7):
        start_d, count_d = level_info[d]
        start_next, count_next = level_info[d + 1]
        for i in range(count_d):
            parent = start_d + i
            child1 = start_next + 2 * i
            child2 = start_next + 2 * i + 1
            edges.append((parent, child1))
            edges.append((parent, child2))
    
    # Matching edges at level 8
    # B0: nodes 382-509 (128 nodes), B1: 510-637, B2: 638-765
    # B0G0: 382-445, B0G1: 446-509
    # B1G0: 510-573, B1G1: 574-637
    # B2G0: 638-701, B2G1: 702-765
    
    # Cycle 1: B0G0 <-> B1G0 (64 pairs, forming a 128-cycle)
    # x_i = 382 + i, y_i = 510 + i for i = 0..63
    # Edges: x_i - y_i and y_i - x_{i+1} (mod 64)
    for i in range(64):
        x_i = 382 + i
        y_i = 510 + i
        x_next = 382 + ((i + 1) % 64)
        edges.append((x_i, y_i))
        edges.append((y_i, x_next))
    
    # Cycle 2: B0G1 <-> B2G0
    for i in range(64):
        x_i = 446 + i
        y_i = 638 + i
        x_next = 446 + ((i + 1) % 64)
        edges.append((x_i, y_i))
        edges.append((y_i, x_next))
    
    # Cycle 3: B1G1 <-> B2G1
    for i in range(64):
        x_i = 574 + i
        y_i = 702 + i
        x_next = 574 + ((i + 1) % 64)
        edges.append((x_i, y_i))
        edges.append((y_i, x_next))
    
    # Verify edge count
    if len(edges) != 1149:
        # Fallback
        return None
    
    # Verify 3-regularity and girth
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    
    # Quick girth check: BFS from a few vertices
    for src in [0, 1, 382, 510, 638]:
        if _bfs_short_cycle_nb(src, adj, n, 16):
            return None
    
    return n, edges
=======
def _direct_bfs_construct(k, g, deadline):
    """
    Build a k-regular graph with girth >= g using the tree-to-depth-8 approach.
    Tree to depth 8 gives 766 vertices. Level-8 vertices (384 of them) need 2 more
    edges each. We add a 2-regular subgraph on these 384 vertices where each cycle
    alternates between different root-branches to avoid short cycles.
    
    The matching subgraph consists of 3 cycles of length 128:
    - Cycle 1: 64 from B0 + 64 from B1, alternating
    - Cycle 2: 64 from B0 + 64 from B2, alternating  
    - Cycle 3: 64 from B1 + 64 from B2, alternating
    
    This ensures no path of length 2 in the matching subgraph has same-branch endpoints,
    preventing cycles shorter than 16.
    """
    if k != 3:
        return None
    
    if time.time() > deadline - 5:
        return None
    
    n = 766
    edges = []
    
    # Level starts
    level_info = [
        (1, 3),    # level 1
        (4, 6),    # level 2
        (10, 12),  # level 3
        (22, 24),  # level 4
        (46, 48),  # level 5
        (94, 96),  # level 6
        (190, 192),# level 7
        (382, 384) # level 8
    ]
    
    # Tree edges
    for i in range(3):
        edges.append((0, 1 + i))
    
    for d in range(7):
        start_d, count_d = level_info[d]
        start_next, count_next = level_info[d + 1]
        for i in range(count_d):
            parent = start_d + i
            child1 = start_next + 2 * i
            child2 = start_next + 2 * i + 1
            edges.append((parent, child1))
            edges.append((parent, child2))
    
    # Level 8: 384 nodes, IDs 382-765
    # Branch 0: 382-509, Branch 1: 510-637, Branch 2: 638-765
    # Each branch split into 2 groups of 64
    
    # Cycle 1: B0[0:64] <-> B1[0:64] alternating (128-cycle)
    for i in range(64):
        x_i = 382 + i
        y_i = 510 + i
        x_next = 382 + ((i + 1) % 64)
        edges.append((x_i, y_i))
        edges.append((y_i, x_next))
    
    # Cycle 2: B0[64:128] <-> B2[0:64] alternating
    for i in range(64):
        x_i = 446 + i
        y_i = 638 + i
        x_next = 446 + ((i + 1) % 64)
        edges.append((x_i, y_i))
        edges.append((y_i, x_next))
    
    # Cycle 3: B1[64:128] <-> B2[64:128] alternating
    for i in range(64):
        x_i = 574 + i
        y_i = 702 + i
        x_next = 574 + ((i + 1) % 64)
        edges.append((x_i, y_i))
        edges.append((y_i, x_next))
    
    # Verify
    if len(edges) != 1149:
        return None
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    
    # Full girth verification
    if not _verify_girth_full(n, adj, 16):
        return None
    
    return n, edges
>>>>>>> REPLACE

