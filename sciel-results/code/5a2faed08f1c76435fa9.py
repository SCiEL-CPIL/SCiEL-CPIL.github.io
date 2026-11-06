def construct():
    # Define the generators for the Mathieu group M12
    # Using 0-based indexing
    # g1 is a 12-cycle
    g1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]
    # g2 is a product of 4-cycles: (0,2,3,5)(11,7,6,10)(4,9,8,1)
    # Let's map this to a list where perm[i] = j
    g2 = [0] * 12
    # (0,2,3,5) -> 0->2, 2->3, 3->5, 5->0
    g2[0] = 2
    g2[2] = 3
    g2[3] = 5
    g2[5] = 0
    # (11,7,6,10) -> 11->7, 7->6, 6->10, 10->11
    g2[11] = 7
    g2[7] = 6
    g2[6] = 10
    g2[10] = 11
    # (4,9,8,1) -> 4->9, 9->8, 8->1, 1->4
    g2[4] = 9
    g2[9] = 8
    g2[8] = 1
    g2[1] = 4
    # Fixpoints: 0,2,3,5,11,7,6,10,4,9,8,1 are all moved. 
    # Wait, are there fixpoints?
    # The cycles cover: 0,2,3,5,11,7,6,10,4,9,8,1. That's 12 points.
    # So no fixpoints.

    def perm_multiply(p, q):
        # Returns p * q, meaning apply q then p? Or p then q?
        # Standard group action: (p*q)(x) = p(q(x))
        return [p[q[i]] for i in range(12)]

    def perm_compose_block(perm, block):
        return tuple(sorted(perm[i] for i in block))

    # Base block
    base_block = (0, 1, 2, 3, 4, 5)
    
    # Generate the group M12 using BFS
    # Identity permutation
    identity = list(range(12))
    
    visited_perms = {tuple(identity)}
    queue = [identity]
    blocks = set()
    
    while queue:
        p = queue.pop(0)
        # Apply p to base block
        b = perm_compose_block(p, base_block)
        blocks.add(b)
        
        # Generate new permutations
        for g in [g1, g2]:
            new_p = perm_multiply(g, p)
            key = tuple(new_p)
            if key not in visited_perms:
                visited_perms.add(key)
                queue.append(new_p)
                
    # The orbit size should be 132
    # Convert to list of lists for the required return format
    return [list(b) for b in blocks]
