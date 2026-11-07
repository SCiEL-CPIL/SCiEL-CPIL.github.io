"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def color_instance(n, edges, lists):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert lists to bitmasks
        # Colors are 0..4, so we need 5 bits.
        doms = [0] * n
        for i in range(n):
            mask = 0
            for c in lists[i]:
                mask |= (1 << c)
            doms[i] = mask
        
        assignment = [-1] * n
        
        # We use an explicit stack for backtracking.
        # Each stack entry: (vertex, color_index_to_try_next, changes_made)
        # changes_made: list of (neighbor, original_mask) to restore
        
        # To pick the next vertex, we use MRV (Minimum Remaining Values) on unassigned vertices.
        # We maintain a set of unassigned vertices.
        unassigned = set(range(n))
        
        # Precompute degrees for tie-breaking in MRV
        degrees = [len(adj[i]) for i in range(n)]
        
        def get_mrv_vertex():
            min_val = 10
            best_v = -1
            best_deg = -1
            for v in unassigned:
                # Count bits in mask
                # Python 3.10+ has int.bit_count()
                count = doms[v].bit_count()
                if count < min_val:
                    min_val = count
                    best_v = v
                    best_deg = degrees[v]
                elif count == min_val:
                    if degrees[v] > best_deg:
                        best_v = v
                        best_deg = degrees[v]
            return best_v, min_val

        # Stack elements: [vertex, next_color_bit_pos, changes]
        # We will iterate through colors in the domain.
        # Instead of storing "next color", we can store the current domain mask and pop the lowest set bit.
        
        stack = []
        
        # Initial step
        v0, size0 = get_mrv_vertex()
        if size0 == 0:
            return None
        stack.append([v0, doms[v0], []])
        unassigned.remove(v0)
        assignment[v0] = -1 # Placeholder, will be set when we pick a color

        while stack:
            v, current_mask, changes = stack[-1]
            
            # Find the lowest set bit in current_mask
            # If current_mask is 0, we need to backtrack
            if current_mask == 0:
                # Backtrack
                stack.pop()
                # Restore changes made by this vertex
                for u, orig_mask in changes:
                    doms[u] = orig_mask
                # Restore v to unassigned
                unassigned.add(v)
                assignment[v] = -1
                continue
            
            # Pick the lowest color
            c_bit = current_mask & -current_mask
            c = c_bit.bit_length() - 1
            
            # Try assigning color c to v
            assignment[v] = c
            
            # Check neighbors and update domains
            new_changes = []
            conflict = False
            for u in adj[v]:
                if u in unassigned:
                    if doms[u] & c_bit:
                        new_changes.append((u, doms[u]))
                        doms[u] ^= c_bit # Remove the bit
                        if doms[u] == 0:
                            conflict = True
                            break
            
            if not conflict:
                # Success for this step, push to stack if there are more unassigned
                if unassigned:
                    # Find next MRV vertex
                    vn, size_n = get_mrv_vertex()
                    if size_n == 0:
                        conflict = True # Should not happen if forward checking works, but safe
                    else:
                        unassigned.remove(vn)
                        # Push state for vn
                        # We need to save the state of v's remaining colors?
                        # No, we are going deeper. The current state of v is "assigned c".
                        # When we backtrack from vn, we restore v's neighbors?
                        # Wait, the stack structure needs to be careful.
                        
                        # Let's restructure the stack to hold the full state or just the delta.
                        # Current stack item: [v, remaining_mask_for_v, changes_made_by_v]
                        # If we push vn, we don't remove v from the stack.
                        # The "remaining_mask_for_v" should be current_mask without c_bit.
                        # So, we update the top of the stack to have the remaining mask.
                        stack[-1][1] = current_mask ^ c_bit
                        stack[-1][2] = new_changes # Save changes made by v for this color choice
                        # Wait, if we backtrack from vn, we need to restore the changes made by v?
                        # No, the changes made by v are "active" while we are exploring vn's subtree.
                        # If vn fails, we pop vn, restore vn's changes, and then we are back at v.
                        # Then we try the next color for v.
                        # So, the `changes` in the stack item for v should be the changes made by v's current choice.
                        # When we pop v (because all its colors failed), we restore those changes.
                        
                        stack.append([vn, doms[vn], []])
                        continue # Loop to process vn
                else:
                    # All assigned
                    return [int(c) for c in assignment]
            
            # If conflict or no more unassigned (but we handled success above), we need to try next color for v
            # Restore changes made by v for this color
            for u, orig_mask in new_changes:
                doms[u] = orig_mask
            
            # Update the mask for v to exclude c
            stack[-1][1] = current_mask ^ c_bit
            
            # If stack[-1][1] becomes 0, the next iteration will pop it.
            
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = color_instance(n, edges, lists)
        if coloring is None:
            coloring = [0] * n
        out.append([int(c) for c in coloring])
    return out
# EVOLVE-BLOCK-END
