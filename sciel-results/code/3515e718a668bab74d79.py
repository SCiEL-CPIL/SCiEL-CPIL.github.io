    def creates_c6_bitset(u, v):
        # Check if there is a simple path of length 5 from u to v
        # S_k: set of nodes reachable by simple path of length k
        # We maintain a list of bitsets for visited nodes at each step to exclude them
        
        visited_mask = 1 << u
        current_set = 1 << u
        
        for step in range(5):
            next_set = 0
            # Iterate over nodes in current_set
            # To iterate bits in a python int:
            temp = current_set
            while temp:
                lsb = temp & -temp
                node = lsb.bit_length() - 1
                temp ^= lsb
                # Get neighbors of node
                neighbors = adj[node] & ~visited_mask
                next_set |= neighbors
            
            # Update visited
            visited_mask |= next_set
            current_set = next_set
            
            if current_set == 0:
                return False # No more paths
            
            # Early exit: if v is in current_set and step == 4 (so path length is 5)
            # Wait, step 0 is u.
            # Step 1: neighbors of u (path len 1)
            # Step 5: neighbors of S_4 (path len 5)
            # So we check at the end of the loop if step == 4?
            # No, the loop runs 5 times.
            # After 5 iterations, current_set contains nodes at distance 5.
            
        return (current_set >> v) & 1
