    def greedy_clique_cover(candidates_mask):
        if candidates_mask == 0:
            return 0
        remaining = candidates_mask
        cover_count = 0
        while remaining:
            # Pick the lowest set bit as the seed for the clique
            lsb = remaining & -remaining
            seed_idx = lsb.bit_length() - 1
            
            # Remove seed from remaining
            remaining &= ~lsb
            
            # Candidates for this clique must be adjacent to seed
            clique_candidates = remaining & adj[seed_idx]
            
            while clique_candidates:
                # Pick lowest set bit
                lsb_c = clique_candidates & -clique_candidates
                u_idx = lsb_c.bit_length() - 1
                
                # Add u to clique
                # Next candidates must be adjacent to u
                clique_candidates &= adj[u_idx]
                
                # Remove u from remaining
                remaining &= ~lsb_c
                
            cover_count += 1
        return cover_count
