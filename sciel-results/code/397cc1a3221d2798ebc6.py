    def find_independent_set(adj, n, k):
        def search(cand_mask, size_needed):
            if size_needed == 0:
                return True
            if popcount(cand_mask) < size_needed:
                return False
            
            # Find vertex with max degree in cand_mask
            max_deg = -1
            best_v = -1
            temp_mask = cand_mask
            while temp_mask:
                v_bit = temp_mask & -temp_mask
                v = v_bit.bit_length() - 1
                deg = popcount(adj[v] & cand_mask)
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp_mask &= temp_mask - 1
            
            if best_v == -1:
                return False
            
            v_bit = 1 << best_v
            
            # Option 1: Include best_v
            new_cand = cand_mask & ~(adj[best_v] | v_bit)
            if search(new_cand, size_needed - 1):
                return True
            
            # Option 2: Exclude best_v
            new_cand = cand_mask & ~v_bit
            if search(new_cand, size_needed):
                return True
            
            return False
        
        full_mask = (1 << n) - 1
        return search(full_mask, k)
