    def search(cands, needed):
        if needed == 0:
            return []
        if cands == 0:
            return None
        if cands.bit_count() < needed:
            return None
            
        # Pick vertex with highest degree in subgraph induced by cands
        best_v = -1
        best_deg = -1
        temp = cands
        while temp:
            v_bit = temp & -temp
            v = v_bit.bit_length() - 1
            deg = (adj[v] & cands).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= v_bit
            
        v = best_v
        
        new_cands = cands & ~adj[v] & ~(1 << v)
        res = search(new_cands, needed - 1)
        if res is not None:
            return [v] + res
            
        res = search(cands & ~(1 << v), needed)
        if res is not None:
            return res
            
        return None
