    def greedy_color(candidates_mask):
        # Get active vertices
        active = []
        m = candidates_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            active.append(idx)
            m ^= lsb
            
        if not active:
            return []
            
        # Calculate degree in subgraph for sorting
        # degree(v) = popcount(adj_bits[v] & candidates_mask)
        # Python 3.8+ has int.bit_count()
        
        # Sort by degree descending
        # To avoid calculating popcount for all if it's too slow, we can just use index order.
        # But better coloring helps a lot.
        
        # Let's try sorting by degree.
        # This requires O(N log N) per call.
        
        def get_deg(v):
            return (adj_bits[v] & candidates_mask).bit_count()
            
        # Sorting 5000 items might take ~1-2ms.
        # If we have 1000 nodes, that's 1-2 seconds. Might be tight.
        # Let's stick to index order for safety, or a hybrid.
        # Actually, the standard Tomita implementation often uses a fixed order or degree order.
        # Let's use index order first. If it's too slow, we can optimize.
        
        color_of = {}
        for v in active:
            used_colors = set()
            for u in neighbor_lists[v]:
                if (1 << u) & candidates_mask:
                    if u in color_of:
                        used_colors.add(color_of[u])
            
            c = 0
            while c in used_colors:
                c += 1
            color_of[v] = c
            
        max_color = max(color_of.values()) if color_of else 0
        color_classes = [[] for _ in range(max_color + 1)]
        for v, c in color_of.items():
            color_classes[c].append(v)
            
        return color_classes
