    for p in range(NUM_POINTS):
        row = dist2[p]
        # Convert to list for faster indexing? 
        # Actually, numpy scalar access is okay, but list is faster.
        # Let's stick to numpy scalar for now, it's usually optimized.
        dist_map = {}
        for q in range(NUM_POINTS):
            if p == q:
                continue
            d = int(row[q])
            if d not in dist_map:
                dist_map[d] = []
            dist_map[d].append(q)
        # ... rest of precomputation
