    quad_edge_indices = []
    for q in quads:
        a, b, c, d = q
        quad_edge_indices.append([
            edge_to_idx[(a,b)], edge_to_idx[(a,c)], edge_to_idx[(a,d)],
            edge_to_idx[(b,c)], edge_to_idx[(b,d)], edge_to_idx[(c,d)]
        ])
        
    def quad_mask(coloring, quad_idx):
        mask = 0
        for idx in quad_edge_indices[quad_idx]:
            mask |= 1 << coloring[idx]
        return mask
