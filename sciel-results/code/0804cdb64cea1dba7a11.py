K4_EDGE_INDICES = []
for a, b, c, d in K4S:
    K4_EDGE_INDICES.append([
        EDGE_INDEX[(a, b)], EDGE_INDEX[(a, c)], EDGE_INDEX[(a, d)],
        EDGE_INDEX[(b, c)], EDGE_INDEX[(b, d)], EDGE_INDEX[(c, d)]
    ])

def is_valid_k4_fast(k4_idx, colors):
    e_idxs = K4_EDGE_INDICES[k4_idx]
    return len({colors[e_idxs[0]], colors[e_idxs[1]], colors[e_idxs[2]], colors[e_idxs[3]], colors[e_idxs[4]], colors[e_idxs[5]]}) >= 5
