import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 115.0

    def time_exceeded():
        return time.time() - start_time > deadline

    q = 29
    squares_29 = set()
    for x in range(1, q):
        squares_29.add((x * x) % q)

    def build_paley_colors(n):
        squares = set()
        for x in range(1, n):
            squares.add((x * x) % n)
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d in squares:
                    colors.append(0)
                else:
                    colors.append(1)
        return colors

    colors_29 = build_paley_colors(q)

    def build_adj(n, colors):
        """Build bitmask adjacency lists for both color classes."""
        adj = [[0] * n for _ in range(2)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        return adj

    def count_k222_conflicts(n, adj):
        """
        Count the number of K_{2,2,2} subgraphs in the given adjacency.
        Uses efficient bitmask operations.
        Returns the count.
        """
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    continue
                cn_mask = (adj[i] & adj[j]) & ~(1 << i) & ~(1 << j)
                # Count vertices in common neighborhood
                cn_count = cn_mask.bit_count()
                if cn_count < 4:
                    continue
                
                # Get common neighbors as list
                cn_list = []
                mask = cn_mask
                while mask:
                    lsb = mask & (-mask)
                    v = lsb.bit_length() - 1
                    cn_list.append(v)
                    mask ^= lsb
                
                # Find pairs (k,l) in cn_list with no edge between them
                for a in range(len(cn_list)):
                    k = cn_list[a]
                    for b in range(a + 1, len(cn_list)):
                        l = cn_list[b]
                        if adj[k] & (1 << l):
                            continue
                        # Now find common neighbors of i,j,k,l
                        cn2_mask = (adj[i] & adj[j] & adj[k] & adj[l])
                        cn2_mask &= ~(1 << i) & ~(1 << j) & ~(1 << k) & ~(1 << l)
                        cn2_count = cn2_mask.bit_count()
                        if cn2_count < 2:
                            continue
                        
                        # Count pairs (m,o) in cn2 with no edge
                        cn2_list = []
                        mask2 = cn2_mask
                        while mask2:
                            lsb2 = mask2 & (-mask2)
                            v2 = lsb2.bit_length() - 1
                            cn2_list.append(v2)
                            mask2 ^= lsb2
                        
                        for c in range(len(cn2_list)):
                            m = cn2_list[c]
                            for d in range(c + 1, len(cn2_list)):
                                o = cn2_list[d]
                                if not (adj[m] & (1 << o)):
                                    count += 1
        return count

    def find_k222_witness(n, adj):
        """Find one K_{2,2,2} witness or return None."""
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    continue
                cn_mask = (adj[i] & adj[j]) & ~(1 << i) & ~(1 << j)
                cn_count = cn_mask.bit_count()
                if cn_count < 4:
                    continue
                
                cn_list = []
                mask = cn_mask
                while mask:
                    lsb = mask & (-mask)
                    v = lsb.bit_length() - 1
                    cn_list.append(v)
                    mask ^= lsb
                
                for a in range(len(cn_list)):
                    k = cn_list[a]
                    for b in range(a + 1, len(cn_list)):
                        l = cn_list[b]
                        if adj[k] & (1 << l):
                            continue
                        cn2_mask = (adj[i] & adj[j] & adj[k] & adj[l])
                        cn2_mask &= ~(1 << i) & ~(1 << j) & ~(1 << k) & ~(1 << l)
                        cn2_count = cn2_mask.bit_count()
                        if cn2_count < 2:
                            continue
                        
                        cn2_list = []
                        mask2 = cn2_mask
                        while mask2:
                            lsb2 = mask2 & (-mask2)
                            v2 = lsb2.bit_length() - 1
                            cn2_list.append(v2)
                            mask2 ^= lsb2
                        
                        for c in range(len(cn2_list)):
                            m = cn2_list[c]
                            for d in range(c + 1, len(cn2_list)):
                                o = cn2_list[d]
                                if not (adj[m] & (1 << o)):
                                    return (i, j, k, l, m, o)
        return None

    # Precompute edge lookup for n=29 and n=30
    def make_edge_lookup(n):
        lookup = {}
        idx = 0
        for a in range(n):
            for b in range(a + 1, n):
                lookup[(a, b)] = idx
                idx += 1
        return lookup

    edge_lookup_29 = make_edge_lookup(29)
    edge_lookup_30 = make_edge_lookup(30)

    # Verify Paley(29) is valid
    adj_29 = build_adj(29, colors_29)
    conflicts_29 = count_k222_conflicts(29, adj_29[0]) + count_k222_conflicts(29, adj_29[1])

    if conflicts_29 > 0:
        # Paley(29) is invalid - repair it
        random.seed(42)
        colors = colors_29[:]
        for iteration in range(5000):
            if time_exceeded():
                break
            adj = build_adj(29, colors)
            conf = count_k222_conflicts(29, adj[0]) + count_k222_conflicts(29, adj[1])
            if conf == 0:
                return {'n': 29, 'colors': colors}
            witness = find_k222_witness(29, adj[0]) or find_k222_witness(29, adj[1])
            if witness is None:
                break
            verts = list(witness)
            v1, v2 = random.sample(verts, 2)
            if v1 > v2:
                v1, v2 = v2, v1
            edge_idx = edge_lookup_29[(v1, v2)]
            colors[edge_idx] = 1 - colors[edge_idx]
        return {'n': 29, 'colors': colors}

    # Paley(29) is valid! Try to extend to n=30
    best_n = 29
    best_colors = colors_29[:]

    random.seed(42)
    
    max_attempts = 50
    for attempt in range(max_attempts):
        if time_exceeded():
            break

        # Create initial coloring for n=30
        colors_30 = colors_29[:]
        for i in range(29):
            colors_30.append(random.randint(0, 1))

        # Local search with conflict counting
        adj_30 = build_adj(30, colors_30)
        prev_conflicts = count_k222_conflicts(30, adj_30[0]) + count_k222_conflicts(30, adj_30[1])
        
        # Track edges involved in conflicts for better flip selection
        conflict_edges = [0] * 435  # C(30,2) = 435
        
        for iteration in range(3000):
            if time_exceeded():
                break

            adj_30 = build_adj(30, colors_30)
            cur_conflicts = count_k222_conflicts(30, adj_30[0]) + count_k222_conflicts(30, adj_30[1])
            
            if cur_conflicts == 0:
                return {'n': 30, 'colors': colors_30}

            # Find a witness to identify conflict edges
            witness = find_k222_witness(30, adj_30[0])
            if witness is None:
                witness = find_k222_witness(30, adj_30[1])
            
            if witness is not None:
                verts = list(witness)
                # Mark all edges among these 6 vertices as conflict edges
                for a in range(6):
                    for b in range(a + 1, 6):
                        v1, v2 = verts[a], verts[b]
                        if v1 > v2:
                            v1, v2 = v2, v1
                        edge_idx = edge_lookup_30[(v1, v2)]
                        conflict_edges[edge_idx] += 1

            # Choose an edge to flip: preferentially pick from conflict edges
            # Use weighted random selection
            total_weight = sum(conflict_edges)
            if total_weight > 0 and random.random() < 0.7:
                # Weighted selection from conflict edges
                r = random.uniform(0, total_weight)
                cumulative = 0
                chosen_edge = -1
                for ei in range(435):
                    cumulative += conflict_edges[ei]
                    if cumulative >= r:
                        chosen_edge = ei
                        break
                if chosen_edge < 0:
                    chosen_edge = random.randint(0, 434)
            else:
                chosen_edge = random.randint(0, 434)

            # Flip the edge
            colors_30[chosen_edge] = 1 - colors_30[chosen_edge]
            conflict_edges[chosen_edge] = 0  # Reset after flip
            
            # Decay other conflict counts
            for ei in range(435):
                conflict_edges[ei] = conflict_edges[ei] * 0.9

        # After local search, check final state
        adj_30 = build_adj(30, colors_30)
        final_conflicts = count_k222_conflicts(30, adj_30[0]) + count_k222_conflicts(30, adj_30[1])
        if final_conflicts == 0:
            return {'n': 30, 'colors': colors_30}

    return {'n': best_n, 'colors': best_colors}


if __name__ == "__main__":
    result = construct()
    print(result['n'])
