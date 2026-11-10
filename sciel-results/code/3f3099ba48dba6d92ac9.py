import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 16
    from itertools import combinations
    
    # Precompute all K4s and their edge indices
    all_k4s = list(combinations(range(n), 4))
    k4_edge_indices = []
    for quad in all_k4s:
        ei = []
        for (i,j) in combinations(quad, 2):
            ei.append(i * 15 - i*(i-1)//2 + (j - i - 1))  # need proper indexing
        k4_edge_indices.append(ei)
    
    # Simpler: precompute edge index for each (i,j)
    edge_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_idx[(i,j)] = idx
            idx += 1
    
    k4_eis = []
    for quad in all_k4s:
        eis = []
        for (i,j) in combinations(quad, 2):
            eis.append(edge_idx[(i,j)])
        k4_eis.append(eis)
    
    # For each edge, precompute which K4s it belongs to
    edge_to_k4s = [[] for _ in range(120)]
    for ki, eis in enumerate(k4_eis):
        for ei in eis:
            edge_to_k4s[ei].append(ki)
    
    def k4_color_count(coloring, eis):
        """Count distinct colors in a K4 given edge indices."""
        s = set()
        for ei in eis:
            s.add(coloring[ei])
        return len(s)
    
    def is_valid_fast(coloring):
        for eis in k4_eis:
            s = set()
            for ei in eis:
                s.add(coloring[ei])
            if len(s) < 5:
                return False
        return True
    
    def check_k4(coloring, eis):
        s = set()
        for ei in eis:
            s.add(coloring[ei])
        return len(s)
    
    def local_search(coloring, max_colors, time_limit):
        """Improve coloring to reduce violations and color count."""
        t_end = time.time() + time_limit
        while time.time() < t_end:
            # Find a violation
            found = False
            for ki in random.sample(range(len(k4_eis)), min(200, len(k4_eis))):
                eis = k4_eis[ki]
                s = set()
                for ei in eis:
                    s.add(coloring[ei])
                if len(s) < 5:
                    # Recolor one of the edges
                    # Find edge with duplicate color
                    counts = {}
                    for ei in eis:
                        c = coloring[ei]
                        counts[c] = counts.get(c, 0) + 1
                    dup_edges = [ei for ei in eis if counts[coloring[ei]] >= 2]
                    if dup_edges:
                        ei = random.choice(dup_edges)
                        new_c = random.randint(0, max_colors - 1)
                        if new_c != coloring[ei]:
                            coloring[ei] = new_c
                    found = True
                    break
            if not found:
                # No violations found in sample, try to reduce colors
                # Find a color that can be eliminated
                color_counts = {}
                for c in coloring:
                    color_counts[c] = color_counts.get(c, 0) + 1
                # Try to recolor edges of a rare color
                rare_colors = sorted(color_counts.keys(), key=lambda c: color_counts[c])
                for rc in rare_colors[:3]:
                    # Try to recolor all edges of this color to other colors
                    edges_of_rc = [ei for ei in range(120) if coloring[ei] == rc]
                    success = True
                    for ei in edges_of_rc:
                        # Try each other color
                        placed = False
                        for new_c in range(max_colors):
                            if new_c == rc:
                                continue
                            coloring[ei] = new_c
                            # Check affected K4s
                            ok = True
                            for k4i in edge_to_k4s[ei]:
                                if check_k4(coloring, k4_eis[k4i]) < 5:
                                    ok = False
                                    break
                            if ok:
                                placed = True
                                break
                        if not placed:
                            coloring[ei] = rc  # revert
                            success = False
                            break
                    if success:
                        break
                if not success:
                    # Do a random perturbation
                    ei = random.randint(0, 119)
                    coloring[ei] = random.randint(0, max_colors - 1)
        
        return coloring
    
    best = None
    best_cnt = 99
    
    for max_c in [13, 14, 15]:
        if time.time() > deadline - 20:
            break
        for attempt in range(2000):
            if time.time() > deadline - 5:
                break
            coloring = [random.randint(0, max_c - 1) for _ in range(120)]
            # Quick repair: iterate a few times
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                # Check a random K4
                ki = random.randint(0, len(k4_eis) - 1)
                eis = k4_eis[ki]
                s = set()
                for ei in eis:
                    s.add(coloring[ei])
                if len(s) < 5:
                    counts = {}
                    for ei in eis:
                        c = coloring[ei]
                        counts[c] = counts.get(c, 0) + 1
                    dup = [ei for ei in eis if counts[coloring[ei]] >= 2]
                    if dup:
                        ei = random.choice(dup)
                        coloring[ei] = random.randint(0, max_c - 1)
            
            # Full validity check
            if is_valid_fast(coloring):
                used = len(set(coloring))
                if used < best_cnt:
                    best_cnt = used
                    best = coloring[:]
                    if used <= 13:
                        return best  # Can't do better than 13
    
    if best is not None:
        color_map = {}
        nxt = 0
        for c in best:
            if c not in color_map:
                color_map[c] = nxt
                nxt += 1
        return [color_map[c] for c in best]
    else:
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
