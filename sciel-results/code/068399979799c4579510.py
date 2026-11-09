"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140
    n = 16
    
    edge_idx = {}
    edge_verts = []
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_idx[(i,j)] = idx
            edge_verts.append((i,j))
            idx += 1
    total_edges = 120
    
    edge_k4_partners = [[] for _ in range(total_edges)]
    for e in range(total_edges):
        i, j = edge_verts[e]
        others = [v for v in range(n) if v != i and v != j]
        for a in range(len(others)):
            for b in range(a+1, len(others)):
                edge_k4_partners[e].append((others[a], others[b]))
    
    k4_edge_lists = {}
    def get_k4_edges(a, b, c, d):
        verts = tuple(sorted((a,b,c,d)))
        if verts not in k4_edge_lists:
            pairs = [(verts[0],verts[1]), (verts[0],verts[2]), (verts[0],verts[3]),
                     (verts[1],verts[2]), (verts[1],verts[3]), (verts[2],verts[3])]
            k4_edge_lists[verts] = [edge_idx[p] for p in pairs]
        return k4_edge_lists[verts]
    
    # Precompute all K4 edge lists
    all_k4_edges = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    all_k4_edges.append(get_k4_edges(a, b, c, d))
    
    colors = np.zeros(total_edges, dtype=np.int32)
    for e in range(total_edges):
        i, j = edge_verts[e]
        colors[e] = (i + j) % 17
    
    def check_edge_change_valid(e, new_color, arr):
        i, j = edge_verts[e]
        old = arr[e]
        if old == new_color:
            return True
        arr[e] = new_color
        for (k, l) in edge_k4_partners[e]:
            edges = get_k4_edges(i, j, k, l)
            if len(set(arr[edges].tolist())) < 5:
                arr[e] = old
                return False
        arr[e] = old
        return True
    
    def count_colors(arr):
        return len(set(arr.tolist()))
    
    def verify(arr):
        for edges in all_k4_edges:
            if len(set(arr[edges].tolist())) < 5:
                return False
        return True
    
    best = colors.copy()
    best_count = count_colors(best)
    
    # Greedy elimination
    for target_k in range(16, 12, -1):
        if time.time() > deadline - 25:
            break
        trial = best.copy()
        used = sorted(set(trial.tolist()))
        if len(used) <= target_k:
            mapping = {c: i for i, c in enumerate(used)}
            trial = np.array([mapping[c] for c in trial], dtype=np.int32)
            if verify(trial):
                best = trial.copy()
                best_count = count_colors(best)
            continue
        color_counts = {}
        for c in trial:
            color_counts[c] = color_counts.get(c, 0) + 1
        to_elim = sorted(color_counts.keys(), key=lambda c: color_counts[c])[:len(used)-target_k]
        ok = True
        for col in to_elim:
            if time.time() > deadline - 25:
                ok = False
                break
            for e in [x for x in range(total_edges) if trial[x] == col]:
                found = False
                for nc in used:
                    if nc == col:
                        continue
                    if check_edge_change_valid(e, nc, trial):
                        trial[e] = nc
                        found = True
                        break
                if not found:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            used_new = sorted(set(trial.tolist()))
            if len(used_new) <= target_k:
                mapping = {c: i for i, c in enumerate(used_new)}
                trial = np.array([mapping[c] for c in trial], dtype=np.int32)
                if verify(trial):
                    best = trial.copy()
                    best_count = count_colors(best)
    
    # SA
    if time.time() < deadline - 30:
        sa = best.copy()
        T = 20.0
        for step in range(30000):
            if time.time() > deadline - 25:
                break
            if T < 0.5:
                break
            e = random.randint(0, total_edges-1)
            old_c = sa[e]
            new_c = random.randint(0, 119)
            if new_c == old_c:
                continue
            if check_edge_change_valid(e, new_c, sa):
                old_count = count_colors(sa)
                sa[e] = new_c
                new_count = count_colors(sa)
                delta = new_count - old_count
                if delta <= 0 or random.random() < np.exp(-delta/T):
                    pass
                else:
                    sa[e] = old_c
            T *= 0.995
        if verify(sa) and count_colors(sa) < best_count:
            best = sa.copy()
            best_count = count_colors(best)
    
    used = sorted(set(best.tolist()))
    mapping = {c: i for i, c in enumerate(used)}
    final = np.array([mapping[c] for c in best], dtype=np.int32)
    
    result = []
    for i in range(n):
        for j in range(i+1, n):
            result.append(int(final[edge_idx[(i,j)]]))
    return result

# EVOLVE-BLOCK-END
