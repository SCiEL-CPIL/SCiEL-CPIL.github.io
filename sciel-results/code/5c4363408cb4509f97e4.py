import time
import random
import numpy as np

N = 16
EDGE_COUNT = N * (N - 1) // 2  # 120

# Precompute edges
edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
edge_idx = {(i, j): k for k, (i, j) in enumerate(edges)}

# Precompute all K4s and their edge indices
k4_edges = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                e = [edge_idx[(a,b)], edge_idx[(a,c)], edge_idx[(a,d)],
                     edge_idx[(b,c)], edge_idx[(b,d)], edge_idx[(c,d)]]
                k4_edges.append(e)

NUM_K4 = len(k4_edges)

# Map edge -> list of K4 indices
edge_to_k4s = [[] for _ in range(EDGE_COUNT)]
for k4i, eids in enumerate(k4_edges):
    for eid in eids:
        edge_to_k4s[eid].append(k4i)

def count_distinct(colours, eids):
    return len(set(colours[e] for e in eids))

def construct():
    start = time.time()
    deadline = start + 140  # 10s safety margin
    
    for K in [15, 14, 13]:
        if time.time() > deadline - 5:
            break
        
        # Initialize
        colours = np.array([(i + j) % K for i in range(N) for j in range(i + 1, N)], dtype=np.int32)
        
        # Compute initial violations
        violations = np.zeros(NUM_K4, dtype=bool)
        for k4i, eids in enumerate(k4_edges):
            if count_distinct(colours, eids) < 5:
                violations[k4i] = True
        
        viol_count = int(np.sum(violations))
        
        if viol_count == 0:
            return colours.tolist()
        
        # Hill climbing
        best_viol = viol_count
        best_colours = colours.copy()
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() < deadline - 5 and no_improve < max_no_improve:
            if viol_count == 0:
                return colours.tolist()
            
            # Pick a random violating K4
            viol_idx = np.where(violations)[0]
            if len(viol_idx) == 0:
                break
            k4i = viol_idx[random.randint(0, len(viol_idx) - 1)]
            eids = k4_edges[k4i]
            
            # Try changing each edge in this K4
            found_improve = False
            for eid in eids:
                old_c = colours[eid]
                affected = edge_to_k4s[eid]
                
                # Try a few random new colours
                for _ in range(5):
                    new_c = random.randint(0, K - 1)
                    if new_c == old_c:
                        continue
                    colours[eid] = new_c
                    delta = 0
                    for ak4 in affected:
                        old_v = violations[ak4]
                        new_v = count_distinct(colours, k4_edges[ak4]) < 5
                        if old_v and not new_v:
                            delta -= 1
                        elif not old_v and new_v:
                            delta += 1
                        violations[ak4] = new_v
                    total = viol_count + delta
                    if total < viol_count:
                        # Accept
                        viol_count = total
                        if viol_count < best_viol:
                            best_viol = viol_count
                            best_colours = colours.copy()
                            no_improve = 0
                        else:
                            no_improve += 1
                        found_improve = True
                        break
                    else:
                        # Revert
                        colours[eid] = old_c
                        for ak4 in affected:
                            violations[ak4] = count_distinct(colours, k4_edges[ak4]) < 5
                
                if found_improve:
                    break
            
            if not found_improve:
                no_improve += 1
                # Random perturbation
                for _ in range(3):
                    eid = random.randint(0, EDGE_COUNT - 1)
                    old_c = colours[eid]
                    new_c = random.randint(0, K - 1)
                    colours[eid] = new_c
                    for ak4 in edge_to_k4s[eid]:
                        violations[ak4] = count_distinct(colours, k4_edges[ak4]) < 5
                    viol_count = int(np.sum(violations))
                    if viol_count < best_viol:
                        best_viol = viol_count
                        best_colours = colours.copy()
                        no_improve = 0
                    colours[eid] = old_c
                    for ak4 in edge_to_k4s[eid]:
                        violations[ak4] = count_distinct(colours, k4_edges[ak4]) < 5
                viol_count = int(np.sum(violations))
        
        if viol_count == 0:
            return colours.tolist()
    
    # Fallback: return seed solution (17 colours)
    return [(i + j) % 17 for i in range(N) for j in range(i + 1, N)]
