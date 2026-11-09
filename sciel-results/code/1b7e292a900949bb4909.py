"""Four-cycle-free edge sets of Q7: maximize edges with no 2-face fully selected."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    d = 7
    n = 1 << d  # 128
    total_edges = n * d // 2  # 448

    # 1. Generate all edges
    all_edges = []
    edge_index = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_index[(x, y)] = idx

    # 2. Precompute all C4s (faces)
    all_c4s = []
    for i in range(d):
        for j in range(i + 1, d):
            mask = (1 << i) | (1 << j)
            for x in range(n):
                if (x & mask) == 0:
                    v1 = x
                    v2 = x ^ (1 << i)
                    v3 = x ^ (1 << j)
                    v4 = x ^ (1 << i) ^ (1 << j)
                    
                    e1 = edge_index[(min(v1, v2), max(v1, v2))]
                    e2 = edge_index[(min(v3, v4), max(v3, v4))]
                    e3 = edge_index[(min(v1, v3), max(v1, v3))]
                    e4 = edge_index[(min(v2, v4), max(v2, v4))]
                    
                    all_c4s.append((e1, e2, e3, e4))

    num_c4s = len(all_c4s)
    
    # 3. Precompute C4 membership for each edge
    c4_membership = [[] for _ in range(total_edges)]
    for c4_idx, (e1, e2, e3, e4) in enumerate(all_c4s):
        for e in (e1, e2, e3, e4):
            c4_membership[e].append(c4_idx)

    # Precompute C4 edges as numpy array for fast checks
    c4_edges = np.array(all_c4s, dtype=np.int32)  # shape (num_c4s, 4)

    def greedy_add(seed=None):
        """Greedy addition: shuffle edges, add if no C4 becomes fully selected."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        present = np.zeros(total_edges, dtype=bool)
        # For each C4, track how many of its 4 edges are present
        c4_fill = np.zeros(num_c4s, dtype=np.int32)
        
        edge_order = list(range(total_edges))
        rng.shuffle(edge_order)
        
        for e in edge_order:
            # Check if adding e would complete any C4
            can_add = True
            for c4_idx in c4_membership[e]:
                if c4_fill[c4_idx] == 3:
                    can_add = False
                    break
            if can_add:
                present[e] = True
                for c4_idx in c4_membership[e]:
                    c4_fill[c4_idx] += 1
        
        return present

    def local_search(present, max_time=20.0):
        """Local search with 1-swap and 2-swap moves."""
        start_ls = time.time()
        present = present.copy()
        m = int(np.sum(present))
        rng = random.Random()
        
        # Precompute removed edges
        removed = [e for e in range(total_edges) if not present[e]]
        
        # Phase 1: Try to add back removed edges
        for _ in range(20):
            if time.time() > deadline - 5 or (time.time() - start_ls) > max_time:
                break
            progress = False
            rng.shuffle(removed)
            for e in removed:
                if present[e]:
                    continue
                ok = True
                for c4_idx in c4_membership[e]:
                    e1, e2, e3, e4 = all_c4s[c4_idx]
                    others = [x for x in (e1, e2, e3, e4) if x != e]
                    if all(present[x] for x in others):
                        ok = False
                        break
                if ok:
                    present[e] = True
                    m += 1
                    removed.remove(e)
                    progress = True
            if not progress:
                break
        
        # Phase 2: 1-swap moves
        for _ in range(100):
            if time.time() > deadline - 5 or (time.time() - start_ls) > max_time:
                break
            progress = False
            rng.shuffle(removed)
            for e2 in removed:
                if present[e2]:
                    continue
                blocking_c4s = []
                for c4_idx in c4_membership[e2]:
                    e1, e2_, e3, e4 = all_c4s[c4_idx]
                    others = [x for x in (e1, e2_, e3, e4) if x != e2]
                    if all(present[x] for x in others):
                        blocking_c4s.append(c4_idx)
                
                if not blocking_c4s:
                    present[e2] = True
                    m += 1
                    removed.remove(e2)
                    progress = True
                    continue
                
                for c4_idx in blocking_c4s:
                    e1, e2_, e3, e4 = all_c4s[c4_idx]
                    others = [x for x in (e1, e2_, e3, e4) if x != e2]
                    for e1_rm in others:
                        if not present[e1_rm]:
                            continue
                        # Try swap: remove e1_rm, add e2
                        present[e1_rm] = False
                        present[e2] = True
                        valid = True
                        for c4_idx2 in c4_membership[e2]:
                            ea, eb, ec, ed = all_c4s[c4_idx2]
                            if all(present[x] for x in (ea, eb, ec, ed)):
                                valid = False
                                break
                        if valid:
                            for c4_idx2 in c4_membership[e1_rm]:
                                ea, eb, ec, ed = all_c4s[c4_idx2]
                                if all(present[x] for x in (ea, eb, ec, ed)):
                                    valid = False
                                    break
                        if valid:
                            removed.remove(e2)
                            progress = True
                            break
                        else:
                            present[e1_rm] = True
                            present[e2] = False
                    if progress:
                        break
            if not progress:
                break
        
        return present, m

    # Main: multiple restarts with greedy addition + local search
    best_present = None
    best_m = 0
    
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        seed = restart * 7919 + 12345
        present = greedy_add(seed=seed)
        m = int(np.sum(present))
        
        if m > best_m:
            best_m = m
            best_present = present.copy()
        
        # Local search
        present_ls, m_ls = local_search(present, max_time=15.0)
        if m_ls > best_m:
            best_m = m_ls
            best_present = present_ls.copy()
    
    # Final polish
    if best_present is not None and time.time() < deadline - 10:
        present_final, m_final = local_search(best_present, max_time=40.0)
        if m_final > best_m:
            best_m = m_final
            best_present = present_final.copy()

    # Construct output
    out = []
    for idx, (u, v) in enumerate(all_edges):
        if best_present[idx]:
            out.append([u, v])
    
    return out

# EVOLVE-BLOCK-END
