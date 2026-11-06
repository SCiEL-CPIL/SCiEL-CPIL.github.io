import time
import random
import numpy as np

# Q3 structure
Q3_EDGES = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
Q3_NEIGHBORS = [
    [1, 2, 4],
    [0, 3, 5],
    [0, 3, 6],
    [1, 2, 7],
    [0, 5, 6],
    [1, 4, 7],
    [2, 4, 7],
    [3, 5, 6]
]

def _find_q3(masks, n):
    """
    Finds a Q3 subgraph in the graph represented by masks.
    Returns a dict {q_v: g_v} if found, else None.
    """
    # Quick check: if max degree < 3, no Q3
    max_deg = max(bin(masks[i]).count('1') for i in range(n))
    if max_deg < 3:
        return None

    def backtrack(assign, used_mask):
        if len(assign) == 8:
            return assign
        
        # Select next Q3 vertex to assign
        # Heuristic: Max assigned neighbors
        best_qv = -1
        max_assigned = -1
        for qv in range(8):
            if qv in assign:
                continue
            cnt = 0
            for nb in Q3_NEIGHBORS[qv]:
                if nb in assign:
                    cnt += 1
            if cnt > max_assigned:
                max_assigned = cnt
                best_qv = qv
        
        if best_qv == -1:
            return None
            
        assigned_nbs = [nb for nb in Q3_NEIGHBORS[best_qv] if nb in assign]
        
        # Determine candidates
        if not assigned_nbs:
            avail = ~used_mask & ((1 << n) - 1)
            cands = []
            while avail:
                lsb = avail & -avail
                gv = lsb.bit_length() - 1
                cands.append(gv)
                avail ^= lsb
        else:
            mask = (1 << n) - 1
            for nb in assigned_nbs:
                g_nb = assign[nb]
                mask &= masks[g_nb]
            mask &= ~used_mask
            cands = []
            avail = mask
            while avail:
                lsb = avail & -avail
                gv = lsb.bit_length() - 1
                cands.append(gv)
                avail ^= lsb
                
        if not cands:
            return None
            
        # Randomize order to find different instances
        random.shuffle(cands)
        
        for gv in cands:
            assign[best_qv] = gv
            res = backtrack(assign, used_mask | (1 << gv))
            if res:
                return res
            del assign[best_qv]
            
        return None

    # Try all start vertices for Q3 vertex 0
    # Optimization: Only start with vertices of degree >= 3
    for start in range(n):
        if bin(masks[start]).count('1') < 3:
            continue
        assign = {0: start}
        res = backtrack(assign, 1 << start)
        if res:
            return res
    return None

def _has_q3(masks, n):
    return _find_q3(masks, n) is not None

def construct():
    start_time = time.time()
    deadline = start_time + 115
    n = 16
    random.seed(42)
    
    best_edges = None
    best_count = 0
    
    def set_to_masks(edge_set):
        masks = [0] * n
        for u, v in edge_set:
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
        return masks

    for trial in range(100):
        if time.time() > deadline - 10:
            break
            
        # Random dense graph
        edge_set = set()
        # Start dense
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.85:
                    edge_set.add((i, j))
                    
        masks = set_to_masks(edge_set)
        
        # Phase 1: Remove edges
        iters = 0
        while time.time() < deadline - 20 and iters < 500:
            q3 = _find_q3(masks, n)
            if not q3:
                break
            # Pick random edge from this Q3
            # Map Q3 edges to graph edges
            q3_g_edges = []
            for u, v in Q3_EDGES:
                gu, gv = q3[u], q3[v]
                if gu > gv: gu, gv = gv, gu
                q3_g_edges.append((gu, gv))
            
            # Remove one
            rem_u, rem_v = random.choice(q3_g_edges)
            if (rem_u, rem_v) in edge_set:
                edge_set.remove((rem_u, rem_v))
                masks[rem_u] &= ~(1 << rem_v)
                masks[rem_v] &= ~(1 << rem_u)
            iters += 1
            
        if time.time() > deadline - 10:
            continue
            
        # Phase 2: Add edges
        added = True
        while added and time.time() < deadline - 5:
            added = False
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
            random.shuffle(missing)
            for u, v in missing:
                if time.time() > deadline - 5:
                    break
                # Try add
                edge_set.add((u, v))
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                
                if not _has_q3(masks, n):
                    added = True
                else:
                    edge_set.remove((u, v))
                    masks[u] &= ~(1 << v)
                    masks[v] &= ~(1 << u)
                    
        if len(edge_set) > best_count:
            best_count = len(edge_set)
            best_edges = list(edge_set)
            
    if best_edges is None:
        # Fallback: Star
        return [[0, i] for i in range(1, 16)]
        
    return [[u, v] for u, v in best_edges]
