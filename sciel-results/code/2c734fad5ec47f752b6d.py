import numpy as np
import time
import random
import sys

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    deadline = start_time + 590
    
    # Squares mod 5
    sq5 = {0, 1, 4}
    
    # Squares mod 41
    n41 = 41
    Q41 = set()
    for x in range(n41):
        Q41.add((x * x) % n41)
    
    # Forbidden: d != 0, d mod 5 in sq5, d mod 41 in Q41
    # Build adjacency on 205 nodes
    full_205 = (1 << 205) - 1
    
    # Precompute for each node, which other nodes are forbidden
    # Use the grid structure: node x corresponds to (x mod 5, x mod 41)
    # Two nodes (a5,a41) and (b5,b41) are adjacent if:
    #   (a5-b5) mod 5 in sq5 AND (a41-b41) mod 41 in Q41 AND not (a5==b5 and a41==b41)
    
    # Map from (mod5, mod41) to node in 0..204
    # node = a5 + 5 * j where j = (a41 - a5) * inv5_41 mod 41
    # Actually simpler: for a in 0..4, b in 0..40, find x in 0..204 with x%5=a, x%41=b
    # x = a + 5*j, need (a+5j)%41 = b => 5j = b-a mod 41 => j = (b-a)*33 mod 41
    inv5_41 = 33  # 5*33 = 165 = 4*41+1
    
    grid_to_node = {}
    for a in range(5):
        for b in range(41):
            j = ((b - a) * inv5_41) % 41
            x = (a + 5 * j) % 205
            grid_to_node[(a, b)] = x
    
    # For each grid cell (a,b), compute which other cells are forbidden
    # forbidden if (a'-a) mod 5 in sq5 AND (b'-b) mod 41 in Q41 AND (a',b') != (a,b)
    
    # Precompute forbidden masks for each of the 205 nodes
    adj = [0] * 205
    for a in range(5):
        for b in range(41):
            x = grid_to_node[(a, b)]
            mask = 0
            for a2 in range(5):
                da = (a2 - a) % 5
                if da not in sq5:
                    continue
                for b2 in range(41):
                    db = (b2 - b) % 41
                    if db not in Q41:
                        continue
                    if a2 == a and b2 == b:
                        continue
                    y = grid_to_node[(a2, b2)]
                    mask |= (1 << y)
            adj[x] = mask
    
    # Now do structured search on the grid
    # Column = a (mod 5), Row = b (mod 41)
    # We select a subset of the 205 grid cells
    
    best_mask = 0
    best_size = 0
    
    # Strategy: column-by-column greedy with backtracking/local search
    # For each column a, choose a subset of rows b such that:
    # 1. Within same column: (b1-b2) mod 41 not in Q41 (since da=0 in sq5)
    # 2. Across columns: (a1-a2) mod 5 in sq5 AND (b1-b2) mod 41 in Q41 => forbidden
    
    # For column a, given which rows are used in other columns, determine valid rows
    
    # Let's do multi-start with column-by-column construction
    
    for restart in range(200):
        if time.time() > deadline - 30:
            break
        
        rng = random.Random(restart * 7919 + 13)
        
        # Column order
        col_order = list(range(5))
        rng.shuffle(col_order)
        
        selected = {}  # a -> set of b's
        for a in range(5):
            selected[a] = set()
        
        # For each column in order, greedily pick rows
        for a in col_order:
            # Determine which rows b are valid in column a
            # A row b is valid if:
            # 1. For all (a2, b2) in selected[a2], a2 != a:
            #    (a2-a) mod 5 in sq5 AND (b2-b) mod 41 in Q41 => forbidden
            #    So we need: NOT ((a2-a) mod 5 in sq5 AND (b2-b) mod 41 in Q41)
            #    If (a2-a) mod 5 not in sq5, no constraint
            #    If (a2-a) mod 5 in sq5, then (b2-b) mod 41 must NOT be in Q41
            # 2. Within same column (other b's already picked in this column):
            #    (b-b2) mod 41 not in Q41
            
            valid_rows = set(range(41))
            
            # Constraints from other columns
            for a2 in range(5):
                if a2 == a:
                    continue
                da = (a2 - a) % 5
                if da not in sq5:
                    continue
                # For each b2 in selected[a2], b must satisfy (b-b2) mod 41 not in Q41
                for b2 in selected[a2]:
                    for b in list(valid_rows):
                        if (b - b2) % 41 in Q41:
                            valid_rows.discard(b)
            
            # Now pick rows from valid_rows such that pairwise differences not in Q41
            # Greedy: sort by some heuristic, pick one, remove conflicts
            candidates = sorted(valid_rows, key=lambda b: rng.random())
            picked = []
            for b in candidates:
                # Check against already picked in this column
                ok = True
                for b_picked in picked:
                    if (b - b_picked) % 41 in Q41:
                        ok = False
                        break
                if ok:
                    picked.append(b)
            
            selected[a] = set(picked)
        
        # Build mask
        mask = 0
        total = 0
        for a in range(5):
            for b in selected[a]:
                x = grid_to_node[(a, b)]
                mask |= (1 << x)
                total += 1
        
        if total > best_size:
            best_size = total
            best_mask = mask
        
        # Local search: try to add a node
        improved = True
        while improved:
            if time.time() > deadline - 10:
                break
            improved = False
            # Find all addable nodes
            blocked = 0
            s = mask
            while s:
                lsb = s & -s
                u = lsb.bit_length() - 1
                s ^= lsb
                blocked |= adj[u]
            avail = full_205 & ~blocked & ~mask
            if avail:
                # Pick one
                c = avail
                lsb = c & -c
                v = lsb.bit_length() - 1
                mask |= (1 << v)
                total += 1
                if total > best_size:
                    best_size = total
                    best_mask = mask
                improved = True
        
        # SA to improve
        current_mask = mask
        current_size = total
        T = 1.0
        for step in range(30000):
            if time.time() > deadline - 10:
                break
            
            # Get nodes in current set
            set_nodes = []
            c = current_mask
            while c:
                lsb = c & -c
                u = lsb.bit_length() - 1
                set_nodes.append(u)
                c ^= lsb
            
            if not set_nodes:
                break
            
            u = rng.choice(set_nodes)
            new_mask = current_mask & ~(1 << u)
            
            # Find addable
            blocked = 0
            s = new_mask
            while s:
                lsb = s & -s
                w = lsb.bit_length() - 1
                s ^= lsb
                blocked |= adj[w]
            
            avail = full_205 & ~blocked & ~new_mask
            if avail:
                count = avail.bit_count()
                k = rng.randint(0, count - 1)
                temp = avail
                for i in range(k):
                    lsb = temp & -temp
                    temp ^= lsb
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                candidate_mask = new_mask | (1 << v)
                delta = 0  # same size (swap)
            else:
                candidate_mask = new_mask
                delta = -1
            
            if delta >= 0 or (T > 0.01 and rng.random() < np.exp(delta / T)):
                current_mask = candidate_mask
                current_size += delta
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
            
            T *= 0.99997
            if T < 0.01:
                T = 0.01
    
    # Extract result
    result_nodes = []
    c = best_mask
    while c:
        lsb = c & -c
        u = lsb.bit_length() - 1
        result_nodes.append(u)
        c ^= lsb
    
    if not result_nodes:
        return np.array([0], dtype=np.int64)
    
    return np.array(result_nodes, dtype=np.int64)
