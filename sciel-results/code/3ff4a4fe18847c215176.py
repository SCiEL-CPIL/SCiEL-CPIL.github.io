import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 82.0  # safety margin before 90s kill

    # 255 = 3 * 5 * 17
    # Candidates must share a factor with 255
    candidates = [i for i in range(1, 256) if math.gcd(i, 255) > 1]
    if 255 not in candidates:
        candidates.append(255)
    
    n = len(candidates)
    idx = {v: i for i, v in enumerate(candidates)}
    idx_255 = idx[255]
    
    # Build adjacency as bitmasks
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    best_size = 1
    best_mask = (1 << idx_255)
    
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count("1")
    
    def bron_kerbosch(R, P, X):
        nonlocal best_size, best_mask
        if time.time() - start_time > DEADLINE:
            return
        
        r_size = popcount(R)
        if r_size > best_size:
            best_size = r_size
            best_mask = R
        
        if P == 0 and X == 0:
            return
        
        # Pivot: choose from P|X with max degree in P
        U = P | X
        if U == 0:
            return
        # Find pivot
        pivot = -1
        max_deg = -1
        temp = U
        while temp:
            lb = temp & (-temp)
            v = lb.bit_length() - 1
            deg = popcount(P & adj[v])
            if deg > max_deg:
                max_deg = deg
                pivot = v
            temp ^= lb
        
        # Candidates: P vertices not adjacent to pivot
        cand = P & ~adj[pivot]
        
        while cand:
            if time.time() - start_time > DEADLINE:
                return
            lb = cand & (-cand)
            v = lb.bit_length() - 1
            bron_kerbosch(R | lb, P & adj[v], X & adj[v])
            P ^= lb
            X |= lb
            cand ^= lb
    
    # Greedy initialization: build a large clique quickly
    # Start with all multiples of 3 (they form a clique with 255)
    greedy_clique = set()
    greedy_clique.add(255)
    for c in candidates:
        if c == 255:
            continue
        if c % 3 == 0:
            greedy_clique.add(c)
    
    # Try to add more elements that are compatible with all in greedy_clique
    for c in candidates:
        if c in greedy_clique:
            continue
        ok = True
        for g in greedy_clique:
            if math.gcd(c, g) == 1:
                ok = False
                break
        if ok:
            greedy_clique.add(c)
    
    # Convert greedy clique to mask
    greedy_mask = 0
    for c in greedy_clique:
        greedy_mask |= (1 << idx[c])
    
    if popcount(greedy_mask) > best_size:
        best_size = popcount(greedy_mask)
        best_mask = greedy_mask
    
    # Also try greedy with multiples of 5
    greedy_clique2 = set()
    greedy_clique2.add(255)
    for c in candidates:
        if c == 255:
            continue
        if c % 5 == 0:
            greedy_clique2.add(c)
    for c in candidates:
        if c in greedy_clique2:
            continue
        ok = True
        for g in greedy_clique2:
            if math.gcd(c, g) == 1:
                ok = False
                break
        if ok:
            greedy_clique2.add(c)
    
    greedy_mask2 = 0
    for c in greedy_clique2:
        greedy_mask2 |= (1 << idx[c])
    
    if popcount(greedy_mask2) > best_size:
        best_size = popcount(greedy_mask2)
        best_mask = greedy_mask2
    
    # Also try greedy with multiples of 17
    greedy_clique3 = set()
    greedy_clique3.add(255)
    for c in candidates:
        if c == 255:
            continue
        if c % 17 == 0:
            greedy_clique3.add(c)
    for c in candidates:
        if c in greedy_clique3:
            continue
        ok = True
        for g in greedy_clique3:
            if math.gcd(c, g) == 1:
                ok = False
                break
        if ok:
            greedy_clique3.add(c)
    
    greedy_mask3 = 0
    for c in greedy_clique3:
        greedy_mask3 |= (1 << idx[c])
    
    if popcount(greedy_mask3) > best_size:
        best_size = popcount(greedy_mask3)
        best_mask = greedy_mask3
    
    # Local search: try to improve by swapping elements
    def local_search(clique_mask, max_iters=5000):
        nonlocal best_size, best_mask
        current_mask = clique_mask
        current_size = popcount(current_mask)
        
        # Get list of vertices in clique
        clique_verts = []
        m = current_mask
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            clique_verts.append(v)
            m ^= lb
        
        # Try to add any candidate not in clique
        added = True
        while added and (time.time() - start_time) < DEADLINE:
            added = False
            for c in candidates:
                ci = idx[c]
                if (current_mask >> ci) & 1:
                    continue
                # Check if c is compatible with all in clique
                ok = True
                for v in clique_verts:
                    if not (adj[ci] >> v) & 1:
                        ok = False
                        break
                if ok:
                    current_mask |= (1 << ci)
                    current_size += 1
                    clique_verts.append(ci)
                    added = True
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = current_mask
        
        # Try swaps: remove one, add two
        for _ in range(max_iters):
            if (time.time() - start_time) > DEADLINE:
                break
            improved = False
            for i in range(len(clique_verts)):
                vi = clique_verts[i]
                new_mask = current_mask & ~(1 << vi)
                new_verts = [v for j, v in enumerate(clique_verts) if j != i]
                # Try adding two vertices
                for a in range(n):
                    if (new_mask >> a) & 1:
                        continue
                    if not all((adj[a] >> v) & 1 for v in new_verts):
                        continue
                    temp_mask = new_mask | (1 << a)
                    temp_verts = new_verts + [a]
                    for b in range(a + 1, n):
                        if (temp_mask >> b) & 1:
                            continue
                        if not all((adj[b] >> v) & 1 for v in temp_verts):
                            continue
                        # Also check a and b are compatible
                        if not (adj[a] >> b) & 1:
                            continue
                        final_mask = temp_mask | (1 << b)
                        final_size = popcount(final_mask)
                        if final_size > current_size:
                            current_mask = final_mask
                            current_size = final_size
                            clique_verts = temp_verts + [b]
                            improved = True
                            if current_size > best_size:
                                best_size = current_size
                                best_mask = current_mask
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        
        return current_mask
    
    # Run local search from best greedy solution
    if (time.time() - start_time) < DEADLINE:
        local_search(best_mask)
    
    # Run Bron-Kerbosch from 255 with remaining time
    if (time.time() - start_time) < DEADLINE - 5:
        R = (1 << idx_255)
        P = adj[idx_255]
        X = 0
        bron_kerbosch(R, P, X)
    
    result = []
    m = best_mask
    while m:
        lb = m & (-m)
        v = lb.bit_length() - 1
        result.append(candidates[v])
        m ^= lb
    result.sort()
    
    if 255 not in result:
        result.append(255)
        result.sort()
    
    return result
