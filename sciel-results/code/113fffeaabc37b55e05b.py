import random
import time
import itertools

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
NUM_POINTS = len(POINTS)

def dist2(p, q):
    return (p[0] - q[0])**2 + (p[1] - q[1])**2

# Precompute rev_iso_pairs[u][v] = list of p
rev_iso_pairs = [[[] for _ in range(NUM_POINTS)] for _ in range(NUM_POINTS)]

for p_idx, p in enumerate(POINTS):
    for u_idx in range(NUM_POINTS):
        if u_idx == p_idx: continue
        u = POINTS[u_idx]
        du = dist2(p, u)
        for v_idx in range(u_idx + 1, NUM_POINTS):
            if v_idx == p_idx: continue
            v = POINTS[v_idx]
            dv = dist2(p, v)
            dvu = dist2(u, v)
            if du == dv or du == dvu or dv == dvu:
                rev_iso_pairs[u_idx][v_idx].append(p_idx)
                rev_iso_pairs[v_idx][u_idx].append(p_idx)

def construct():
    start_time = time.time()
    deadline = start_time + 300
    
    # Initial Greedy
    order = list(range(NUM_POINTS))
    random.shuffle(order)
    in_set = [False] * NUM_POINTS
    S = []
    for p_idx in order:
        conflict = False
        for i in range(len(S)):
            u = S[i]
            for v in S[i+1:]:
                if p_idx in rev_iso_pairs[u][v]:
                    conflict = True
                    break
            if conflict: break
        if not conflict:
            in_set[p_idx] = True
            S.append(p_idx)
            
    # Init C
    C = [0] * NUM_POINTS
    for p_idx in range(NUM_POINTS):
        if in_set[p_idx]: continue
        for i in range(len(S)):
            u = S[i]
            for j in range(i+1, len(S)):
                v = S[j]
                for p in rev_iso_pairs[u][v]:
                    if p == p_idx:
                        C[p_idx] += 1
                        break # p_idx is unique in the list? No, p_idx is the point.
                        # Actually, p_idx is in rev_iso_pairs[u][v] at most once.
                        
    # Wait, the loop above is O(N * |S|^2 * deg).
    # Better:
    C = [0] * NUM_POINTS
    for i in range(len(S)):
        u = S[i]
        for j in range(i+1, len(S)):
            v = S[j]
            for p in rev_iso_pairs[u][v]:
                if not in_set[p]:
                    C[p] += 1

    best_size = len(S)
    
    while time.time() < deadline:
        # Try Removals
        S_copy = S[:]
        random.shuffle(S_copy)
        removed = False
        for u in S_copy:
            if time.time() > deadline:
                break
            
            # Calculate deltas for removing u
            dec_counts = {}
            for v in S:
                if v == u: continue
                for p in rev_iso_pairs[u][v]:
                    if not in_set[p]:
                        dec_counts[p] = dec_counts.get(p, 0) + 1
            
            valid = True
            for p, dec in dec_counts.items():
                if C[p] - dec <= 0:
                    valid = False
                    break
            
            if valid:
                # Apply
                in_set[u] = False
                S.remove(u)
                for p, dec in dec_counts.items():
                    C[p] -= dec
                best_size = len(S)
                removed = True
                break
        
        if removed:
            continue
            
        # Try Swaps
        if len(S) == 0: break
        u = random.choice(S)
        not_in_S = [i for i in range(NUM_POINTS) if not in_set[i]]
        if not not_in_S: break
        w = random.choice(not_in_S)
        
        # Check Independence of w with S\{u}
        S_new = [x for x in S if x != u]
        indep_ok = True
        for i in range(len(S_new)):
            a = S_new[i]
            for b in S_new[i+1:]:
                if w in rev_iso_pairs[a][b]:
                    indep_ok = False
                    break
            if not indep_ok: break
        
        if not indep_ok:
            continue
            
        # Check Maximality
        # Deltas
        dec_counts = {}
        inc_counts = {}
        for v in S_new:
            for p in rev_iso_pairs[u][v]:
                if not in_set[p]:
                    dec_counts[p] = dec_counts.get(p, 0) + 1
            for p in rev_iso_pairs[w][v]:
                if not in_set[p]:
                    inc_counts[p] = inc_counts.get(p, 0) + 1
        
        # Check p not in S_new
        # p in S_new are excluded.
        # p == u is not in S_new, so check it.
        # p == w is in S_new, so skip.
        
        valid_max = True
        for p in range(NUM_POINTS):
            if in_set[p] and p != u: # p in S_new
                continue
            # p is not in S_new
            # New count = C[p] - dec + inc
            # Note: C[u] was 0 (or not tracked). 
            # If p == u, C[p] is effectively 0.
            base = C[p] if not in_set[p] else 0
            dec = dec_counts.get(p, 0)
            inc = inc_counts.get(p, 0)
            if base - dec + inc <= 0:
                valid_max = False
                break
        
        if valid_max:
            # Apply Swap
            in_set[u] = False
            in_set[w] = True
            S.remove(u)
            S.append(w)
            
            # Update C
            for p in range(NUM_POINTS):
                if in_set[p]:
                    C[p] = 0
                else:
                    C[p] = C[p] - dec_counts.get(p, 0) + inc_counts.get(p, 0)
                    # Wait, this update is only valid if we applied the swap.
                    # But we need to be careful: C[p] for p in S_new should be 0?
                    # No, C is only defined for p not in S.
                    # So for p in S_new, C[p] should be 0.
                    # For p not in S_new, C[p] = C_old[p] - dec + inc.
                    
    return [(POINTS[i][0], POINTS[i][1]) for i in S]
