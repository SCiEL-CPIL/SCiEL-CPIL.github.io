import itertools
import random
import sys
import time

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = len(POINTS)
POINT_TO_IDX = {p: i for i, p in enumerate(POINTS)}

def dist2(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[0])**2

# Precompute isosceles relationships
# pair_to_qs: dict mapping (idx_i, idx_j) to set of idx_k such that {i,j,k} is isosceles
pair_to_qs = {}

for i in range(NUM_POINTS):
    for j in range(i + 1, NUM_POINTS):
        p_i = POINTS[i]
        p_j = POINTS[j]
        d_ij = dist2(p_i, p_j)
        
        blocked = set()
        for k in range(NUM_POINTS):
            if k == i or k == j:
                continue
            p_k = POINTS[k]
            d_ik = dist2(p_i, p_k)
            d_jk = dist2(p_j, p_k)
            
            if d_ik == d_jk or d_ik == d_ij or d_jk == d_ij:
                blocked.add(k)
        
        pair_to_qs[(i, j)] = blocked
        pair_to_qs[(j, i)] = blocked

def construct():
    start_time = time.time()
    time_limit = 250 # Leave some buffer for 300s limit
    
    best_size = NUM_POINTS + 1
    best_S = None
    
    # Run multiple restarts
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        random.seed(restart * 123 + 456)
        
        S = set()
        blocked_count = [0] * NUM_POINTS
        
        def is_unblocked(q):
            return q not in S and blocked_count[q] == 0

        def add_point(p):
            S.add(p)
            for r in S:
                if r == p:
                    continue
                key = (p, r) if p < r else (r, p)
                for q in pair_to_qs[key]:
                    if q != p and q not in S:
                        blocked_count[q] += 1

        def remove_point(p):
            S.remove(p)
            for r in S:
                key = (p, r) if p < r else (r, p)
                for q in pair_to_qs[key]:
                    if q != p and q not in S:
                        blocked_count[q] -= 1

        # 1. Greedy Construction
        while True:
            unblocked = [q for q in range(NUM_POINTS) if is_unblocked(q)]
            if not unblocked:
                break
            
            # Select best candidate: one that blocks the most unblocked points
            best_p = None
            best_score = -1
            
            for p in unblocked:
                new_blocked = 0
                for r in S:
                    key = (p, r) if p < r else (r, p)
                    for q in pair_to_qs[key]:
                        if q != p and is_unblocked(q):
                            new_blocked += 1
                if new_blocked > best_score:
                    best_score = new_blocked
                    best_p = p
            
            if best_p is None:
                break
            add_point(best_p)
            
        # 2. Local Search
        improved = True
        while improved and (time.time() - start_time < time_limit):
            improved = False
            s_list = list(S)
            random.shuffle(s_list)
            
            # Try 1-removals
            for p in s_list:
                if p not in S:
                    continue
                
                # Check if S\{p} is maximal
                # Temporarily remove p
                old_S = set(S)
                old_blocked = blocked_count[:]
                
                remove_point(p)
                
                # Check if any point is unblocked
                is_maximal = True
                for q in range(NUM_POINTS):
                    if q not in S and blocked_count[q] == 0:
                        is_maximal = False
                        break
                
                if is_maximal:
                    # Accept removal
                    improved = True
                    # S and blocked_count are already updated to S\{p}
                    break
                else:
                    # Restore
                    S.clear()
                    S.update(old_S)
                    for i in range(NUM_POINTS):
                        blocked_count[i] = old_blocked[i]
            
            if improved:
                continue
                
            # Try 2-swaps (remove p, add q)
            # To be efficient, we only try swaps if we haven't found a 1-removal
            # We limit the number of swaps tried per iteration to avoid slowdown
            swap_found = False
            # Sample a subset of pairs if S is large, otherwise all
            # For N=16, |S| is likely small (< 50), so all pairs is fine
            s_list = list(S)
            random.shuffle(s_list)
            
            for p in s_list:
                if swap_found:
                    break
                for q in range(NUM_POINTS):
                    if q in S or q == p:
                        continue
                    
                    # Check if (S\{p}) U {q} is valid maximal independent
                    # 1. Independence check
                    # For all r in S\{p}, pair (q, r) must not have any point in S\{p}
                    # i.e., pair_to_qs[(q, r)] is disjoint from S\{p}
                    
                    valid_indep = True
                    for r in S:
                        if r == p:
                            continue
                        key = (q, r) if q < r else (r, q)
                        # Check if any point in pair_to_qs[key] is in S and != p
                        # Note: q is not in pair_to_qs[key] as q is distinct from r
                        # We just need to check intersection with S
                        # Since S is a set, we can check membership
                        # But pair_to_qs[key] is a set.
                        # If (pair_to_qs[key] & (S - {p})) is not empty, invalid.
                        # Optimization: precompute? No, just check.
                        # S is small.
                        # We can break early.
                        for s in pair_to_qs[key]:
                            if s in S and s != p:
                                valid_indep = False
                                break
                        if not valid_indep:
                            break
                    
                    if not valid_indep:
                        continue
                    
                    # 2. Maximality check
                    # Temporarily apply swap
                    old_S = set(S)
                    old_blocked = blocked_count[:]
                    
                    remove_point(p)
                    # Add q
                    # Note: add_point assumes q is not in S. It isn't.
                    # But add_point updates blocked_count based on current S.
                    # We removed p, so S is S\{p}.
                    add_point(q)
                    
                    # Check maximality
                    is_maximal = True
                    for v in range(NUM_POINTS):
                        if v not in S and blocked_count[v] == 0:
                            is_maximal = False
                            break
                    
                    if is_maximal:
                        # Accept swap
                        improved = True
                        swap_found = True
                        break
                    else:
                        # Restore
                        S.clear()
                        S.update(old_S)
                        for i in range(NUM_POINTS):
                            blocked_count[i] = old_blocked[i]
                
            if not improved and not swap_found:
                # No moves found. Try a random perturbation?
                # Or just break and restart
                # For now, break to restart
                break
            
            # Update best
            if len(S) < best_size:
                best_size = len(S)
                best_S = set(S)
                
    if best_S is None:
        # Fallback: return a valid set, even if not minimal
        # The greedy construction in the last restart left S in some state
        # But we might have restored it.
        # Let's just run one more greedy to be safe if best_S is None
        S = set()
        blocked_count = [0] * NUM_POINTS
        while True:
            unblocked = [q for q in range(NUM_POINTS) if q not in S and blocked_count[q] == 0]
            if not unblocked:
                break
            # Pick first
            p = unblocked[0]
            S.add(p)
            for r in S:
                if r == p:
                    continue
                key = (p, r) if p < r else (r, p)
                for q in pair_to_qs[key]:
                    if q != p and q not in S:
                        blocked_count[q] += 1
        best_S = S
        
    return [POINTS[q] for q in best_S]
