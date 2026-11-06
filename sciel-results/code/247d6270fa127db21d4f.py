import itertools
import random
import sys

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = len(POINTS)
POINT_TO_IDX = {p: i for i, p in enumerate(POINTS)}

def dist2(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

# Precompute isosceles relationships
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
    best_size = NUM_POINTS
    best_S = None
    
    # Run multiple greedy passes with different random seeds
    for seed in range(20):
        rng = random.Random(seed)
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

        # Greedy Construction
        while True:
            unblocked = [q for q in range(NUM_POINTS) if is_unblocked(q)]
            if not unblocked:
                break
            
            # Select best candidate
            best_p = None
            best_score = -1
            candidates = []
            
            for p in unblocked:
                # Calculate how many unblocked points p will block
                new_blocked = 0
                for r in S:
                    key = (p, r) if p < r else (r, p)
                    for q in pair_to_qs[key]:
                        if q != p and is_unblocked(q):
                            new_blocked += 1
                
                # Also consider if p is in S? No, p is unblocked, so p not in S.
                # Adding p to S removes p from unblocked.
                # The "score" is how many OTHER unblocked points get blocked.
                
                if new_blocked > best_score:
                    best_score = new_blocked
                    candidates = [p]
                elif new_blocked == best_score:
                    candidates.append(p)
            
            if not candidates:
                break
                
            # Random tie-breaking
            best_p = rng.choice(candidates)
            add_point(best_p)

        # Local Search: Try to shrink S
        improved = True
        while improved:
            improved = False
            s_list = list(S)
            rng.shuffle(s_list)
            
            for p in s_list:
                if p not in S:
                    continue
                
                # Save state
                old_S = set(S)
                old_blocked = blocked_count[:]
                
                # Remove p
                remove_point(p)
                
                # Check if maximal
                unblocked = [q for q in range(NUM_POINTS) if is_unblocked(q)]
                
                if not unblocked:
                    # S\{p} is maximal. Size reduced.
                    improved = True
                    break
                else:
                    # Try to swap: add q from unblocked to see if it restores maximality
                    # We want to see if there is a q such that S\{p} U {q} is maximal.
                    # This means q blocks all other unblocked points (and p is removed).
                    # Note: p is not in S, so p is not "blocked" in the sense of being in S.
                    # But p is a grid point. If p is not in S, it must be blocked by S\{p} U {q}.
                    # Wait, p was in S. Now p is not in S.
                    # So p is a candidate for being blocked.
                    # The unblocked set includes p?
                    # is_unblocked(p) checks if p not in S and blocked_count[p] == 0.
                    # Since p is not in S, we check blocked_count[p].
                    # blocked_count[p] was decremented when we removed p?
                    # No, remove_point(p) iterates r in S (which no longer has p).
                    # It updates blocked_count for q based on pairs (p, r) being removed.
                    # So blocked_count[p] reflects how many pairs in S\{p} block p.
                    
                    # If unblocked is not empty, we try to find q in unblocked
                    # such that adding q makes unblocked empty.
                    found_swap = False
                    for q in unblocked:
                        # Check if adding q blocks all points in unblocked
                        # Note: q itself will be in S, so it won't be in unblocked anymore.
                        # We need to check if all u in unblocked \ {q} are blocked by S\{p} U {q}.
                        
                        # Simulate adding q
                        # We can just check if q blocks all other unblocked points.
                        # A point u is blocked by q if there exists r in S\{p} such that {q, r, u} is isosceles.
                        # OR if q itself is u? No, u != q.
                        
                        # Efficient check:
                        # For each u in unblocked:
                        #   if u == q: continue (q is added to S, so it's "handled")
                        #   Check if u is blocked by q and S\{p}.
                        #   u is blocked if blocked_count[u] > 0 (from S\{p}) OR (q, r) blocks u for some r in S\{p}.
                        
                        # This simulation is expensive if done naively for each q.
                        # Given small size of S, it's okay.
                        
                        valid_swap = True
                        for u in unblocked:
                            if u == q:
                                continue
                            # Is u blocked by S\{p}?
                            if blocked_count[u] > 0:
                                continue
                            # Is u blocked by q and some r in S\{p}?
                            blocked_by_q = False
                            for r in S: # S is S\{p}
                                key = (q, r) if q < r else (r, q)
                                if u in pair_to_qs[key]:
                                    blocked_by_q = True
                                    break
                            if not blocked_by_q:
                                valid_swap = False
                                break
                        
                        if valid_swap:
                            # Perform swap
                            add_point(q)
                            # Note: add_point(q) updates S and blocked_count.
                            # S is now S\{p} U {q}.
                            # Size is same.
                            # We accept this neutral move to potentially allow further reductions.
                            found_swap = True
                            break
                    
                    if found_swap:
                        improved = True
                        break
                    else:
                        # Restore state
                        S.clear()
                        S.update(old_S)
                        for i in range(NUM_POINTS):
                            blocked_count[i] = old_blocked[i]

        if len(S) < best_size:
            best_size = len(S)
            best_S = set(S)
            
    return [POINTS[q] for q in best_S]
