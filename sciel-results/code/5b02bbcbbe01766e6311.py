"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 64
TOTAL_POINTS = N * N  # 4096
DEADLINE_SECONDS = 330
SAFETY_MARGIN = 30.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS - SAFETY_MARGIN
    
    # Precompute all squared distances
    xs = np.array([i // N for i in range(TOTAL_POINTS)], dtype=np.int32)
    ys = np.array([i % N for i in range(TOTAL_POINTS)], dtype=np.int32)
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist2 = dx * dx + dy * dy
    del dx, dy
    
    def try_add(p, S_list, dists_from, dist2):
        """Check if p can be added. Returns True if yes."""
        seen_from_p = set()
        for q in S_list:
            d = dist2[p, q]
            if d in seen_from_p:
                return False
            seen_from_p.add(d)
        for q in S_list:
            if dist2[q, p] in dists_from[q]:
                return False
        return True
    
    def do_add(p, S_list, S_set, dists_from, dist2):
        """Add p to S."""
        S_list.append(p)
        S_set.add(p)
        for q in S_list:
            if q != p:
                dists_from[q].add(dist2[q, p])
                dists_from[p].add(dist2[p, q])
    
    def greedy_with_order(order, dist2, deadline):
        S_list = []
        S_set = set()
        dists_from = [set() for _ in range(TOTAL_POINTS)]
        
        for p in order:
            if time.time() > deadline:
                break
            if try_add(p, S_list, dists_from, dist2):
                do_add(p, S_list, S_set, dists_from, dist2)
        
        return S_list, S_set, dists_from
    
    def local_search(S_list, S_set, dists_from, dist2, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding
            candidates = [p for p in range(TOTAL_POINTS) if p not in S_set]
            random.shuffle(candidates)
            for p in candidates:
                if time.time() > deadline:
                    break
                if try_add(p, S_list, dists_from, dist2):
                    do_add(p, S_list, S_set, dists_from, dist2)
                    improved = True
                    break
            # Try swapping
            if not improved and time.time() < deadline:
                for i, q in enumerate(S_list):
                    if time.time() > deadline:
                        break
                    # Remove q
                    S_list.remove(q)
                    S_set.discard(q)
                    for r in S_list:
                        dists_from[r].discard(dist2[r, q])
                    dists_from[q] = set()
                    
                    candidates = [p for p in range(TOTAL_POINTS) if p not in S_set]
                    random.shuffle(candidates)
                    for p in candidates:
                        if time.time() > deadline:
                            break
                        if try_add(p, S_list, dists_from, dist2):
                            do_add(p, S_list, S_set, dists_from, dist2)
                            improved = True
                            break
                    
                    if not improved:
                        # Re-add q
                        do_add(q, S_list, S_set, dists_from, dist2)
                    
                    if improved:
                        break
        return S_list
    
    best_S = []
    
    # Multiple greedy runs
    num_greedy = 20
    for i in range(num_greedy):
        if time.time() > deadline:
            break
        order = list(range(TOTAL_POINTS))
        random.shuffle(order)
        S_list, S_set, dists_from = greedy_with_order(order, dist2, deadline)
        if len(S_list) > len(best_S):
            best_S = S_list
            best_S_set = S_set
            best_dists = dists_from
    
    # Local search on best
    if time.time() < deadline and best_S:
        best_S = local_search(best_S, best_S_set, best_dists, dist2, deadline)
    
    result = [(p // N, p % N) for p in best_S]
    return result

# EVOLVE-BLOCK-END
