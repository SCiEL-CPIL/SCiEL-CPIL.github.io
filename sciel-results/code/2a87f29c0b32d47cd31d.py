import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 275  # Safety margin for 300s limit

    N = 64
    M = N * N

    # Precompute all points
    points = [(x, y) for x in range(N) for y in range(N)]
    coords = np.array(points, dtype=np.int32)

    # Precompute squared distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sum(diff**2, axis=-1)

    def try_add(S_set, dist_to_S, p):
        """Check if point p can be added to set S_set."""
        if p in S_set:
            return False

        if not S_set:
            return True

        # Check 1: All distances from p to S must be unique
        s_indices = np.array(list(S_set), dtype=np.int32)
        d_vals = dist[p, s_indices]

        if len(np.unique(d_vals)) < len(d_vals):
            return False

        # Check 2: For each s in S, d(p,s) must not be in dist_to_S[s]
        # We can vectorize this check using a set of (s, d) pairs? 
        # No, dist_to_S[s] is a Python set. 
        # But we can iterate over s_indices and check membership.
        # Optimization: If we had a global set of forbidden (s, d) pairs, it would be faster, 
        # but maintaining that is complex. 
        # Let's stick to the loop but ensure it's tight.
        for i, s in enumerate(s_indices):
            if d_vals[i] in dist_to_S[s]:
                return False

        return True

    def add_point(S_set, dist_to_S, p):
        """Add point p to S_set and update dist_to_S."""
        S_set.add(p)
        dist_to_S[p] = set()
        for q in S_set:
            if q == p:
                continue
            d = int(dist[p, q])
            dist_to_S[p].add(d)
            dist_to_S[q].add(d)

    def remove_point(S_set, dist_to_S, p):
        """Remove point p from S_set and update dist_to_S."""
        if p not in S_set:
            return
        for q in S_set:
            if q == p:
                continue
            d = int(dist[p, q])
            dist_to_S[q].discard(d)
        dist_to_S.pop(p, None)
        S_set.discard(p)

    def greedy_construct(order):
        """Greedy construction following a given order."""
        S_set = set()
        dist_to_S = {}
        for p in order:
            if time.time() > deadline - 5:
                break
            if try_add(S_set, dist_to_S, p):
                add_point(S_set, dist_to_S, p)
        return S_set, dist_to_S

    def local_search(S_set, dist_to_S, max_iters=5000):
        """
        Local search with:
        1. Add
        2. 1-swap (remove 1, add 1)
        3. 2-swap (remove 2, add 2)
        4. Kick (remove k, re-greedy)
        """
        S_set = set(S_set)
        dist_to_S = {k: set(v) for k, v in dist_to_S.items()}
        
        improved = True
        iter_count = 0
        while improved and time.time() < deadline - 10 and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # 1. Try to add any point
            # Instead of creating a full list of candidates every time (O(M)), 
            # we can just iterate over a shuffled range and check membership.
            # However, try_add is expensive. We need a good candidate list.
            # Let's create the candidate list once per iteration if needed, or cache it?
            # Caching is hard because S changes.
            # Let's just create it. M=4096 is small.
            candidates = [i for i in range(M) if i not in S_set]
            random.shuffle(candidates)
            
            added = False
            for p in candidates:
                if try_add(S_set, dist_to_S, p):
                    add_point(S_set, dist_to_S, p)
                    improved = True
                    added = True
                    break
            
            if improved:
                continue

            # 2. If no addition, try 1-swap
            if S_set:
                s_list = list(S_set)
                random.shuffle(s_list)
                
                # Try removing up to 10 random points
                swap_success = False
                for r in s_list[:10]:
                    remove_point(S_set, dist_to_S, r)
                    
                    # Re-fetch candidates (S changed)
                    candidates = [i for i in range(M) if i not in S_set]
                    random.shuffle(candidates)
                    
                    for p in candidates:
                        if p == r:
                            continue
                        if try_add(S_set, dist_to_S, p):
                            add_point(S_set, dist_to_S, p)
                            improved = True
                            swap_success = True
                            break
                    
                    if not swap_success:
                        add_point(S_set, dist_to_S, r)
                    else:
                        break
                
                if improved:
                    continue

            # 3. Try 2-swap (remove 2, add 2)
            if len(S_set) >= 2:
                s_list = list(S_set)
                random.shuffle(s_list)
                
                # Try a few pairs of removals
                num_pairs = min(5, len(s_list))
                for i in range(num_pairs):
                    r1 = s_list[i]
                    r2 = s_list[(i + 1) % len(s_list)]
                    if r1 == r2:
                        continue
                    
                    remove_point(S_set, dist_to_S, r1)
                    remove_point(S_set, dist_to_S, r2)
                    
                    cand_list = [i for i in range(M) if i not in S_set]
                    random.shuffle(cand_list)
                    
                    success = False
                    # Limit search for speed
                    search_limit = min(300, len(cand_list))
                    for p1 in cand_list[:search_limit]:
                        if try_add(S_set, dist_to_S, p1):
                            add_point(S_set, dist_to_S, p1)
                            for p2 in cand_list[:search_limit]:
                                if p2 == p1:
                                    continue
                                if try_add(S_set, dist_to_S, p2):
                                    add_point(S_set, dist_to_S, p2)
                                    success = True
                                    improved = True
                                    break
                            if not success:
                                remove_point(S_set, dist_to_S, p1)
                            else:
                                break
                    
                    if not success:
                        add_point(S_set, dist_to_S, r1)
                        add_point(S_set, dist_to_S, r2)
                    else:
                        break
                
                if improved:
                    continue

            # 4. Kick: Remove 3-5 random points, re-greedy
            if S_set:
                s_list = list(S_set)
                random.shuffle(s_list)
                kick_size = min(5, len(s_list))
                kick_points = s_list[:kick_size]
                for r in kick_points:
                    remove_point(S_set, dist_to_S, r)
                
                # Re-greedy: Try to add as many points as possible
                cand_list = [i for i in range(M) if i not in S_set]
                random.shuffle(cand_list)
                
                kick_improved = False
                added_count = 0
                # Try to add at least as many as we removed to consider it improved?
                # Or just any improvement in size?
                # Let's try to add up to kick_size points.
                for p in cand_list:
                    if time.time() > deadline - 10:
                        break
                    if try_add(S_set, dist_to_S, p):
                        add_point(S_set, dist_to_S, p)
                        kick_improved = True
                        added_count += 1
                        if added_count >= kick_size:
                            break
                        
                if kick_improved:
                    improved = True
                else:
                    for r in kick_points:
                        add_point(S_set, dist_to_S, r)

        return S_set, dist_to_S

    def generate_orders():
        """Generate diverse orderings for greedy initialization."""
        orders = []
        
        # 1. Pure random orders
        for _ in range(10):
            o = list(range(M))
            random.shuffle(o)
            orders.append(o)
        
        # 2. Spiral/center-out ordering
        center = N // 2
        order = []
        for r in range(max(center, N - center)):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) == r or abs(dy) == r:
                        x = center + dx
                        y = center + dy
                        if 0 <= x < N and 0 <= y < N:
                            idx = y * N + x
                            if idx not in order:
                                order.append(idx)
        for i in range(M):
            if i not in order:
                order.append(i)
        # Light shuffle
        for i in range(len(order) - 1, 0, -1):
            j = random.randint(0, i)
            order[i], order[j] = order[j], order[i]
        orders.append(order)
        
        # 3. Row-major with perturbation
        o = list(range(M))
        for i in range(len(o) - 1, 0, -1):
            if random.random() < 0.3:
                j = random.randint(0, i)
                o[i], o[j] = o[j], o[i]
        orders.append(o)
        
        # 4. Column-major
        col_order = []
        for x in range(N):
            for y in range(N):
                col_order.append(y * N + x)
        random.shuffle(col_order)
        orders.append(col_order)
        
        # 5. Diagonal
        diag_order = []
        for d in range(2 * N - 1):
            for x in range(N):
                y = d - x
                if 0 <= y < N:
                    diag_order.append(y * N + x)
        random.shuffle(diag_order)
        orders.append(diag_order)

        return orders

    best_S_set = set()
    best_dist_to_S = {}
    
    orders = generate_orders()
    
    # Run greedy + local search for each order
    for o in orders:
        if time.time() > deadline - 30:
            break
            
        S_set, dist_to_S = greedy_construct(o)
        S_set, dist_to_S = local_search(S_set, dist_to_S, max_iters=2000)
        
        if len(S_set) > len(best_S_set):
            best_S_set = S_set.copy()
            best_dist_to_S = {k: set(v) for k, v in dist_to_S.items()}
            
    # Final polish on the best set found
    if time.time() < deadline - 5 and best_S_set:
        best_S_set, best_dist_to_S = local_search(best_S_set, best_dist_to_S, max_iters=5000)

    return [points[i] for i in best_S_set]

# For testing
if __name__ == "__main__":
    res = construct()
    print(len(res))
