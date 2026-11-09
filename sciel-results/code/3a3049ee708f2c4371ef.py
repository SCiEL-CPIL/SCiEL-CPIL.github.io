import time
import random

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Precompute conflict triples: for each x, list of (o1,o2,o3) 
    # such that {x,o1,o2,o3} is a 4-AP
    ap_trios = [[] for _ in range(N)]
    for a in range(N):
        for d in range(1, (N - a) // 3 + 1):
            a3 = a + 3 * d
            if a3 >= N:
                break
            p0, p1, p2, p3 = a, a + d, a + 2 * d, a + 3 * d
            ap_trios[p0].append((p1, p2, p3))
            ap_trios[p1].append((p0, p2, p3))
            ap_trios[p2].append((p0, p1, p3))
            ap_trios[p3].append((p0, p1, p2))

    def can_add(mask, x):
        for (o1, o2, o3) in ap_trios[x]:
            if (mask >> o1) & 1 and (mask >> o2) & 1 and (mask >> o3) & 1:
                return False
        return True

    def polish(mask, order=None):
        if order is None:
            order = range(N)
        improved = True
        while improved:
            improved = False
            for x in order:
                if not (mask >> x) & 1:
                    if can_add(mask, x):
                        mask |= (1 << x)
                        improved = True
        return mask

    def greedy_fill(order):
        mask = 0
        for x in order:
            if can_add(mask, x):
                mask |= (1 << x)
        return mask

    # Phase 1: Multiple diverse starting points
    random.seed(42)
    best_mask = 0
    best_size = 0

    # Algebraic: base-4 digits with 0s and 1s only
    base4_mask = 0
    for i in range(N):
        n = i
        valid = True
        while n > 0:
            if n % 4 > 1:
                valid = False
                break
            n //= 4
        if valid:
            base4_mask |= (1 << i)
    base4_mask = polish(base4_mask)
    if base4_mask.bit_count() > best_size:
        best_size = base4_mask.bit_count()
        best_mask = base4_mask

    # Randomized greedies
    for _ in range(80):
        if time.time() > deadline:
            break
        order = list(range(N))
        random.shuffle(order)
        m = greedy_fill(order)
        m = polish(m)
        sz = m.bit_count()
        if sz > best_size:
            best_size = sz
            best_mask = m

    # Phase 2: Iterated local search with smart perturbation
    def local_search(mask, max_iters=40):
        in_set = [x for x in range(N) if (mask >> x) & 1]
        not_in_set = [x for x in range(N) if not (mask >> x) & 1]
        
        for K in [1, 2, 3, 4, 5, 6]:
            for _ in range(max_iters):
                if time.time() > deadline:
                    break
                if len(in_set) < K or len(not_in_set) < K:
                    break
                removals = random.sample(in_set, K)
                additions = random.sample(not_in_set, K)
                
                new_mask = mask
                for x in removals:
                    new_mask &= ~(1 << x)
                for x in additions:
                    if can_add(new_mask, x):
                        new_mask |= (1 << x)
                
                new_mask = polish(new_mask)
                new_size = new_mask.bit_count()
                if new_size > mask.bit_count():
                    return new_mask
        return mask

    current_mask = polish(best_mask)
    current_size = current_mask.bit_count()

    for _ in range(400):
        if time.time() > deadline:
            break
        
        # Perturbation: remove a cluster of elements
        perturbed = current_mask
        num_remove = random.randint(5, 12)
        in_set = [x for x in range(N) if (perturbed >> x) & 1]
        to_remove = random.sample(in_set, min(num_remove, len(in_set)))
        for x in to_remove:
            perturbed &= ~(1 << x)
        
        # Re-add greedily
        for x in range(N):
            if not (perturbed >> x) & 1:
                if can_add(perturbed, x):
                    perturbed |= (1 << x)
        
        # Local search
        improved = local_search(perturbed, max_iters=30)
        
        if improved.bit_count() > current_size:
            current_mask = improved
            current_size = improved.bit_count()
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask

    # Final polish
    best_mask = polish(best_mask)
    best_size = best_mask.bit_count()

    result = [x + 1 for x in range(N) if (best_mask >> x) & 1]
    return sorted(result)
