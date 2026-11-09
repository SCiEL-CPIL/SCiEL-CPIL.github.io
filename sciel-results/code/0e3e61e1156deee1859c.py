import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    n_max = 180

    # Precompute forbidden triples: (a,b,c) with a<b<c and 1/a = 1/b + 1/c
    # Equivalent: (b-a)(c-a) = a^2
    forbidden_triples = []
    for a in range(1, n_max + 1):
        a_sq = a * a
        for u in range(1, int(a_sq**0.5) + 1):
            if a_sq % u == 0:
                v = a_sq // u
                if u >= v:
                    continue
                b = a + u
                c = a + v
                if b <= n_max and c <= n_max:
                    forbidden_triples.append((a, b, c))

    # For each vertex i (1-indexed), store pair conflicts
    pair_conflicts = [[] for _ in range(n_max + 1)]
    
    for a, b, c in forbidden_triples:
        ai, bi, ci = a - 1, b - 1, c - 1
        pair_conflicts[a].append((bi, ci))
        pair_conflicts[b].append((ai, ci))
        pair_conflicts[c].append((ai, bi))

    def is_valid_addition(S, num):
        """Check if adding num to set S creates no forbidden triple."""
        idx = num - 1
        if (S >> idx) & 1:
            return False
        for j, k in pair_conflicts[num]:
            if (S >> j) & 1 and (S >> k) & 1:
                return False
        return True

    # Precompute bit masks for each triple for fast violation counting
    triple_masks = [(1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)) for a, b, c in forbidden_triples]

    def count_violations(S):
        """Count number of forbidden triples fully contained in S."""
        count = 0
        for mask in triple_masks:
            if (S & mask) == mask:
                count += 1
        return count

    # Greedy construction
    def greedy_order(order):
        S = 0
        for num in order:
            idx = num - 1
            if S & (1 << idx):
                continue
            if is_valid_addition(S, num):
                S |= (1 << idx)
        return S

    # Triple count for ordering
    triple_count = [0] * (n_max + 1)
    for a, b, c in forbidden_triples:
        triple_count[a] += 1
        triple_count[b] += 1
        triple_count[c] += 1

    order_deg = list(range(1, n_max + 1))
    order_deg.sort(key=lambda x: triple_count[x])
    order_asc = list(range(1, n_max + 1))
    order_desc = list(range(n_max, 0, -1))

    best_S = 0
    best_size = 0

    for order in [order_deg, order_asc, order_desc]:
        S = greedy_order(order)
        if S.bit_count() > best_size:
            best_size = S.bit_count()
            best_S = S

    # Random greedy orders
    for _ in range(300):
        if time.time() - start_time > deadline:
            break
        order_rand = list(range(1, n_max + 1))
        random.shuffle(order_rand)
        S = greedy_order(order_rand)
        if S.bit_count() > best_size:
            best_size = S.bit_count()
            best_S = S

    # Local search: add any valid vertex, then try swaps
    def local_search(S):
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > deadline:
                break
            # Try adding any valid vertex
            for num in range(1, n_max + 1):
                if is_valid_addition(S, num):
                    S |= (1 << (num - 1))
                    improved = True
                    break
            if not improved:
                # Try swap: remove one, add another that enables a net gain
                for r in range(1, n_max + 1):
                    ridx = r - 1
                    if not ((S >> ridx) & 1):
                        continue
                    S2 = S & ~(1 << ridx)
                    for a in range(1, n_max + 1):
                        aidx = a - 1
                        if (S2 >> aidx) & 1:
                            continue
                        if is_valid_addition(S2, a):
                            S3 = S2 | (1 << aidx)
                            if S3.bit_count() > S.bit_count():
                                S = S3
                                improved = True
                                break
                    if improved:
                        break
        return S

    best_S = local_search(best_S)
    best_size = best_S.bit_count()

    # Simulated annealing on Max-SAT formulation
    lambda_val = 100.0  # heavy penalty for violations
    
    def objective(S):
        """Higher is better: size minus penalty for violations."""
        violations = count_violations(S)
        return S.bit_count() - lambda_val * violations

    def simulated_annealing(S, T_start=10.0, T_end=0.1, n_iters=50000):
        """SA to optimize the objective."""
        current_S = S
        current_obj = objective(current_S)
        best_S_local = current_S
        best_obj = current_obj
        
        for it in range(n_iters):
            if time.time() - start_time > deadline:
                break
            
            # Choose a random move
            move_type = random.random()
            if move_type < 0.6:
                # Flip one bit
                idx = random.randint(0, n_max - 1)
                new_S = current_S ^ (1 << idx)
            elif move_type < 0.8:
                # Remove one bit
                bits = [i for i in range(n_max) if (current_S >> i) & 1]
                if not bits:
                    continue
                idx = random.choice(bits)
                new_S = current_S & ~(1 << idx)
            else:
                # Add one bit
                bits = [i for i in range(n_max) if not ((current_S >> i) & 1)]
                if not bits:
                    continue
                idx = random.choice(bits)
                new_S = current_S | (1 << idx)
            
            new_obj = objective(new_S)
            delta = new_obj - current_obj
            
            # Temperature schedule
            T = T_start * (T_end / T_start) ** (it / n_iters)
            
            if delta >= 0 or random.random() < (2.718281828 ** (delta / max(T, 1e-10))):
                current_S = new_S
                current_obj = new_obj
                if current_obj > best_obj:
                    best_obj = current_obj
                    best_S_local = current_S
        
        return best_S_local, best_obj

    # Run SA with multiple restarts
    for sa_round in range(30):
        if time.time() - start_time > deadline:
            break
        
        # Start from best or perturbed best
        if sa_round == 0:
            S_start = best_S
        else:
            # Perturb: flip 3-8 random bits
            S_start = best_S
            n_flips = random.randint(3, 8)
            for _ in range(n_flips):
                idx = random.randint(0, n_max - 1)
                S_start ^= (1 << idx)
        
        S_new, obj_new = simulated_annealing(S_start, T_start=5.0, T_end=0.05, n_iters=40000)
        
        # Polish: try to add any valid vertex
        for num in range(1, n_max + 1):
            if is_valid_addition(S_new, num):
                S_new |= (1 << (num - 1))
        
        # Check if valid (no violations) and larger
        if count_violations(S_new) == 0 and S_new.bit_count() > best_size:
            best_size = S_new.bit_count()
            best_S = S_new

    # Final polish: add any remaining valid vertices
    for num in range(1, n_max + 1):
        if is_valid_addition(best_S, num):
            best_S |= (1 << (num - 1))

    best_size = best_S.bit_count()
    result = sorted(i + 1 for i in range(n_max) if (best_S >> i) & 1)
    
    if not result:
        result = [1]
    
    return result
