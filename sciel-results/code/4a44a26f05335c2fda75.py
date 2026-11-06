"""Sparse squares of twelve-term integer polynomials - collision-aware local search with sparse convolution."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start = time.time()
    deadline = start + 87.0

    best_poly = None
    best_nz = float('inf')
    best_exps = None
    best_coefs = None

    def square_nonzero_sparse(exponents, coeffs):
        """Fast sparse square using dict-based accumulation."""
        # Build dict: degree -> sum of products
        sq = defaultdict(int)
        n = len(exponents)
        for i in range(n):
            ei = exponents[i]
            ci = coeffs[i]
            sq[ei + ei] += ci * ci
            for j in range(i + 1, n):
                ej = exponents[j]
                cj = coeffs[j]
                s = ei + ej
                sq[s] += 2 * ci * cj
        return sum(1 for v in sq.values() if v != 0)

    def square_full_sparse(exponents, coeffs):
        """Return full dict of square coefficients."""
        sq = defaultdict(int)
        n = len(exponents)
        for i in range(n):
            ei = exponents[i]
            ci = coeffs[i]
            sq[ei + ei] += ci * ci
            for j in range(i + 1, n):
                ej = exponents[j]
                cj = coeffs[j]
                s = ei + ej
                sq[s] += 2 * ci * cj
        return dict(sq)

    def count_collisions(exponents):
        """Count number of sum collisions among pairs."""
        pair_sums = defaultdict(int)
        n = len(exponents)
        for i in range(n):
            for j in range(i + 1, n):
                pair_sums[exponents[i] + exponents[j]] += 1
        collisions = 0
        for c in pair_sums.values():
            if c > 1:
                collisions += c * (c - 1) // 2
        return collisions

    def validate(poly_coeff_list):
        if not poly_coeff_list or len(poly_coeff_list) > 41:
            return False
        if poly_coeff_list[-1] == 0:
            return False
        for c in poly_coeff_list:
            if not isinstance(c, int) or c < -64 or c > 64:
                return False
        if sum(1 for c in poly_coeff_list if c != 0) != 12:
            return False
        return True

    def update_best(exponents, coeffs, nz):
        nonlocal best_poly, best_nz, best_exps, best_coefs
        if nz < best_nz:
            best_nz = nz
            best_exps = list(exponents)
            best_coefs = list(coeffs)
            max_deg = max(exponents)
            poly = [0] * (max_deg + 1)
            for e, c in zip(exponents, coeffs):
                poly[e] = c
            best_poly = poly

    random.seed(42)

    # Phase 1: Generate diverse structured seeds
    seeds = []

    # Arithmetic progressions
    for d in range(1, 8):
        max_a = max(0, 40 - 11 * d)
        for a in range(0, max_a + 1, 2):
            exps = [a + i * d for i in range(12)]
            if exps[-1] >= 41:
                continue
            for coefs in [[1]*12, [1 if i%2==0 else -1 for i in range(12)],
                         [1,1,-1,-1,1,1,-1,-1,1,1,-1,-1],
                         [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1],
                         [2,1,-1,-2,1,2,-1,-2,1,2,-1,-2]]:
                seeds.append((exps, coefs))

    # Two-AP structures
    for d1 in range(1, 5):
        for d2 in range(1, 5):
            for a1 in range(0, 12, 2):
                for gap in range(3, 18, 2):
                    a2 = a1 + gap
                    exps = sorted([a1 + i*d1 for i in range(6)] + [a2 + i*d2 for i in range(6)])
                    if len(set(exps)) != 12 or any(e >= 41 for e in exps):
                        continue
                    for s in [1, -1]:
                        coefs = [s]*12
                        seeds.append((exps, coefs))
                    # Mixed signs
                    coefs = [s if i < 6 else -s for i in range(12)]
                    seeds.append((exps, coefs))

    # Symmetric structures
    for center in range(6, 36, 2):
        for offsets in [[1,2,3,4,5,6],[1,2,3,4,5,8],[1,2,4,5,7,9],[2,3,4,5,6,8],
                       [1,3,4,6,8,10],[1,2,5,6,7,9]]:
            exps = sorted([center-o for o in offsets] + [center+o for o in offsets])
            if any(e < 0 or e >= 41 for e in exps) or len(set(exps)) != 12:
                continue
            for coefs in [[1]*12, [1]*6+[-1]*6, [1,-1,1,-1,1,-1]*2,
                         [1,1,1,1,1,1,-1,-1,-1,-1,-1,-1]]:
                seeds.append((exps, coefs))

    # Evaluate seeds
    for exps, coefs in seeds:
        if time.time() > deadline - 55:
            break
        if len(set(exps)) != 12 or any(e < 0 or e >= 41 for e in exps):
            continue
        if any(c == 0 or abs(c) > 64 for c in coefs):
            continue
        nz = square_nonzero_sparse(exps, coefs)
        update_best(exps, coefs, nz)

    # Phase 2: Collision-aware local search with hill climbing
    def local_search(init_exps, init_coefs, max_iters=8000):
        nonlocal best_poly, best_nz, best_exps, best_coefs
        cur_exps = sorted(init_exps)
        cur_coefs = list(init_coefs)
        cur_nz = square_nonzero_sparse(cur_exps, cur_coefs)
        update_best(cur_exps, cur_coefs, cur_nz)

        T = 4.0
        T_min = 0.05
        cooling = 0.9997
        no_improve = 0
        patience = 1500

        for it in range(max_iters):
            if time.time() > deadline - 3:
                break

            new_exps = list(cur_exps)
            new_coefs = list(cur_coefs)
            move_type = random.random()

            if move_type < 0.25:
                # Single exponent move
                idx = random.randint(0, 11)
                shift = random.randint(-6, 6)
                new_e = new_exps[idx] + shift
                if 0 <= new_e < 41 and new_e not in new_exps:
                    new_exps[idx] = new_e
                    new_exps.sort()
                else:
                    continue
            elif move_type < 0.40:
                # Two exponent move
                i, j = random.sample(range(12), 2)
                si = random.randint(-5, 5)
                sj = random.randint(-5, 5)
                ne_i = new_exps[i] + si
                ne_j = new_exps[j] + sj
                if (0 <= ne_i < 41 and 0 <= ne_j < 41 and
                    ne_i not in new_exps and ne_j not in new_exps and ne_i != ne_j):
                    new_exps[i] = ne_i
                    new_exps[j] = ne_j
                    new_exps.sort()
                else:
                    continue
            elif move_type < 0.50:
                # Three exponent move
                i, j, k = random.sample(range(12), 3)
                si = random.randint(-4, 4)
                sj = random.randint(-4, 4)
                sk = random.randint(-4, 4)
                ne_i = new_exps[i] + si
                ne_j = new_exps[j] + sj
                ne_k = new_exps[k] + sk
                if (0 <= ne_i < 41 and 0 <= ne_j < 41 and 0 <= ne_k < 41 and
                    ne_i not in new_exps and ne_j not in new_exps and ne_k not in new_exps and
                    ne_i != ne_j and ne_i != ne_k and ne_j != ne_k):
                    new_exps[i] = ne_i
                    new_exps[j] = ne_j
                    new_exps[k] = ne_k
                    new_exps.sort()
                else:
                    continue
            elif move_type < 0.65:
                # Single coefficient modification
                idx = random.randint(0, 11)
                new_val = random.choice([-8,-6,-4,-3,-2,-1,1,2,3,4,5,6,8,12,16])
                if new_val != new_coefs[idx] and abs(new_val) <= 64:
                    new_coefs[idx] = new_val
                else:
                    continue
            elif move_type < 0.75:
                # Two coefficient modification
                i, j = random.sample(range(12), 2)
                new_coefs[i] = random.choice([-6,-4,-3,-2,-1,1,2,3,4,5,6])
                new_coefs[j] = random.choice([-6,-4,-3,-2,-1,1,2,3,4,5,6])
            elif move_type < 0.85:
                # Collision-guided: pick a pair and try to create another pair with same sum
                i, j = random.sample(range(12), 2)
                target_sum = new_exps[i] + new_exps[j]
                # Find a pair (k,l) not involving i,j that can be adjusted
                candidates = [x for x in range(12) if x not in (i, j)]
                if len(candidates) >= 2:
                    k, l = random.sample(candidates, 2)
                    # Try to move k so that new_exps[k] + new_exps[l] == target_sum
                    new_k = target_sum - new_exps[l]
                    if 0 <= new_k < 41 and new_k not in new_exps and new_k != new_exps[l]:
                        new_exps[k] = new_k
                        new_exps.sort()
                    else:
                        # Try to move l
                        new_l = target_sum - new_exps[k]
                        if 0 <= new_l < 41 and new_l not in new_exps and new_l != new_exps[k]:
                            new_exps[l] = new_l
                            new_exps.sort()
                        else:
                            continue
                else:
                    continue
            else:
                # Triple collision-guided: try to create a 3-way sum collision
                i, j = random.sample(range(12), 2)
                target_sum = new_exps[i] + new_exps[j]
                candidates = [x for x in range(12) if x not in (i, j)]
                if len(candidates) >= 2:
                    k, l = random.sample(candidates, 2)
                    # Move both k and l to hit target_sum
                    # k + l = target_sum, so try k = target_sum - l
                    new_k = target_sum - new_exps[l]
                    if 0 <= new_k < 41 and new_k not in new_exps and new_k != new_exps[l]:
                        new_exps[k] = new_k
                        new_exps.sort()
                    else:
                        continue

            if len(set(new_exps)) != 12 or any(e < 0 or e >= 41 for e in new_exps):
                continue
            if any(c == 0 or abs(c) > 64 for c in new_coefs):
                continue

            new_nz = square_nonzero_sparse(new_exps, new_coefs)
            delta = new_nz - cur_nz

            if delta <= 0 or random.random() < min(1.0, max(0.0, -delta / max(T, T_min))):
                cur_nz = new_nz
                cur_exps = list(new_exps)
                cur_coefs = list(new_coefs)
                if delta < 0:
                    no_improve = 0
                else:
                    no_improve += 1
                update_best(cur_exps, cur_coefs, cur_nz)
            else:
                no_improve += 1

            T = max(T * cooling, T_min)

            # Restart if stuck
            if no_improve > patience:
                T = 4.0
                no_improve = 0

        return cur_exps, cur_coefs, cur_nz

    # Phase 3: Run local search from multiple diverse starts
    num_runs = 20
    for i in range(num_runs):
        if time.time() > deadline - 8:
            break

        if i < 6 and best_exps is not None:
            # Perturb best solution
            exps = list(best_exps)
            coefs = list(best_coefs)
            for _ in range(random.randint(1, 4)):
                if random.random() < 0.5:
                    idx = random.randint(0, 11)
                    shift = random.randint(-4, 4)
                    new_e = exps[idx] + shift
                    if 0 <= new_e < 41 and new_e not in exps:
                        exps[idx] = new_e
                        exps.sort()
                else:
                    idx = random.randint(0, 11)
                    coefs[idx] = random.choice([-6,-4,-3,-2,-1,1,2,3,4,5,6,8])
                    if coefs[idx] == 0:
                        coefs[idx] = 1
            if len(set(exps)) == 12 and all(0 <= e < 41 for e in exps) and \
               all(c != 0 and abs(c) <= 64 for c in coefs):
                local_search(exps, coefs, max_iters=5000)
        else:
            # Random structured start
            rtype = random.random()
            if rtype < 0.35:
                d = random.randint(1, 7)
                max_a = max(0, 40 - 11 * d)
                a = random.randint(0, max_a)
                exps = [a + i * d for i in range(12)]
                if exps[-1] >= 41:
                    continue
            elif rtype < 0.65:
                d1 = random.randint(1, 4)
                d2 = random.randint(1, 4)
                a1 = random.randint(0, 10)
                gap = random.randint(3, 16)
                a2 = a1 + gap
                exps = sorted([a1 + i*d1 for i in range(6)] + [a2 + i*d2 for i in range(6)])
                if len(set(exps)) != 12 or any(e >= 41 for e in exps):
                    continue
            elif rtype < 0.85:
                center = random.randint(8, 34)
                offsets = sorted(random.sample(range(1, 20), 6))
                exps = sorted([center-o for o in offsets] + [center+o for o in offsets])
                if any(e < 0 or e >= 41 for e in exps) or len(set(exps)) != 12:
                    continue
            else:
                exps = sorted(random.sample(range(41), 12))
            coefs = [random.choice([-4,-3,-2,-1,1,2,3,4]) for _ in range(12)]
            local_search(exps, coefs, max_iters=4000)

    # Phase 4: Intensive final search on best
    if best_exps is not None and time.time() < deadline - 5:
        local_search(best_exps, best_coefs, max_iters=10000)

    # Phase 5: Targeted coefficient optimization for best exponent set
    if best_exps is not None and time.time() < deadline - 3:
        cur_exps = list(best_exps)
        cur_coefs = list(best_coefs)
        cur_nz = square_nonzero_sparse(cur_exps, cur_coefs)
        update_best(cur_exps, cur_coefs, cur_nz)

        # Try all sign patterns for small coefficient sets
        sign_patterns = []
        for pattern in [[1]*12, [1 if i%2==0 else -1 for i in range(12)],
                       [1,1,-1,-1,1,1,-1,-1,1,1,-1,-1],
                       [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1],
                       [1,1,1,-1,-1,-1,1,1,1,-1,-1,-1],
                       [1,-1,-1,1,1,-1,-1,1,1,-1,-1,1]]:
            sign_patterns.append(pattern)

        for sp in sign_patterns:
            if time.time() > deadline - 2:
                break
            for mag in [1, 2, 3]:
                for coefs in [[sp[i]*mag for i in range(12)],
                             [sp[i]*(mag if i%3==0 else 1) for i in range(12)]]:
                    if any(c == 0 or abs(c) > 64 for c in coefs):
                        continue
                    nz = square_nonzero_sparse(cur_exps, coefs)
                    update_best(cur_exps, coefs, nz)

    # Final fallback
    if best_poly is None:
        best_poly = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    if not validate(best_poly):
        best_poly = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    return best_poly

# EVOLVE-BLOCK-END
