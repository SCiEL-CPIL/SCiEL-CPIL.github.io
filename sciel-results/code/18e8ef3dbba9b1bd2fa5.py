import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # Safety margin for 90s limit

    best_poly = None
    best_nz = float('inf')
    
    def to_list(exps, coefs):
        max_deg = max(exps) if exps else 0
        poly = [0] * (max_deg + 1)
        for e, c in zip(exps, coefs):
            poly[e] = c
        # Ensure length <= 41
        if len(poly) > 41:
             return None
        # Ensure final coefficient is nonzero
        if poly[-1] == 0:
             return None
        return poly

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

    def square_nonzero(poly_list):
        """
        Calculate number of nonzero terms in the square of the polynomial.
        poly_list is a list of coefficients in ascending degree.
        """
        if not poly_list:
            return 0
        # Pad to ensure length is manageable, though convolve handles it
        # We only care about exact integer arithmetic
        c = np.asarray(poly_list, dtype=np.int64)
        sq = np.convolve(c, c, mode='full')
        return int(np.count_nonzero(sq))

    def update_best(exps, coefs):
        nonlocal best_poly, best_nz
        if len(set(exps)) != 12:
            return
        if any(e < 0 or e >= 41 for e in exps):
            return
        if any(c == 0 or abs(c) > 64 for c in coefs):
            return
        
        poly_list = to_list(exps, coefs)
        if poly_list is None:
            return
            
        nz = square_nonzero(poly_list)
        if nz < best_nz:
            best_nz = nz
            best_poly = poly_list

    def local_search(initial_exps, initial_coefs, max_iters=20000, t0=5.0, t_final=0.01, cooling=0.9995):
        """
        Local search with simulated annealing.
        """
        nonlocal best_poly, best_nz
        
        cur_exps = sorted(initial_exps)
        cur_coefs = list(initial_coefs)
        cur_poly = to_list(cur_exps, cur_coefs)
        if cur_poly is None:
            return
        cur_nz = square_nonzero(cur_poly)
        
        if cur_nz < best_nz:
            best_nz = cur_nz
            best_poly = cur_poly

        best_in_run_poly = cur_poly
        best_in_run_nz = cur_nz
        best_in_run_exps = list(cur_exps)
        best_in_run_coefs = list(cur_coefs)

        for it in range(max_iters):
            if time.time() > deadline - 2.0:
                break

            t = t0 * (cooling ** it)
            if t < t_final:
                t = t_final

            new_exps = list(cur_exps)
            new_coefs = list(cur_coefs)
            
            r = random.random()
            
            # Move Types
            if r < 0.40:
                # Single exponent shift
                idx = random.randint(0, 11)
                # Bias towards small shifts
                shift = random.randint(-5, 5)
                new_e = new_exps[idx] + shift
                if 0 <= new_e < 41 and new_e not in new_exps:
                    new_exps[idx] = new_e
                    new_exps.sort()
                else:
                    continue
            elif r < 0.60:
                # Coefficient change
                idx = random.randint(0, 11)
                # Try small integers, but allow larger range occasionally
                if random.random() < 0.8:
                    new_val = random.choice([-4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8, 12, 16, 32])
                else:
                    new_val = random.randint(-64, 64)
                if new_val != 0 and new_val != new_coefs[idx]:
                    new_coefs[idx] = new_val
            elif r < 0.80:
                # Two exponent shift (coordinated)
                i, j = random.sample(range(12), 2)
                si = random.randint(-3, 3)
                sj = random.randint(-3, 3)
                ne_i = new_exps[i] + si
                ne_j = new_exps[j] + sj
                if (0 <= ne_i < 41 and 0 <= ne_j < 41 and 
                    ne_i not in new_exps and ne_j not in new_exps and ne_i != ne_j):
                    new_exps[i] = ne_i
                    new_exps[j] = ne_j
                    new_exps.sort()
                else:
                    continue
            else:
                # Large jump: reset one exponent to random
                idx = random.randint(0, 11)
                new_e = random.randint(0, 40)
                if new_e not in new_exps:
                    new_exps[idx] = new_e
                    new_exps.sort()
                else:
                    continue

            # Validate
            if len(set(new_exps)) != 12 or any(e < 0 or e >= 41 for e in new_exps):
                continue
            if any(c == 0 or abs(c) > 64 for c in new_coefs):
                continue

            new_poly = to_list(new_exps, new_coefs)
            if new_poly is None:
                continue

            new_nz = square_nonzero(new_poly)
            delta = new_nz - cur_nz

            # Accept if better, or with probability exp(-delta/t)
            if delta < 0 or (t > 0 and random.random() < np.exp(-delta / t)):
                cur_nz = new_nz
                cur_exps = new_exps
                cur_coefs = new_coefs
                cur_poly = new_poly

                if cur_nz < best_in_run_nz:
                    best_in_run_nz = cur_nz
                    best_in_run_poly = cur_poly
                    best_in_run_exps = list(cur_exps)
                    best_in_run_coefs = list(cur_coefs)

                if cur_nz < best_nz:
                    best_nz = cur_nz
                    best_poly = cur_poly

        return best_in_run_exps, best_in_run_coefs, best_in_run_poly

    # Phase 1: Generate diverse starting points
    random.seed(42)
    starting_points = []

    # 1. Clustered structures (Two groups of 6)
    # These are strong candidates for cancellation
    for d1 in [1, 2, 3]:
        for d2 in [1, 2, 3]:
            for gap in [2, 3, 4, 5, 6, 8, 10, 15, 20]:
                for a1 in range(0, 10):
                    a2 = a1 + gap
                    exp1 = [a1 + i * d1 for i in range(6)]
                    exp2 = [a2 + i * d2 for i in range(6)]
                    all_exps = sorted(exp1 + exp2)
                    if len(set(all_exps)) == 12 and all(e < 41 for e in all_exps):
                        # Try different coefficient patterns
                        # Pattern 1: +1, -1
                        coefs = [1]*6 + [-1]*6
                        starting_points.append((all_exps, coefs))
                        # Pattern 2: Alternating
                        coefs_alt = [1, -1, 1, -1, 1, -1] * 2
                        starting_points.append((all_exps, coefs_alt))
                        # Pattern 3: Random small ints
                        coefs_rand = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
                        starting_points.append((all_exps, coefs_rand))

    # 2. Arithmetic Progressions
    for d in [1, 2, 3, 4, 5]:
        max_a = max(0, 40 - 11 * d)
        for a in range(0, min(max_a + 1, 5)):
            exps = [a + i * d for i in range(12)]
            if exps[-1] < 41:
                # Alternating signs
                coefs = [1 if i % 2 == 0 else -1 for i in range(12)]
                starting_points.append((exps, coefs))
                # Block signs
                coefs2 = [1]*6 + [-1]*6
                starting_points.append((exps, coefs2))
                # Random signs
                coefs3 = [random.choice([-1, 1]) for _ in range(12)]
                starting_points.append((exps, coefs3))

    # 3. Symmetric structures
    for center in [10, 15, 20, 25, 30]:
        for offsets in [[1,2,3,4,5,6], [1,2,3,4,6,8], [2,3,4,5,6,7], [1,3,5,7,9,11], [1,2,4,8,16,32]]:
            exps = sorted([center - o for o in offsets] + [center + o for o in offsets])
            if all(0 <= e < 41 for e in exps) and len(set(exps)) == 12:
                coefs = [1]*6 + [-1]*6
                starting_points.append((exps, coefs))
                coefs2 = [1, -1, 1, -1, 1, -1] * 2
                starting_points.append((exps, coefs2))

    # 4. Random starts
    for _ in range(50):
        exps = sorted(random.sample(range(41), 12))
        coefs = [random.choice([-3, -2, -1, 1, 2, 3, 4, 5]) for _ in range(12)]
        starting_points.append((exps, coefs))

    # Phase 2: Evaluate and Refine
    scored_starts = []
    for exps, coefs in starting_points:
        poly = to_list(exps, coefs)
        if poly is None:
            continue
        nz = square_nonzero(poly)
        scored_starts.append((nz, exps, coefs, poly))
        
    scored_starts.sort(key=lambda x: x[0])
    
    # Run SA on the best N starting points
    num_sa_runs = min(len(scored_starts), 60)
    time_per_run = (deadline - time.time() - 5) / max(num_sa_runs, 1)
    
    for i in range(num_sa_runs):
        if time.time() > deadline - 5:
            break
        
        nz, exps, coefs, poly = scored_starts[i]
        # If we already found a very good solution, we can reduce iterations
        # but for now, give each a decent chance
        est_iters = int(time_per_run * 3000)
        est_iters = max(2000, min(est_iters, 20000))
        
        local_search(exps, coefs, max_iters=est_iters)

    # Phase 3: Continue from best found with more aggressive search
    if best_poly is not None and time.time() < deadline - 10:
        best_exp = [i for i, c in enumerate(best_poly) if c != 0]
        best_coef = [c for c in best_poly if c != 0]
        
        # Multiple restarts from perturbed best
        remaining_time = deadline - time.time() - 5
        num_extra = max(1, int(remaining_time / 3.0))
        
        for restart in range(min(num_extra, 50)):
            if time.time() > deadline - 5:
                break
            
            pert_exp = list(best_exp)
            pert_coef = list(best_coef)
            
            # Perturb
            num_pert = random.randint(2, 4)
            for _ in range(num_pert):
                if random.random() < 0.6:
                    idx = random.randint(0, 11)
                    shift = random.randint(-8, 8)
                    new_e = pert_exp[idx] + shift
                    if 0 <= new_e < 41 and new_e not in pert_exp:
                        pert_exp[idx] = new_e
                        pert_exp.sort()
                else:
                    idx = random.randint(0, 11)
                    pert_coef[idx] = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8])
                    if pert_coef[idx] == 0:
                        pert_coef[idx] = 1
            
            if len(set(pert_exp)) == 12 and all(0 <= e < 41 for e in pert_exp) and \
               all(c != 0 and abs(c) <= 64 for c in pert_coef):
                remaining = deadline - time.time() - 5
                est_iters = max(2000, int(remaining * 2500))
                local_search(pert_exp, pert_coef, max_iters=min(est_iters, 15000))

    # Final fallback
    if best_poly is None:
        best_poly = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    if not validate(best_poly):
        best_poly = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    return best_poly
