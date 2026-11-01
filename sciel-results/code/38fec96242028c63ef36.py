import math
import time
import numpy as np
import random

def construct():
    START = time.time()
    DEADLINE = START + 140
    
    def elapsed():
        return time.time() - START
    
    def time_ok(margin=5):
        return elapsed() < DEADLINE - margin
    
    def compute_score_bitset(u):
        """Fast scoring using bitsets for small sets."""
        u = sorted(set(u))
        n = len(u)
        if n < 2 or 0 not in u:
            return -1.0
        max_u = max(u)
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0
        # Use bitsets for sumset and diffset
        sumset_bits = 0
        diffset_bits = 0
        offset = max_u  # offset for negative differences
        for a in u:
            for b in u:
                s = a + b
                d = a - b + offset
                sumset_bits |= (1 << s)
                diffset_bits |= (1 << d)
        sumset_size = bin(sumset_bits).count('1')
        diffset_size = bin(diffset_bits).count('1')
        r = diffset_size / sumset_size
        if r <= 0:
            return -1.0
        return 1.0 + math.log(r) / math.log(m)
    
    def compute_score_fast(u):
        u_arr = np.array(u, dtype=np.int64)
        n = len(u_arr)
        if n == 0:
            return -1.0
        if 0 not in u_arr:
            return -1.0
        max_u = int(u_arr.max())
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0
        if n <= 50:
            return compute_score_bitset(u)
        sums = u_arr[:, None] + u_arr[None, :]
        sumset_size = len(np.unique(sums))
        diffs = u_arr[:, None] - u_arr[None, :]
        diffset_size = len(np.unique(diffs))
        r = diffset_size / sumset_size
        if r <= 0:
            return -1.0
        return 1.0 + math.log(r) / math.log(m)
    
    def compute_score(u):
        u = sorted(set(u))
        if not u or 0 not in u:
            return -1.0
        return compute_score_fast(u)
    
    # Generate Golomb rulers
    def gen_golomb_ruler(n):
        """Generate a Golomb ruler with n marks using greedy approach."""
        if n <= 1:
            return [0]
        if n == 2:
            return [0, 1]
        
        # Try to build a Golomb ruler
        best = None
        best_score = -1
        
        for seed in range(50):
            random.seed(seed)
            marks = [0]
            for i in range(1, n):
                # Try to add next mark
                found = False
                for candidate in range(1, 200):
                    new_marks = sorted(marks + [candidate])
                    # Check if it's a Golomb ruler (all differences unique)
                    diffs = set()
                    valid = True
                    for j in range(len(new_marks)):
                        for k in range(j + 1, len(new_marks)):
                            d = new_marks[k] - new_marks[j]
                            if d in diffs:
                                valid = False
                                break
                            diffs.add(d)
                        if not valid:
                            break
                    if valid:
                        marks = new_marks
                        found = True
                        break
                if not found:
                    break
            
            if len(marks) == n:
                s = compute_score(marks)
                if s > best_score:
                    best_score = s
                    best = list(marks)
        
        if best:
            return best
        return [0, 1, 3]
    
    # Known good starting points
    golomb_candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 29, 34],
        [0, 1, 4, 10, 18, 23, 29, 37, 45],
        [0, 1, 4, 10, 18, 23, 29, 37, 45, 52],
        [0, 1, 4, 10, 18, 23, 29, 37, 45, 52, 60],
        [0, 1, 4, 10, 18, 23, 29, 37, 45, 52, 60, 72],
    ]
    
    # Try to generate better Golomb rulers
    for n in [7, 8, 9, 10, 11, 12]:
        if not time_ok(10):
            break
        gr = gen_golomb_ruler(n)
        if gr and gr not in golomb_candidates:
            golomb_candidates.append(gr)
    
    best_set = [0, 1, 3]
    best_score = compute_score(best_set)
    
    for u in golomb_candidates:
        if not time_ok(10):
            break
        s = compute_score(u)
        if s > best_score:
            best_score = s
            best_set = list(u)
    
    # Simulated annealing with structural moves
    def simulated_annealing(start_u, time_budget=30):
        start_t = time.time()
        current_u = sorted(set(start_u))
        current_score = compute_score(current_u)
        best_u = list(current_u)
        best_s = current_score
        
        T = 0.3
        T_min = 1e-6
        cooling = 0.997
        
        while time.time() - start_t < time_budget and time_ok(10):
            if T < T_min:
                break
            
            move_type = random.random()
            candidate = list(current_u)
            
            if move_type < 0.25 and len(candidate) > 2:
                idx = random.randint(1, len(candidate) - 1)
                candidate.pop(idx)
            elif move_type < 0.45:
                max_u = max(candidate)
                # Add element that's likely to increase differences
                new_val = random.randint(max_u + 1, max_u + len(candidate) * len(candidate) + 10)
                candidate.append(new_val)
            elif move_type < 0.70:
                idx = random.randint(1, len(candidate) - 1)
                # Small perturbations
                delta = random.randint(-3, 3)
                new_val = candidate[idx] + delta
                if new_val >= 1:
                    candidate[idx] = new_val
                    candidate = sorted(set(candidate))
                    if len(candidate) != len(current_u):
                        candidate = list(current_u)
                        continue
            elif move_type < 0.85:
                # Swap: remove one, add another
                if len(candidate) > 2:
                    idx = random.randint(1, len(candidate) - 1)
                    candidate.pop(idx)
                    max_u = max(candidate) if candidate else 0
                    new_val = random.randint(max_u + 1, max_u + len(candidate) * len(candidate) + 10)
                    candidate.append(new_val)
            else:
                # Insert in the middle
                if len(candidate) > 1:
                    idx = random.randint(1, len(candidate))
                    mid_val = candidate[idx - 1] + 1
                    new_val = random.randint(mid_val, mid_val + len(candidate) + 5)
                    candidate.insert(idx, new_val)
                    candidate = sorted(set(candidate))
            
            if not candidate or 0 not in candidate or len(candidate) < 2:
                continue
            
            candidate = sorted(set(candidate))
            cand_score = compute_score(candidate)
            
            if cand_score == -1.0:
                continue
            
            delta_s = cand_score - current_score
            if delta_s > 0 or random.random() < math.exp(delta_s / T):
                current_u = candidate
                current_score = cand_score
                if cand_score > best_s:
                    best_s = cand_score
                    best_u = list(candidate)
            
            T *= cooling
        
        return best_u, best_s
    
    if time_ok(25):
        random.seed(42)
        starting_points = [best_set] + golomb_candidates[:6]
        for sp in starting_points:
            if not time_ok(15):
                break
            sa_u, sa_s = simulated_annealing(sp, time_budget=12)
            if sa_s > best_score:
                best_score = sa_s
                best_set = sa_u
    
    # Genetic algorithm
    if time_ok(35):
        def crossover(u1, u2):
            u1 = sorted(set(u1))
            u2 = sorted(set(u2))
            all_vals = sorted(set(u1 + u2))
            child = [0]
            for v in all_vals:
                if v == 0:
                    continue
                if v in u1 or v in u2:
                    if random.random() < 0.5:
                        child.append(v)
            child = sorted(set(child))
            if len(child) < 2:
                child = [0, 1]
            return child
        
        def mutate(u):
            u = list(u)
            for _ in range(random.randint(1, 2)):
                mtype = random.random()
                if mtype < 0.4 and len(u) > 2:
                    idx = random.randint(1, len(u) - 1)
                    u.pop(idx)
                elif mtype < 0.7:
                    max_u = max(u)
                    u.append(random.randint(max_u + 1, max_u + len(u) * len(u) + 10))
                else:
                    idx = random.randint(1, len(u) - 1)
                    u[idx] = max(1, u[idx] + random.randint(-2, 2))
                    u = sorted(set(u))
                    if len(u) < 2:
                        u = [0, 1]
            return sorted(set(u))
        
        population = [best_set]
        for u in golomb_candidates[:5]:
            population.append(list(u))
        for _ in range(3):
            n = random.randint(4, 12)
            vals = sorted(random.sample(range(1, 80), n - 1))
            population.append([0] + vals)
        
        pop_size = len(population)
        for gen in range(30):
            if not time_ok(25):
                break
            
            scored = []
            for u in population:
                s = compute_score(u)
                scored.append((s, u))
            scored.sort(key=lambda x: -x[0])
            
            if scored[0][0] > best_score:
                best_score = scored[0][0]
                best_set = list(scored[0][1])
            
            survivors = [u for _, u in scored[:max(2, pop_size // 3)]]
            
            new_pop = list(survivors)
            while len(new_pop) < pop_size:
                p1 = scored[random.randint(0, min(7, pop_size - 1))][1]
                p2 = scored[random.randint(0, min(7, pop_size - 1))][1]
                child = crossover(p1, p2)
                child = mutate(child)
                new_pop.append(child)
            
            population = new_pop[:pop_size]
        
        for u in population:
            s = compute_score(u)
            if s > best_score:
                best_score = s
                best_set = list(u)
    
    # Final polish: intensive local search on the best found
    if time_ok(20):
        current_u = list(best_set)
        current_score = best_score
        
        # Phase 1: Try adding elements
        for _ in range(50):
            if not time_ok(10):
                break
            improved = False
            max_u = max(current_u)
            for new_val in range(max_u + 1, max_u + len(current_u) ** 2 + 10):
                candidate = sorted(current_u + [new_val])
                s = compute_score(candidate)
                if s > current_score + 1e-12:
                    current_u = candidate
                    current_score = s
                    improved = True
                    break
            if improved:
                continue
            
            # Phase 2: Try removing elements
            if len(current_u) > 2:
                for i in range(1, len(current_u)):
                    candidate = [x for j, x in enumerate(current_u) if j != i]
                    s = compute_score(candidate)
                    if s > current_score + 1e-12:
                        current_u = candidate
                        current_score = s
                        improved = True
                        break
            if improved:
                continue
            
            # Phase 3: Try small perturbations
            for i in range(1, len(current_u)):
                for delta in [-4, -3, -2, -1, 1, 2, 3, 4]:
                    new_val = current_u[i] + delta
                    if new_val < 1:
                        continue
                    candidate = list(current_u)
                    candidate[i] = new_val
                    candidate = sorted(set(candidate))
                    if len(candidate) != len(current_u) or 0 not in candidate:
                        continue
                    s = compute_score(candidate)
                    if s > current_score + 1e-12:
                        current_u = candidate
                        current_score = s
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        if current_score > best_score:
            best_score = current_score
            best_set = current_u
    
    best_set = sorted(set(best_set))
    if 0 not in best_set:
        best_set = [0] + best_set
    
    return best_set
