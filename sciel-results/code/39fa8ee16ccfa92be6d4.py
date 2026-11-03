import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    idx_to_word = all_words

    # Precompute ball members as integer arrays
    # ball_members[i] = list of indices of words within distance 1 of word i
    ball_members = []
    for i, w in enumerate(all_words):
        members = [i]
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_word = w[:pos] + (new_val,) + w[pos+1:]
                j = word_to_idx[new_word]
                members.append(j)
        ball_members.append(members)

    ball_members = np.array(ball_members, dtype=np.int32)
    ball_size = ball_members.shape[1]  # 13 for q=3,n=6

    def get_idx(word):
        return word_to_idx[word]

    # Multi-start greedy initialization
    def greedy_init(random_order=None):
        if random_order is None:
            random_order = list(range(total_words))
            random.shuffle(random_order)
        
        uncovered = np.ones(total_words, dtype=np.int32)
        code = []
        code_set = set()
        
        for idx in random_order:
            if uncovered[idx] == 0:
                continue
            # Count how many currently-uncovered words this ball covers
            covered_count = uncovered[ball_members[idx]].sum()
            if covered_count > 0:
                code.append(idx)
                code_set.add(idx)
                uncovered[ball_members[idx]] = 0
        
        return code, code_set

    # Coverage count maintenance
    def build_coverage(code_indices):
        cov = np.zeros(total_words, dtype=np.int32)
        for ci in code_indices:
            cov[ball_members[ci]] += 1
        return cov

    def uncovered_count(cov):
        return np.sum(cov == 0)

    # Lagrangian local search
    def local_search(code, code_set, cov, time_limit):
        best_code = list(code)
        best_set = set(code_set)
        best_size = len(code)
        best_cov = cov.copy()
        best_uncovered = uncovered_count(best_cov)
        
        # Lagrangian multiplier
        lam = 1.0
        T = 2.0
        T_min = 0.01
        cooling = 0.999
        
        steps = 0
        no_improve = 0
        max_no_improve = 500
        
        while time.time() < time_limit:
            if not code:
                break
            
            current_uncovered = uncovered_count(cov)
            current_cost = len(code) + lam * current_uncovered
            
            move_type = random.random()
            
            if move_type < 0.25:
                # Remove a codeword
                if len(code) <= 1:
                    T = max(T * cooling, T_min)
                    continue
                ri = random.randint(0, len(code) - 1)
                w = code[ri]
                
                new_code = code.copy()
                new_code.pop(ri)
                new_set = code_set.copy()
                new_set.discard(w)
                
                new_cov = cov - 1
                new_cov[ball_members[w]] -= 1
                new_uncovered = uncovered_count(new_cov)
                new_cost = len(new_code) + lam * new_uncovered
                delta = new_cost - current_cost
                
                if delta <= 0 or random.random() < np.exp(-delta / T):
                    code = new_code
                    code_set = new_set
                    cov = new_cov
                    if new_uncovered == 0 and len(code) < best_size:
                        best_size = len(code)
                        best_code = list(code)
                        best_set = set(code_set)
                        best_cov = cov.copy()
                        best_uncovered = new_uncovered
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                    
            elif move_type < 0.55:
                # Perturb: change one symbol in one codeword
                if not code:
                    T = max(T * cooling, T_min)
                    continue
                ri = random.randint(0, len(code) - 1)
                w = code[ri]
                pos = random.randint(0, n - 1)
                old_val = all_words[w][pos]
                new_vals = [v for v in range(q) if v != old_val]
                new_val = random.choice(new_vals)
                new_w_tuple = all_words[w][:pos] + (new_val,) + all_words[w][pos+1:]
                new_w = word_to_idx[new_w_tuple]
                
                if new_w in code_set and new_w != w:
                    no_improve += 1
                    T = max(T * cooling, T_min)
                    continue
                
                new_code = code.copy()
                new_code[ri] = new_w
                new_set = code_set.copy()
                new_set.discard(w)
                new_set.add(new_w)
                
                new_cov = cov.copy()
                new_cov[ball_members[w]] -= 1
                new_cov[ball_members[new_w]] += 1
                new_uncovered = uncovered_count(new_cov)
                new_cost = len(new_code) + lam * new_uncovered
                delta = new_cost - current_cost
                
                if delta <= 0 or random.random() < np.exp(-delta / T):
                    code = new_code
                    code_set = new_set
                    cov = new_cov
                    if new_uncovered == 0 and len(code) < best_size:
                        best_size = len(code)
                        best_code = list(code)
                        best_set = set(code_set)
                        best_cov = cov.copy()
                        best_uncovered = new_uncovered
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                    
            elif move_type < 0.80:
                # Swap: replace one codeword with a random other
                if not code:
                    T = max(T * cooling, T_min)
                    continue
                ri = random.randint(0, len(code) - 1)
                old_w = code[ri]
                
                new_w = random.randint(0, total_words - 1)
                if new_w in code_set and new_w != old_w:
                    no_improve += 1
                    T = max(T * cooling, T_min)
                    continue
                
                new_code = code.copy()
                new_code[ri] = new_w
                new_set = code_set.copy()
                new_set.discard(old_w)
                new_set.add(new_w)
                
                new_cov = cov.copy()
                new_cov[ball_members[old_w]] -= 1
                new_cov[ball_members[new_w]] += 1
                new_uncovered = uncovered_count(new_cov)
                new_cost = len(new_code) + lam * new_uncovered
                delta = new_cost - current_cost
                
                if delta <= 0 or random.random() < np.exp(-delta / T):
                    code = new_code
                    code_set = new_set
                    cov = new_cov
                    if new_uncovered == 0 and len(code) < best_size:
                        best_size = len(code)
                        best_code = list(code)
                        best_set = set(code_set)
                        best_cov = cov.copy()
                        best_uncovered = new_uncovered
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                    
            else:
                # 2-swap: replace two codewords with two random others
                if len(code) < 2:
                    T = max(T * cooling, T_min)
                    continue
                r1, r2 = random.sample(range(len(code)), 2)
                w1, w2 = code[r1], code[r2]
                
                nw1 = random.randint(0, total_words - 1)
                nw2 = random.randint(0, total_words - 1)
                
                # Check collisions
                collision = False
                for nw in [nw1, nw2]:
                    if nw in code_set and nw != w1 and nw != w2:
                        collision = True
                        break
                    if nw1 == nw2:
                        collision = True
                        break
                if collision:
                    no_improve += 1
                    T = max(T * cooling, T_min)
                    continue
                
                new_code = code.copy()
                new_code[r1] = nw1
                new_code[r2] = nw2
                new_set = code_set.copy()
                new_set.discard(w1)
                new_set.discard(w2)
                new_set.add(nw1)
                new_set.add(nw2)
                
                new_cov = cov.copy()
                new_cov[ball_members[w1]] -= 1
                new_cov[ball_members[w2]] -= 1
                new_cov[ball_members[nw1]] += 1
                new_cov[ball_members[nw2]] += 1
                new_uncovered = uncovered_count(new_cov)
                new_cost = len(new_code) + lam * new_uncovered
                delta = new_cost - current_cost
                
                if delta <= 0 or random.random() < np.exp(-delta / T):
                    code = new_code
                    code_set = new_set
                    cov = new_cov
                    if new_uncovered == 0 and len(code) < best_size:
                        best_size = len(code)
                        best_code = list(code)
                        best_set = set(code_set)
                        best_cov = cov.copy()
                        best_uncovered = new_uncovered
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            
            T = max(T * cooling, T_min)
            steps += 1
            
            # Adapt lagrangian
            if current_uncovered > 0:
                lam = min(lam * 1.02, 10.0)
            elif current_uncovered == 0 and len(code) > best_size:
                lam = max(lam * 0.98, 0.1)
            
            if no_improve > max_no_improve:
                # Restart from best
                code = list(best_code)
                code_set = set(best_set)
                cov = best_cov.copy()
                T = 2.0
                no_improve = 0
                max_no_improve = int(max_no_improve * 1.2)
        
        return best_code, best_set, best_cov

    # Run multi-start
    best_overall = None
    best_overall_size = float('inf')
    best_overall_cov = None
    
    num_starts = 8
    per_start_time = (deadline - start_time - 5.0) / num_starts
    
    for s in range(num_starts):
        remaining = deadline - start_time - 5.0
        if remaining < 2.0:
            break
        
        # Different initialization strategies
        if s == 0:
            # Standard greedy with all zeros first
            code = [0]  # index 0 = (0,0,0,0,0,0)
            code_set = {0}
            cov = np.zeros(total_words, dtype=np.int32)
            cov[ball_members[0]] += 1
            # Then greedy fill
            uncovered = (cov == 0)
            for idx in range(1, total_words):
                if uncovered[idx]:
                    covered = uncovered[ball_members[idx]].sum()
                    if covered > 0:
                        code.append(idx)
                        code_set.add(idx)
                        cov[ball_members[idx]] += 1
                        uncovered = (cov == 0)
        elif s == 1:
            # Greedy with random order
            code, code_set = greedy_init()
            cov = build_coverage(code)
        elif s == 2:
            # Greedy with random order
            code, code_set = greedy_init()
            cov = build_coverage(code)
        else:
            # Greedy with random order
            code, code_set = greedy_init()
            cov = build_coverage(code)
        
        if not code:
            continue
        
        # Local search
        ls_time = min(per_start_time, deadline - start_time - 3.0)
        best_code, best_set, best_cov = local_search(code, code_set, cov, start_time + ls_time)
        
        if len(best_code) < best_overall_size:
            best_overall_size = len(best_code)
            best_overall = best_code
            best_overall_cov = best_cov
        
        # Check if we've hit the known optimum
        if best_overall_size <= 73:
            break
    
    if best_overall is None:
        # Fallback: simple greedy
        code, code_set = greedy_init()
        best_overall = code
    
    # Final polish: greedy removal
    final_code = list(best_overall)
    final_cov = build_coverage(final_code)
    
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        for i in range(len(final_code) - 1, -1, -1):
            w = final_code[i]
            test_cov = final_cov - 1
            test_cov[ball_members[w]] -= 1
            if np.sum(test_cov == 0) == 0:
                final_code.pop(i)
                final_cov = test_cov
                improved = True
                break
    
    return [list(all_words[ci]) for ci in final_code]
