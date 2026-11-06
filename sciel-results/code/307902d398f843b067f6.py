import numpy as np
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 146.0  # Safety margin from 150s

    N = 256

    # Precompute all 4-term arithmetic progressions (0-indexed)
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            ap_list.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))

    aps = np.array(ap_list, dtype=np.int32)
    num_aps = len(aps)

    # For each position, precompute the other 3 positions in each AP containing it
    aps_at = [[] for _ in range(N)]
    for idx in range(num_aps):
        for pos in aps[idx]:
            aps_at[pos].append(idx)

    # Build others_for_pos: for position i, array of shape (num_aps_containing_i, 3)
    others_for_pos = [None] * N
    for i in range(N):
        if not aps_at[i]:
            others_for_pos[i] = np.array([], dtype=np.int32).reshape(0, 3)
        else:
            others_list = []
            for ap_idx in aps_at[i]:
                row = aps[ap_idx]
                others = [p for p in row if p != i]
                others_list.append(others)
            others_for_pos[i] = np.array(others_list, dtype=np.int32)

    def count_mono(col):
        c = col[aps]
        return int(np.sum((c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])))

    def subgradient_step(col, score, lr, rng, max_iters=1000):
        """Vectorized stochastic subgradient descent on the spin system."""
        col_float = col.astype(np.float64)
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            # Compute subgradient: for each position, count how many APs it's in
            # where the other 3 are monochromatic, weighted by whether flipping helps
            grad = np.zeros(N, dtype=np.float64)
            for i in range(N):
                others = others_for_pos[i]
                if others.shape[0] == 0:
                    continue
                oc = col[others]
                all_same = (oc[:, 0] == oc[:, 1]) & (oc[:, 1] == oc[:, 2])
                if np.any(all_same):
                    c_other = oc[:, 0]
                    c_i = col[i]
                    # If other 3 match c_i, flipping i breaks mono APs (good if score decreases)
                    # If other 3 match 1-c_i, flipping i creates mono APs (bad)
                    before = (c_other == c_i) & all_same
                    after = (c_other == (1 - c_i)) & all_same
                    delta = np.sum(after) - np.sum(before)
                    grad[i] = -delta  # negative gradient = direction to decrease score
            
            # Stochastic update: pick random subset and update
            num_updates = rng.randint(1, 10)
            update_indices = rng.sample(range(N), num_updates)
            for i in update_indices:
                if grad[i] != 0:
                    # Flip if gradient suggests improvement
                    if grad[i] < 0:
                        col[i] = 1 - col[i]
                        score += int(-grad[i])
                    elif rng.random() < 0.1:  # occasional random flip for exploration
                        col[i] = 1 - col[i]
                        # Recompute score change
                        d = 0
                        others = others_for_pos[i]
                        if others.shape[0] > 0:
                            oc = col[others]
                            all_same = (oc[:, 0] == oc[:, 1]) & (oc[:, 1] == oc[:, 2])
                            if np.any(all_same):
                                c_other = oc[:, 0]
                                c_i = col[i]
                                before = (c_other == c_i) & all_same
                                after = (c_other == (1 - c_i)) & all_same
                                d = int(np.sum(after) - np.sum(before))
                        score += d
        return col, score

    def vectorized_multi_flip(col, score, num_flips, rng, max_rounds=200):
        """Try flipping multiple bits simultaneously to escape local minima."""
        for _ in range(max_rounds):
            if time.time() > deadline - 3.0:
                break
            # Pick a random subset of positions to flip
            flip_mask = np.zeros(N, dtype=bool)
            indices = rng.sample(range(N), num_flips)
            for i in indices:
                flip_mask[i] = True
            
            # Compute delta for the combined flip
            old_score = score
            new_col = col.copy()
            new_col[flip_mask] = 1 - new_col[flip_mask]
            new_score = count_mono(new_col)
            
            if new_score < score:
                col = new_col
                score = new_score
            # Accept with probability based on temperature-like criterion
            elif rng.random() < np.exp(-(new_score - score) / 3.0):
                col = new_col
                score = new_score
        
        return col, score

    def high_temp_anneal(col, score, rng, max_rounds=500):
        """High-temperature annealing with large multi-flip moves."""
        for r in range(max_rounds):
            if time.time() > deadline - 5.0:
                break
            # Temperature schedule: start high, decrease
            t = 10.0 * (0.95 ** r)
            
            # Larger flip sets at high temperature
            num_flips = rng.randint(5, 20)
            flip_mask = np.zeros(N, dtype=bool)
            indices = rng.sample(range(N), num_flips)
            for i in indices:
                flip_mask[i] = True
            
            new_col = col.copy()
            new_col[flip_mask] = 1 - new_col[flip_mask]
            new_score = count_mono(new_col)
            
            delta = new_score - score
            if delta < 0 or rng.random() < np.exp(-delta / t):
                col = new_col
                score = new_score
        
        return col, score

    def fast_delta(col, i):
        """Change in mono count if we flip position i."""
        others = others_for_pos[i]
        if others.shape[0] == 0:
            return 0
        oc = col[others]
        all_same = (oc[:, 0] == oc[:, 1]) & (oc[:, 1] == oc[:, 2])
        if not np.any(all_same):
            return 0
        c_other = oc[:, 0]
        c_i = col[i]
        before = (c_other == c_i) & all_same
        after = (c_other == (1 - c_i)) & all_same
        return int(np.sum(after) - np.sum(before))

    def local_search(col, score, max_passes=50):
        improved = True
        passes = 0
        while improved and passes < max_passes and time.time() < deadline - 2.0:
            improved = False
            passes += 1
            order = list(range(N))
            random.shuffle(order)
            for i in order:
                if time.time() > deadline - 2.0:
                    break
                d = fast_delta(col, i)
                if d < 0:
                    col[i] = 1 - col[i]
                    score += d
                    improved = True
        return col, score

    def sa_run(seed, max_steps, t0, t1):
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)
        col = np_rng.randint(0, 2, size=N).astype(np.int32)
        cur = count_mono(col)
        best = col.copy()
        best_s = cur
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            t = t0 * (t1 / t0) ** (step / max_steps)
            i = rng.randint(0, N - 1)
            d = fast_delta(col, i)
            if d <= 0 or (t > 1e-12 and rng.random() < np.exp(-d / t)):
                col[i] = 1 - col[i]
                cur += d
                if cur < best_s:
                    best_s = cur
                    best = col.copy()
        return best, best_s

    def ga_run(pop_size, generations, seed):
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)
        
        # Initialize population
        pop = []
        scores = []
        for _ in range(pop_size):
            if rng.random() < 0.5:
                col = np_rng.randint(0, 2, size=N).astype(np.int32)
            else:
                # Structured start
                block = 1 + rng.randint(1, 10)
                col = np.array([(i // block) % 2 for i in range(N)], dtype=np.int32)
            s = count_mono(col)
            pop.append(col)
            scores.append(s)
            
        best_idx = np.argmin(scores)
        global_best_col = pop[best_idx].copy()
        global_best_score = scores[best_idx]

        for gen in range(generations):
            if time.time() > deadline - 10.0:
                break
            
            # Selection: Tournament
            next_pop = []
            next_scores = []
            
            for _ in range(pop_size):
                # Pick 3 parents
                parents = []
                for _ in range(3):
                    idx = rng.randint(0, pop_size - 1)
                    parents.append(idx)
                # Select best parent
                p1_idx = parents[0]
                p2_idx = parents[1]
                p3_idx = parents[2]
                
                best_p = min(parents, key=lambda x: scores[x])
                # Crossover with another random parent
                other_p = rng.randint(0, pop_size - 1)
                
                p1 = pop[best_p]
                p2 = pop[other_p]
                
                # Uniform crossover
                mask = np_rng.randint(0, 2, size=N).astype(bool)
                child = np.where(mask, p1, p2).astype(np.int32)
                
                # Mutation
                if rng.random() < 0.1:
                    mut_idx = rng.randint(0, N - 1)
                    child[mut_idx] = 1 - child[mut_idx]
                
                s = count_mono(child)
                
                # Local search on child (shallow)
                child, s = local_search(child, s, max_passes=5)
                
                next_pop.append(child)
                next_scores.append(s)
                
                if s < global_best_score:
                    global_best_score = s
                    global_best_col = child.copy()

            pop = next_pop
            scores = next_scores
            
            # Elitism: Keep the best individual from current gen
            best_in_gen = np.argmin(scores)
            pop[0] = pop[best_in_gen]
            scores[0] = scores[best_in_gen]

        return global_best_col, global_best_score

    best_col = None
    best_score = float('inf')

    # Phase 1: Multi-start Greedy/LS with subgradient steps
    for trial in range(30):
        if time.time() > deadline - 30.0:
            break
        seed = 42 + trial * 97
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)

        if trial % 2 == 0:
            col = np_rng.randint(0, 2, size=N).astype(np.int32)
        else:
            block = 1 + (trial % 8)
            col = np.array([(i // block) % 2 for i in range(N)], dtype=np.int32)

        score = count_mono(col)
        col, score = local_search(col, score, max_passes=50)

        # Add subgradient step for every 2nd trial
        if trial % 2 == 0 and time.time() < deadline - 25.0:
            col, score = subgradient_step(col, score, 0.5, rng, max_iters=200)
            col, score = local_search(col, score, max_passes=30)

        # Add multi-flip escape attempts for every 3rd trial
        if trial % 3 == 0 and time.time() < deadline - 25.0:
            col, score = vectorized_multi_flip(col, score, 8, rng, max_rounds=50)
            col, score = local_search(col, score, max_passes=30)

        if score < best_score:
            best_score = score
            best_col = col.copy()

    # Phase 2: SA Runs with high-temp annealing
    for r in range(15):
        if time.time() > deadline - 20.0:
            break
        seed = 1000 + r * 211
        col, s = sa_run(seed, 50000, 2.0, 0.01)
        col, s = local_search(col, s, max_passes=30)
        
        # Add high-temp annealing after SA
        if time.time() < deadline - 15.0:
            rng = random.Random(seed + 777)
            col, s = high_temp_anneal(col, s, rng, max_rounds=100)
            col, s = local_search(col, s, max_passes=20)
        
        # Add multi-flip escape
        if time.time() < deadline - 12.0:
            rng = random.Random(seed + 888)
            col, s = vectorized_multi_flip(col, s, 12, rng, max_rounds=30)
            col, s = local_search(col, s, max_passes=20)
        
        if s < best_score:
            best_score = s
            best_col = col.copy()

    # Phase 3: Genetic Algorithm
    if time.time() < deadline - 15.0:
        col, s = ga_run(20, 20, 9999)
        col, s = local_search(col, s, max_passes=50)
        if s < best_score:
            best_score = s
            best_col = col.copy()

    # Phase 4: Final Polish
    if best_col is not None and time.time() < deadline - 2.0:
        col = best_col.copy()
        col, best_score = local_search(col, best_score, max_passes=100)
        
        # Final subgradient polish
        if time.time() < deadline - 3.0:
            rng = random.Random(11111)
            col, s = subgradient_step(col, best_score, 0.3, rng, max_iters=300)
            col, s = local_search(col, s, max_passes=50)
            if s < best_score:
                best_score = s
                best_col = col
        
        # Final multi-flip polish
        if time.time() < deadline - 1.0:
            rng = random.Random(12345)
            col, s = vectorized_multi_flip(col, best_score, 6, rng, max_rounds=20)
            col, s = local_search(col, s, max_passes=50)
            if s < best_score:
                best_score = s
                best_col = col
        else:
            best_col = col

    return [int(c) for c in best_col]
