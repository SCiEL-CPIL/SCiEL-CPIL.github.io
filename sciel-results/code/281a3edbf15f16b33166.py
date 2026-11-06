import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 90s limit, 5s margin
    
    n = 128
    target_start = 128
    target_end = 256
    
    def evaluate(A_set):
        """
        Returns (max_mult, num_uncovered)
        """
        if not A_set:
            return 1000, 129
        v = np.zeros(129, dtype=np.int32)
        for x in A_set:
            if 0 <= x <= 128:
                v[x] = 1
            else:
                return 1000, 129
        
        conv = np.convolve(v, v, mode='full')
        
        max_mult = 0
        num_uncovered = 0
        
        for t in range(target_start, target_end + 1):
            ordered = conv[t]
            diag = 1 if (t % 2 == 0 and (t // 2) in A_set) else 0
            unordered = (ordered + diag) // 2
            
            if unordered == 0:
                num_uncovered += 1
            elif unordered > max_mult:
                max_mult = unordered
                
        return max_mult, num_uncovered

    def fast_cost(A_set):
        """Fast cost: heavily penalize uncovered, then max multiplicity"""
        m, u = evaluate(A_set)
        return u * 10000 + m

    def greedy_constructive(target_size, seed_val=None):
        """
        Constructive approach: start with mandatory elements, then greedily
        add elements that cover the most uncovered targets or reduce peak multiplicity.
        """
        if seed_val is not None:
            random.seed(seed_val)
        
        A = set()
        # 128 is mandatory (128+128=256)
        A.add(128)
        
        # Track which targets are covered and their multiplicities
        # targets[t] = list of pairs (a,b) with a+b=t, a<=b, a,b in A
        # For efficiency, track count per target
        target_counts = np.zeros(target_end + 1, dtype=np.int32)
        # Also track which pairs contribute to each target for removal
        pair_to_targets = {}  # (a,b) -> list of targets
        
        def add_element(x):
            if x in A:
                return
            A.add(x)
            # For each existing element y in A, add pair (min(x,y), max(x,y))
            for y in list(A):
                if y < x:
                    continue
                a, b = min(x, y), max(x, y)
                s = a + b
                target_counts[s] += 1
                if (a, b) not in pair_to_targets:
                    pair_to_targets[(a, b)] = []
                pair_to_targets[(a, b)].append(s)
        
        def remove_element(x):
            if x not in A:
                return
            # Find all pairs involving x
            pairs_to_remove = []
            for y in list(A):
                a, b = min(x, y), max(x, y)
                pairs_to_remove.append((a, b))
            
            for (a, b) in pairs_to_remove:
                if (a, b) in pair_to_targets:
                    for s in pair_to_targets[(a, b)]:
                        target_counts[s] -= 1
                    del pair_to_targets[(a, b)]
            
            A.discard(x)
        
        # Add 128
        add_element(128)
        
        # Greedily add elements
        candidates = list(range(0, 128))
        random.shuffle(candidates)
        
        while len(A) < target_size and time.time() < deadline - 3:
            best_candidate = None
            best_score = -float('inf')
            
            for x in candidates:
                if x in A:
                    continue
                
                # Evaluate adding x
                old_counts = target_counts.copy()
                add_element(x)
                
                # Compute score: prefer covering uncovered targets, then reduce peak
                unc_before = np.sum(target_counts[target_start:target_end+1] == 0)
                unc_after = np.sum(target_counts[target_start:target_end+1] == 0)
                covered_new = unc_before - unc_after
                
                peak_before = np.max(target_counts[target_start:target_end+1])
                peak_after = np.max(target_counts[target_start:target_end+1])
                
                # Score: heavily weight new coverage, slightly penalize peak increase
                score = covered_new * 1000 - (peak_after - peak_before)
                
                # Remove x to restore state
                remove_element(x)
                
                if score > best_score:
                    best_score = score
                    best_candidate = x
            
            if best_candidate is None:
                break
            
            add_element(best_candidate)
        
        return A

    def local_search(initial_A, max_iters=500):
        """Local search with add/remove/swap moves"""
        current_A = set(initial_A)
        cur_cost = fast_cost(current_A)
        best_A = set(current_A)
        best_cost = cur_cost
        
        T = 5.0
        T_min = 0.1
        cooling = 0.995
        
        it = 0
        while it < max_iters and time.time() < deadline - 3:
            it += 1
            move_type = random.random()
            new_A = set(current_A)
            
            if move_type < 0.35 and len(current_A) > 3:
                # Remove
                x = random.choice(list(current_A))
                new_A.remove(x)
            elif move_type < 0.75:
                # Add
                x = random.randint(0, 128)
                if x in current_A:
                    continue
                new_A.add(x)
            else:
                # Swap
                if len(current_A) < 2:
                    continue
                x = random.choice(list(current_A))
                y = random.randint(0, 128)
                if y == x:
                    continue
                new_A.remove(x)
                new_A.add(y)
            
            if len(new_A) == 0:
                continue
                
            new_cost = fast_cost(new_A)
            delta = new_cost - cur_cost
            
            if delta < 0 or random.random() < np.exp(-delta / max(T, 0.01)):
                current_A = new_A
                cur_cost = new_cost
                
                if cur_cost < best_cost:
                    best_A = set(current_A)
                    best_cost = cur_cost
                    
            T = max(T * cooling, T_min)
            
        return best_A

    # Phase 1: Greedy constructive with different sizes and seeds
    best_overall_A = set()
    best_overall_cost = float('inf')
    
    sizes_to_try = [18, 20, 22, 24, 26]
    
    for size in sizes_to_try:
        if time.time() > deadline - 10:
            break
        
        for seed in range(3):
            if time.time() > deadline - 8:
                break
            
            greedy_A = greedy_constructive(size, seed_val=42 + seed * 1000 + size)
            
            if not greedy_A:
                continue
            
            # Polish with local search
            polished_A = local_search(greedy_A, max_iters=200)
            
            cost = fast_cost(polished_A)
            if cost < best_overall_cost:
                best_overall_cost = cost
                best_overall_A = set(polished_A)
                
    # Phase 2: Additional local search restarts from random starts
    for size in [20, 22, 24]:
        if time.time() > deadline - 5:
            break
        
        for restart in range(3):
            if time.time() > deadline - 4:
                break
            
            init_A = set()
            init_A.add(128)
            while len(init_A) < size:
                init_A.add(random.randint(0, 128))
            
            res_A = local_search(init_A, max_iters=300)
            cost = fast_cost(res_A)
            if cost < best_overall_cost:
                best_overall_cost = cost
                best_overall_A = set(res_A)
                
    # Fallback
    if not best_overall_A:
        best_overall_A = set(range(64, 129))
        
    # Final polish: try removing redundant elements
    m, u = evaluate(best_overall_A)
    if u == 0:
        for x in sorted(list(best_overall_A), key=lambda v: -v):  # Try removing larger elements first
            if time.time() > deadline - 2:
                break
            test_A = set(best_overall_A) - {x}
            if not test_A:
                continue
            m_t, u_t = evaluate(test_A)
            if u_t == 0 and m_t <= m:
                best_overall_A = test_A
                m, u = evaluate(best_overall_A)

    return sorted(best_overall_A)
