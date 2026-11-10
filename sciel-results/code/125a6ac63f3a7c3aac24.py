import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    N = 128
    TARGETS = list(range(128, 257))  # 129 targets
    n_targets = len(TARGETS)
    n_elements = N + 1  # 0..128
    
    # Precompute for each target t, the list of unordered pairs (a,b) with a<=b, a+b=t, 0<=a,b<=N
    pair_list = [[] for _ in range(n_targets)]
    for i, t in enumerate(TARGETS):
        a_min = max(0, t - N)
        a_max = min(N, t // 2)
        for a in range(a_min, a_max + 1):
            b = t - a
            pair_list[i].append((a, b))
    
    # For each element x, precompute which (target_idx, partner) pairs it forms
    elem_pairs = [[] for _ in range(n_elements)]
    for x in range(n_elements):
        for i, t in enumerate(TARGETS):
            y = t - x
            if 0 <= y <= N and x <= y:
                elem_pairs[x].append((i, y))
    
    def compute_mult(A_set):
        """Return max multiplicity over targets. Returns inf if not covered."""
        max_mult = 0
        for i, t in enumerate(TARGETS):
            count = 0
            for (a, b) in pair_list[i]:
                if a in A_set and b in A_set:
                    count += 1
            if count == 0:
                return float('inf')
            if count > max_mult:
                max_mult = count
        return max_mult
    
    def get_mults(A_set):
        """Return list of multiplicities for all targets. -1 if not covered."""
        mults = []
        for i, t in enumerate(TARGETS):
            count = 0
            for (a, b) in pair_list[i]:
                if a in A_set and b in A_set:
                    count += 1
            if count == 0:
                return None
            mults.append(count)
        return mults
    
    def is_complete(A_set):
        """Check if all targets are covered."""
        for i, t in enumerate(TARGETS):
            found = False
            for (a, b) in pair_list[i]:
                if a in A_set and b in A_set:
                    found = True
                    break
            if not found:
                return False
        return True
    
    def get_missing(A_set):
        missing = []
        for i, t in enumerate(TARGETS):
            found = False
            for (a, b) in pair_list[i]:
                if a in A_set and b in A_set:
                    found = True
                    break
            if not found:
                missing.append(i)
        return missing
    
    def greedy_complete(A_set, deadline):
        """Greedily add elements to ensure completeness."""
        for _ in range(200):
            if time.time() > deadline - 3:
                break
            if is_complete(A_set):
                break
            missing = get_missing(A_set)
            if not missing:
                break
            
            best_gain = -1
            best_elem = None
            for x in range(n_elements):
                if x in A_set:
                    continue
                gain = 0
                for mi in missing:
                    t = TARGETS[mi]
                    y = t - x
                    if 0 <= y <= N and y in A_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_elem = x
            
            if best_gain > 0 and best_elem is not None:
                A_set.add(best_elem)
            else:
                mi = missing[0]
                t = TARGETS[mi]
                for (a, b) in pair_list[mi]:
                    if a not in A_set or b not in A_set:
                        A_set.add(a)
                        A_set.add(b)
                        break
                else:
                    A_set.add(t // 2)
                    A_set.add(t - t // 2)
        return A_set
    
    def local_search(A_set, deadline, max_iters=5000):
        """Local search to minimize max multiplicity."""
        current_mult = compute_mult(A_set)
        if current_mult == float('inf'):
            return A_set, current_mult
        
        no_improve = 0
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            improved = False
            best_new_set = None
            best_new_mult = current_mult
            
            # Try removing elements (most impactful for reducing peak)
            if len(A_set) > 1:
                for a in list(A_set):
                    new_set = A_set - {a}
                    if not new_set:
                        continue
                    nm = compute_mult(new_set)
                    if nm < best_new_mult:
                        best_new_mult = nm
                        best_new_set = new_set
            
            # Try swapping elements
            if len(A_set) > 0:
                candidates = list(A_set)
                random.shuffle(candidates)
                for a_in in candidates[:min(5, len(candidates))]:
                    for _ in range(3):
                        a_out = random.randint(0, N)
                        if a_out in A_set:
                            continue
                        new_set = A_set - {a_in} | {a_out}
                        nm = compute_mult(new_set)
                        if nm < best_new_mult:
                            best_new_mult = nm
                            best_new_set = new_set
            
            # Try adding an element
            for x in range(n_elements):
                if x in A_set:
                    continue
                new_set = A_set | {x}
                nm = compute_mult(new_set)
                if nm < best_new_mult:
                    best_new_mult = nm
                    best_new_set = new_set
            
            if best_new_set is not None and best_new_mult < current_mult:
                A_set = best_new_set
                current_mult = best_new_mult
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 100:
                    break
        
        return A_set, current_mult
    
    best_A = None
    best_mult = float('inf')
    
    # Generate diverse structured seeds
    seeds = []
    
    # Seed 1: Even numbers (structured base)
    evens = set(range(0, N + 1, 2))
    seeds.append(set(evens))
    
    # Seed 2: Even numbers + a few strategic odds
    evens2 = set(range(0, N + 1, 2))
    for _ in range(8):
        odds_candidate = random.choice([x for x in range(1, N + 1, 2)])
        evens2.add(odds_candidate)
    seeds.append(evens2)
    
    # Seed 3: Arithmetic progression step 3
    ap3 = set(range(0, N + 1, 3))
    for _ in range(10):
        x = random.randint(0, N)
        ap3.add(x)
    seeds.append(ap3)
    
    # Seed 4: Two-cluster structure
    cluster1 = set(range(0, 32))
    cluster2 = set(range(96, N + 1))
    seeds.append(cluster1 | cluster2)
    
    # Seed 5: Random medium set
    k = random.randint(20, 30)
    rand_set = set(random.sample(range(0, N + 1), k))
    seeds.append(rand_set)
    
    # Seed 6: Symmetric around 64
    sym = set()
    for _ in range(15):
        x = random.randint(0, 64)
        sym.add(x)
        sym.add(128 - x)
    seeds.append(sym)
    
    # Seed 7: Dense middle
    mid = set(range(32, 97))
    seeds.append(mid)
    
    # Seed 8: Sparse structured
    sparse = set()
    for i in range(0, N + 1, 4):
        sparse.add(i)
    for i in range(1, N + 1, 8):
        if random.random() < 0.5:
            sparse.add(i)
    seeds.append(sparse)
    
    # Process each seed
    for seed_set in seeds:
        if time.time() > deadline - 5:
            break
        
        A_set = set(seed_set)
        A_set = greedy_complete(A_set, deadline)
        
        if not is_complete(A_set):
            continue
        
        A_set, mult = local_search(A_set, deadline, max_iters=3000)
        
        if mult < best_mult:
            best_mult = mult
            best_A = list(A_set)
    
    # If we still have time, do a few more rounds of perturbation + local search
    if best_A is not None and time.time() < deadline - 10:
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            
            # Perturb the best solution
            A_set = set(best_A)
            n_perturb = random.randint(3, 8)
            for _ in range(n_perturb):
                a_in = random.choice(list(A_set))
                A_set.remove(a_in)
                a_out = random.randint(0, N)
                A_set.add(a_out)
            
            A_set = greedy_complete(A_set, deadline)
            if not is_complete(A_set):
                continue
            
            A_set, mult = local_search(A_set, deadline, max_iters=2000)
            
            if mult < best_mult:
                best_mult = mult
                best_A = list(A_set)
    
    if best_A is None:
        # Fallback: return a valid set
        return list(range(0, N + 1, 2))
    
    return best_A
