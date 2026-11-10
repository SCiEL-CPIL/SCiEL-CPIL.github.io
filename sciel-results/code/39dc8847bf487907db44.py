"""Max-SAT-style search for 4-AP-free subset of [1,256] maximizing 3-AP count."""
import time
import random
import numpy as np

N = 256
# Precompute all 4-APs (a, a+d, a+2d, a+3d) with positive d
FOUR_APS = []
for d in range(1, (N - 1) // 3 + 1):
    for a in range(1, N - 3 * d + 1):
        FOUR_APS.append((a, a + d, a + 2 * d, a + 3 * d))

# Precompute all 3-APs (a, a+d, a+2d) with positive d
THREE_APS = []
for d in range(1, (N - 1) // 2 + 1):
    for a in range(1, N - 2 * d + 1):
        THREE_APS.append((a, a + d, a + 2 * d))

# Map each element to the 4-APs it participates in
ELEM_TO_4APS = {i: [] for i in range(1, N + 1)}
for idx, ap in enumerate(FOUR_APS):
    for x in ap:
        ELEM_TO_4APS[x].append(idx)

# Map each element to the 3-APs it participates in
ELEM_TO_3APS = {i: [] for i in range(1, N + 1)}
for idx, ap in enumerate(THREE_APS):
    for x in ap:
        ELEM_TO_3APS[x].append(idx)

def count_3aps(state):
    """Count 3-APs where all three elements are present. state is a boolean array of size N+1."""
    cnt = 0
    for a, b, c in THREE_APS:
        if state[a] and state[b] and state[c]:
            cnt += 1
    return cnt

def count_4aps(state):
    """Count 4-APs where all four elements are present."""
    cnt = 0
    for a, b, c, d in FOUR_APS:
        if state[a] and state[b] and state[c] and state[d]:
            cnt += 1
    return cnt

def objective(state):
    """Reward 3-APs, heavily penalize 4-APs."""
    three = count_3aps(state)
    four = count_4aps(state)
    return three - 10000 * four

def greedy_seed():
    """Greedy construction: add elements that don't create a 4-AP, prefer those that maximize 3-APs."""
    state = np.zeros(N + 1, dtype=bool)
    # Try all single elements, pick the one that gives most 3-APs while keeping 4-AP-free
    # Actually, do a greedy: iterate, at each step add the element that maximizes marginal 3-AP gain
    # subject to 4-AP constraint
    remaining = set(range(1, N + 1))
    # We'll do a simple greedy: repeatedly add the element that gives max new 3-APs without creating a 4-AP
    for _ in range(N):
        if not remaining:
            break
        best_elem = None
        best_gain = -1
        for x in remaining:
            # Check if adding x creates a 4-AP
            ok = True
            for ap_idx in ELEM_TO_4APS[x]:
                a, b, c, d = FOUR_APS[ap_idx]
                others = [v for v in (a, b, c, d) if v != x]
                if all(state[v] for v in others):
                    ok = False
                    break
            if not ok:
                continue
            # Count new 3-APs
            gain = 0
            for ap_idx in ELEM_TO_3APS[x]:
                a, b, c = THREE_APS[ap_idx]
                others = [v for v in (a, b, c) if v != x]
                if all(state[v] for v in others):
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_elem = x
        if best_elem is None:
            break
        state[best_elem] = True
        remaining.discard(best_elem)
    return state

def parallel_tempering(deadline, seed=42):
    """Parallel tempering with simulated annealing over multiple replicas."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    
    n_replicas = 8
    # Geometrically spaced temperatures
    T_cold = 0.5
    T_hot = 50.0
    temps = [T_cold * (T_hot / T_cold) ** (i / (n_replicas - 1)) for i in range(n_replicas)]
    
    # Initialize replicas: cold replica gets greedy seed, others get random perturbations
    base_state = greedy_seed()
    replicas = []
    for i in range(n_replicas):
        state = base_state.copy()
        if i > 0:
            # Perturb: flip some random elements
            n_flips = int(N * 0.1 * i)
            for _ in range(n_flips):
                x = rng.randint(1, N)
                state[x] = not state[x]
        replicas.append(state)
    
    best_state = base_state.copy()
    best_obj = objective(best_state)
    
    # Track best valid (4-AP-free) solution
    best_valid_state = None
    best_valid_3aps = -1
    
    # Check initial
    for i, state in enumerate(replicas):
        obj = objective(state)
        if count_4aps(state) == 0:
            three = count_3aps(state)
            if three > best_valid_3aps:
                best_valid_3aps = three
                best_valid_state = state.copy()
    
    step = 0
    while time.time() < deadline - 2.0:
        # Annealing step for each replica
        for i in range(n_replicas):
            T = temps[i]
            state = replicas[i]
            current_obj = objective(state)
            
            # Pick a random element to flip
            x = rng.randint(1, N)
            state[x] = not state[x]
            new_obj = objective(state)
            delta = new_obj - current_obj
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if delta >= 0:
                accept = True
            else:
                accept = np_rng.random() < np.exp(delta / T)
            
            if not accept:
                state[x] = not state[x]
                continue
            
            # Track best
            if new_obj > best_obj:
                best_obj = new_obj
                best_state = state.copy()
            
            if count_4aps(state) == 0:
                three = count_3aps(state)
                if three > best_valid_3aps:
                    best_valid_3aps = three
                    best_valid_state = state.copy()
        
        # Periodic swaps between adjacent replicas
        if step % 10 == 0:
            for i in range(n_replicas - 1):
                T1, T2 = temps[i], temps[i + 1]
                obj1 = objective(replicas[i])
                obj2 = objective(replicas[i + 1])
                # Swap with probability based on energy difference
                delta = obj2 - obj1
                if delta >= 0 or np_rng.random() < np.exp(-delta / T1):
                    replicas[i], replicas[i + 1] = replicas[i + 1], replicas[i]
        
        # Cool down slowly
        if step % 200 == 0 and step > 0:
            for i in range(1, n_replicas):
                temps[i] *= 0.95
            # Re-establish geometric spacing
            T_cold = temps[0]
            T_hot = temps[-1]
            temps = [T_cold * (T_hot / T_cold) ** (i / (n_replicas - 1)) for i in range(n_replicas)]
        
        step += 1
    
    # Final polish: local search on best valid state
    if best_valid_state is not None:
        # Try adding elements that don't create 4-APs
        for _ in range(50):
            improved = False
            for x in range(1, N + 1):
                if best_valid_state[x]:
                    continue
                # Check if adding x creates a 4-AP
                ok = True
                for ap_idx in ELEM_TO_4APS[x]:
                    a, b, c, d = FOUR_APS[ap_idx]
                    others = [v for v in (a, b, c, d) if v != x]
                    if all(best_valid_state[v] for v in others):
                        ok = False
                        break
                if ok:
                    # Check 3-AP gain
                    gain = 0
                    for ap_idx in ELEM_TO_3APS[x]:
                        a, b, c = THREE_APS[ap_idx]
                        others = [v for v in (a, b, c) if v != x]
                        if all(best_valid_state[v] for v in others):
                            gain += 1
                    if gain > 0:
                        best_valid_state[x] = True
                        improved = True
            if not improved:
                break
        
        # Try removing elements that are in few 3-APs and freeing up space
        for _ in range(20):
            improved = False
            for x in range(1, N + 1):
                if not best_valid_state[x]:
                    continue
                # Count 3-APs containing x
                ap_count = 0
                for ap_idx in ELEM_TO_3APS[x]:
                    a, b, c = THREE_APS[ap_idx]
                    others = [v for v in (a, b, c) if v != x]
                    if all(best_valid_state[v] for v in others):
                        ap_count += 1
                if ap_count <= 1:
                    # Try removing x and see if we can add more
                    test_state = best_valid_state.copy()
                    test_state[x] = False
                    # Try to add other elements
                    added = 0
                    for y in range(1, N + 1):
                        if test_state[y]:
                            continue
                        ok = True
                        for ap_idx in ELEM_TO_4APS[y]:
                            a, b, c, d = FOUR_APS[ap_idx]
                            others = [v for v in (a, b, c, d) if v != y]
                            if all(test_state[v] for v in others):
                                ok = False
                                break
                        if ok:
                            gain = 0
                            for ap_idx in ELEM_TO_3APS[y]:
                                a, b, c = THREE_APS[ap_idx]
                                others = [v for v in (a, b, c) if v != y]
                                if all(test_state[v] for v in others):
                                    gain += 1
                            if gain > 0:
                                test_state[y] = True
                                added += 1
                    if added > 1:
                        new_three = count_3aps(test_state)
                        if new_three > best_valid_3aps:
                            best_valid_state = test_state
                            best_valid_3aps = new_three
                            improved = True
            if not improved:
                break
    
    # Return best valid solution
    if best_valid_state is not None:
        return sorted([i for i in range(1, N + 1) if best_valid_state[i]])
    else:
        # Fallback: greedy seed
        return sorted([i for i in range(1, N + 1) if base_state[i]])

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin before 90s kill
    
    # Use parallel tempering
    result = parallel_tempering(deadline, seed=42)
    
    # Ensure nonempty
    if not result:
        result = [1]
    
    return result
