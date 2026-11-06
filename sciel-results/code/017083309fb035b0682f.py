import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 128
    positions = list(range(1, n+1))
    
    # Precompute all nonconstant 4-APs
    ap_positions = []
    for a in range(1, n+1):
        for d in range(1, n):
            if a + 3*d <= n:
                ap_positions.append([a, a+d, a+2*d, a+3*d])
            else:
                break
    
    num_aps = len(ap_positions)
    
    # For each position, list of AP indices it belongs to
    pos_to_aps = [[] for _ in range(n+1)]
    for idx in range(num_aps):
        for pos in ap_positions[idx]:
            pos_to_aps[pos].append(idx)
    
    def solve_with_k(k, seed, max_iters=50000):
        rng = random.Random(seed)
        
        colors = [0] * (n+1)
        for p in positions:
            colors[p] = rng.randint(0, k-1)
        
        violated = [False] * num_aps
        num_violations = 0
        for idx in range(num_aps):
            ap = ap_positions[idx]
            cset = set(colors[p] for p in ap)
            if len(cset) < 3:
                violated[idx] = True
                num_violations += 1
        
        if num_violations == 0:
            return colors[1:]
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            if num_violations == 0:
                return colors[1:]
            
            v_indices = [i for i in range(num_aps) if violated[i]]
            if not v_indices:
                return colors[1:]
            ap_idx = v_indices[rng.randint(0, len(v_indices)-1)]
            ap = ap_positions[ap_idx]
            
            pos_order = list(ap)
            rng.shuffle(pos_order)
            
            improved = False
            for pos in pos_order:
                old_color = colors[pos]
                for c in range(k):
                    if c == old_color:
                        continue
                    colors[pos] = c
                    delta = 0
                    for ap_i in pos_to_aps[pos]:
                        was_v = violated[ap_i]
                        cset = set(colors[q] for q in ap_positions[ap_i])
                        is_v = len(cset) < 3
                        violated[ap_i] = is_v
                        if is_v and not was_v:
                            delta += 1
                        elif not is_v and was_v:
                            delta -= 1
                    
                    if num_violations + delta < num_violations:
                        num_violations += delta
                        improved = True
                        break
                    else:
                        colors[pos] = old_color
                        for ap_i in pos_to_aps[pos]:
                            cset = set(colors[q] for q in ap_positions[ap_i])
                            violated[ap_i] = len(cset) < 3
                        num_violations = sum(violated)
                        break
                if improved:
                    break
            
            if not improved:
                pos = rng.randint(1, n)
                new_c = rng.randint(0, k-1)
                colors[pos] = new_c
                for ap_i in pos_to_aps[pos]:
                    cset = set(colors[q] for q in ap_positions[ap_i])
                    violated[ap_i] = len(cset) < 3
                num_violations = sum(violated)
        
        return None
    
    # Try k=3 with many seeds
    for seed in range(500):
        if time.time() > deadline:
            break
        result = solve_with_k(3, seed, max_iters=20000)
        if result is not None:
            return result
    
    # Try k=4
    for seed in range(500):
        if time.time() > deadline:
            break
        result = solve_with_k(4, seed, max_iters=20000)
        if result is not None:
            return result
    
    # Try k=5
    for seed in range(500):
        if time.time() > deadline:
            break
        result = solve_with_k(5, seed, max_iters=20000)
        if result is not None:
            return result
    
    # Fallback: all distinct
    return list(range(n))
