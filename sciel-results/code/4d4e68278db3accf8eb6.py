import time
import math
import random
import numpy as np

BASE = 51975

def _divisors(n):
    ds = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            ds.append(i)
            if i != n // i:
                ds.append(n // i)
    return sorted(ds)

MODULI = [m for m in _divisors(BASE) if m > 1]

def score_classes(classes):
    covered = bytearray(BASE)
    for m, r in classes:
        covered[r::m] = b'\x01' * len(range(r, BASE, m))
    return BASE - sum(covered)

def construct():
    deadline = time.time() + 140.0
    rng = random.Random(98765)

    def greedy_init(order):
        cnt = np.zeros(BASE, dtype=np.int32)
        covered = 0
        used = {}
        for m in order:
            best_r, best_gain = 0, -1
            for r in range(m):
                idx = np.arange(r, BASE, m)
                g = int(np.count_nonzero(cnt[idx] == 0))
                if g > best_gain:
                    best_gain, best_r = g, r
            idx = np.arange(best_r, BASE, m)
            gain = int(np.count_nonzero(cnt[idx] == 0))
            covered += gain
            cnt[idx] += 1
            used[m] = best_r
        return used, covered, cnt

    orders = [
        sorted(MODULI, reverse=True),
        sorted(MODULI),
    ]
    for _ in range(4):
        o = MODULI[:]; rng.shuffle(o); orders.append(o)

    best_score = BASE
    best_classes = None
    best_state = None # (used_dict, covered_count, cnt_array)

    for o in orders:
        used, covered, cnt = greedy_init(o)
        sc = BASE - covered
        if sc < best_score:
            best_score = sc
            best_classes = [[m, used[m]] for m in used]
            best_state = (dict(used), covered, cnt.copy())

    if best_state is None:
        # Fallback
        used = {m: 0 for m in MODULI}
        cnt = np.zeros(BASE, dtype=np.int32)
        for m, r in used.items():
            cnt[r::m] += 1
        covered = int(np.count_nonzero(cnt))
        best_state = (used, covered, cnt)
        best_score = BASE - covered
        best_classes = [[m, 0] for m in MODULI]

    def simulate_annealing(initial_used, initial_covered, initial_cnt, max_time):
        used = dict(initial_used)
        cnt = initial_cnt.copy()
        covered = initial_covered
        best_used = dict(used)
        best_covered = covered
        best_score_sa = BASE - covered
        
        T = 100.0
        T_min = 0.01
        alpha = 0.9995 # Cooling rate
        
        unused_moduli = [m for m in MODULI if m not in used]
        
        start_time = time.time()
        
        while time.time() < start_time + max_time:
            if T < T_min:
                break
                
            # Generate a move
            move_type = rng.random()
            
            if move_type < 0.6 and used:
                # Change residue of existing modulus
                m = rng.choice(list(used.keys()))
                old_r = used[m]
                new_r = rng.randint(0, m - 1)
                if new_r == old_r:
                    continue
                
                # Calculate Delta
                # Remove old
                idx_old = np.arange(old_r, BASE, m)
                loss = int(np.count_nonzero(cnt[idx_old] == 1))
                # Add new
                idx_new = np.arange(new_r, BASE, m)
                gain = int(np.count_nonzero(cnt[idx_new] == 0))
                
                delta = gain - loss
                new_covered = covered + delta
                
                if delta >= 0 or rng.random() < math.exp(delta / T):
                    # Accept
                    cnt[idx_old] -= 1
                    cnt[idx_new] += 1
                    used[m] = new_r
                    covered = new_covered
                    
                    if covered > best_covered:
                        best_covered = covered
                        best_used = dict(used)
                        best_score_sa = BASE - covered
                        
            elif move_type < 0.8 and unused_moduli:
                # Add new modulus
                m = rng.choice(unused_moduli)
                new_r = rng.randint(0, m - 1)
                
                idx_new = np.arange(new_r, BASE, m)
                gain = int(np.count_nonzero(cnt[idx_new] == 0))
                delta = gain
                
                if delta >= 0 or rng.random() < math.exp(delta / T):
                    cnt[idx_new] += 1
                    used[m] = new_r
                    covered += delta
                    unused_moduli.remove(m)
                    
                    if covered > best_covered:
                        best_covered = covered
                        best_used = dict(used)
                        best_score_sa = BASE - covered
                        
            elif move_type < 0.9 and used:
                # Remove existing modulus
                m = rng.choice(list(used.keys()))
                old_r = used[m]
                
                idx_old = np.arange(old_r, BASE, m)
                loss = int(np.count_nonzero(cnt[idx_old] == 1))
                delta = -loss
                
                if delta >= 0 or rng.random() < math.exp(delta / T):
                    cnt[idx_old] -= 1
                    del used[m]
                    covered -= loss
                    unused_moduli.append(m)
                    
                    if covered > best_covered:
                        best_covered = covered
                        best_used = dict(used)
                        best_score_sa = BASE - covered
                        
            T *= alpha
            
        return best_used, best_score_sa

    # Run SA with remaining time
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        used_sa, score_sa = simulate_annealing(
            best_state[0], 
            best_state[1], 
            best_state[2], 
            remaining_time - 5
        )
        if score_sa < best_score:
            best_score = score_sa
            best_classes = [[m, used_sa[m]] for m in used_sa]
            
    return best_classes
