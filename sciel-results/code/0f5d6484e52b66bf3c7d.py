import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    N = 128
    T_START = 128
    T_END = 256
    
    def get_initial_mults(A_bool):
        conv = np.convolve(A_bool, A_bool, mode='full')
        mults = np.zeros(257, dtype=int)
        for t in range(257):
            D_t = 0
            if t % 2 == 0:
                half = t // 2
                if 0 <= half <= N and A_bool[half]:
                    D_t = 1
            mults[t] = (conv[t] + D_t) // 2
        return mults

    def update_mults(mults, A_bool, x, is_add):
        new_mults = mults.copy()
        indices = np.where(A_bool > 0)[0]
        if is_add:
            for y in indices:
                new_mults[x + y] += 1
            new_mults[2 * x] += 1
        else:
            for y in indices:
                new_mults[x + y] -= 1
        return new_mults

    def calc_energy(mults):
        target_mults = mults[T_START:T_END+1]
        if np.min(target_mults) == 0:
            missing_count = np.sum(target_mults == 0)
            penalty = missing_count * 10000
            max_mult = np.max(target_mults) if np.max(target_mults) > 0 else 0
            return penalty + max_mult
        return np.max(target_mults)

    def simulated_annealing(init_A, max_iters=2000):
        A_bool = np.zeros(N+1, dtype=int)
        for a in init_A:
            A_bool[a] = 1
            
        mults = get_initial_mults(A_bool)
        current_energy = calc_energy(mults)
        
        best_energy = current_energy
        best_A = A_bool.copy()
        
        T_start = 10.0
        T_end = 0.1
        
        for i in range(max_iters):
            if time.time() > deadline - 2:
                break
                
            T = T_start * (T_end / T_start) ** (i / max_iters)
            
            x = random.randint(0, N)
            is_add = (A_bool[x] == 0)
            
            new_mults = update_mults(mults, A_bool, x, is_add)
            new_energy = calc_energy(new_mults)
            
            delta = new_energy - current_energy
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                A_bool[x] = 1 if is_add else 0
                mults = new_mults
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_A = A_bool.copy()
            
        best_set = set(np.where(best_A > 0)[0])
        best_mults = get_initial_mults(best_A)
        if np.min(best_mults[T_START:T_END+1]) > 0:
             return best_set, np.max(best_mults[T_START:T_END+1])
        else:
             return None, float('inf')

    best_A = None
    best_mult = float('inf')
    
    seeds = []
    
    # 1. Random sets
    for _ in range(5):
        k = random.randint(15, 25)
        seeds.append(set(random.sample(range(0, N+1), k)))
        
    # 2. Structured: Arithmetic Progressions
    for step in [1, 2, 3]:
        seeds.append(set(range(0, N+1, step)))
        
    # 3. Hybrid
    seeds.append(set(range(0, 10)) | set(range(118, 129)))
    seeds.append(set(range(0, 15)) | set(range(113, 129)))
    
    # 4. Greedy
    A_greedy = set()
    mults_greedy = np.zeros(257, dtype=int)
    for _ in range(20):
        best_x = -1
        best_max_mult = float('inf')
        for x in range(0, N+1):
            if x in A_greedy: continue
            new_mults = mults_greedy.copy()
            for y in A_greedy:
                new_mults[x+y] += 1
            new_mults[2*x] += 1
            
            if np.min(new_mults[T_START:T_END+1]) == 0:
                continue
                
            m = np.max(new_mults[T_START:T_END+1])
            if m < best_max_mult:
                best_max_mult = m
                best_x = x
                
        if best_x != -1:
            A_greedy.add(best_x)
            for y in A_greedy:
                mults_greedy[best_x+y] += 1
            mults_greedy[2*best_x] += 1
        else:
            break
    seeds.append(A_greedy)
    
    for seed in seeds:
        if time.time() > deadline - 5:
            break
            
        A_set, mult = simulated_annealing(seed, max_iters=2000)
        
        if A_set is not None and mult < best_mult:
            best_mult = mult
            best_A = A_set
            
    if best_A is None:
        best_A = set(range(0, N+1))
        
    return list(best_A)
