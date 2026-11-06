import time
import random

# Global precomputation
def _precompute():
    def get_idx(i, j):
        if i > j: i, j = j, i
        return 15*i - i*(i-1)//2 + j - i - 1

    k4_edges = []
    edge_to_k4s = [[] for _ in range(120)]
    
    for a in range(16):
        for b in range(a+1, 16):
            for c in range(b+1, 16):
                for d in range(c+1, 16):
                    idxs = [
                        get_idx(a,b), get_idx(a,c), get_idx(a,d),
                        get_idx(b,c), get_idx(b,d), get_idx(c,d)
                    ]
                    k_idx = len(k4_edges)
                    k4_edges.append(idxs)
                    for idx in idxs:
                        edge_to_k4s[idx].append(k_idx)
                        
    return k4_edges, edge_to_k4s

K4_EDGES, EDGE_TO_K4S = _precompute()

def _count_colours(colours, idxs):
    c0 = colours[idxs[0]]
    c1 = colours[idxs[1]]
    c2 = colours[idxs[2]]
    c3 = colours[idxs[3]]
    c4 = colours[idxs[4]]
    c5 = colours[idxs[5]]
    return len({c0, c1, c2, c3, c4, c5})

def _calc_energy(colours):
    energy = 0
    for idxs in K4_EDGES:
        cnt = _count_colours(colours, idxs)
        if cnt < 5:
            energy += 5 - cnt
    return energy

def _is_valid(colours):
    for idxs in K4_EDGES:
        if _count_colours(colours, idxs) < 5:
            return False
    return True

def _sa_search(init_colours, num_colours, time_limit):
    current = init_colours[:]
    current_energy = _calc_energy(current)
    best_energy = current_energy
    best_colouring = current[:]
    
    T = 5.0
    cool = 0.9995
    start = time.time()
    
    while time.time() - start < time_limit:
        e = random.randint(0, 119)
        old_c = current[e]
        new_c = random.randint(0, num_colours - 1)
        if old_c == new_c:
            continue
            
        current[e] = new_c
        
        delta = 0
        for k_idx in EDGE_TO_K4S[e]:
            idxs = K4_EDGES[k_idx]
            old_cnt = _count_colours(current, idxs) # Wait, current is already modified
            # We need old energy contribution
            # This approach is flawed because we modified current before calculating delta.
            
        # Correct approach:
        # 1. Calculate old contribution of affected K4s
        # 2. Modify
        # 3. Calculate new contribution
        # 4. Delta = New - Old
        
        # Let's restart the loop logic properly.
        
        # Undo change for calculation
        current[e] = old_c
        
        old_contrib = 0
        for k_idx in EDGE_TO_K4S[e]:
            idxs = K4_EDGES[k_idx]
            cnt = _count_colours(current, idxs)
            if cnt < 5:
                old_contrib += 5 - cnt
                
        current[e] = new_c
        
        new_contrib = 0
        for k_idx in EDGE_TO_K4S[e]:
            idxs = K4_EDGES[k_idx]
            cnt = _count_colours(current, idxs)
            if cnt < 5:
                new_contrib += 5 - cnt
                
        delta = new_contrib - old_contrib
        
        if delta < 0 or random.random() < pow(2.718281828, -delta / T):
            current_energy += delta
            if current_energy < best_energy:
                best_energy = current_energy
                best_colouring = current[:]
        else:
            current[e] = old_c
            
        T *= cool
        if T < 0.1:
            T = 0.1
            
    return best_colouring, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Helper to create 15-colour Hamming
    def get_hamming_15():
        cols = [0] * 120
        for i in range(16):
            for j in range(i+1, 16):
                # Index calculation
                idx = 15*i - i*(i-1)//2 + j - i - 1
                cols[idx] = i ^ j
        return cols

    # 1. Try 14 colours
    if time.time() < deadline - 20:
        init_15 = get_hamming_15()
        # Merge colour 1 and 2
        init_14 = []
        for c in init_15:
            if c == 2:
                init_14.append(1)
            else:
                init_14.append(c)
                
        # Pass time limit explicitly
        time_left = deadline - 10 - time.time()
        if time_left > 0:
            best_14, energy_14 = _sa_search(init_14, 14, time_left)
            if energy_14 == 0:
                return best_14

    # 2. Try 15 colours
    if time.time() < deadline - 5:
        init_15 = get_hamming_15()
        time_left = deadline - 5 - time.time()
        if time_left > 0:
            best_15, energy_15 = _sa_search(init_15, 15, time_left)
            if energy_15 == 0:
                return best_15

    # 3. Fallback: Seed
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
