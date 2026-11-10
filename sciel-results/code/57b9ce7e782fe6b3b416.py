import time
import random
import math

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Precompute all 4-term arithmetic progressions
    aps = []
    index_to_aps = [[] for _ in range(N)]
    
    for d in range(1, (N - 3) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            idx1 = a - 1
            idx2 = a + d - 1
            idx3 = a + 2 * d - 1
            idx4 = a + 3 * d - 1
            
            ap = (idx1, idx2, idx3, idx4)
            ap_id = len(aps)
            aps.append(ap)
            
            index_to_aps[idx1].append((ap_id, 0))
            index_to_aps[idx2].append((ap_id, 1))
            index_to_aps[idx3].append((ap_id, 2))
            index_to_aps[idx4].append((ap_id, 3))
    
    num_aps = len(aps)
    
    # Helper to count monochromatic 4-APs
    def count_mono(colors):
        count = 0
        for ap in aps:
            if colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
                count += 1
        return count
    
    # Fast local search: hill climbing with 1-bit flips
    def local_search(colors, max_iters=5000):
        current_score = count_mono(colors)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for k in range(N):
                c_k = colors[k]
                new_c_k = 1 - c_k
                
                delta = 0
                for ap_id, pos in index_to_aps[k]:
                    ap = aps[ap_id]
                    if pos == 0:
                        o1, o2, o3 = ap[1], ap[2], ap[3]
                    elif pos == 1:
                        o1, o2, o3 = ap[0], ap[2], ap[3]
                    elif pos == 2:
                        o1, o2, o3 = ap[0], ap[1], ap[3]
                    else:
                        o1, o2, o3 = ap[0], ap[1], ap[2]
                        
                    c_o1 = colors[o1]
                    c_o2 = colors[o2]
                    c_o3 = colors[o3]
                    
                    if c_o1 == c_o2 == c_o3:
                        if c_o1 == c_k:
                            delta -= 1
                        else:
                            delta += 1
                            
                if delta < 0:
                    colors[k] = new_c_k
                    current_score += delta
                    improved = True
        return colors, current_score
    
    # Simulated annealing for one run
    def simulated_annealing(init_colors, T_initial, T_final, cooling_rate, max_no_improve):
        colors = init_colors[:]
        current_score = count_mono(colors)
        best_score = current_score
        best_colors = colors[:]
        
        T = T_initial
        no_improve_steps = 0
        steps = 0
        
        while time.time() < deadline - 2.0:
            if T < T_final or no_improve_steps > max_no_improve:
                break
            
            steps += 1
            
            # Choose move: 1-bit flip
            k = random.randint(0, N - 1)
            c_k = colors[k]
            new_c_k = 1 - c_k
            
            delta = 0
            for ap_id, pos in index_to_aps[k]:
                ap = aps[ap_id]
                if pos == 0:
                    o1, o2, o3 = ap[1], ap[2], ap[3]
                elif pos == 1:
                    o1, o2, o3 = ap[0], ap[2], ap[3]
                elif pos == 2:
                    o1, o2, o3 = ap[0], ap[1], ap[3]
                else:
                    o1, o2, o3 = ap[0], ap[1], ap[2]
                    
                c_o1 = colors[o1]
                c_o2 = colors[o2]
                c_o3 = colors[o3]
                
                if c_o1 == c_o2 == c_o3:
                    if c_o1 == c_k:
                        delta -= 1
                    else:
                        delta += 1
                        
            accept = False
            if delta < 0:
                accept = True
            elif T > 0:
                # Avoid overflow in exp
                if delta / T < 700:
                    if random.random() < math.exp(-delta / T):
                        accept = True
            
            if accept:
                colors[k] = new_c_k
                current_score += delta
                
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors[:]
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                no_improve_steps += 1
                
            T *= cooling_rate
            
        return best_colors, best_score
    
    best_score = float('inf')
    best_colors = None
    
    # Phase 1: Try several diverse initializations with local search
    num_init = 10
    for i in range(num_init):
        if time.time() > deadline - 30:
            break
        
        if i == 0:
            # Random initialization
            colors = [random.randint(0, 1) for _ in range(N)]
        elif i == 1:
            # Alternating pattern
            colors = [i % 2 for i in range(N)]
        elif i == 2:
            # Block pattern
            colors = [(i // 8) % 2 for i in range(N)]
        elif i == 3:
            # Another block pattern
            colors = [(i // 16) % 2 for i in range(N)]
        elif i == 4:
            # Thue-Morse-like
            colors = [bin(i).count('1') % 2 for i in range(N)]
        else:
            # Random
            colors = [random.randint(0, 1) for _ in range(N)]
        
        # Local search polish
        colors, score = local_search(colors, max_iters=3000)
        
        if score < best_score:
            best_score = score
            best_colors = colors[:]
    
    # Phase 2: SA runs from best solution and perturbed versions
    num_sa_runs = 5
    for i in range(num_sa_runs):
        if time.time() > deadline - 5:
            break
        
        if i == 0:
            init_colors = best_colors[:]
        else:
            # Perturb best solution
            init_colors = best_colors[:]
            num_flips = random.randint(10, 30)
            indices_to_flip = random.sample(range(N), num_flips)
            for idx in indices_to_flip:
                init_colors[idx] = 1 - init_colors[idx]
        
        # SA parameters - slower cooling for more exploration
        T_initial = 2.0
        T_final = 0.001
        cooling_rate = 0.9999
        max_no_improve = 20000
        
        sa_colors, sa_score = simulated_annealing(init_colors, T_initial, T_final, cooling_rate, max_no_improve)
        
        if sa_score < best_score:
            best_score = sa_score
            best_colors = sa_colors[:]
    
    # Phase 3: Final local search polish on best solution
    if time.time() < deadline - 3:
        final_colors, final_score = local_search(best_colors[:], max_iters=10000)
        if final_score < best_score:
            best_score = final_score
            best_colors = final_colors[:]
    
    return best_colors
