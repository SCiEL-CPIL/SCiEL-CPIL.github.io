import time
import random

def construct():
    N = 256
    deadline = time.time() + 115
    
    # Precompute all 4-APs
    ap_list = []
    pos_in_aps = [[] for _ in range(N)]
    
    for d in range(1, N // 3 + 1):
        for a in range(N - 3 * d):
            i0 = a
            i1 = a + d
            i2 = a + 2 * d
            i3 = a + 3 * d
            ap_idx = len(ap_list)
            ap_list.append((i0, i1, i2, i3))
            pos_in_aps[i0].append(ap_idx)
            pos_in_aps[i1].append(ap_idx)
            pos_in_aps[i2].append(ap_idx)
            pos_in_aps[i3].append(ap_idx)
    
    num_aps = len(ap_list)
    
    # Precompute for each position k, the APs it's in with the other 3 positions
    pos_ap_info = [[] for _ in range(N)]
    for k in range(N):
        for ap_idx in pos_in_aps[k]:
            i0, i1, i2, i3 = ap_list[ap_idx]
            if k == i0:
                others = (i1, i2, i3)
            elif k == i1:
                others = (i0, i2, i3)
            elif k == i2:
                others = (i0, i1, i3)
            else:
                others = (i0, i1, i2)
            pos_ap_info[k].append((ap_idx, others))
    
    def count_mono_from_colors(colors):
        mono = 0
        for (i0, i1, i2, i3) in ap_list:
            c = colors[i0]
            if colors[i1] == c and colors[i2] == c and colors[i3] == c:
                mono += 1
        return mono
    
    def greedy_polish(colors, max_iterations=100):
        """Try all single flips, take best improvement, repeat until no improvement."""
        current_score = count_mono_from_colors(colors)
        for _ in range(max_iterations):
            if time.time() > deadline - 5:
                break
            best_delta = 0
            best_k = -1
            for k in range(N):
                old_color = colors[k]
                new_color = 1 - old_color
                delta = 0
                for (ap_idx, others) in pos_ap_info[k]:
                    co1 = colors[others[0]]
                    co2 = colors[others[1]]
                    co3 = colors[others[2]]
                    old_mono = 1 if (old_color == co1 == co2 == co3) else 0
                    new_mono = 1 if (new_color == co1 == co2 == co3) else 0
                    delta += new_mono - old_mono
                if delta < best_delta:
                    best_delta = delta
                    best_k = k
            if best_k == -1:
                break
            colors[best_k] = 1 - colors[best_k]
            current_score += best_delta
        return current_score
    
    best_score = float('inf')
    best_colors = None
    
    max_restarts = 40
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            colors = [i % 2 for i in range(N)]
        elif restart == 1:
            colors = [1 - (i % 2) for i in range(N)]
        else:
            colors = [random.randint(0, 1) for _ in range(N)]
        
        current_score = count_mono_from_colors(colors)
        
        # Greedy polish first
        current_score = greedy_polish(colors, max_iterations=50)
        
        if current_score < best_score:
            best_score = current_score
            best_colors = colors[:]
        
        # Simulated annealing
        if time.time() > deadline - 10:
            break
        
        T = 2.0
        alpha = 0.999
        max_steps = 10000
        step = 0
        
        while step < max_steps and T > 0.05:
            if time.time() > deadline - 5:
                break
            
            k = random.randint(0, N - 1)
            new_color = 1 - colors[k]
            old_color = colors[k]
            
            delta = 0
            for (ap_idx, others) in pos_ap_info[k]:
                co1 = colors[others[0]]
                co2 = colors[others[1]]
                co3 = colors[others[2]]
                old_mono = 1 if (old_color == co1 == co2 == co3) else 0
                new_mono = 1 if (new_color == co1 == co2 == co3) else 0
                delta += new_mono - old_mono
            
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                colors[k] = new_color
                current_score += delta
                
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors[:]
            
            T *= alpha
            step += 1
        
        # Final greedy polish
        if time.time() > deadline - 5:
            break
        current_score = greedy_polish(colors, max_iterations=20)
        if current_score < best_score:
            best_score = current_score
            best_colors = colors[:]
    
    return best_colors if best_colors is not None else [0] * N
