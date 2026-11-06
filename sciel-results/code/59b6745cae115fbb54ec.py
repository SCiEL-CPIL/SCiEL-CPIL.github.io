import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    def get_seed():
        n = 9
        return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n), 3)]}
    
    best_result = get_seed()
    
    n = 34
    triples = list(itertools.combinations(range(n), 3))
    num_triples = len(triples)
    triple_idx = {t: i for i, t in enumerate(triples)}
    
    k4_indices = []
    for quad in itertools.combinations(range(n), 4):
        t_indices = [triple_idx[(quad[i], quad[j], quad[k])] for i in range(4) for j in range(i+1, 4) for k in range(j+1, 4)]
        k4_indices.append(t_indices)
        
    k5_indices = []
    for quint in itertools.combinations(range(n), 5):
        t_indices = [triple_idx[(quint[c0], quint[c1], quint[c2])] for c0, c1, c2 in itertools.combinations(range(5), 3)]
        k5_indices.append(t_indices)
        
    triple_to_k4 = [[] for _ in range(num_triples)]
    for i, inds in enumerate(k4_indices):
        for t in inds:
            triple_to_k4[t].append(i)
            
    triple_to_k5 = [[] for _ in range(num_triples)]
    for i, inds in enumerate(k5_indices):
        for t in inds:
            triple_to_k5[t].append(i)
            
    def solve_n(n):
        start_time = time.time()
        triples = list(itertools.combinations(range(n), 3))
        num_triples = len(triples)
        triple_idx = {t: i for i, t in enumerate(triples)}
        
        k4_indices = []
        for quad in itertools.combinations(range(n), 4):
            t_indices = [triple_idx[(quad[i], quad[j], quad[k])] for i in range(4) for j in range(i+1, 4) for k in range(j+1, 4)]
            k4_indices.append(t_indices)
            
        k5_indices = []
        for quint in itertools.combinations(range(n), 5):
            t_indices = [triple_idx[(quint[c0], quint[c1], quint[c2])] for c0, c1, c2 in itertools.combinations(range(5), 3)]
            k5_indices.append(t_indices)
            
        triple_to_k4 = [[] for _ in range(num_triples)]
        for i, inds in enumerate(k4_indices):
            for t in inds:
                triple_to_k4[t].append(i)
                
        triple_to_k5 = [[] for _ in range(num_triples)]
        for i, inds in enumerate(k5_indices):
            for t in inds:
                triple_to_k5[t].append(i)
                
        while time.time() < deadline:
            colors = [random.randint(0, 1) for _ in range(num_triples)]
            is_red_k4 = [False] * len(k4_indices)
            is_blue_k5 = [False] * len(k5_indices)
            red_k4_count = 0
            blue_k5_count = 0
            
            for i, inds in enumerate(k4_indices):
                if all(colors[t] == 0 for t in inds):
                    is_red_k4[i] = True
                    red_k4_count += 1
                    
            for i, inds in enumerate(k5_indices):
                if all(colors[t] == 1 for t in inds):
                    is_blue_k5[i] = True
                    blue_k5_count += 1
                    
            total_viol = red_k4_count + blue_k5_count
            if total_viol == 0:
                return colors
                
            current_temp = 1.0
            cooling_rate = 0.9999
            min_temp = 1e-6
            
            while time.time() < deadline:
                if total_viol == 0:
                    return colors
                    
                t_idx = random.randint(0, num_triples - 1)
                delta = 0
                affected_k4 = []
                affected_k5 = []
                
                for k4_i in triple_to_k4[t_idx]:
                    affected_k4.append(k4_i)
                    if is_red_k4[k4_i]:
                        delta -= 1
                    else:
                        if colors[t_idx] == 1:
                            all_others_red = True
                            for t in k4_indices[k4_i]:
                                if t != t_idx and colors[t] != 0:
                                    all_others_red = False
                                    break
                            if all_others_red:
                                delta += 1
                                
                for k5_i in triple_to_k5[t_idx]:
                    affected_k5.append(k5_i)
                    if is_blue_k5[k5_i]:
                        delta -= 1
                    else:
                        if colors[t_idx] == 0:
                            all_others_blue = True
                            for t in k5_indices[k5_i]:
                                if t != t_idx and colors[t] != 1:
                                    all_others_blue = False
                                    break
                            if all_others_blue:
                                delta += 1
                                
                if delta <= 0 or random.random() < pow(2.71828, -delta / current_temp):
                    colors[t_idx] = 1 - colors[t_idx]
                    total_viol += delta
                    for k4_i in affected_k4:
                        if is_red_k4[k4_i]:
                            is_red_k4[k4_i] = False
                        else:
                            if colors[t_idx] == 0: # It was flipped to 0
                                all_others_red = True
                                for t in k4_indices[k4_i]:
                                    if t != t_idx and colors[t] != 0:
                                        all_others_red = False
                                        break
                                if all_others_red:
                                    is_red_k4[k4_i] = True
                    for k5_i in affected_k5:
                        if is_blue_k5[k5_i]:
                            is_blue_k5[k5_i] = False
                        else:
                            if colors[t_idx] == 1: # It was flipped to 1
                                all_others_blue = True
                                for t in k5_indices[k5_i]:
                                    if t != t_idx and colors[t] != 1:
                                        all_others_blue = False
                                        break
                                if all_others_blue:
                                    is_blue_k5[k5_i] = True
                                    
                current_temp *= cooling_rate
                if current_temp < min_temp:
                    current_temp = min_temp
            # End inner while
        # End outer while (restarts)
        return None

    res = solve_n(34)
    if res:
        return {'n': 34, 'colors': res}
        
    # Fallback to seed
    return best_result
