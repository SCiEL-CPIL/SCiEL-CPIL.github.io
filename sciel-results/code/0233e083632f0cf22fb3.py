import time
import random
import itertools
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    N = 16
    vertices = range(N)
    all_triples = list(itertools.combinations(vertices, 3))
    num_triples = len(all_triples)  # 560
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute K(2,2,2) configurations
    k222_configs = []
    triple_to_k222 = [[] for _ in range(num_triples)]
    
    for six_vertices in itertools.combinations(vertices, 6):
        a, b, c, d, e, f = six_vertices
        pairs_list = [
            ((a,b),(c,d),(e,f)),
            ((a,b),(c,e),(d,f)),
            ((a,b),(c,f),(d,e)),
            ((a,c),(b,d),(e,f)),
            ((a,c),(b,e),(d,f)),
            ((a,c),(b,f),(d,e)),
            ((a,d),(b,c),(e,f)),
            ((a,d),(b,e),(c,f)),
            ((a,d),(b,f),(c,e)),
            ((a,e),(b,c),(d,f)),
            ((a,e),(b,d),(c,f)),
            ((a,e),(b,f),(c,d)),
            ((a,f),(b,c),(d,e)),
            ((a,f),(b,d),(c,e)),
            ((a,f),(b,e),(c,d)),
        ]
        
        for p1, p2, p3 in pairs_list:
            config_triples = []
            for x in p1:
                for y in p2:
                    for z in p3:
                        t = tuple(sorted((x, y, z)))
                        idx = triple_to_idx[t]
                        config_triples.append(idx)
            
            config_id = len(k222_configs)
            k222_configs.append(config_triples)
            
            for t_idx in config_triples:
                triple_to_k222[t_idx].append(config_id)
                
    num_k222 = len(k222_configs)
    
    def greedy_init(rng):
        present = [False] * num_triples
        k222_counts = [0] * num_k222
        count = 0
        
        triple_degree = [len(triple_to_k222[i]) for i in range(num_triples)]
        order = list(range(num_triples))
        rng.shuffle(order)
        order.sort(key=lambda i: triple_degree[i])
        
        for ti in order:
            ok = True
            for cid in triple_to_k222[ti]:
                if k222_counts[cid] >= 7:
                    ok = False
                    break
            if ok:
                present[ti] = True
                count += 1
                for cid in triple_to_k222[ti]:
                    k222_counts[cid] += 1
                    
        return present, k222_counts, count

    def simulated_annealing(present, k222_counts, count, rng, time_limit):
        def count_violations():
            v = 0
            for c in k222_counts:
                if c >= 8:
                    v += 1
            return v
        
        num_viol = count_violations()
        current_score = float(count) - 100.0 * num_viol
        
        T = 5.0
        cooling_rate = 0.99
        min_T = 0.01
        reheat_T = 3.0
        
        best_score = current_score
        best_present = present[:]
        best_count = count
        best_viol = num_viol
        
        steps_since_improve = 0
        max_steps_no_improve = 5000
        
        present_list = [i for i in range(num_triples) if present[i]]
        
        while T > min_T and time.time() < time_limit:
            if steps_since_improve > max_steps_no_improve:
                T = reheat_T
                steps_since_improve = 0
                max_steps_no_improve = int(max_steps_no_improve * 1.2)
            
            move_type = rng.random()
            
            if move_type < 0.35:
                ti = rng.randint(0, num_triples - 1)
                if present[ti]:
                    continue
                
                delta_viol = 0
                for cid in triple_to_k222[ti]:
                    if k222_counts[cid] == 7:
                        delta_viol += 1
                delta_score = 1.0 - 100.0 * delta_viol
                
                if delta_score > 0 or rng.random() < math.exp(delta_score / T):
                    present[ti] = True
                    present_list.append(ti)
                    count += 1
                    for cid in triple_to_k222[ti]:
                        k222_counts[cid] += 1
                    num_viol += delta_viol
                    current_score += delta_score
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_present = present[:]
                        best_count = count
                        best_viol = num_viol
                        steps_since_improve = 0
                    else:
                        steps_since_improve += 1
                        
            elif move_type < 0.70:
                if not present_list:
                    continue
                idx = rng.randint(0, len(present_list) - 1)
                ti = present_list[idx]
                present_list[idx] = present_list[-1]
                present_list.pop()
                
                delta_viol = 0
                for cid in triple_to_k222[ti]:
                    if k222_counts[cid] == 8:
                        delta_viol -= 1
                delta_score = -1.0 - 100.0 * delta_viol
                
                if delta_score > 0 or rng.random() < math.exp(delta_score / T):
                    present[ti] = False
                    count -= 1
                    for cid in triple_to_k222[ti]:
                        k222_counts[cid] -= 1
                    num_viol += delta_viol
                    current_score += delta_score
                    
            else:
                if not present_list:
                    continue
                idx_rm = rng.randint(0, len(present_list) - 1)
                ti_rm = present_list[idx_rm]
                
                ti_add = rng.randint(0, num_triples - 1)
                if present[ti_add]:
                    continue
                
                net_changes = {}
                for cid in triple_to_k222[ti_rm]:
                    net_changes[cid] = net_changes.get(cid, 0) - 1
                for cid in triple_to_k222[ti_add]:
                    net_changes[cid] = net_changes.get(cid, 0) + 1
                net_changes = {k: v for k, v in net_changes.items() if v != 0}
                
                delta_viol = 0
                for cid, delta in net_changes.items():
                    old_count = k222_counts[cid]
                    new_count = old_count + delta
                    if old_count < 8 and new_count >= 8:
                        delta_viol += 1
                    elif old_count >= 8 and new_count < 8:
                        delta_viol -= 1
                delta_score = -100.0 * delta_viol
                
                if delta_score > 0 or rng.random() < math.exp(delta_score / T):
                    present[ti_rm] = False
                    present[ti_add] = True
                    present_list[idx_rm] = ti_add
                    for cid, delta in net_changes.items():
                        k222_counts[cid] += delta
                    num_viol += delta_viol
                    current_score += delta_score
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_present = present[:]
                        best_count = count
                        best_viol = num_viol
                        steps_since_improve = 0
                    else:
                        steps_since_improve += 1
            
            T *= cooling_rate
            
        return best_present, best_count, best_viol

    best_global_count = 0
    best_global_result = []
    
    num_restarts = 20
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 2:
            break
        
        rng = random.Random(12345 + r * 7)
        
        present, k222_counts, count = greedy_init(rng)
        
        if count > best_global_count:
            best_global_count = count
            best_global_result = [list(all_triples[i]) for i in range(num_triples) if present[i]]
        
        sa_time = time.time() + min(time_per_restart * 0.85, deadline - time.time() - 1)
        best_present, best_count, best_viol = simulated_annealing(present, k222_counts, count, rng, sa_time)
        
        if best_count > best_global_count:
            best_global_count = best_count
            best_global_result = [list(all_triples[i]) for i in range(num_triples) if best_present[i]]
    
    if not best_global_result:
        best_global_result = [[0, 1, 2]]
    
    return best_global_result

if __name__ == "__main__":
    res = construct()
    print(f"Number of triples: {len(res)}")
