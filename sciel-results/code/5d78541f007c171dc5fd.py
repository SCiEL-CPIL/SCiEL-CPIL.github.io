import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    def find_coloring(n, time_limit):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_index = {t: i for i, t in enumerate(triples)}
        
        quad_list = list(combinations(range(n), 4))
        quint_list = list(combinations(range(n), 5))
        
        num_quads = len(quad_list)
        num_quints = len(quint_list)
        
        quad_triples = [[] for _ in range(num_quads)]
        triple_to_quads = [[] for _ in range(num_triples)]
        
        for qi, q in enumerate(quad_list):
            t1 = (q[0], q[1], q[2])
            t2 = (q[0], q[1], q[3])
            t3 = (q[0], q[2], q[3])
            t4 = (q[1], q[2], q[3])
            
            ti1 = triple_index[t1]
            ti2 = triple_index[t2]
            ti3 = triple_index[t3]
            ti4 = triple_index[t4]
            
            quad_triples[qi] = [ti1, ti2, ti3, ti4]
            
            triple_to_quads[ti1].append(qi)
            triple_to_quads[ti2].append(qi)
            triple_to_quads[ti3].append(qi)
            triple_to_quads[ti4].append(qi)
            
        quint_triples = [[] for _ in range(num_quints)]
        triple_to_quints = [[] for _ in range(num_triples)]
        
        for qi, q in enumerate(quint_list):
            verts = list(q)
            ts = []
            for i in range(5):
                for j in range(i+1, 5):
                    for k in range(j+1, 5):
                        t = (verts[i], verts[j], verts[k])
                        ts.append(triple_index[t])
            quint_triples[qi] = ts
            
            for ti in ts:
                triple_to_quints[ti].append(qi)
                
        best_colors = None
        
        for attempt in range(30):
            if time.time() > time_limit:
                break
                
            if attempt % 2 == 0:
                colors = [random.randint(0, 1) for _ in range(num_triples)]
            else:
                colors = [0 if (a+b+c) % 3 == 0 else 1 for a,b,c in triples]
                
            red_violations = 0
            blue_violations = 0
            
            for qi in range(num_quads):
                ts = quad_triples[qi]
                if colors[ts[0]] == 0 and colors[ts[1]] == 0 and colors[ts[2]] == 0 and colors[ts[3]] == 0:
                    red_violations += 1
                    
            for qi in range(num_quints):
                ts = quint_triples[qi]
                if all(colors[t] == 1 for t in ts):
                    blue_violations += 1
                    
            total_violations = red_violations + blue_violations
            
            if total_violations == 0:
                return colors
                
            quad_red_count = [0] * num_quads
            quint_blue_count = [0] * num_quints
            
            for qi in range(num_quads):
                ts = quad_triples[qi]
                quad_red_count[qi] = sum(1 for t in ts if colors[t] == 0)
                
            for qi in range(num_quints):
                ts = quint_triples[qi]
                quint_blue_count[qi] = sum(1 for t in ts if colors[t] == 1)
                
            max_iters = 10000
            for _ in range(max_iters):
                if time.time() > time_limit:
                    break
                    
                if total_violations == 0:
                    return colors
                    
                ti = random.randint(0, num_triples - 1)
                
                delta_red_viol = 0
                delta_blue_viol = 0
                
                if colors[ti] == 0:
                    for qi in triple_to_quads[ti]:
                        if quad_red_count[qi] == 4:
                            delta_red_viol -= 1
                    for qi in triple_to_quints[ti]:
                        if quint_blue_count[qi] == 9:
                            delta_blue_viol += 1
                else:
                    for qi in triple_to_quads[ti]:
                        if quad_red_count[qi] == 3:
                            delta_red_viol += 1
                    for qi in triple_to_quints[ti]:
                        if quint_blue_count[qi] == 10:
                            delta_blue_viol -= 1
                            
                delta_total = delta_red_viol + delta_blue_viol
                
                if delta_total <= 0:
                    colors[ti] ^= 1
                    total_violations += delta_total
                    
                    if colors[ti] == 1:
                        for qi in triple_to_quads[ti]:
                            quad_red_count[qi] -= 1
                        for qi in triple_to_quints[ti]:
                            quint_blue_count[qi] += 1
                    else:
                        for qi in triple_to_quads[ti]:
                            quad_red_count[qi] += 1
                        for qi in triple_to_quints[ti]:
                            quint_blue_count[qi] -= 1
                            
                    if total_violations == 0:
                        return colors
                            
        return None

    for n in range(40, 9, -1):
        if time.time() > deadline - 5.0:
            break
            
        t_limit = time.time() + 2.0
        colors = find_coloring(n, t_limit)
        if colors is not None:
            return {'n': n, 'colors': colors}
            
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
