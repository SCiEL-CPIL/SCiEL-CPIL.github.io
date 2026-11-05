import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    def build_indices(n):
        triples = list(combinations(range(n), 3))
        idx3 = {}
        for i, t in enumerate(triples):
            idx3[t] = i
            
        quads = list(combinations(range(n), 4))
        idx4 = {}
        for i, q in enumerate(quads):
            idx4[q] = i
            
        quints = list(combinations(range(n), 5))
        idx5 = {}
        for i, q in enumerate(quints):
            idx5[q] = i
            
        tri_to_quads = [[] for _ in range(len(triples))]
        tri_to_quints = [[] for _ in range(len(triples))]
        
        for i, (a,b,c) in enumerate(triples):
            for d in range(n):
                if d in (a,b,c): continue
                quad = tuple(sorted((a,b,c,d)))
                tri_to_quads[i].append(idx4[quad])
                
        for q_idx, (a,b,c,d,e) in enumerate(quints):
            t1 = (a,b,c); t2 = (a,b,d); t3 = (a,b,e); t4 = (a,c,d)
            t5 = (a,c,e); t6 = (a,d,e); t7 = (b,c,d); t8 = (b,c,e)
            t9 = (b,d,e); t10 = (c,d,e)
            for t in [t1,t2,t3,t4,t5,t6,t7,t8,t9,t10]:
                tri_to_quints[idx3[t]].append(q_idx)
                
        return triples, idx3, quads, idx4, quints, idx5, tri_to_quads, tri_to_quints

    def try_solve(n, time_limit):
        triples, idx3, quads, idx4, quints, idx5, tri_to_quads, tri_to_quints = build_indices(n)
        num_triples = len(triples)
        num_quads = len(quads)
        num_quints = len(quints)
        
        colors = bytearray(random.getrandbits(1) for _ in range(num_triples))
        
        red_count_4 = [0] * num_quads
        blue_count_5 = [0] * num_quints
        
        for q_idx, (a,b,c,d) in enumerate(quads):
            r = 0
            r += 1 if colors[idx3[(a,b,c)]] == 0 else 0
            r += 1 if colors[idx3[(a,b,d)]] == 0 else 0
            r += 1 if colors[idx3[(a,c,d)]] == 0 else 0
            r += 1 if colors[idx3[(b,c,d)]] == 0 else 0
            red_count_4[q_idx] = r
            
        for q_idx, (a,b,c,d,e) in enumerate(quints):
            b_cnt = 0
            t1 = (a,b,c); t2 = (a,b,d); t3 = (a,b,e); t4 = (a,c,d)
            t5 = (a,c,e); t6 = (a,d,e); t7 = (b,c,d); t8 = (b,c,e)
            t9 = (b,d,e); t10 = (c,d,e)
            for t in [t1,t2,t3,t4,t5,t6,t7,t8,t9,t10]:
                if colors[idx3[t]] == 1:
                    b_cnt += 1
            blue_count_5[q_idx] = b_cnt
            
        viol_red = sum(1 for c in red_count_4 if c == 4)
        viol_blue = sum(1 for c in blue_count_5 if c == 10)
        current_violations = viol_red + viol_blue
        
        if current_violations == 0:
            return list(colors)
            
        T = 1.0
        cooling = 0.9999
        min_T = 0.01
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            if current_violations == 0:
                return list(colors)
                
            i = random.randint(0, num_triples - 1)
            old_color = colors[i]
            new_color = 1 - old_color
            
            delta_viol = 0
            
            for q_idx in tri_to_quads[i]:
                old_cnt = red_count_4[q_idx]
                if old_color == 0:
                    new_cnt = old_cnt - 1
                    if old_cnt == 4:
                        delta_viol -= 1
                    # new_cnt cannot be 4 if old_cnt was 4 and we subtracted
                    # if old_cnt was 3, new is 2, no change
                    # if old_cnt was 5? impossible, max is 4.
                    red_count_4[q_idx] = new_cnt
                else:
                    new_cnt = old_cnt + 1
                    if new_cnt == 4:
                        delta_viol += 1
                    red_count_4[q_idx] = new_cnt
                    
            for q_idx in tri_to_quints[i]:
                old_cnt = blue_count_5[q_idx]
                if old_color == 0:
                    new_cnt = old_cnt + 1
                    if new_cnt == 10:
                        delta_viol += 1
                    blue_count_5[q_idx] = new_cnt
                else:
                    new_cnt = old_cnt - 1
                    if old_cnt == 10:
                        delta_viol -= 1
                    blue_count_5[q_idx] = new_cnt
                        
            new_violations = current_violations + delta_viol
            
            if new_violations < current_violations:
                colors[i] = new_color
                current_violations = new_violations
            else:
                if delta_viol == 0:
                    colors[i] = new_color
                else:
                    prob = math.exp(-delta_viol / T)
                    if random.random() < prob:
                        colors[i] = new_color
                        current_violations = new_violations
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return None

    best_n = 0
    best_colors = []
    
    # Try n=20, 25, 30
    targets = [20, 25, 30]
    for n in targets:
        if time.time() > deadline - 10:
            break
        # Allocate time based on remaining budget
        remaining = deadline - time.time()
        # Spend up to 1/3 of remaining time for each target, but not more than 40s
        t_limit = min(remaining / 3, 40)
        if t_limit < 5:
            break
            
        result = try_solve(n, t_limit)
        if result is not None:
            best_n = n
            best_colors = result
            
    if best_n == 0:
        n = 9
        return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
        
    return {'n': best_n, 'colors': best_colors}
