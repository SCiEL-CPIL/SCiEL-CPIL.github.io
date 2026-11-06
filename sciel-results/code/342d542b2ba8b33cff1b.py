import random
import time
from itertools import combinations

def construct():
    def solve_for_n(n, time_limit):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        four_sets = list(combinations(range(n), 4))
        five_sets = list(combinations(range(n), 5))
        
        four_set_triples = []
        for s in four_sets:
            a,b,c,d = s
            idxs = [
                triple_to_idx[(a,b,c)],
                triple_to_idx[(a,b,d)],
                triple_to_idx[(a,c,d)],
                triple_to_idx[(b,c,d)]
            ]
            four_set_triples.append(idxs)
            
        five_set_triples = []
        for s in five_sets:
            a,b,c,d,e = s
            idxs = [
                triple_to_idx[(a,b,c)],
                triple_to_idx[(a,b,d)],
                triple_to_idx[(a,b,e)],
                triple_to_idx[(a,c,d)],
                triple_to_idx[(a,c,e)],
                triple_to_idx[(a,d,e)],
                triple_to_idx[(b,c,d)],
                triple_to_idx[(b,c,e)],
                triple_to_idx[(b,d,e)],
                triple_to_idx[(c,d,e)]
            ]
            five_set_triples.append(idxs)
            
        colors = [random.choice([0, 1]) for _ in range(num_triples)]
        
        red_K4_count = 0
        blue_K5_count = 0
        
        four_set_red_counts = [0] * len(four_sets)
        for i, idxs in enumerate(four_set_triples):
            cnt = 0
            for idx in idxs:
                if colors[idx] == 0:
                    cnt += 1
            four_set_red_counts[i] = cnt
            if cnt == 4:
                red_K4_count += 1
                
        five_set_blue_counts = [0] * len(five_sets)
        for i, idxs in enumerate(five_set_triples):
            cnt = 0
            for idx in idxs:
                if colors[idx] == 1:
                    cnt += 1
            five_set_blue_counts[i] = cnt
            if cnt == 10:
                blue_K5_count += 1
                
        triple_to_fours = [[] for _ in range(num_triples)]
        triple_to_fives = [[] for _ in range(num_triples)]
        
        for i, idxs in enumerate(four_set_triples):
            for idx in idxs:
                triple_to_fours[idx].append(i)
                
        for i, idxs in enumerate(five_set_triples):
            for idx in idxs:
                triple_to_fives[idx].append(i)
                
        start_time = time.time()
        current_temp = 1.0
        min_temp = 0.01
        cooling = 0.999
        
        while time.time() - start_time < time_limit:
            if red_K4_count == 0 and blue_K5_count == 0:
                return colors
                
            idx = random.randint(0, num_triples - 1)
            old_color = colors[idx]
            new_color = 1 - old_color
            
            delta_red = 0
            delta_blue = 0
            
            for fs_idx in triple_to_fours[idx]:
                old_cnt = four_set_red_counts[fs_idx]
                if old_color == 0:
                    new_cnt = old_cnt - 1
                    if old_cnt == 4: delta_red -= 1
                    if new_cnt == 4: delta_red += 1
                else:
                    new_cnt = old_cnt + 1
                    if old_cnt == 4: delta_red -= 1
                    if new_cnt == 4: delta_red += 1
                four_set_red_counts[fs_idx] = new_cnt
                
            for fs_idx in triple_to_fives[idx]:
                old_cnt = five_set_blue_counts[fs_idx]
                if old_color == 1:
                    new_cnt = old_cnt - 1
                    if old_cnt == 10: delta_blue -= 1
                    if new_cnt == 10: delta_blue += 1
                else:
                    new_cnt = old_cnt + 1
                    if old_cnt == 10: delta_blue -= 1
                    if new_cnt == 10: delta_blue += 1
                five_set_blue_counts[fs_idx] = new_cnt
                
            delta_cost = delta_red + delta_blue
            
            if delta_cost < 0 or random.random() < pow(2.71828, -delta_cost / current_temp):
                colors[idx] = new_color
                red_K4_count += delta_red
                blue_K5_count += delta_blue
            else:
                for fs_idx in triple_to_fours[idx]:
                    if old_color == 0: four_set_red_counts[fs_idx] += 1
                    else: four_set_red_counts[fs_idx] -= 1
                for fs_idx in triple_to_fives[idx]:
                    if old_color == 1: five_set_blue_counts[fs_idx] += 1
                    else: five_set_blue_counts[fs_idx] -= 1
                
            current_temp *= cooling
            if current_temp < min_temp:
                current_temp = min_temp
                
        return None

    # Try n=34 with most of the time
    colors = solve_for_n(34, 110)
    if colors is not None:
        return {'n': 34, 'colors': colors}
        
    # Fallback to seed
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
