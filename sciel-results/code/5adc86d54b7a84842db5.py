import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_score = float('inf')
    best_poly = None
    
    def dict_to_list(p):
        if not p:
            return []
        max_deg = max(p.keys())
        res = [0] * (max_deg + 1)
        for deg, coeff in p.items():
            res[deg] = coeff
        return res

    def get_score(p):
        square_coeffs = {}
        keys = list(p.keys())
        for i in keys:
            ci = p[i]
            for j in keys:
                cj = p[j]
                deg = i + j
                square_coeffs[deg] = square_coeffs.get(deg, 0) + ci * cj
        count = 0
        for v in square_coeffs.values():
            if v != 0:
                count += 1
        return count

    def random_poly():
        p = {}
        degrees = random.sample(range(0, 41), 12)
        for d in degrees:
            c = random.randint(-5, 5)
            if c == 0:
                c = 1
            p[d] = c
        return p

    def local_search(p):
        current_score = get_score(p)
        no_improve_count = 0
        
        while time.time() < deadline:
            if no_improve_count > 100:
                break
                
            new_p = p.copy()
            keys = list(new_p.keys())
            idx = random.randint(0, len(keys)-1)
            key = keys[idx]
            
            if random.random() < 0.5:
                old_c = new_p[key]
                new_c = old_c + random.randint(-2, 2)
                if new_c == 0:
                    new_c = 1
                if new_c < -64: new_c = -64
                if new_c > 64: new_c = 64
                new_p[key] = new_c
            else:
                new_d = key + random.randint(-5, 5)
                if new_d < 0: new_d = 0
                if new_d > 40: new_d = 40
                if new_d in new_p:
                    continue
                c = new_p.pop(key)
                new_p[new_d] = c
            
            new_score = get_score(new_p)
            
            if new_score < current_score:
                p = new_p
                current_score = new_score
                no_improve_count = 0
            else:
                no_improve_count += 1
                
        return p, current_score

    while time.time() < deadline:
        p = random_poly()
        p, score = local_search(p)
        
        if score < best_score:
            best_score = score
            best_poly = p

    if best_poly is None:
        return [1] * 12

    return dict_to_list(best_poly)
