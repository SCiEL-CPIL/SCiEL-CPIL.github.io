import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    def build_color_map(colors, n):
        color_map = {}
        idx = 0
        for a, b, c in itertools.combinations(range(n), 3):
            color_map[(a, b, c)] = colors[idx]
            idx += 1
        return color_map
    
    def find_violations(color_map, n):
        """Find all red K4 and blue K5 violations."""
        red_k4s = []
        blue_k5s = []
        
        for quad in itertools.combinations(range(n), 4):
            a, b, c, d = quad
            if (color_map[(a,b,c)] == 0 and 
                color_map[(a,b,d)] == 0 and 
                color_map[(a,c,d)] == 0 and 
                color_map[(b,c,d)] == 0):
                red_k4s.append(quad)
        
        for quint in itertools.combinations(range(n), 5):
            all_blue = True
            for tri in itertools.combinations(quint, 3):
                if color_map[tri] != 1:
                    all_blue = False
                    break
            if all_blue:
                blue_k5s.append(quint)
        
        return red_k4s, blue_k5s
    
    def is_valid(colors, n):
        color_map = build_color_map(colors, n)
        red_k4s, blue_k5s = find_violations(color_map, n)
        return len(red_k4s) == 0 and len(blue_k5s) == 0
    
    def repair_search(n, rng, max_iter=100):
        """Use repair-based local search."""
        triples = list(itertools.combinations(range(n), 3))
        triple_index = {t: i for i, t in enumerate(triples)}
        colors = [rng.choice([0, 1]) for _ in triples]
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            color_map = build_color_map(colors, n)
            red_k4s, blue_k5s = find_violations(color_map, n)
            
            if not red_k4s and not blue_k5s:
                return colors
            
            # Pick a random violation and try to fix it
            if red_k4s:
                quad = rng.choice(red_k4s)
                # Flip one of the 4 triples in this quad to blue
                tri_options = list(itertools.combinations(quad, 3))
                tri = rng.choice(tri_options)
                idx = triple_index[tuple(sorted(tri))]
                colors[idx] = 1
            else:
                quint = rng.choice(blue_k5s)
                # Flip one of the 10 triples in this quint to red
                tri_options = list(itertools.combinations(quint, 3))
                tri = rng.choice(tri_options)
                idx = triple_index[tuple(sorted(tri))]
                colors[idx] = 0
        
        return None
    
    best_n = 1
    best_colors = None
    
    # Try repair search for various n, starting from larger n
    for n in range(34, 19, -1):
        if time.time() > deadline:
            break
        
        max_attempts = 3 if n > 30 else 5
        success = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            rng = random.Random(hash((n, attempt)) & 0xFFFFFFFF)
            colors = repair_search(n, rng, max_iter=50)
            if colors is not None:
                if n > best_n:
                    best_n = n
                    best_colors = colors
                success = True
                break
        
        if success:
            continue
        else:
            continue
    
    # If no success, try greedy
    if best_colors is None or best_n < 20:
        def greedy_construct(n, rng):
            triples = list(itertools.combinations(range(n), 3))
            triple_index = {t: i for i, t in enumerate(triples)}
            colors = [None] * len(triples)
            
            order = list(range(len(triples)))
            rng.shuffle(order)
            
            for idx in order:
                a, b, c = triples[idx]
                placed = False
                
                for color in [0, 1]:
                    valid = True
                    
                    if color == 0:  # Red
                        for d in range(n):
                            if d == a or d == b or d == c:
                                continue
                            t1 = tuple(sorted((a, b, d)))
                            t2 = tuple(sorted((a, c, d)))
                            t3 = tuple(sorted((b, c, d)))
                            
                            i1 = triple_index.get(t1)
                            i2 = triple_index.get(t2)
                            i3 = triple_index.get(t3)
                            
                            if i1 is not None and i2 is not None and i3 is not None:
                                if colors[i1] == 0 and colors[i2] == 0 and colors[i3] == 0:
                                    valid = False
                                    break
                    
                    elif color == 1:  # Blue
                        for d in range(n):
                            if d == a or d == b or d == c:
                                continue
                            for e in range(d + 1, n):
                                if e == a or e == b or e == c:
                                    continue
                                five = tuple(sorted((a, b, c, d, e)))
                                all_blue = True
                                for tri in itertools.combinations(five, 3):
                                    t = tuple(sorted(tri))
                                    i = triple_index.get(t)
                                    if i is None or colors[i] is None or colors[i] != 1:
                                        all_blue = False
                                        break
                                if all_blue:
                                    valid = False
                                    break
                            if not valid:
                                break
                    
                    if valid:
                        colors[idx] = color
                        placed = True
                        break
                
                if not placed:
                    return None
            
            if is_valid(colors, n):
                return colors
            return None
        
        for n in range(25, 15, -1):
            if time.time() > deadline:
                break
            
            max_attempts = 10
            success = False
            
            for attempt in range(max_attempts):
                if time.time() > deadline:
                    break
                
                rng = random.Random(hash((n, attempt)) & 0xFFFFFFFF)
                colors = greedy_construct(n, rng)
                if colors is not None:
                    if n > best_n:
                        best_n = n
                        best_colors = colors
                    success = True
                    break
            
            if success:
                break
    
    # If still no success, return a small valid coloring
    if best_colors is None:
        n = 9
        return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n), 3)]}
    
    return {'n': best_n, 'colors': best_colors}
