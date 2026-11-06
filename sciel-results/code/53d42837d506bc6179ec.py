import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(9), 3)]
    
    def _build_index(n):
        triples = list(itertools.combinations(range(n), 3))
        idx = {}
        for i, t in enumerate(triples):
            idx[t] = i
        return triples, idx
    
    def _build_mappings(n):
        triples, idx = _build_index(n)
        num_edges = len(triples)
        
        sets4 = list(itertools.combinations(range(n), 4))
        sets5 = list(itertools.combinations(range(n), 5))
        num_sets4 = len(sets4)
        num_sets5 = len(sets5)
        
        edge_to_4sets = [[] for _ in range(num_edges)]
        for s4_idx, s4 in enumerate(sets4):
            for i in range(4):
                triple = s4[:i] + s4[i+1:]
                e_idx = idx[triple]
                edge_to_4sets[e_idx].append(s4_idx)
                
        edge_to_5sets = [[] for _ in range(num_edges)]
        for s5_idx, s5 in enumerate(sets5):
            for triple in itertools.combinations(s5, 3):
                e_idx = idx[triple]
                edge_to_5sets[e_idx].append(s5_idx)
                
        return triples, idx, num_edges, sets4, sets5, num_sets4, num_sets5, edge_to_4sets, edge_to_5sets
    
    def _local_search(colors, n, edge_to_4sets, edge_to_5sets, num_sets4, num_sets5, num_edges, deadline):
        """Efficient local search with violation counts."""
        colors = list(colors)
        
        red_count_4 = [0] * num_sets4
        blue_count_5 = [0] * num_sets5
        
        for e_idx, color in enumerate(colors):
            if color == 0:
                for s4_idx in edge_to_4sets[e_idx]:
                    red_count_4[s4_idx] += 1
            else:
                for s5_idx in edge_to_5sets[e_idx]:
                    blue_count_5[s5_idx] += 1
                    
        viol4 = sum(1 for c in red_count_4 if c == 4)
        viol5 = sum(1 for c in blue_count_5 if c == 10)
        total_viol = viol4 + viol5
        
        if total_viol == 0:
            return colors, 0
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for e_idx in range(num_edges):
                if time.time() > deadline:
                    break
                    
                old_color = colors[e_idx]
                new_color = 1 - old_color
                delta_viol = 0
                
                if old_color == 0:
                    for s4_idx in edge_to_4sets[e_idx]:
                        old_c = red_count_4[s4_idx]
                        new_c = old_c - 1
                        if old_c == 4:
                            delta_viol -= 1
                        if new_c == 4:
                            delta_viol += 1
                        red_count_4[s4_idx] = new_c
                    for s5_idx in edge_to_5sets[e_idx]:
                        old_c = blue_count_5[s5_idx]
                        new_c = old_c + 1
                        if old_c == 10:
                            delta_viol -= 1
                        if new_c == 10:
                            delta_viol += 1
                        blue_count_5[s5_idx] = new_c
                else:
                    for s5_idx in edge_to_5sets[e_idx]:
                        old_c = blue_count_5[s5_idx]
                        new_c = old_c - 1
                        if old_c == 10:
                            delta_viol -= 1
                        if new_c == 10:
                            delta_viol += 1
                        blue_count_5[s5_idx] = new_c
                    for s4_idx in edge_to_4sets[e_idx]:
                        old_c = red_count_4[s4_idx]
                        new_c = old_c + 1
                        if old_c == 4:
                            delta_viol -= 1
                        if new_c == 4:
                            delta_viol += 1
                        red_count_4[s4_idx] = new_c
                        
                total_viol += delta_viol
                
                if total_viol < 0:
                    colors[e_idx] = new_color
                    improved = True
                    if total_viol == 0:
                        return colors, 0
                else:
                    if old_color == 0:
                        for s4_idx in edge_to_4sets[e_idx]:
                            red_count_4[s4_idx] += 1
                        for s5_idx in edge_to_5sets[e_idx]:
                            blue_count_5[s5_idx] -= 1
                    else:
                        for s5_idx in edge_to_5sets[e_idx]:
                            blue_count_5[s5_idx] += 1
                        for s4_idx in edge_to_4sets[e_idx]:
                            red_count_4[s4_idx] -= 1
                    total_viol -= delta_viol
                    
        return colors, total_viol
    
    def _try_structured(n, deadline):
        """Try structured algebraic constructions then refine."""
        try:
            triples, idx, num_edges, sets4, sets5, num_sets4, num_sets5, edge_to_4sets, edge_to_5sets = _build_mappings(n)
        except Exception:
            return None, None
            
        # Try multiple structured initializations
        inits = []
        
        # Init 1: (a+b+c) % 3 == 0 -> red
        c1 = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
        inits.append(c1)
        
        # Init 2: (a+b+c) % 2 == 0 -> red
        c2 = [0 if (a + b + c) % 2 == 0 else 1 for a, b, c in triples]
        inits.append(c2)
        
        # Init 3: Quadratic residue mod 5
        qr5 = set()
        for x in range(5):
            qr5.add((x*x) % 5)
        c3 = [0 if ((a - b) * (b - c) * (c - a)) % 5 in qr5 else 1 for a, b, c in triples]
        inits.append(c3)
        
        # Init 4: Based on differences mod 7
        c4 = [0 if ((a + 2*b + 3*c) % 7) in {0, 1, 2} else 1 for a, b, c in triples]
        inits.append(c4)
        
        # Init 5: Random
        c5 = [random.randint(0, 1) for _ in range(num_edges)]
        inits.append(c5)
        
        # Try each initialization with local search
        for init_colors in inits:
            if time.time() > deadline:
                break
            result, viol = _local_search(init_colors, n, edge_to_4sets, edge_to_5sets, num_sets4, num_sets5, num_edges, deadline)
            if viol == 0:
                return n, result
                
        return None, None
    
    # Try from n=34 down
    for n in range(34, 9, -1):
        if time.time() > deadline - 10.0:
            break
            
        result_n, result_colors = _try_structured(n, deadline - 5.0)
        
        if result_n is not None:
            best_n = result_n
            best_colors = result_colors
            break
    
    return {'n': best_n, 'colors': best_colors}
